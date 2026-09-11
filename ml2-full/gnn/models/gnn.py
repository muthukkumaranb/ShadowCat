"""
GNN model: GraphSAGE-based architecture for graph-level attack risk prediction.

Architecture:
    Graph → GraphSAGE layers → Node embeddings → Global pooling → MLP → Sigmoid → P(attack)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder that produces node embeddings from graph structure + features.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            if i < self.num_layers - 1:  # no dropout on last layer
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class AttackRiskGNN(nn.Module):
    """
    Full GNN model for graph-level binary attack risk prediction.

    Pipeline:
        1. GraphSAGE encoder → node embeddings
        2. Global pooling (mean + max) → graph embedding
        3. MLP classifier → logit → sigmoid → P(attack)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        pooling: str = "mean_max",
    ):
        """
        Args:
            in_channels: Number of input node features
            hidden_channels: GNN hidden dimension
            num_layers: Number of GraphSAGE layers
            dropout: Dropout rate
            pooling: 'mean', 'max', or 'mean_max' (concatenated)
        """
        super().__init__()
        self.pooling = pooling
        self.encoder = GraphSAGEEncoder(
            in_channels, hidden_channels, num_layers, dropout
        )

        # Pooling output dimension
        if pooling == "mean_max":
            pool_dim = hidden_channels * 2
        else:
            pool_dim = hidden_channels

        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def get_graph_embedding(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute graph-level embedding (for downstream temporal models).

        Returns:
            Graph embedding tensor of shape (batch_size, embed_dim)
        """
        node_emb = self.encoder(x, edge_index)

        if self.pooling == "mean":
            graph_emb = global_mean_pool(node_emb, batch)
        elif self.pooling == "max":
            graph_emb = global_max_pool(node_emb, batch)
        elif self.pooling == "mean_max":
            mean_pool = global_mean_pool(node_emb, batch)
            max_pool = global_max_pool(node_emb, batch)
            graph_emb = torch.cat([mean_pool, max_pool], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return graph_emb

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: graph → logit.

        Args:
            x: Node features (N_total, F)
            edge_index: Edge indices (2, E_total)
            batch: Batch assignment vector (N_total,)

        Returns:
            Logits of shape (batch_size, 1)
        """
        graph_emb = self.get_graph_embedding(x, edge_index, batch)
        logits = self.classifier(graph_emb)
        return logits

    def predict_proba(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Return probabilities instead of logits."""
        logits = self.forward(x, edge_index, batch)
        return torch.sigmoid(logits)


class TemporalAttackGNN(nn.Module):
    """
    Temporal GNN for future attack forecasting from graph sequences.

    Pipeline:
        1. Per-snapshot GNN → graph embedding
        2. Temporal aggregation (mean pooling over sequence) → temporal embedding
        3. MLP → logit → P(future attack)

    This is deliberately simple — a team LSTM/GRU can replace step 2.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        pooling: str = "mean_max",
        temporal_method: str = "mean",
    ):
        super().__init__()
        self.temporal_method = temporal_method

        self.gnn = AttackRiskGNN(
            in_channels, hidden_channels, num_layers, dropout, pooling
        )

        # Embedding dimension from the GNN
        if pooling == "mean_max":
            embed_dim = hidden_channels * 2
        else:
            embed_dim = hidden_channels

        # Temporal classifier
        self.temporal_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def get_sequence_embedding(self, graph_sequence: list) -> torch.Tensor:
        """
        Compute temporal embedding from a sequence of graphs.

        Args:
            graph_sequence: List of PyG Data objects (one per timestep)

        Returns:
            Temporal embedding tensor (1, embed_dim)
        """
        embeddings = []
        for data in graph_sequence:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
            emb = self.gnn.get_graph_embedding(data.x, data.edge_index, batch)
            embeddings.append(emb)

        # Stack and aggregate temporally
        stacked = torch.stack(embeddings, dim=0)  # (W, 1, D)
        stacked = stacked.squeeze(1)  # (W, D)

        if self.temporal_method == "mean":
            temporal_emb = stacked.mean(dim=0, keepdim=True)  # (1, D)
        elif self.temporal_method == "last":
            temporal_emb = stacked[-1:, :]  # (1, D)
        elif self.temporal_method == "attention":
            # Simple learnable attention
            attn_weights = F.softmax(stacked.norm(dim=1), dim=0)  # (W,)
            temporal_emb = (stacked * attn_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
        else:
            temporal_emb = stacked.mean(dim=0, keepdim=True)

        return temporal_emb

    def forward(self, graph_sequence: list) -> torch.Tensor:
        """
        Forward pass for temporal forecasting.

        Args:
            graph_sequence: List of PyG Data objects

        Returns:
            Logit of shape (1, 1)
        """
        temporal_emb = self.get_sequence_embedding(graph_sequence)
        logits = self.temporal_classifier(temporal_emb)
        return logits

    def predict_proba(self, graph_sequence: list) -> torch.Tensor:
        logits = self.forward(graph_sequence)
        return torch.sigmoid(logits)
