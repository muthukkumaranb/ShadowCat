"""
GraphSAGE-based graph encoder for cyber attack risk forecasting.

Architecture (World Model contract):
    UCS(t) → Graph Construction → GraphEncoder → 64-D Graph Embedding

Primary interface:
    graph_embedding = encoder.encode(graph)
    shape: [batch_size, embedding_dim]  (default embedding_dim=64)

The encoder produces a REPRESENTATION — it is NOT the final classifier.
Downstream Fusion combines graph_embedding with temporal_embedding for the World Model.

Edge Feature Handling:
    Standard SAGEConv does not consume edge features. We implement EdgeAwareSAGEConv
    which concatenates edge features with source node features before aggregation,
    effectively making edge information part of the message-passing computation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_remaining_self_loops
from typing import Optional, Dict, Any, List


class EdgeAwareSAGEConv(MessagePassing):
    """
    GraphSAGE-style convolution that incorporates edge features.

    Standard SAGEConv:
        h_v^(l+1) = W_self * h_v^(l)  +  W_neigh * AGG({h_u^(l) : u ∈ N(v)})

    EdgeAwareSAGEConv:
        h_v^(l+1) = W_self * h_v^(l)  +  W_neigh * AGG({MLP_edge([h_u^(l) || e_uv]) : u ∈ N(v)})

    The edge MLP transforms [source_features || edge_features] into a message vector
    before mean aggregation, allowing edge information (flow count, bytes, packets,
    protocol ratios) to influence neighbor representations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        normalize: bool = True,
        bias: bool = True,
    ):
        """
        Args:
            in_channels: Input node feature dimension
            out_channels: Output node feature dimension
            edge_dim: Edge feature dimension (0 means no edge features)
            normalize: Whether to L2-normalize output embeddings
            bias: Whether to use bias in linear layers
        """
        super().__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.normalize = normalize

        # Self-transform
        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)

        # Neighbor message transform
        if edge_dim > 0:
            # Edge-aware: concatenate source node features + edge features → message
            self.edge_mlp = nn.Sequential(
                nn.Linear(in_channels + edge_dim, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
        else:
            # No edge features: standard SAGEConv behavior
            self.lin_neigh = nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_self.reset_parameters()
        if self.edge_dim > 0:
            for layer in self.edge_mlp:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        else:
            self.lin_neigh.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim] or None

        Returns:
            Updated node features [N, out_channels]
        """
        # Self-representation
        out_self = self.lin_self(x)

        # Neighbor aggregation (with edge features in messages)
        out_neigh = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        out = out_self + out_neigh

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

    def message(self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Construct messages from source nodes to target nodes.

        Args:
            x_j: Source node features [E, in_channels]
            edge_attr: Edge features [E, edge_dim] or None

        Returns:
            Messages [E, out_channels]
        """
        if self.edge_dim > 0 and edge_attr is not None:
            # Concatenate source features with edge features
            msg_input = torch.cat([x_j, edge_attr], dim=-1)  # [E, in_channels + edge_dim]
            return self.edge_mlp(msg_input)  # [E, out_channels]
        else:
            if hasattr(self, 'lin_neigh'):
                return self.lin_neigh(x_j)
            else:
                return self.edge_mlp(x_j)


class GraphEncoder(nn.Module):
    """
    Primary graph representation encoder.

    Contract:
        G_t → h_graph_t  (shape: [batch_size, embedding_dim])

    Architecture:
        Node Features + Edge Connectivity + Edge Features
            ↓
        EdgeAwareSAGEConv layers (message passing with edge features)
            ↓
        Node Embeddings
            ↓
        Global Mean Pool + Global Max Pool
            ↓
        Concatenate → Graph Embedding (default: 64-D)

    This encoder is the GNN component's primary output interface.
    It does NOT classify attacks — that is the World Model's responsibility.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int = 0,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        pooling: str = "mean_max",
    ):
        """
        Args:
            node_feature_dim: Number of input node features (F_node)
            edge_feature_dim: Number of edge features (F_edge), 0 = no edge features
            hidden_dim: Hidden dimension for GraphSAGE layers
            embedding_dim: Final graph embedding dimension (default: 64)
            num_layers: Number of message-passing layers
            dropout: Dropout rate
            pooling: 'mean', 'max', or 'mean_max' (concatenated)
        """
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # Message-passing layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer: node_feature_dim → hidden_dim
        self.convs.append(EdgeAwareSAGEConv(
            in_channels=node_feature_dim,
            out_channels=hidden_dim,
            edge_dim=edge_feature_dim,
        ))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Subsequent layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(EdgeAwareSAGEConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_dim=edge_feature_dim,
            ))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Pooling output dimension
        if pooling == "mean_max":
            pool_out_dim = hidden_dim * 2
        else:
            pool_out_dim = hidden_dim

        # Projection to final embedding dimension
        # If pool_out_dim != embedding_dim, add a projection layer
        if pool_out_dim != embedding_dim:
            self.projection = nn.Sequential(
                nn.Linear(pool_out_dim, embedding_dim),
                nn.ReLU(),
            )
        else:
            self.projection = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass: nodes → graph embedding.

        Args:
            x: Node features [N_total, node_feature_dim]
            edge_index: Edge indices [2, E_total]
            batch: Batch assignment vector [N_total]
            edge_attr: Edge features [E_total, edge_feature_dim] or None

        Returns:
            Graph embedding [batch_size, embedding_dim]
        """
        return self.encode(x, edge_index, batch, edge_attr)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Primary interface: compute graph-level embedding.

        This is the key output of the GNN component.
        Another engineer can call this without understanding PyG internals.

        Args:
            x: Node features [N_total, node_feature_dim]
            edge_index: Edge indices [2, E_total]
            batch: Batch assignment vector [N_total]
            edge_attr: Edge features [E_total, edge_feature_dim] or None

        Returns:
            Graph embedding tensor of shape [batch_size, embedding_dim]
        """
        # Message passing layers
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, edge_attr=edge_attr)
            h = self.bns[i](h)
            h = F.relu(h)
            if i < self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Global pooling: node embeddings → graph embedding
        if self.pooling == "mean":
            graph_emb = global_mean_pool(h, batch)
        elif self.pooling == "max":
            graph_emb = global_max_pool(h, batch)
        elif self.pooling == "mean_max":
            mean_pool = global_mean_pool(h, batch)
            max_pool = global_max_pool(h, batch)
            graph_emb = torch.cat([mean_pool, max_pool], dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Project to final embedding dimension
        graph_emb = self.projection(graph_emb)

        return graph_emb

    def encode_single(self, data: Data) -> torch.Tensor:
        """
        Convenience method for encoding a single graph (no batching needed).

        Args:
            data: A single PyG Data object

        Returns:
            Graph embedding [1, embedding_dim]
        """
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        return self.encode(data.x, data.edge_index, batch, edge_attr=data.edge_attr)


class DiagnosticClassifier(nn.Module):
    """
    Optional diagnostic snapshot-risk classifier head.

    This is NOT the World Model's final attack prediction.
    It exists solely for ablation experiments and diagnostic evaluation.

    Architecture:
        Graph Embedding (from GraphEncoder)
            ↓
        MLP → logit → sigmoid → P(attack)
    """

    def __init__(self, embedding_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_embedding: [batch_size, embedding_dim]

        Returns:
            Logits [batch_size, 1]
        """
        return self.head(graph_embedding)


class SnapshotGNN(nn.Module):
    """
    Combined encoder + diagnostic classifier for training/evaluation.

    This wraps GraphEncoder + DiagnosticClassifier for convenience during training.
    The primary interface for downstream use remains GraphEncoder.encode().
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int = 0,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        pooling: str = "mean_max",
    ):
        super().__init__()
        self.encoder = GraphEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
        )
        self.classifier = DiagnosticClassifier(
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass returning classification logits."""
        graph_emb = self.encoder.encode(x, edge_index, batch, edge_attr=edge_attr)
        return self.classifier(graph_emb)

    def get_graph_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get graph embedding (for downstream use)."""
        return self.encoder.encode(x, edge_index, batch, edge_attr=edge_attr)

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return probabilities instead of logits."""
        logits = self.forward(x, edge_index, batch, edge_attr=edge_attr)
        return torch.sigmoid(logits)


# ── Backward Compatibility Aliases ────────────────────────────────────────────
# These allow existing code (run_experiments.py, checkpoints) to work
# while migrating to the new architecture.

AttackRiskGNN = SnapshotGNN


class TemporalAttackGNN(nn.Module):
    """
    OPTIONAL temporal GNN for sequence-based forecasting experiments.

    This is a diagnostic/experimental model that aggregates graph embeddings
    from consecutive snapshots. It does NOT replace the LSTM/Fusion system.

    Architecture:
        [G_t-29, ..., G_t] → GraphEncoder → [h_t-29, ..., h_t] → Temporal Aggregation → MLP → P(attack)
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int = 0,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        pooling: str = "mean_max",
        temporal_method: str = "mean",
    ):
        super().__init__()
        self.temporal_method = temporal_method
        self.embedding_dim = embedding_dim

        self.gnn = SnapshotGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
        )

        # Temporal classifier operates on aggregated embeddings
        self.temporal_classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def get_sequence_embedding(self, graph_sequence: List[Data]) -> torch.Tensor:
        """
        Compute temporal embedding from a sequence of graphs.

        Args:
            graph_sequence: List of PyG Data objects (one per timestep)

        Returns:
            Temporal embedding tensor [1, embedding_dim]
        """
        embeddings = []
        for data in graph_sequence:
            emb = self.gnn.encoder.encode_single(data)
            embeddings.append(emb)

        stacked = torch.stack(embeddings, dim=0).squeeze(1)  # [W, embedding_dim]

        if self.temporal_method == "mean":
            temporal_emb = stacked.mean(dim=0, keepdim=True)
        elif self.temporal_method == "last":
            temporal_emb = stacked[-1:, :]
        elif self.temporal_method == "attention":
            attn_weights = F.softmax(stacked.norm(dim=1), dim=0)
            temporal_emb = (stacked * attn_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
        else:
            temporal_emb = stacked.mean(dim=0, keepdim=True)

        return temporal_emb

    def forward(self, graph_sequence: List[Data]) -> torch.Tensor:
        """Forward pass for temporal forecasting."""
        temporal_emb = self.get_sequence_embedding(graph_sequence)
        return self.temporal_classifier(temporal_emb)

    def predict_proba(self, graph_sequence: List[Data]) -> torch.Tensor:
        return torch.sigmoid(self.forward(graph_sequence))
