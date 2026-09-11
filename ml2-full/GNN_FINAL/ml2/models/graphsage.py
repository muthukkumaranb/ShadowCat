import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, embedding_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Additional layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
        # Final layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, embedding_dim))
        else:
            self.convs[0] = SAGEConv(in_channels, embedding_dim)
            
    def forward(self, x, edge_index):
        """
        x: [N, in_channels]
        edge_index: [2, E]
        Returns: h_v [N, embedding_dim]
        """
        h = x
        for i in range(self.num_layers - 1):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
        # Output layer (no activation or dropout, typical for embeddings)
        h = self.convs[-1](h, edge_index)
        return h

class GraphBranch(nn.Module):
    def __init__(self, in_channels: int = 11, hidden_dim: int = 64, embedding_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        """
        Wraps the GraphSAGE encoder and provides graph-level pooling.
        """
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_dim, embedding_dim, num_layers, dropout)
        
    def forward(self, x, edge_index, batch=None):
        """
        x: Node features [N, F]
        edge_index: Graph connectivity [2, E]
        batch: Batch vector [N] assigning each node to a graph in the batch
        
        Returns: 
        h_v: node embeddings [N, embedding_dim]
        g: pooled graph embedding [num_graphs, embedding_dim]
        """
        h_v = self.encoder(x, edge_index)
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # Mean pooling is deterministic, interpretable, and mitigates overfitting on small corpora
        g = global_mean_pool(h_v, batch)
        
        return h_v, g
