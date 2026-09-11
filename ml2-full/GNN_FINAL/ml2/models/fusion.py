"""
Fusion models for combining ML2's graph embeddings with ML1's temporal embeddings.

Two variants for ablation:
    - FusedModel: concatenates graph + temporal embeddings → projection head
    - TemporalOnlyBaseline: uses only temporal embedding (no graph component)

Both expose an identical .forward() interface so they can be swapped in the
ablation harness without any calling-code changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml2.models.graphsage import GraphBranch


class FusedModel(nn.Module):
    """
    Fuses ML2's graph-level embedding g_t with ML1's temporal embedding z_t.

    Architecture:
        g_t = GraphBranch(x, edge_index, batch)           → [B, graph_dim]
        z_t = ML1 temporal embedding                       → [B, z_dim]
        h   = [g_t ∥ z_t]                                  → [B, graph_dim + z_dim]
        out = MLP(h)                                       → [B, output_dim]

    The output is a predicted next-state embedding, used with MSE loss for
    dynamics prediction (next-state error metric).
    """

    def __init__(
        self,
        node_feature_dim: int = 11,
        graph_hidden_dim: int = 64,
        graph_embedding_dim: int = 64,
        graph_num_layers: int = 2,
        graph_dropout: float = 0.3,
        z_dim: int = 64,
        fusion_hidden_dim: int = 128,
        output_dim: int = 11,
    ):
        super().__init__()

        self.graph_branch = GraphBranch(
            in_channels=node_feature_dim,
            hidden_dim=graph_hidden_dim,
            embedding_dim=graph_embedding_dim,
            num_layers=graph_num_layers,
            dropout=graph_dropout,
        )

        fused_dim = graph_embedding_dim + z_dim

        self.projection = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, output_dim),
        )

        # Rollout projection: for autoregressive multi-step rollout,
        # the input is [pred(output_dim) ; z_t(z_dim)], NOT [g(graph_dim) ; z_t].
        rollout_dim = output_dim + z_dim
        self.rollout_projection = nn.Sequential(
            nn.Linear(rollout_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, output_dim),
        )

        self._graph_embedding_dim = graph_embedding_dim
        self._z_dim = z_dim
        self._output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        z_t: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N, F].
            edge_index: Graph connectivity [2, E].
            z_t: ML1 temporal embedding [B, z_dim].
            batch: Batch vector [N], assigning each node to a graph.

        Returns:
            Predicted next-state [B, output_dim] (raw-feature space).
        """
        _, g_t = self.graph_branch(x, edge_index, batch)  # [B, graph_dim]
        fused = torch.cat([g_t, z_t], dim=-1)             # [B, graph_dim + z_dim]
        return self.projection(fused)                      # [B, output_dim]

    @property
    def model_type(self) -> str:
        return "fused"


class TemporalOnlyBaseline(nn.Module):
    """
    Baseline that uses ONLY ML1's temporal embedding z_t (no graph component).
    Matched architecture (same MLP capacity) so the ablation is fair.
    """

    def __init__(
        self,
        z_dim: int = 64,
        fusion_hidden_dim: int = 128,
        output_dim: int = 11,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(z_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, output_dim),
        )

        # Rollout projection: for autoregressive multi-step rollout,
        # the input is [pred(output_dim) ; z_t(z_dim)].
        rollout_dim = output_dim + z_dim
        self.rollout_projection = nn.Sequential(
            nn.Linear(rollout_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, output_dim),
        )

        self._z_dim = z_dim
        self._output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        z_t: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Same signature as FusedModel.forward() -- graph inputs are accepted
        but ignored, enabling drop-in swapping for ablation.

        Args:
            x: Node features [N, F] (IGNORED).
            edge_index: Graph connectivity [2, E] (IGNORED).
            z_t: ML1 temporal embedding [B, z_dim].
            batch: Batch vector [N] (IGNORED).

        Returns:
            Predicted next-state [B, output_dim] (raw-feature space).
        """
        return self.projection(z_t)

    @property
    def model_type(self) -> str:
        return "temporal_only"


class GraphOnlyBaseline(nn.Module):
    """
    Additional baseline: uses ONLY the graph embedding g_t (no temporal component).
    Useful to quantify the isolated contribution of the graph branch.
    """

    def __init__(
        self,
        node_feature_dim: int = 11,
        graph_hidden_dim: int = 64,
        graph_embedding_dim: int = 64,
        graph_num_layers: int = 2,
        graph_dropout: float = 0.3,
        fusion_hidden_dim: int = 128,
        output_dim: int = 11,
    ):
        super().__init__()

        self.graph_branch = GraphBranch(
            in_channels=node_feature_dim,
            hidden_dim=graph_hidden_dim,
            embedding_dim=graph_embedding_dim,
            num_layers=graph_num_layers,
            dropout=graph_dropout,
        )

        self.projection = nn.Sequential(
            nn.Linear(graph_embedding_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, output_dim),
        )

        # Rollout projection (output_dim + z_dim input for autoregressive steps)
        rollout_dim = output_dim + 64  # z_dim default
        self.rollout_projection = nn.Sequential(
            nn.Linear(rollout_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        z_t: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Same signature -- z_t is accepted but ignored.
        """
        _, g_t = self.graph_branch(x, edge_index, batch)
        return self.projection(g_t)

    @property
    def model_type(self) -> str:
        return "graph_only"
