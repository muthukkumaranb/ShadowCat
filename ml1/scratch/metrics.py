"""
Dynamics metrics for ML2 evaluation.

These are the exact metrics used in the fused-vs-temporal-only ablation:
    1. next_state_error: 1-step MSE between predicted and actual next embedding
    2. rollout_error: K-step autoregressive accumulated MSE vs ground truth

Both operate on graph-level pooled embeddings (not node-level).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader


@torch.no_grad()
def next_state_error(
    model: nn.Module,
    pairs: List[Tuple[Data, Data]],
    z_dim: int = 64,
    device: str = "cpu",
) -> float:
    """
    Computes 1-step next-state prediction MSE.

    For each (graph_t, graph_{t+1}) pair:
        pred = model(graph_t, z_t)
        target = global_mean_pool(graph_{t+1}.x)   [fixed raw-feature target]
        error += MSE(pred, target)

    Target is always mean-pooled raw node features (fixed 11-dim space).
    This ensures no self-referential target and identical target space
    for all model variants (fused, temporal-only, graph-only).

    Args:
        model: A model with .forward(x, edge_index, z_t, batch) -> [B, output_dim].
        pairs: List of (graph_t, graph_{t+1}) tuples.
        z_dim: Dimension of the temporal embedding z_t.
        device: Device to run on.

    Returns:
        Mean MSE across all pairs.
    """
    model.eval()
    total_mse = 0.0
    count = 0

    for graph_t, graph_tp1 in pairs:
        graph_t = graph_t.to(device)
        graph_tp1 = graph_tp1.to(device)

        # Mock z_t (in real pipeline, comes from ML1)
        z_t = torch.zeros(1, z_dim, device=device)

        # Prediction
        batch_t = torch.zeros(graph_t.x.size(0), dtype=torch.long, device=device)
        pred = model(graph_t.x, graph_t.edge_index, z_t, batch_t)

        # Target: mean-pooled raw node features of graph_{t+1}
        batch_tp1 = torch.zeros(graph_tp1.x.size(0), dtype=torch.long, device=device)
        target = global_mean_pool(graph_tp1.x, batch_tp1)

        mse = torch.nn.functional.mse_loss(pred, target).item()
        total_mse += mse
        count += 1

    return total_mse / max(count, 1)


@torch.no_grad()
def rollout_error(
    model: nn.Module,
    graphs: List[Data],
    K: int,
    z_dim: int = 64,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Computes K-step autoregressive rollout error.

    Starting from graph_t, autoregressively predicts state for K steps:
        pred_1 = model(graph_t.x, graph_t.edge_index, z_t, batch)
        pred_2 = model.projection([pred_1 ; z_t])   (autoregressive)
        ...
        pred_K = model.projection([pred_{K-1} ; z_t])

    Each pred_k is compared against the mean-pooled raw node features
    of graph_{t+k} (fixed target space, no learned component).

    Args:
        model: A model with .forward(x, edge_index, z_t, batch) -> predicted state.
        graphs: Chronologically ordered list of PyG Data objects.
        K: Number of rollout steps.
        z_dim: Dimension of ML1's temporal embedding.
        device: Device to run on.

    Returns:
        Dict with:
            'per_step_mse': list of MSE at each step k=1..K
            'cumulative_mse': accumulated MSE over K steps
            'mean_mse': mean of per-step MSE
    """
    model.eval()

    if len(graphs) <= K:
        return {
            "per_step_mse": [float("nan")] * K,
            "cumulative_mse": float("nan"),
            "mean_mse": float("nan"),
        }

    def get_target_embedding(graph: Data) -> torch.Tensor:
        """Get mean-pooled raw node features as the fixed target."""
        graph = graph.to(device)
        batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
        return global_mean_pool(graph.x, batch)

    per_step_errors = [[] for _ in range(K)]

    # Slide the rollout window
    for start_idx in range(len(graphs) - K):
        graph_t = graphs[start_idx].to(device)
        batch_t = torch.zeros(graph_t.x.size(0), dtype=torch.long, device=device)
        z_t = torch.zeros(1, z_dim, device=device)

        # Initial prediction (step 1)
        current_pred = model(graph_t.x, graph_t.edge_index, z_t, batch_t)

        for k in range(K):
            target_graph = graphs[start_idx + k + 1]
            target_emb = get_target_embedding(target_graph)

            mse = torch.nn.functional.mse_loss(current_pred, target_emb).item()
            per_step_errors[k].append(mse)

            # Autoregressive: use current prediction as input for next step
            # Graph is frozen — we do NOT re-run graph_branch on future graphs.
            if k < K - 1:
                fused_input = torch.cat([current_pred, z_t], dim=-1)
                if hasattr(model, "rollout_projection"):
                    current_pred = model.rollout_projection(fused_input)
                elif hasattr(model, "projection"):
                    current_pred = model.projection(fused_input)

    per_step_mse = [
        np.mean(errors) if errors else float("nan") for errors in per_step_errors
    ]
    cumulative_mse = sum(m for m in per_step_mse if not np.isnan(m))
    mean_mse = np.nanmean(per_step_mse)

    return {
        "per_step_mse": per_step_mse,
        "cumulative_mse": float(cumulative_mse),
        "mean_mse": float(mean_mse),
    }


def compute_all_rollout_errors(
    model: nn.Module,
    graphs: List[Data],
    K_values: List[int] = None,
    z_dim: int = 64,
    device: str = "cpu",
) -> Dict[int, Dict[str, float]]:
    """
    Computes rollout error for multiple values of K.

    Args:
        model: The model to evaluate.
        graphs: Chronologically ordered graphs.
        K_values: List of rollout horizons. Defaults to [1, 2, 3, 5, 10].
        z_dim: ML1 temporal embedding dimension.
        device: Device to run on.

    Returns:
        Dict mapping K → rollout error results.
    """
    if K_values is None:
        K_values = [1, 2, 3, 5, 10]

    results = {}
    for K in K_values:
        if K >= len(graphs):
            results[K] = {
                "per_step_mse": [float("nan")] * K,
                "cumulative_mse": float("nan"),
                "mean_mse": float("nan"),
            }
        else:
            results[K] = rollout_error(model, graphs, K, z_dim, device)

    return results
