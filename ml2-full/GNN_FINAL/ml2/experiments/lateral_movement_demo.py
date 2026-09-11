"""
Lateral-Movement Case Study (Demo-Only, n=1).

This is a qualitative demonstration — no trained detection head, no Precision@K.
It runs a single episode through the trained GraphSAGE encoder and visualizes:
    1. Graph snapshots over time (node-link diagrams)
    2. Embedding trajectory (how the graph embedding evolves)
    3. Anomaly score via embedding velocity (||g_t - g_{t-1}||)

The purpose is to show that graph-structural changes during lateral movement
produce visually distinct embedding shifts, providing evidence that the
graph branch captures meaningful structural dynamics.
"""

import torch
import numpy as np
import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from ml2.models.graphsage import GraphBranch
from ml2.data.dataset import SnapshotDataset

logger = logging.getLogger(__name__)


def compute_embedding_trajectory(
    model: torch.nn.Module,
    graphs: List[Data],
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Computes the graph-level embedding for each snapshot.

    Returns:
        Dict with:
            'embeddings': np.ndarray [T, embedding_dim]
            'velocities': np.ndarray [T-1] — ||g_t - g_{t-1}||
            'timestamps': list of timestamp values (if available)
    """
    model.eval()
    model.to(device)

    embeddings = []
    timestamps = []

    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)

            if hasattr(model, "graph_branch"):
                _, g = model.graph_branch(graph.x, graph.edge_index, batch)
            elif hasattr(model, "encoder"):
                h = model.encoder(graph.x, graph.edge_index)
                g = global_mean_pool(h, batch)
            else:
                # Raw fallback — just pool features
                g = global_mean_pool(graph.x, batch)

            embeddings.append(g.cpu().numpy().flatten())

            if hasattr(graph, "timestamp"):
                timestamps.append(str(graph.timestamp))

    embeddings = np.array(embeddings)  # [T, D]

    # Compute velocity: L2 norm of embedding differences
    velocities = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)  # [T-1]

    return {
        "embeddings": embeddings,
        "velocities": velocities,
        "timestamps": timestamps if timestamps else list(range(len(graphs))),
    }


def compute_graph_statistics(graphs: List[Data]) -> List[Dict]:
    """
    Computes per-snapshot structural statistics for the demo.

    Returns list of dicts with keys:
        num_nodes, num_edges, density, mean_degree, max_in_byte_ratio
    """
    stats = []
    for graph in graphs:
        N = graph.x.shape[0]
        E = graph.edge_index.shape[1] if graph.edge_index.numel() > 0 else 0
        density = E / max(N * (N - 1), 1)

        # Compute degree from edge_index
        if E > 0:
            src = graph.edge_index[0]
            unique_src, counts = torch.unique(src, return_counts=True)
            mean_deg = counts.float().mean().item()
        else:
            mean_deg = 0.0

        # Byte ratio from features (col 2)
        max_byte_ratio = graph.x[:, 2].max().item() if N > 0 else 0.0

        stats.append({
            "num_nodes": N,
            "num_edges": E,
            "density": density,
            "mean_out_degree": mean_deg,
            "max_in_byte_ratio": max_byte_ratio,
        })

    return stats


def detect_anomalous_transitions(
    velocities: np.ndarray,
    threshold_sigma: float = 2.0,
) -> List[int]:
    """
    Identifies anomalous transitions as those where embedding velocity
    exceeds mean + threshold_sigma * std.

    Args:
        velocities: Array of embedding velocities [T-1].
        threshold_sigma: Number of standard deviations above mean.

    Returns:
        List of transition indices (0-indexed, representing t→t+1).
    """
    if len(velocities) == 0:
        return []

    mean_v = np.mean(velocities)
    std_v = np.std(velocities)
    threshold = mean_v + threshold_sigma * std_v

    anomalous = [i for i, v in enumerate(velocities) if v > threshold]
    return anomalous


def run_lateral_movement_demo(
    graphs: List[Data],
    model: torch.nn.Module = None,
    config: dict = None,
    output_dir: str = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """
    Runs the n=1 lateral-movement case study.

    This is a demo-only pipeline:
        1. Builds graph snapshots (assumed already built)
        2. Computes embedding trajectory via trained model
        3. Identifies anomalous transitions via velocity threshold
        4. Computes structural statistics per snapshot
        5. Saves results (no Precision@K, no trained head)

    Args:
        graphs: Chronologically ordered PyG Data objects for one episode.
        model: Trained model (e.g., FusedModel or GraphBranch).
            If None, creates a fresh (untrained) GraphBranch for demo purposes.
        config: ML2 config dict.
        output_dir: Directory for saving demo results.
        device: 'cuda' or 'cpu'.
        verbose: Print progress.

    Returns:
        Dict with:
            'embeddings': [T, D] array
            'velocities': [T-1] array
            'anomalous_transitions': list of indices
            'graph_stats': list of per-snapshot stats
            'timestamps': list
    """
    if model is None:
        # Create an untrained GraphBranch for structural demo
        in_channels = graphs[0].x.shape[1] if graphs else 11
        model = GraphBranch(
            in_channels=in_channels,
            hidden_dim=64,
            embedding_dim=64,
        )
        if verbose:
            logger.info("No trained model provided — using untrained GraphBranch for demo.")

    if verbose:
        logger.info(f"Lateral-movement demo: {len(graphs)} snapshots")

    # 1. Compute embedding trajectory
    trajectory = compute_embedding_trajectory(model, graphs, device=device)

    # 2. Detect anomalous transitions
    anomalous = detect_anomalous_transitions(trajectory["velocities"])

    # 3. Structural statistics
    graph_stats = compute_graph_statistics(graphs)

    results = {
        "embeddings": trajectory["embeddings"],
        "velocities": trajectory["velocities"],
        "anomalous_transitions": anomalous,
        "graph_stats": graph_stats,
        "timestamps": trajectory["timestamps"],
        "n_snapshots": len(graphs),
        "n_anomalous": len(anomalous),
    }

    if verbose:
        logger.info(f"  Embedding dim: {trajectory['embeddings'].shape[1]}")
        logger.info(f"  Anomalous transitions: {len(anomalous)} / {len(trajectory['velocities'])}")
        if anomalous:
            logger.info(f"  Anomalous indices: {anomalous}")

    # 4. Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save embeddings
        np.save(os.path.join(output_dir, "embeddings.npy"), trajectory["embeddings"])
        np.save(os.path.join(output_dir, "velocities.npy"), trajectory["velocities"])

        # Save summary JSON
        summary = {
            "n_snapshots": len(graphs),
            "n_anomalous_transitions": len(anomalous),
            "anomalous_transition_indices": anomalous,
            "velocity_mean": float(np.mean(trajectory["velocities"])) if len(trajectory["velocities"]) > 0 else None,
            "velocity_std": float(np.std(trajectory["velocities"])) if len(trajectory["velocities"]) > 0 else None,
            "graph_stats": graph_stats,
        }
        with open(os.path.join(output_dir, "demo_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        if verbose:
            logger.info(f"  Results saved to {output_dir}")

    return results
