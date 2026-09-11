"""
Graph dataset: PyTorch Geometric dataset wrapper for temporal graph snapshots.
Supports sequence access for temporal forecasting models.
"""
import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
from typing import List, Optional, Tuple


class SnapshotGraphDataset(Dataset):
    """
    Dataset of graph snapshots stored as individual .pt files.
    Each graph is a PyG Data object with node features, edge features, and label.
    """

    def __init__(
        self,
        root: str,
        graphs: Optional[List[Data]] = None,
        split: str = "all",
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            root: Directory to store/load processed graph files
            graphs: Optional list of pre-built Data objects to save
            split: 'train', 'val', 'test', or 'all'
        """
        self._split = split
        self._graphs = graphs
        super().__init__(root, transform, pre_transform)

        if graphs is not None:
            self._save_graphs(graphs)

    @property
    def processed_file_names(self) -> List[str]:
        """Return list of processed file names."""
        processed_dir = self.processed_dir
        if os.path.exists(processed_dir):
            return [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
        return []

    def _save_graphs(self, graphs: List[Data]):
        """Save graphs to disk."""
        os.makedirs(self.processed_dir, exist_ok=True)
        for i, g in enumerate(graphs):
            torch.save(g, os.path.join(self.processed_dir, f"graph_{i:06d}.pt"))

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> Data:
        path = os.path.join(self.processed_dir, f"graph_{idx:06d}.pt")
        return torch.load(path, weights_only=False)


class TemporalGraphDataset:
    """
    Wraps a list of chronologically ordered graph snapshots for temporal forecasting.
    Provides sliding window access: each sample is a sequence of W consecutive graphs.

    For forecasting: features come from [t-W+1, ..., t], label is future_attack at t.
    """

    def __init__(
        self,
        graphs: List[Data],
        window_size: int = 30,
        split_indices: Optional[dict] = None,
    ):
        """
        Args:
            graphs: Chronologically ordered list of PyG Data objects
            window_size: Number of consecutive graphs in each sequence
            split_indices: Dict with 'train', 'val', 'test' index ranges
        """
        self.graphs = graphs
        self.window_size = window_size
        self.split_indices = split_indices or {}

        # Build valid sequence indices (need W consecutive graphs)
        self._valid_indices = list(range(window_size - 1, len(graphs)))

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Tuple[List[Data], torch.Tensor]:
        """
        Returns:
            sequence: List of W consecutive Data objects
            label: Target label from the last graph in the sequence
        """
        end_idx = self._valid_indices[idx]
        start_idx = end_idx - self.window_size + 1
        sequence = self.graphs[start_idx:end_idx + 1]
        label = sequence[-1].y
        return sequence, label

    def get_split(self, split: str) -> List[int]:
        """Get indices for a chronological split."""
        if split in self.split_indices:
            start, end = self.split_indices[split]
            return [i for i, vi in enumerate(self._valid_indices)
                    if start <= vi <= end]
        return list(range(len(self)))

    @property
    def labels(self) -> np.ndarray:
        """All labels for valid sequences."""
        return np.array([self.graphs[vi].y.item() for vi in self._valid_indices])


def create_chronological_splits(
    n_graphs: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    window_size: int = 30,
    horizon_windows: int = 10,
) -> dict:
    """
    Create chronological train/val/test splits with purge and embargo.

    Args:
        n_graphs: Total number of graph snapshots
        train_frac: Fraction for training
        val_frac: Fraction for validation
        window_size: Lookback window size (for embargo)
        horizon_windows: Forecast horizon in windows (for purge)

    Returns:
        Dict with 'train', 'val', 'test' tuples of (start_idx, end_idx)
    """
    train_end = int(np.floor(n_graphs * train_frac))
    val_end = int(np.floor(n_graphs * (train_frac + val_frac)))

    # Purge: remove train samples whose label horizon crosses into val
    train_end_purged = train_end - horizon_windows

    # Embargo: remove early val samples whose lookback reaches into train
    val_start_embargoed = train_end + window_size

    # Similar for val→test boundary
    val_end_purged = val_end - horizon_windows
    test_start_embargoed = val_end + window_size

    splits = {
        "train": (0, max(0, train_end_purged - 1)),
        "val": (min(val_start_embargoed, val_end), max(0, val_end_purged - 1)),
        "test": (min(test_start_embargoed, n_graphs - 1), n_graphs - 1),
    }

    return splits
