"""
SnapshotDataset: wraps chronologically-ordered PyG Data objects as a dataset,
and provides temporal (non-shuffled) train/val/test splits.
"""

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Dict, Optional
import copy


class SnapshotDataset(InMemoryDataset):
    """
    In-memory dataset of chronologically ordered communication-graph snapshots.

    Each element is a PyG `Data` object produced by `build_communication_graph()`.
    Ordering is strictly temporal — no shuffling is permitted for time-series integrity.
    """

    def __init__(self, graphs: List[Data], transform=None, pre_transform=None):
        # We bypass the usual root/download/process flow because graphs are
        # already materialised in memory by graph_builder.
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self._data_list = graphs

    # --- PyG InMemoryDataset interface overrides ---

    def len(self) -> int:
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        return self._data_list[idx]

    @property
    def num_node_features(self) -> int:
        if len(self._data_list) == 0:
            return 0
        return self._data_list[0].x.shape[1]

    @property
    def num_edge_features(self) -> int:
        if len(self._data_list) == 0:
            return 0
        return self._data_list[0].edge_attr.shape[1] if self._data_list[0].edge_attr is not None else 0


def get_temporal_splits(
    graphs: List[Data], train_frac: float = 0.7, val_frac: float = 0.15, test_frac: float = 0.15
) -> Tuple[SnapshotDataset, SnapshotDataset, SnapshotDataset]:
    """
    Splits a chronological list of graphs into train, validation, and test datasets.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-5, "Fractions must sum to 1.0"
    n = len(graphs)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))

    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train : n_train + n_val]
    test_graphs = graphs[n_train + n_val :]

    if len(test_graphs) == 0:
        test_graphs = val_graphs[-1:]
        val_graphs = val_graphs[:-1]
    if len(val_graphs) == 0:
        val_graphs = train_graphs[-1:]
        train_graphs = train_graphs[:-1]

    return SnapshotDataset(train_graphs), SnapshotDataset(val_graphs), SnapshotDataset(test_graphs)

def get_loeo_splits(
    graphs: List[Data],
    train_indices: List[int],
    test_indices: List[int],
    purge_window: int = 0,
    embargo_window: int = 0
) -> Tuple[SnapshotDataset, SnapshotDataset]:
    """
    Splits a chronologically ordered list of graphs into train and test sets
    based on exact indices from a LOEO manifest.
    
    Optionally applies purge (removing elements before test_indices) 
    and embargo (removing elements after test_indices) to the train set 
    to prevent temporal leakage.

    Args:
        graphs: List of PyG Data objects, already sorted by time.
        train_indices: List of integer indices for training.
        test_indices: List of integer indices for testing.
        purge_window: Number of indices to remove from train set *before* the first test index.
        embargo_window: Number of indices to remove from train set *after* the last test index.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    if len(test_indices) == 0:
        return SnapshotDataset([graphs[i] for i in train_indices]), SnapshotDataset([])

    # Apply purge/embargo dynamically to train_indices
    if purge_window > 0 or embargo_window > 0:
        min_test = min(test_indices)
        max_test = max(test_indices)
        
        filtered_train = []
        for idx in train_indices:
            # If index is before the test set, check purge window
            if idx < min_test:
                if idx < (min_test - purge_window):
                    filtered_train.append(idx)
            # If index is after the test set, check embargo window
            elif idx > max_test:
                if idx > (max_test + embargo_window):
                    filtered_train.append(idx)
        train_indices = filtered_train

    train_graphs = [graphs[i] for i in train_indices]
    test_graphs = [graphs[i] for i in test_indices]

    return SnapshotDataset(train_graphs), SnapshotDataset(test_graphs)


def build_paired_sequences(
    dataset: SnapshotDataset,
) -> List[Tuple[Data, Data]]:
    """
    Creates (graph_t, graph_{t+1}) pairs for next-state prediction training.

    Each pair consists of consecutive snapshots. The model predicts graph_{t+1}'s
    pooled embedding from graph_t's embedding.

    Args:
        dataset: A SnapshotDataset of chronologically ordered graphs.

    Returns:
        List of (current_graph, next_graph) tuples.
    """
    pairs = []
    for i in range(len(dataset) - 1):
        pairs.append((dataset[i], dataset[i + 1]))
    return pairs


def get_dataloader(
    dataset: SnapshotDataset,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    """
    Creates a PyG DataLoader. Shuffle is False by default to preserve temporal order.
    Only set shuffle=True for training when the model doesn't depend on sequence order
    (e.g., individual snapshot classification).
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
