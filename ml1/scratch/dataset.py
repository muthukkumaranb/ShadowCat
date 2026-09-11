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
    graphs: List[Data],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Tuple[SnapshotDataset, SnapshotDataset, SnapshotDataset]:
    """
    Splits a chronologically ordered list of graphs into train/val/test sets.

    No shuffling — the split respects temporal ordering:
      [0..train_end) | [train_end..val_end) | [val_end..N)

    Args:
        graphs: List of PyG Data objects, already sorted by time.
        train_frac: Fraction of data for training.
        val_frac: Fraction of data for validation.
        test_frac: Fraction of data for testing.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        f"Split fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"

    N = len(graphs)
    train_end = int(N * train_frac)
    val_end = train_end + int(N * val_frac)

    # Guarantee at least 1 sample in each split when N >= 3
    if N >= 3:
        train_end = max(train_end, 1)
        train_end = min(train_end, N - 2)  # Leave room for val + test
        val_end = max(val_end, train_end + 1)
        val_end = min(val_end, N - 1)  # Leave room for test

    train_graphs = graphs[:train_end]
    val_graphs = graphs[train_end:val_end]
    test_graphs = graphs[val_end:]

    return (
        SnapshotDataset(train_graphs),
        SnapshotDataset(val_graphs),
        SnapshotDataset(test_graphs),
    )


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
