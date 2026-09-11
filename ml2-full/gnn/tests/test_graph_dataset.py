"""
Unit tests for graph dataset pipeline: temporal sequences, chronological ordering,
and no-future-information leakage.
"""
import pytest
import torch
from torch_geometric.data import Data
from gnn.graph_dataset import (
    TemporalGraphDataset,
    create_chronological_splits,
)


def _make_labeled_graph(label: float, timestamp: str):
    """Create a minimal test graph with a label and timestamp."""
    return Data(
        x=torch.randn(3, 4),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.randn(2, 2),
        y=torch.tensor([label]),
        timestamp=timestamp,
        num_nodes=3,
    )


class TestTemporalGraphDataset:

    def test_sequence_length(self):
        """Each temporal sequence should have exactly window_size graphs."""
        graphs = [_make_labeled_graph(0.0, f"2016-08-29 00:{i:02d}:00") for i in range(10)]
        window_size = 3
        dataset = TemporalGraphDataset(graphs, window_size=window_size)

        for idx in range(len(dataset)):
            sequence, label = dataset[idx]
            assert len(sequence) == window_size, \
                f"Sequence at index {idx} has {len(sequence)} graphs, expected {window_size}"

    def test_chronological_order_preserved(self):
        """Graphs within each sequence should be in chronological order."""
        timestamps = [f"2016-08-29 00:{i:02d}:00" for i in range(10)]
        graphs = [_make_labeled_graph(0.0, ts) for ts in timestamps]
        dataset = TemporalGraphDataset(graphs, window_size=3)

        for idx in range(len(dataset)):
            sequence, label = dataset[idx]
            for j in range(len(sequence) - 1):
                ts_current = sequence[j].timestamp
                ts_next = sequence[j + 1].timestamp
                assert ts_current < ts_next, \
                    f"Chronological violation at sequence {idx}: {ts_current} >= {ts_next}"

    def test_valid_indices_count(self):
        """Number of valid sequences = n_graphs - window_size + 1."""
        n_graphs = 20
        window_size = 5
        graphs = [_make_labeled_graph(0.0, f"t_{i}") for i in range(n_graphs)]
        dataset = TemporalGraphDataset(graphs, window_size=window_size)

        assert len(dataset) == n_graphs - window_size + 1

    def test_labels_from_last_graph(self):
        """Label for each sequence should come from the last graph in the sequence."""
        labels = [0.0, 0.0, 0.0, 1.0, 0.0]
        graphs = [_make_labeled_graph(labels[i], f"t_{i}") for i in range(5)]
        dataset = TemporalGraphDataset(graphs, window_size=3)

        # Sequence 0: [0, 1, 2] → label from graph 2 = 0.0
        # Sequence 1: [1, 2, 3] → label from graph 3 = 1.0
        # Sequence 2: [2, 3, 4] → label from graph 4 = 0.0
        _, label0 = dataset[0]
        _, label1 = dataset[1]
        _, label2 = dataset[2]

        assert label0.item() == 0.0
        assert label1.item() == 1.0
        assert label2.item() == 0.0


class TestChronologicalSplits:

    def test_splits_are_non_overlapping(self):
        """Train/val/test splits must not overlap."""
        splits = create_chronological_splits(
            n_graphs=200,
            train_frac=0.70,
            val_frac=0.15,
            window_size=5,
            horizon_windows=3,
        )

        train_start, train_end = splits["train"]
        val_start, val_end = splits["val"]
        test_start, test_end = splits["test"]

        # No overlap
        assert train_end < val_start, "Train and val overlap"
        assert val_end < test_start, "Val and test overlap"

    def test_purge_gap_exists(self):
        """Purge gap should remove horizon_windows from train_end."""
        splits = create_chronological_splits(
            n_graphs=100,
            train_frac=0.70,
            val_frac=0.15,
            window_size=5,
            horizon_windows=10,
        )

        _, train_end = splits["train"]
        # Train boundary before purge = index 70
        # After purge: train_end <= 70 - 10
        assert train_end <= 60

    def test_embargo_gap_exists(self):
        """Embargo should push val_start forward by window_size."""
        splits = create_chronological_splits(
            n_graphs=300,
            train_frac=0.70,
            val_frac=0.15,
            window_size=30,
            horizon_windows=2,
        )

        val_start, _ = splits["val"]
        # Train boundary = index 210
        # Embargo: val_start >= 210 + 30 = 240
        assert val_start >= 240, f"Embargo not applied: val_start={val_start}"


class TestNoFutureInformation:

    def test_no_future_graph_in_sequence(self):
        """
        For temporal forecasting: the sequence [t-W+1, ..., t] must not
        contain any graph from time > t.
        """
        n_graphs = 50
        window_size = 5
        graphs = [_make_labeled_graph(0.0, f"t_{i:03d}") for i in range(n_graphs)]
        dataset = TemporalGraphDataset(graphs, window_size=window_size)

        for idx in range(len(dataset)):
            sequence, label = dataset[idx]
            # The last graph in the sequence defines time "t"
            last_ts = sequence[-1].timestamp

            # All other graphs must have timestamps <= last_ts
            for j, g in enumerate(sequence):
                assert g.timestamp <= last_ts, \
                    f"Future information leak at sequence {idx}: " \
                    f"graph {j} timestamp {g.timestamp} > last timestamp {last_ts}"

    def test_splits_chronological_ordering(self):
        """Train indices < val indices < test indices (strictly chronological)."""
        splits = create_chronological_splits(
            n_graphs=500,
            train_frac=0.70,
            val_frac=0.15,
            window_size=30,
            horizon_windows=10,
        )

        train_start, train_end = splits["train"]
        val_start, val_end = splits["val"]
        test_start, test_end = splits["test"]

        assert train_start == 0
        assert train_end < val_start
        assert val_end < test_start
        assert test_end == 499
