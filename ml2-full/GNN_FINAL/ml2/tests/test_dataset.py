"""
Tests for SnapshotDataset and temporal splitting.
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from ml2.data.dataset import (
    SnapshotDataset,
    get_temporal_splits,
    get_loeo_splits,
    build_paired_sequences,
    get_dataloader,
)


def _make_synthetic_graphs(n: int = 20, num_nodes: int = 5, num_features: int = 11) -> list:
    """Creates n synthetic graphs with increasing 'timestamps'."""
    graphs = []
    for t in range(n):
        x = torch.randn(num_nodes, num_features)
        # Random edges
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_attr = torch.randn(num_nodes * 2, 5)
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, timestamp=t)
        graphs.append(g)
    return graphs


class TestSnapshotDataset:
    def test_len(self):
        graphs = _make_synthetic_graphs(10)
        ds = SnapshotDataset(graphs)
        assert len(ds) == 10

    def test_get(self):
        graphs = _make_synthetic_graphs(10)
        ds = SnapshotDataset(graphs)
        g = ds[0]
        assert isinstance(g, Data)
        assert g.x.shape == (5, 11)

    def test_num_features(self):
        graphs = _make_synthetic_graphs(10, num_features=11)
        ds = SnapshotDataset(graphs)
        assert ds.num_node_features == 11
        assert ds.num_edge_features == 5

    def test_empty_dataset(self):
        ds = SnapshotDataset([])
        assert len(ds) == 0
        assert ds.num_node_features == 0


class TestTemporalSplits:
    def test_split_sizes(self):
        graphs = _make_synthetic_graphs(100)
        train_ds, val_ds, test_ds = get_temporal_splits(graphs, 0.7, 0.15, 0.15)
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 100

    def test_split_ordering(self):
        """Verify that splits preserve chronological order."""
        graphs = _make_synthetic_graphs(20)
        train_ds, val_ds, test_ds = get_temporal_splits(graphs, 0.6, 0.2, 0.2)

        # Last train timestamp < first val timestamp
        train_last_ts = train_ds[len(train_ds) - 1].timestamp
        val_first_ts = val_ds[0].timestamp
        assert train_last_ts < val_first_ts

        # Last val timestamp < first test timestamp
        val_last_ts = val_ds[len(val_ds) - 1].timestamp
        test_first_ts = test_ds[0].timestamp
        assert val_last_ts < test_first_ts

    def test_no_overlap(self):
        """Verify no data leakage between splits."""
        graphs = _make_synthetic_graphs(20)
        train_ds, val_ds, test_ds = get_temporal_splits(graphs, 0.6, 0.2, 0.2)

        train_ts = set(train_ds[i].timestamp for i in range(len(train_ds)))
        val_ts = set(val_ds[i].timestamp for i in range(len(val_ds)))
        test_ts = set(test_ds[i].timestamp for i in range(len(test_ds)))

        assert train_ts.isdisjoint(val_ts)
        assert train_ts.isdisjoint(test_ts)
        assert val_ts.isdisjoint(test_ts)

    def test_minimum_split_sizes(self):
        """Even with tiny dataset, each split should have >= 1 sample."""
        graphs = _make_synthetic_graphs(3)
        train_ds, val_ds, test_ds = get_temporal_splits(graphs, 0.7, 0.15, 0.15)
        assert len(train_ds) >= 1
        assert len(val_ds) >= 1
        assert len(test_ds) >= 1

    def test_fractions_sum_validation(self):
        graphs = _make_synthetic_graphs(10)
        with pytest.raises(AssertionError):
            get_temporal_splits(graphs, 0.5, 0.3, 0.3)  # sums to 1.1


class TestLOEOSplits:
    def test_split_sizes(self):
        graphs = _make_synthetic_graphs(100)
        train_idx = list(range(0, 50)) + list(range(60, 100))
        test_idx = list(range(50, 60))
        train_ds, test_ds = get_loeo_splits(graphs, train_idx, test_idx)
        assert len(train_ds) == 90
        assert len(test_ds) == 10

    def test_no_overlap(self):
        """Verify no data leakage between splits."""
        graphs = _make_synthetic_graphs(20)
        train_idx = [0, 1, 2, 3, 4, 10, 11, 12]
        test_idx = [5, 6, 7, 8, 9]
        train_ds, test_ds = get_loeo_splits(graphs, train_idx, test_idx)

        train_ts = set(train_ds[i].timestamp for i in range(len(train_ds)))
        test_ts = set(test_ds[i].timestamp for i in range(len(test_ds)))

        assert train_ts.isdisjoint(test_ts)

    def test_purge_embargo(self):
        """Test that purge and embargo windows successfully remove indices."""
        graphs = _make_synthetic_graphs(20)
        # Test window: 5..9
        # Purge window: 2 elements before (3, 4)
        # Embargo window: 2 elements after (10, 11)
        train_idx = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
        test_idx = [5, 6, 7, 8, 9]
        
        train_ds, test_ds = get_loeo_splits(graphs, train_idx, test_idx, purge_window=2, embargo_window=2)
        
        train_ts = [train_ds[i].timestamp for i in range(len(train_ds))]
        
        assert 3 not in train_ts
        assert 4 not in train_ts
        assert 10 not in train_ts
        assert 11 not in train_ts
        assert set(train_ts) == {0, 1, 2, 12, 13, 14}


class TestPairedSequences:
    def test_pair_count(self):
        graphs = _make_synthetic_graphs(10)
        ds = SnapshotDataset(graphs)
        pairs = build_paired_sequences(ds)
        assert len(pairs) == 9  # n-1 pairs

    def test_pair_ordering(self):
        graphs = _make_synthetic_graphs(10)
        ds = SnapshotDataset(graphs)
        pairs = build_paired_sequences(ds)
        for i, (g_t, g_tp1) in enumerate(pairs):
            assert g_t.timestamp == i
            assert g_tp1.timestamp == i + 1

    def test_single_graph(self):
        graphs = _make_synthetic_graphs(1)
        ds = SnapshotDataset(graphs)
        pairs = build_paired_sequences(ds)
        assert len(pairs) == 0


class TestDataLoader:
    def test_dataloader_creation(self):
        graphs = _make_synthetic_graphs(10)
        ds = SnapshotDataset(graphs)
        loader = get_dataloader(ds, batch_size=4, shuffle=False)
        batches = list(loader)
        assert len(batches) > 0
