"""
Unit tests for graph construction and feature calculation.
"""
import pytest
import pandas as pd
import numpy as np
import torch
from gnn.graph_builder import (
    build_graph_snapshot,
    NODE_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    extract_graph_statistics,
)


def test_build_graph_snapshot_basic():
    flows = pd.DataFrame({
        "source_id": ["192.168.1.1", "192.168.1.1", "192.168.1.2"],
        "destination_id": ["10.0.0.1", "10.0.0.2", "10.0.0.1"],
        "protocol": [6, 17, 6],
        "byte_count": [100, 200, 300],
        "packet_count": [1, 2, 3],
        "duration": [0.1, 0.2, 0.3],
        "source_port": [1024, 1025, 1026],
        "destination_port": [80, 443, 80],
    })

    ts = pd.Timestamp("2016-08-29 01:00:00", tz="UTC")
    data = build_graph_snapshot(flows, ts, label=1)

    assert data.num_nodes == 4
    assert data.x.shape == (4, len(NODE_FEATURE_NAMES))
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] == 3
    assert data.edge_attr.shape == (3, len(EDGE_FEATURE_NAMES))
    assert data.y.item() == 1.0


def test_no_self_loops():
    flows = pd.DataFrame({
        "source_id": ["10.0.0.1", "10.0.0.1"],
        "destination_id": ["10.0.0.1", "10.0.0.2"],  # one self loop, one valid
        "protocol": [6, 6],
        "byte_count": [100, 200],
        "packet_count": [1, 2],
        "duration": [0.1, 0.2],
        "source_port": [80, 80],
        "destination_port": [80, 80],
    })

    ts = pd.Timestamp("2016-08-29 01:00:00", tz="UTC")
    data = build_graph_snapshot(flows, ts)

    # Edge index should not have src == dst
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    assert np.all(src != dst)


def test_extract_graph_statistics():
    flows = pd.DataFrame({
        "source_id": ["A", "B", "C"],
        "destination_id": ["B", "C", "A"],
        "protocol": [6, 6, 6],
        "byte_count": [50, 60, 70],
        "packet_count": [1, 1, 1],
        "duration": [0.0, 0.0, 0.0],
        "source_port": [100, 101, 102],
        "destination_port": [22, 22, 22],
    })

    ts = pd.Timestamp("2016-08-29 01:00:00", tz="UTC")
    data = build_graph_snapshot(flows, ts)
    stats = extract_graph_statistics(data)

    assert "n_nodes" in stats
    assert "n_edges" in stats
    assert "density" in stats
    assert stats["n_nodes"] == 3.0
    assert stats["n_edges"] == 3.0
