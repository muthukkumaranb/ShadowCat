import pytest
import pandas as pd
import numpy as np
import torch
from ml2.data.graph_builder import build_communication_graph
from ml2.data.canonical_ucs import load_canonical_ucs


def test_canonical_split_is_time_ordered_and_disjoint():
    """Canonical UCS split labels must prevent cross-partition time leakage."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "data" / "ucs"
    windows, _, _ = load_canonical_ucs(root)
    timestamps = {
        split: windows.loc[windows["split"] == split, "window_start_utc"]
        for split in ("train", "val", "test")
    }

    assert timestamps["train"].max() < timestamps["val"].min()
    assert timestamps["val"].max() < timestamps["test"].min()
    assert set(timestamps["train"]).isdisjoint(timestamps["val"])
    assert set(timestamps["val"]).isdisjoint(timestamps["test"])
    assert set(timestamps["train"]).isdisjoint(timestamps["test"])
    assert windows["window_start_utc"].is_unique

def test_graph_leakage_constraint():
    """
    Tests that the graph for window t only includes edges from flows strictly BEFORE t.
    Synthetic setup:
    t0 (10:00): A -> B
    t1 (10:01): A -> C
    t2 (10:02): B -> C
    """
    
    t0 = pd.to_datetime('2026-09-04 10:00:00', utc=True)
    t1 = pd.to_datetime('2026-09-04 10:01:00', utc=True)
    t2 = pd.to_datetime('2026-09-04 10:02:00', utc=True)
    
    # Create the flows
    flows = pd.DataFrame([
        {'timestamp_utc': t0, 'source_id': 'A', 'destination_id': 'B', 'byte_count': 100, 'packet_count': 1, 'duration': 0.1, 'protocol': 6},
        {'timestamp_utc': t1, 'source_id': 'A', 'destination_id': 'C', 'byte_count': 200, 'packet_count': 2, 'duration': 0.2, 'protocol': 6},
        {'timestamp_utc': t2, 'source_id': 'B', 'destination_id': 'C', 'byte_count': 300, 'packet_count': 3, 'duration': 0.3, 'protocol': 6},
    ])
    
    # --- Test Window t1 ---
    # Window flows for t1
    window_t1_flows = flows[flows['timestamp_utc'] == t1]
    
    # Active hosts in t1: A, C
    # Flows before t1: A -> B
    # Since B is not active in t1, A->B shouldn't be an edge either, actually! 
    # But let's say B was active in t1 for some other reason, or we just want to verify A->C is NOT an edge
    
    # Let's make B active in t1 as well just to test A->B
    window_t1_flows = pd.concat([window_t1_flows, pd.DataFrame([{
        'timestamp_utc': t1, 'source_id': 'B', 'destination_id': 'B', 'byte_count': 0, 'packet_count': 0, 'duration': 0, 'protocol': 6
    }])])
    
    graph_t1 = build_communication_graph(
        window_flows=window_t1_flows,
        all_flows_before_t=flows[flows['timestamp_utc'] < t1],
        window_start_utc=t1
    )
    
    # Active hosts expected: A, B, C (sorted: A, B, C)
    assert graph_t1.x.shape[0] == 3
    
    edges = set(zip(graph_t1.edge_index[0].tolist(), graph_t1.edge_index[1].tolist()))
    
    # Edge A->B should exist because it happened before t1 (at t0) and both A and B are active in t1
    assert (0, 1) in edges # A(0) -> B(1)
    
    # Edge A->C should NOT exist in the graph because it happens AT t1, not strictly before
    assert (0, 2) not in edges # A(0) -> C(2)
    
    # --- Test Window t2 ---
    window_t2_flows = flows[flows['timestamp_utc'] == t2]
    # Make A active in t2 to check A->B and A->C
    window_t2_flows = pd.concat([window_t2_flows, pd.DataFrame([{
        'timestamp_utc': t2, 'source_id': 'A', 'destination_id': 'A', 'byte_count': 0, 'packet_count': 0, 'duration': 0, 'protocol': 6
    }])])
    
    graph_t2 = build_communication_graph(
        window_flows=window_t2_flows,
        all_flows_before_t=flows[flows['timestamp_utc'] < t2],
        window_start_utc=t2
    )
    
    # Active hosts expected: A, B, C (sorted: A, B, C)
    assert graph_t2.x.shape[0] == 3
    
    edges_t2 = set(zip(graph_t2.edge_index[0].tolist(), graph_t2.edge_index[1].tolist()))
    
    # Edge A->B should exist (from t0)
    assert (0, 1) in edges_t2
    # Edge A->C should exist (from t1)
    assert (0, 2) in edges_t2
    # Edge B->C should NOT exist (happens at t2)
    assert (1, 2) not in edges_t2
    
    print("Leakage constraints verified successfully!")
