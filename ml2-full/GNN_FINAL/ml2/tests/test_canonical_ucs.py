from pathlib import Path

import pandas as pd

from ml2.data.canonical_ucs import build_canonical_graphs, load_canonical_ucs


ROOT = Path(__file__).resolve().parents[1]
UCS_ROOT = ROOT / "data" / "ucs"


def test_canonical_graphs_match_window_timestamps_and_splits():
    windows, edges, node_lookup = load_canonical_ucs(UCS_ROOT)
    graphs = build_canonical_graphs(windows.head(3), edges, node_lookup)

    assert len(windows) == 2787
    assert [graph.timestamp for graph in graphs] == list(windows.head(3)["window_start_utc"])
    assert [graph.window_id for graph in graphs] == list(windows.head(3)["window_id"])
    assert [graph.split for graph in graphs] == list(windows.head(3)["split"])
    assert set(edges["src_node_id"]).union(edges["dst_node_id"]).issubset(set(node_lookup["node_id"]))


def test_canonical_split_counts_and_date_range():
    windows, _, _ = load_canonical_ucs(UCS_ROOT)
    assert windows["split"].value_counts().to_dict() == {"train": 2013, "test": 405, "val": 369}
    assert windows["window_start_utc"].min() == pd.Timestamp("2018-02-14 01:00:00", tz="UTC")
    assert windows["window_start_utc"].max() == pd.Timestamp("2018-03-02 12:59:00", tz="UTC")