"""Load canonical UCS windows and construct graph snapshots from UCS edges."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


WINDOW_COLUMNS = {"window_id", "window_start_utc", "split", "label_binary"}
EDGE_COLUMNS = {
    "window_id", "window_start_utc", "src_node_id", "dst_node_id",
    "flow_count", "byte_count_sum", "packet_count_sum", "duration_mean_sec",
    "protocol_mode",
}
LOOKUP_COLUMNS = {"node_id", "endpoint_identifier"}


def load_canonical_ucs(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate the canonical post-purge UCS artifacts."""
    windows = pd.read_parquet(root / "ucs_windows.parquet")
    edges = pd.read_parquet(root / "ucs_graph_edgelists.parquet")
    node_lookup = pd.read_parquet(root / "node_lookup.parquet")

    for frame, required, name in (
        (windows, WINDOW_COLUMNS, "ucs_windows"),
        (edges, EDGE_COLUMNS, "ucs_graph_edgelists"),
        (node_lookup, LOOKUP_COLUMNS, "node_lookup"),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing canonical columns: {sorted(missing)}")

    windows = windows.copy()
    edges = edges.copy()
    windows["window_start_utc"] = pd.to_datetime(windows["window_start_utc"], utc=True)
    edges["window_start_utc"] = pd.to_datetime(edges["window_start_utc"], utc=True)
    if not windows["window_start_utc"].is_monotonic_increasing:
        raise ValueError("Canonical UCS windows are not chronologically ordered")
    if windows["window_id"].duplicated().any():
        raise ValueError("Canonical UCS window_id values must be unique")

    window_keys = windows[["window_id", "window_start_utc"]]
    edge_keys = edges[["window_id", "window_start_utc"]].drop_duplicates()
    joined = edge_keys.merge(window_keys, on=["window_id", "window_start_utc"], how="left", indicator=True)
    if (joined["_merge"] != "both").any():
        raise ValueError("Graph edges contain windows absent from canonical UCS windows")
    graph_nodes = set(edges["src_node_id"]).union(edges["dst_node_id"])
    if graph_nodes - set(node_lookup["node_id"]):
        raise ValueError("Graph edges contain node IDs absent from node_lookup.parquet")
    if not set(windows["split"].dropna()).issubset({"train", "val", "test"}):
        raise ValueError("Canonical window split values must be train, val, or test")
    return windows, edges, node_lookup


def build_canonical_graphs(
    windows: pd.DataFrame,
    edges: pd.DataFrame,
    node_lookup: pd.DataFrame,
    deviation_by_window: Dict[str, float] | None = None,
) -> List[Data]:
    """Build one PyG graph per canonical UCS window using canonical node IDs."""
    lookup_order = {node_id: index for index, node_id in enumerate(node_lookup["node_id"])}
    graphs = []
    for row in windows.itertuples(index=False):
        window_edges = edges[edges["window_id"] == row.window_id]
        node_ids = sorted(
            set(window_edges["src_node_id"]).union(window_edges["dst_node_id"]),
            key=lookup_order.__getitem__,
        )
        node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
        x = np.zeros((len(node_ids), 11), dtype=np.float32)
        for edge in window_edges.itertuples(index=False):
            source = node_to_index[edge.src_node_id]
            destination = node_to_index[edge.dst_node_id]
            x[source, 1] += 1
            x[destination, 0] += 1
            x[source, 3] += 1
            x[destination, 3] += 1
            x[source, 6] += float(edge.byte_count_sum)
            x[destination, 6] += float(edge.byte_count_sum)
            x[source, 8] += float(edge.packet_count_sum)
            x[destination, 8] += float(edge.packet_count_sum)
        if deviation_by_window is not None:
            x[:, 10] = float(deviation_by_window.get(row.window_id, 0.0))

        edge_index = torch.tensor(
            [[node_to_index[e] for e in window_edges["src_node_id"]],
             [node_to_index[e] for e in window_edges["dst_node_id"]]],
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            window_edges[[
                "flow_count", "byte_count_sum", "packet_count_sum",
                "duration_mean_sec", "protocol_mode",
            ]].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        graphs.append(Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
            edge_attr=edge_attr,
            timestamp=row.window_start_utc,
            window_id=row.window_id,
            split=row.split,
            y=torch.tensor([int(row.label_binary)], dtype=torch.long),
        ))
    return graphs