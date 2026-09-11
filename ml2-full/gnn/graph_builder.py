"""
Graph builder: converts validated network flows into PyTorch Geometric Data objects.
Each 1-minute snapshot becomes one graph with node features, edge features, and labels.
Vectorized and highly efficient.
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Optional, Dict
from gnn.node_mapping import NodeMapper


# ── Node feature names (12 features) ─────────────────────────────────────────
NODE_FEATURE_NAMES = [
    "out_degree", "in_degree",
    "out_flow_count", "in_flow_count",
    "out_bytes", "in_bytes",
    "out_packets", "in_packets",
    "out_tcp_ratio", "out_udp_ratio",
    "unique_dst_ports", "unique_src_ports",
]

# ── Edge feature names (6 features) ──────────────────────────────────────────
EDGE_FEATURE_NAMES = [
    "flow_count", "total_bytes", "total_packets",
    "mean_duration", "tcp_ratio", "udp_ratio",
]


def build_graph_snapshot(
    flows: pd.DataFrame,
    timestamp: pd.Timestamp,
    label: Optional[int] = None,
) -> Data:
    """
    Build a single PyTorch Geometric Data object from flows in one time window.

    Args:
        flows: DataFrame with columns [source_id, destination_id, protocol,
               byte_count, packet_count, duration, source_port, destination_port]
        timestamp: The snapshot timestamp
        label: Binary attack label (0/1) or None

    Returns:
        PyG Data object with x (node features), edge_index, edge_attr, y, timestamp
    """
    if len(flows) == 0:
        return Data(
            x=torch.zeros((1, len(NODE_FEATURE_NAMES)), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=torch.float32),
            y=torch.tensor([label if label is not None else 0], dtype=torch.float32),
            timestamp=str(timestamp),
            num_nodes=1,
        )

    # Remove self-loops
    flows = flows[flows["source_id"] != flows["destination_id"]].copy()
    if len(flows) == 0:
        return Data(
            x=torch.zeros((1, len(NODE_FEATURE_NAMES)), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, len(EDGE_FEATURE_NAMES)), dtype=torch.float32),
            y=torch.tensor([label if label is not None else 0], dtype=torch.float32),
            timestamp=str(timestamp),
            num_nodes=1,
        )

    # ── Node mapping ──────────────────────────────────────────────────────
    mapper = NodeMapper()
    mapper.fit(flows["source_id"].values, flows["destination_id"].values)
    n_nodes = mapper.num_nodes

    src_idx = mapper.transform(flows["source_id"].values)
    dst_idx = mapper.transform(flows["destination_id"].values)

    flows["src_node"] = src_idx
    flows["dst_node"] = dst_idx

    # ── Protocol flags ────────────────────────────────────────────────────
    flows["is_tcp"] = (flows["protocol"] == 6).astype(np.float32)
    flows["is_udp"] = (flows["protocol"] == 17).astype(np.float32)

    # ── Node features (Vectorized Aggregation) ────────────────────────────
    node_features = np.zeros((n_nodes, len(NODE_FEATURE_NAMES)), dtype=np.float32)

    # Outbound stats (per source node)
    out_agg = flows.groupby("src_node").agg(
        out_degree=("dst_node", "nunique"),
        out_flow_count=("dst_node", "size"),
        out_bytes=("byte_count", "sum"),
        out_packets=("packet_count", "sum"),
        tcp_sum=("is_tcp", "sum"),
        udp_sum=("is_udp", "sum"),
        unique_dst_ports=("destination_port", "nunique"),
    )

    if not out_agg.empty:
        src_indices = out_agg.index.values
        node_features[src_indices, 0] = out_agg["out_degree"].values
        node_features[src_indices, 2] = out_agg["out_flow_count"].values
        node_features[src_indices, 4] = out_agg["out_bytes"].values
        node_features[src_indices, 6] = out_agg["out_packets"].values
        flow_cnt = out_agg["out_flow_count"].values
        node_features[src_indices, 8] = np.where(flow_cnt > 0, out_agg["tcp_sum"].values / flow_cnt, 0.0)
        node_features[src_indices, 9] = np.where(flow_cnt > 0, out_agg["udp_sum"].values / flow_cnt, 0.0)
        node_features[src_indices, 10] = out_agg["unique_dst_ports"].values

    # Inbound stats (per destination node)
    in_agg = flows.groupby("dst_node").agg(
        in_degree=("src_node", "nunique"),
        in_flow_count=("src_node", "size"),
        in_bytes=("byte_count", "sum"),
        in_packets=("packet_count", "sum"),
        unique_src_ports=("source_port", "nunique"),
    )

    if not in_agg.empty:
        dst_indices = in_agg.index.values
        node_features[dst_indices, 1] = in_agg["in_degree"].values
        node_features[dst_indices, 3] = in_agg["in_flow_count"].values
        node_features[dst_indices, 5] = in_agg["in_bytes"].values
        node_features[dst_indices, 7] = in_agg["in_packets"].values
        node_features[dst_indices, 11] = in_agg["unique_src_ports"].values

    # ── Edge construction (aggregate duplicate edges) ─────────────────────
    edge_df = flows.groupby(["src_node", "dst_node"]).agg(
        flow_count=("byte_count", "size"),
        total_bytes=("byte_count", "sum"),
        total_packets=("packet_count", "sum"),
        mean_duration=("duration", "mean"),
        tcp_ratio=("is_tcp", "mean"),
        udp_ratio=("is_udp", "mean"),
    ).reset_index()

    edge_src = edge_df["src_node"].values.astype(np.int64)
    edge_dst = edge_df["dst_node"].values.astype(np.int64)
    edge_features = edge_df[EDGE_FEATURE_NAMES].values.astype(np.float32)

    # ── Build PyG Data ────────────────────────────────────────────────────
    edge_index = torch.tensor(np.stack([edge_src, edge_dst], axis=0), dtype=torch.long)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_features, dtype=torch.float32),
        y=torch.tensor([label if label is not None else 0], dtype=torch.float32),
        timestamp=str(timestamp),
        num_nodes=n_nodes,
    )

    return data


def build_all_graphs(
    validated_flows: pd.DataFrame,
    labels_df: pd.DataFrame,
    snapshot_freq: str = "min",
) -> List[Data]:
    """
    Build all graph snapshots from validated flows DataFrame with associated labels.
    """
    validated_flows = validated_flows.copy()
    validated_flows["window_ts"] = validated_flows["timestamp"].dt.floor(snapshot_freq)

    label_lookup = {}
    for _, row in labels_df.iterrows():
        ts = row["timestamp"]
        label_lookup[ts] = int(row["future_attack"])

    grouped = validated_flows.groupby("window_ts", sort=True)
    total_groups = len(grouped)
    print(f"  Found {total_groups} time snapshots to convert into graphs.", flush=True)

    graphs = []
    for i, (window_ts, window_flows) in enumerate(grouped):
        if (i + 1) % 50 == 0 or i == 0 or i == total_groups - 1:
            print(f"  Building graph {i+1}/{total_groups} ({window_ts})", flush=True)

        label = label_lookup.get(window_ts, 0)
        graph = build_graph_snapshot(window_flows, window_ts, label=label)
        graphs.append(graph)

    return graphs


def build_all_graphs_streaming(
    parquet_path: str,
    labels_df: pd.DataFrame,
    snapshot_freq: str = "min",
) -> List[Data]:
    """
    Build all graph snapshots by streaming row-group by row-group from Parquet.
    Keeps memory footprint minimal (< 200MB) while processing 40M+ flows.
    """
    import pyarrow.parquet as pq

    label_lookup = {}
    for _, row in labels_df.iterrows():
        ts = row["timestamp"]
        label_lookup[ts] = int(row["future_attack"])

    pf = pq.ParquetFile(parquet_path)
    graphs = []
    buffer_df = pd.DataFrame()
    cols = [
        "timestamp", "source_id", "destination_id", "protocol",
        "byte_count", "packet_count", "duration", "source_port", "destination_port"
    ]

    print(f"Streaming {pf.num_row_groups} row groups ({pf.metadata.num_rows:,} total flows)...", flush=True)

    for rg_idx in range(pf.num_row_groups):
        t_chunk = pf.read_row_group(rg_idx, columns=cols).to_pandas()
        t_chunk["window_ts"] = t_chunk["timestamp"].dt.floor(snapshot_freq)

        if not buffer_df.empty:
            t_chunk = pd.concat([buffer_df, t_chunk], ignore_index=True)
            buffer_df = pd.DataFrame()

        unique_windows = t_chunk["window_ts"].unique()

        if rg_idx < pf.num_row_groups - 1:
            complete_windows = unique_windows[:-1]
            last_window = unique_windows[-1]
            buffer_df = t_chunk[t_chunk["window_ts"] == last_window]
            t_chunk = t_chunk[t_chunk["window_ts"] != last_window]
        else:
            complete_windows = unique_windows

        for window_ts, window_flows in t_chunk.groupby("window_ts", sort=True):
            label = label_lookup.get(window_ts, 0)
            graph = build_graph_snapshot(window_flows, window_ts, label=label)
            graphs.append(graph)
            if len(graphs) % 50 == 0 or len(graphs) == 1:
                print(f"  Constructed {len(graphs)} graph snapshots...", flush=True)

    if not buffer_df.empty:
        for window_ts, window_flows in buffer_df.groupby("window_ts", sort=True):
            label = label_lookup.get(window_ts, 0)
            graph = build_graph_snapshot(window_flows, window_ts, label=label)
            graphs.append(graph)

    print(f"Successfully constructed all {len(graphs)} graph snapshots via streaming!", flush=True)
    return graphs


def extract_graph_statistics(data: Data) -> Dict[str, float]:
    """Extract graph-level statistical features for baseline models."""
    n_nodes = data.num_nodes
    n_edges = data.edge_index.shape[1] if data.edge_index.numel() > 0 else 0

    stats = {
        "n_nodes": float(n_nodes),
        "n_edges": float(n_edges),
        "density": float(n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0,
    }

    if n_nodes > 0 and data.x is not None:
        x = data.x.numpy()
        for i, name in enumerate(NODE_FEATURE_NAMES):
            col = x[:, i]
            stats[f"node_{name}_mean"] = float(np.mean(col))
            stats[f"node_{name}_std"] = float(np.std(col))
            stats[f"node_{name}_max"] = float(np.max(col))

    if n_edges > 0 and data.edge_attr is not None:
        ea = data.edge_attr.numpy()
        for i, name in enumerate(EDGE_FEATURE_NAMES):
            col = ea[:, i]
            stats[f"edge_{name}_mean"] = float(np.mean(col))
            stats[f"edge_{name}_std"] = float(np.std(col))
            stats[f"edge_{name}_max"] = float(np.max(col))

    if n_edges > 0:
        src = data.edge_index[0].numpy()
        dst = data.edge_index[1].numpy()
        out_deg = np.bincount(src, minlength=n_nodes)
        in_deg = np.bincount(dst, minlength=n_nodes)

        stats["out_degree_mean"] = float(np.mean(out_deg))
        stats["out_degree_std"] = float(np.std(out_deg))
        stats["out_degree_max"] = float(np.max(out_deg))
        stats["in_degree_mean"] = float(np.mean(in_deg))
        stats["in_degree_std"] = float(np.std(in_deg))
        stats["in_degree_max"] = float(np.max(in_deg))

    return stats
