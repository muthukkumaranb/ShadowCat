import torch
try:
    from torch_geometric.data import Data
except ImportError:
    class Data:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def __repr__(self):
            return f"Data({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"

import pandas as pd
import numpy as np
from typing import List, Optional

try:
    from ml2.data.node_features import extract_node_features
except ImportError:
    try:
        from scratch.node_features import extract_node_features
    except ImportError:
        from node_features import extract_node_features

def build_communication_graph(
    window_flows: pd.DataFrame,
    all_flows_before_t: pd.DataFrame,
    window_start_utc: pd.Timestamp,
    novelty_tminus1: np.ndarray = None
) -> Data:
    """
    Builds a leakage-safe communication graph for window t.
    Nodes: hosts active in window_flows.
    Edges: flows between those hosts that occurred strictly BEFORE window_start_utc.
    """
    # Determine column names flexibly
    src_col = 'source_id' if 'source_id' in window_flows.columns else ('src_ip' if 'src_ip' in window_flows.columns else 'source_ip')
    dst_col = 'destination_id' if 'destination_id' in window_flows.columns else ('dst_ip' if 'dst_ip' in window_flows.columns else 'destination_ip')

    # 1. Identify active nodes in window t
    src_hosts = set(window_flows[src_col].unique()) if len(window_flows) > 0 and src_col in window_flows.columns else set()
    dst_hosts = set(window_flows[dst_col].unique()) if len(window_flows) > 0 and dst_col in window_flows.columns else set()
    active_hosts = src_hosts.union(dst_hosts)
    active_hosts = sorted(list(active_hosts))
    
    if len(active_hosts) == 0:
        return Data(
            x=torch.zeros((0, 11), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 5), dtype=torch.float32),
            timestamp=window_start_utc
        )

    host_to_idx = {h: i for i, h in enumerate(active_hosts)}
    
    # 2. Enforce Strict Temporal Leakage Constraint for Edges
    # Edges are flows that occurred strictly BEFORE window_start_utc.
    ts_col = 'timestamp_utc' if 'timestamp_utc' in all_flows_before_t.columns else ('timestamp' if 'timestamp' in all_flows_before_t.columns else 'window_start_utc')
    past_flows = all_flows_before_t[all_flows_before_t[ts_col] < window_start_utc] if len(all_flows_before_t) > 0 and ts_col in all_flows_before_t.columns else pd.DataFrame()
    
    # Only keep edges between hosts that are active in the CURRENT window
    if len(past_flows) > 0 and src_col in past_flows.columns and dst_col in past_flows.columns:
        valid_edges = past_flows[
            past_flows[src_col].isin(active_hosts) & 
            past_flows[dst_col].isin(active_hosts)
        ]
    else:
        valid_edges = pd.DataFrame()
    
    # 3. Construct Edges
    if len(valid_edges) > 0:
        byte_col = 'byte_count' if 'byte_count' in valid_edges.columns else ('bytes' if 'bytes' in valid_edges.columns else 'total_bytes')
        pkt_col = 'packet_count' if 'packet_count' in valid_edges.columns else ('packets' if 'packets' in valid_edges.columns else 'total_packets')
        dur_col = 'duration' if 'duration' in valid_edges.columns else ('flow_duration' if 'flow_duration' in valid_edges.columns else 'duration_ms')
        proto_col = 'protocol' if 'protocol' in valid_edges.columns else ('proto' if 'proto' in valid_edges.columns else 'protocol_type')

        agg_dict = {}
        if byte_col in valid_edges.columns:
            agg_dict['flow_count'] = (byte_col, 'size')
            agg_dict['total_bytes'] = (byte_col, 'sum')
        else:
            agg_dict['flow_count'] = (src_col, 'size')
            agg_dict['total_bytes'] = (src_col, lambda x: 0.0)

        if pkt_col in valid_edges.columns:
            agg_dict['total_packets'] = (pkt_col, 'sum')
        else:
            agg_dict['total_packets'] = (src_col, lambda x: 0.0)

        if dur_col in valid_edges.columns:
            agg_dict['mean_duration'] = (dur_col, 'mean')
        else:
            agg_dict['mean_duration'] = (src_col, lambda x: 0.0)

        if proto_col in valid_edges.columns:
            agg_dict['tcp_ratio'] = (proto_col, lambda x: (x == 6).mean())
        else:
            agg_dict['tcp_ratio'] = (src_col, lambda x: 0.0)

        edge_df = valid_edges.groupby([src_col, dst_col]).agg(**agg_dict).reset_index()
        
        src_indices = [host_to_idx[h] for h in edge_df[src_col]]
        dst_indices = [host_to_idx[h] for h in edge_df[dst_col]]
        
        edge_index = torch.tensor(np.stack([src_indices, dst_indices], axis=0), dtype=torch.long)
        edge_attr = torch.tensor(edge_df[['flow_count', 'total_bytes', 'total_packets', 'mean_duration', 'tcp_ratio']].values, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 5), dtype=torch.float32)
        
    # 4. Construct Node Features
    x_features = extract_node_features(window_flows, novelty_tminus1, active_hosts, host_to_idx)
    x = torch.tensor(x_features, dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, timestamp=window_start_utc)

def build_all_graphs(flows_df: pd.DataFrame, snapshot_freq: str = '60s') -> List[Data]:
    """
    Builds a sequence of graphs chronologically, ensuring past_flows only contains
    records strictly before each window's start.
    """
    flows_df = flows_df.copy()
    ts_col = 'timestamp_utc' if 'timestamp_utc' in flows_df.columns else ('timestamp' if 'timestamp' in flows_df.columns else 'window_start_utc')
    
    if not pd.api.types.is_datetime64_any_dtype(flows_df[ts_col]):
        flows_df[ts_col] = pd.to_datetime(flows_df[ts_col], utc=True)
        
    flows_df['window_start_utc'] = flows_df[ts_col].dt.floor(snapshot_freq)
    
    windows = sorted(flows_df['window_start_utc'].unique())
    
    graphs = []
    for t in windows:
        window_flows = flows_df[flows_df['window_start_utc'] == t]
        all_flows_before_t = flows_df[flows_df[ts_col] < t]
        
        graph = build_communication_graph(
            window_flows=window_flows,
            all_flows_before_t=all_flows_before_t,
            window_start_utc=t,
            novelty_tminus1=None # Will be injected later during fusion
        )
        graphs.append(graph)
        
    return graphs

