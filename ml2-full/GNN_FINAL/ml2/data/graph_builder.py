import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from typing import List, Optional
from ml2.data.node_features import extract_node_features

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
    # 1. Identify active nodes in window t
    src_hosts = set(window_flows['source_id'].unique()) if len(window_flows) > 0 else set()
    dst_hosts = set(window_flows['destination_id'].unique()) if len(window_flows) > 0 else set()
    active_hosts = src_hosts.union(dst_hosts)
    active_hosts = sorted(list(active_hosts))
    
    if len(active_hosts) == 0:
        return Data(
            x=torch.zeros((1, 11), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 5), dtype=torch.float32),
            timestamp=window_start_utc
        )

    host_to_idx = {h: i for i, h in enumerate(active_hosts)}
    
    # 2. Enforce Strict Temporal Leakage Constraint for Edges
    # Edges are flows that occurred strictly BEFORE window_start_utc.
    past_flows = all_flows_before_t[all_flows_before_t['timestamp_utc'] < window_start_utc]
    
    # Only keep edges between hosts that are active in the CURRENT window
    valid_edges = past_flows[
        past_flows['source_id'].isin(active_hosts) & 
        past_flows['destination_id'].isin(active_hosts)
    ]
    
    # 3. Construct Edges
    if len(valid_edges) > 0:
        edge_df = valid_edges.groupby(['source_id', 'destination_id']).agg(
            flow_count=('byte_count', 'size'),
            total_bytes=('byte_count', 'sum'),
            total_packets=('packet_count', 'sum'),
            mean_duration=('duration', 'mean'),
            tcp_ratio=('protocol', lambda x: (x == 6).mean()),
        ).reset_index()
        
        src_indices = [host_to_idx[h] for h in edge_df['source_id']]
        dst_indices = [host_to_idx[h] for h in edge_df['destination_id']]
        
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
    flows_df['window_start_utc'] = flows_df['timestamp_utc'].dt.floor(snapshot_freq)
    
    windows = sorted(flows_df['window_start_utc'].unique())
    
    graphs = []
    # To optimize, we can accumulate past_flows
    for t in windows:
        window_flows = flows_df[flows_df['window_start_utc'] == t]
        all_flows_before_t = flows_df[flows_df['timestamp_utc'] < t]
        
        graph = build_communication_graph(
            window_flows=window_flows,
            all_flows_before_t=all_flows_before_t,
            window_start_utc=t,
            novelty_tminus1=None # Will be injected later during fusion
        )
        graphs.append(graph)
        
    return graphs
