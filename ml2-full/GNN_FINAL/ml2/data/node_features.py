import pandas as pd
import numpy as np

def extract_node_features(
    window_flows: pd.DataFrame,
    novelty_tminus1: np.ndarray,
    active_hosts: set,
    host_to_idx: dict
) -> np.ndarray:
    """
    Extracts node features for a given window.
    Features:
    0: in_degree
    1: out_degree
    2: in_byte_ratio (bytes_in / (bytes_out + 1e-9))
    3: distinct_peer_count
    4: mean_flow_duration
    5: std_flow_duration
    6: mean_bytes
    7: std_bytes
    8: mean_packets
    9: std_packets
    10: novelty_tminus1 (from ML1)
    """
    N = len(active_hosts)
    features = np.zeros((N, 11), dtype=np.float32)
    
    if len(window_flows) == 0:
        return features

    # Calculate degrees and peer counts
    out_peers = window_flows.groupby('source_id')['destination_id'].nunique()
    in_peers = window_flows.groupby('destination_id')['source_id'].nunique()
    
    out_bytes = window_flows.groupby('source_id')['byte_count'].sum()
    in_bytes = window_flows.groupby('destination_id')['byte_count'].sum()
    
    out_dur_mean = window_flows.groupby('source_id')['duration'].mean()
    out_dur_std = window_flows.groupby('source_id')['duration'].std().fillna(0.0)
    out_byte_mean = window_flows.groupby('source_id')['byte_count'].mean()
    out_byte_std = window_flows.groupby('source_id')['byte_count'].std().fillna(0.0)
    out_pkt_mean = window_flows.groupby('source_id')['packet_count'].mean()
    out_pkt_std = window_flows.groupby('source_id')['packet_count'].std().fillna(0.0)

    for host in active_hosts:
        idx = host_to_idx[host]
        
        o_peers = out_peers.get(host, 0)
        i_peers = in_peers.get(host, 0)
        
        features[idx, 0] = window_flows[window_flows['destination_id'] == host].shape[0] # in_degree
        features[idx, 1] = window_flows[window_flows['source_id'] == host].shape[0] # out_degree
        
        b_in = in_bytes.get(host, 0)
        b_out = out_bytes.get(host, 0)
        features[idx, 2] = b_in / (b_out + 1e-9)
        
        features[idx, 3] = o_peers + i_peers # naive distinct peer count sum for simplicity
        
        features[idx, 4] = out_dur_mean.get(host, 0)
        features[idx, 5] = out_dur_std.get(host, 0)
        features[idx, 6] = out_byte_mean.get(host, 0)
        features[idx, 7] = out_byte_std.get(host, 0)
        features[idx, 8] = out_pkt_mean.get(host, 0)
        features[idx, 9] = out_pkt_std.get(host, 0)
        
        # t-1 novelty from ML1
        features[idx, 10] = novelty_tminus1[idx] if novelty_tminus1 is not None and idx < len(novelty_tminus1) else 0.0

    return features

class NodeFeatureScaler:
    """
    Fits standard scaling strictly on training graphs and applies it to test graphs
    to prevent temporal leakage.
    
    Only scales the first 10 features (degrees, bytes, durations, etc.).
    The 11th feature (novelty from ML1) is left untouched because it is pre-normalized or categorical.
    """
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, graphs):
        """Fit on a list of training PyG Data objects."""
        if len(graphs) == 0:
            return
            
        import torch
        # Concatenate all nodes' features across all training graphs
        all_features = torch.cat([g.x[:, :10] for g in graphs], dim=0)
        
        self.mean = all_features.mean(dim=0, keepdim=True)
        self.std = all_features.std(dim=0, keepdim=True)
        # Prevent division by zero
        self.std[self.std < 1e-6] = 1.0
        
    def transform(self, graphs):
        """Transform graphs. Returns a new list of graphs with cloned features to prevent mutating originals."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform")
            
        scaled_graphs = []
        for g in graphs:
            import copy
            g_scaled = copy.copy(g) # Shallow copy of the Data object
            if g.x.shape[1] > 0:
                x_scaled = g.x.clone()
                x_scaled[:, :10] = (x_scaled[:, :10] - self.mean) / self.std
                g_scaled.x = x_scaled
            scaled_graphs.append(g_scaled)
        return scaled_graphs
