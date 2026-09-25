import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

root = Path("data-engineering/data/ucs")
t0 = time.time()
windows = pd.read_parquet(root / "ucs_windows.parquet")
edges = pd.read_parquet(root / "ucs_graph_edgelists.parquet")
lookup = pd.read_parquet(root / "node_lookup.parquet")
print(f"Loaded parquet in {time.time() - t0:.2f}s")

t0 = time.time()
lookup_order = {node_id: index for index, node_id in enumerate(lookup["node_id"])}
grouped_edges = dict(tuple(edges.groupby("window_id")))
print(f"Grouped 729k edges in {time.time() - t0:.2f}s")

t0 = time.time()
graphs = {}
for row in windows.itertuples(index=False):
    win_id = row.window_id
    if win_id not in grouped_edges:
        continue
    w_edges = grouped_edges[win_id]
    srcs = w_edges["src_node_id"].values
    dsts = w_edges["dst_node_id"].values
    all_nodes = sorted(set(srcs).union(dsts), key=lookup_order.__getitem__)
    node_to_idx = {nid: i for i, nid in enumerate(all_nodes)}
    
    x = np.zeros((len(all_nodes), 11), dtype=np.float32)
    src_idx = np.array([node_to_idx[s] for s in srcs])
    dst_idx = np.array([node_to_idx[d] for d in dsts])
    
    np.add.at(x[:, 1], src_idx, 1)
    np.add.at(x[:, 0], dst_idx, 1)
    np.add.at(x[:, 3], src_idx, 1)
    np.add.at(x[:, 3], dst_idx, 1)
    np.add.at(x[:, 6], src_idx, w_edges["byte_count_sum"].values)
    np.add.at(x[:, 6], dst_idx, w_edges["byte_count_sum"].values)
    np.add.at(x[:, 8], src_idx, w_edges["packet_count_sum"].values)
    np.add.at(x[:, 8], dst_idx, w_edges["packet_count_sum"].values)
    
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    graphs[win_id] = Data(x=torch.from_numpy(x), edge_index=edge_index)

print(f"Constructed all {len(graphs)} graphs in {time.time() - t0:.2f}s!")
