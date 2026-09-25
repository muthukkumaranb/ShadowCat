import sys, os, time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.abspath('ml2-full/GNN_FINAL'))
from ml2.models.graphsage import GraphBranch

root = Path("data-engineering/data/ucs")
windows = pd.read_parquet(root / "ucs_windows.parquet")
edges = pd.read_parquet(root / "ucs_graph_edgelists.parquet")
lookup = pd.read_parquet(root / "node_lookup.parquet")

lookup_order = {node_id: index for index, node_id in enumerate(lookup["node_id"])}
grouped_edges = dict(tuple(edges.groupby("window_id")))

# Load trained GraphBranch weights from ML2 fused checkpoint
ckpt_path = "ml2-full/GNN_FINAL/ml2/results/ablation/fused/best_model.pt"
device = torch.device("cpu")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

branch = GraphBranch(in_channels=11, hidden_dim=64, embedding_dim=64)
# Extract graph_branch keys
branch_state = {}
for k, v in state_dict.items():
    if k.startswith("graph_branch."):
        branch_state[k.replace("graph_branch.", "")] = v
missing, unexpected = branch.load_state_dict(branch_state, strict=False)
print("Loaded GraphBranch: missing=", missing, "unexpected=", unexpected)
branch.eval()

t0 = time.time()
embeddings = {}
with torch.no_grad():
    for row in windows.itertuples(index=False):
        win_id = row.window_id
        if win_id not in grouped_edges:
            embeddings[win_id] = np.zeros(64, dtype=np.float32)
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
        h, pooled = branch(torch.from_numpy(x), edge_index)
        embeddings[win_id] = pooled.squeeze(0).numpy()

print(f"Extracted GraphSAGE embeddings for all {len(embeddings)} windows in {time.time() - t0:.2f}s!")
emb_matrix = np.array(list(embeddings.values()))
print(f"Embedding matrix shape: {emb_matrix.shape}, mean: {emb_matrix.mean():.4f}, std: {emb_matrix.std():.4f}")
