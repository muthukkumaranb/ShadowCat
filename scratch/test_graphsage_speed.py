import sys, os, time
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('ml2-full/GNN_FINAL'))
from ml2.models.graphsage import GraphBranch
from ml2.data.canonical_ucs import load_canonical_ucs, build_canonical_graphs

root = Path("data-engineering/data/ucs")
print("Loading canonical UCS...")
t0 = time.time()
windows, edges, lookup = load_canonical_ucs(root)
print(f"Loaded in {time.time() - t0:.2f}s")

print("Building graphs for 100 windows...")
t0 = time.time()
sample_win = windows.head(100)
graphs = build_canonical_graphs(sample_win, edges, lookup)
print(f"Built 100 graphs in {time.time() - t0:.2f}s")

branch = GraphBranch(in_channels=11, hidden_dim=64, embedding_dim=64)
branch.eval()

t0 = time.time()
with torch.no_grad():
    for g in graphs[:50]:
        h, pooled = branch(g.x, g.edge_index)
print(f"Inference on 50 graphs took {time.time() - t0:.2f}s ({(time.time() - t0)/50*1000:.2f}ms/graph)")
