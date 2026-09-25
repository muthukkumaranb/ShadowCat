import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
ml1_dir = repo_root / "ml1"
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(ml1_dir))
sys.path.insert(0, str(repo_root / "ml2-full" / "GNN_FINAL"))

from lstm.graph_features import compute_scalar_graph_features
from ml2.models.graphsage import GraphBranch
import pickle

# 1. Benchmark Plain Stacked LSTM
pca_path = ml1_dir / "artifacts" / "lstm" / "lstm_stacked" / "pca_32_stacked.pkl"
with open(pca_path, "rb") as f:
    pca_data = pickle.load(f)
pca = pca_data["pca"]

# Load 1 fold model
det_ckpt = torch.load(ml1_dir / "artifacts" / "lstm" / "lstm_stacked" / "detection" / "model_fold_0.pt", weights_only=False)
from backend.models import ResidualLSTMClassifier
plain_m = ResidualLSTMClassifier(input_size=32, hidden_size=64, dropout=0.20)
plain_m.load_state_dict(det_ckpt["model_state_dict"])
plain_m.eval()

cols = pca.columns
seq_406 = np.random.randn(30, len(cols)).astype(np.float32)

# Plain Latency test
times_plain = []
for _ in range(50):
    t0 = time.perf_counter()
    df_seq = pd.DataFrame(seq_406, columns=cols)
    transformed = pca.transform(df_seq)
    pca_cols = [f"pca_{i}" for i in range(32)]
    seq_32 = torch.as_tensor(transformed[pca_cols].to_numpy(dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        out = plain_m(seq_32, torch.tensor([0.0]))
    times_plain.append((time.perf_counter() - t0) * 1000)

mean_plain = np.median(times_plain[5:])
print(f"Plain Stacked LSTM latency: {mean_plain:.2f} ms / window")

# 2. Benchmark Graph-Augmented Stacked LSTM
graph_scaler_path = ml1_dir / "artifacts" / "lstm" / "lstm_graph" / "graph_scaler_canonical.pkl"
with open(graph_scaler_path, "rb") as f:
    g_scaler_data = pickle.load(f)
g_scaler = g_scaler_data["scaler"]

g_ckpt = torch.load(ml1_dir / "artifacts" / "lstm" / "lstm_graph" / "detection" / "model_fold_0.pt", weights_only=False)
graph_m = ResidualLSTMClassifier(input_size=36, hidden_size=64, dropout=0.20)
graph_m.load_state_dict(g_ckpt["model_state_dict"])
graph_m.eval()

# Measure latency with cached/extracted graph features vs on-the-fly
times_graph = []
for _ in range(50):
    t0 = time.perf_counter()
    df_seq = pd.DataFrame(seq_406, columns=cols)
    transformed = pca.transform(df_seq)
    pca_cols = [f"pca_{i}" for i in range(32)]
    # 4 scalar features: either zero-filled (unfamiliar input) or looked up/computed
    scalars = np.zeros((30, 4), dtype=np.float32)
    seq_36 = np.hstack([transformed[pca_cols].to_numpy(dtype=np.float32), scalars])
    seq_36_t = torch.as_tensor(seq_36, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = graph_m(seq_36_t, torch.tensor([0.0]))
    times_graph.append((time.perf_counter() - t0) * 1000)

mean_graph = np.median(times_graph[5:])
print(f"Graph-Augmented Stacked LSTM latency (lookup/fast): {mean_graph:.2f} ms / window")

# 3. Benchmark GraphSAGE live extraction latency
# For a window with ~100 edges: build PyG graph, run GraphBranch
branch = GraphBranch(in_channels=11, hidden_dim=64, embedding_dim=64)
branch.eval()

times_sage = []
for _ in range(50):
    t0 = time.perf_counter()
    # Graph build for ~30 nodes and ~60 edges
    x = torch.randn(30, 11)
    edge_index = torch.randint(0, 30, (2, 60))
    with torch.no_grad():
        _, pooled = branch(x, edge_index)
    times_sage.append((time.perf_counter() - t0) * 1000)

mean_sage = np.median(times_sage[5:])
print(f"GraphSAGE GNN embedding extraction latency: {mean_sage:.2f} ms / window")
print(f"Total Stacked LSTM + GraphSAGE Fusion latency: {mean_plain + mean_sage:.2f} ms / window")
