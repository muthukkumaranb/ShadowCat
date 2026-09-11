#!/usr/bin/env python
"""Quick diagnostic: check if targets are normalized."""
import torch
import numpy as np
from pathlib import Path
from ml2.configs import load_config
from ml2.data.canonical_ucs import load_canonical_ucs, build_canonical_graphs
from ml2.data.ml1_interface import ML1Adapter
from ml2.data.dataset import get_temporal_splits
from ml2.data.node_features import NodeFeatureScaler

root = Path('ml2')
print("Checking target normalization...")

# Load minimal data
config = load_config(root / 'configs' / 'ml2_config.yaml')
w, e, lookup = load_canonical_ucs(root / 'data' / 'ucs')
graphs = build_canonical_graphs(w, e, lookup)
window_starts = [graph.timestamp for graph in graphs]
adapter = ML1Adapter(config['ml1_interface']['output_dir'], mock_mode=False)
graphs = adapter.inject_novelty_into_graphs(graphs, window_starts)
graphs = graphs[29:]  # warm-up

# Split
splits_cfg = config.get("splits", {})
train_ds, val_ds, test_ds = get_temporal_splits(graphs, 0.70, 0.15, 0.15)
train_graphs = [train_ds.get(i) for i in range(len(train_ds))]

# Fit scaler and check values
scaler = NodeFeatureScaler()
scaler.fit(train_graphs)
print(f"\n1. Scaler stats (before scaling):")
print(f"   Mean: {scaler.mean[0, :11].numpy()}")
print(f"   Std: {scaler.std[0, :11].numpy()}")

# Apply and check
scaled_train = scaler.transform(train_graphs)
print(f"\n2. Scaled features (after scaling):")
all_scaled = torch.cat([g.x[:, :11] for g in scaled_train], dim=0)
print(f"   Mean: {all_scaled.mean(dim=0).numpy()}")
print(f"   Std: {all_scaled.std(dim=0).numpy()}")
print(f"   Range: [{all_scaled.min().item():.2f}, {all_scaled.max().item():.2f}]")

# Check target (mean-pooled next state)
from torch_geometric.nn import global_mean_pool
targets = []
for g in scaled_train[:50]:  # First 50
    batch = torch.zeros(g.x.size(0), dtype=torch.long)
    target = global_mean_pool(g.x, batch)
    targets.append(target)
targets = torch.cat(targets, dim=0)

print(f"\n3. Target statistics (mean-pooled scaled features):")
print(f"   Mean: {targets.mean(dim=0).numpy()}")
print(f"   Std: {targets.std(dim=0).numpy()}")
print(f"   Range: [{targets.min().item():.2f}, {targets.max().item():.2f}]")

print(f"\n4. CONCLUSION:")
if targets.std() < 10 and abs(targets.mean()) < 10:
    print(f"   ✓ Targets ARE normalized (mean~0, std~1)")
else:
    print(f"   ⚠️  Targets may NOT be normalized")
    print(f"      Mean: {targets.mean().item():.4f}, Std: {targets.std().item():.4f}")
