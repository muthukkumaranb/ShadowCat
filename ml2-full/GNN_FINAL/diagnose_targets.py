#!/usr/bin/env python
"""Diagnose target normalization and scale of prediction targets."""
import torch
import numpy as np
from pathlib import Path
from ml2.configs import load_config
from ml2.data.canonical_ucs import load_canonical_ucs, build_canonical_graphs
from ml2.data.ml1_interface import ML1Adapter
from ml2.data.dataset import get_temporal_splits, build_paired_sequences
from ml2.data.node_features import NodeFeatureScaler
from torch_geometric.nn import global_mean_pool

root = Path('ml2')
print("="*80)
print("TARGET NORMALIZATION DIAGNOSTIC")
print("="*80)

# Load data
config = load_config(root / 'configs' / 'ml2_config.yaml')
w, e, lookup = load_canonical_ucs(root / 'data' / 'ucs')
graphs = build_canonical_graphs(w, e, lookup)
window_starts = [graph.timestamp for graph in graphs]
adapter = ML1Adapter(config['ml1_interface']['output_dir'], mock_mode=False)
graphs = adapter.inject_novelty_into_graphs(graphs, window_starts)
graphs = graphs[29:]  # warm-up

# Create splits
splits_cfg = config.get("splits", {})
train_ds, val_ds, test_ds = get_temporal_splits(
    graphs,
    train_frac=splits_cfg.get("train_frac", 0.70),
    val_frac=splits_cfg.get("val_frac", 0.15),
    test_frac=splits_cfg.get("test_frac", 0.15),
)

train_graphs = [train_ds.get(i) for i in range(len(train_ds))]
val_graphs = [val_ds.get(i) for i in range(len(val_ds))]

print(f"\nDataset splits:")
print(f"  Train: {len(train_graphs)} graphs")
print(f"  Val: {len(val_graphs)} graphs")
print(f"  Test: {len(test_ds)} graphs")

# Fit scaler
scaler = NodeFeatureScaler()
scaler.fit(train_graphs)
print(f"\nNodeFeatureScaler fitted on train set:")
print(f"  Mean of features 0-9: {scaler.mean[0, :10].numpy()}")
print(f"  Std of features 0-9: {scaler.std[0, :10].numpy()}")

# Apply scaler
scaled_train = scaler.transform(train_graphs)
scaled_val = scaler.transform(val_graphs)

# Check scaled feature statistics
print(f"\nScaled feature statistics (after applying scaler):")
all_scaled_features = torch.cat([g.x[:, :10] for g in scaled_train], dim=0)
print(f"  Mean: {all_scaled_features.mean(dim=0).numpy()}")
print(f"  Std: {all_scaled_features.std(dim=0).numpy()}")
print(f"  Min: {all_scaled_features.min(dim=0).values.numpy()}")
print(f"  Max: {all_scaled_features.max(dim=0).values.numpy()}")

# Check novelty feature (slot 10) - not scaled
novelty_features = torch.cat([g.x[:, 10:11] for g in scaled_train], dim=0)
print(f"\nNovelty feature (slot 10, NOT scaled):")
print(f"  Mean: {novelty_features.mean().item():.4f}")
print(f"  Std: {novelty_features.std().item():.4f}")
print(f"  Min: {novelty_features.min().item():.4f}")
print(f"  Max: {novelty_features.max().item():.4f}")

# Compute target statistics
print(f"\nTarget statistics (mean-pooled features from next graph):")
train_pairs = build_paired_sequences(
    type('DS', (), {'get': lambda s, i: scaled_train[i], '__len__': lambda s: len(scaled_train)})()
)

targets = []
for graph_t, graph_tp1 in train_pairs[:100]:  # Sample first 100 pairs
    batch = torch.zeros(graph_tp1.x.size(0), dtype=torch.long)
    target = global_mean_pool(graph_tp1.x, batch)
    targets.append(target)

targets = torch.cat(targets, dim=0)
print(f"  Target shape: {targets.shape}")
print(f"  Mean across all targets: {targets.mean(dim=0).numpy()}")
print(f"  Std across all targets: {targets.std(dim=0).numpy()}")
print(f"  Min: {targets.min(dim=0).values.numpy()}")
print(f"  Max: {targets.max(dim=0).values.numpy()}")
print(f"  Range: {targets.min().item():.4f} to {targets.max().item():.4f}")

# Check if targets are normalized
target_std = targets.std().item()
print(f"\nTarget scale assessment:")
print(f"  Overall target std: {target_std:.4f}")
if target_std < 10:
    print(f"  ✓ Targets appear normalized (std < 10)")
else:
    print(f"  ⚠️  Targets appear unnormalized (std = {target_std:.4f})")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print(f"Input features (0-9) are scaled: mean~0, std~1")
print(f"Targets are {'normalized' if target_std < 10 else 'NOT normalized'}")
print(f"\nIf MSE values are in millions, check if:")
print(f"1. Targets are computed from unnormalized original features")
print(f"2. Loss function uses raw (unscaled) target scale")
print(f"3. Prediction head outputs are in unscaled units")
