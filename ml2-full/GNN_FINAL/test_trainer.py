#!/usr/bin/env python
"""Test trainer with unbuffered output."""
import sys
import os
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)  # line-buffered

from pathlib import Path
from ml2.configs import load_config
from ml2.data.canonical_ucs import load_canonical_ucs, build_canonical_graphs
from ml2.data.ml1_interface import ML1Adapter
from ml2.data.dataset import get_temporal_splits
from ml2.models.fusion import FusedModel
from ml2.training.trainer import ML2Trainer
import torch

print("Starting mini ablation test on GPU...", flush=True)
root = Path('ml2')
config = load_config(root / 'configs' / 'ml2_config.yaml')

# Load and prepare data
w, e, lookup = load_canonical_ucs(root / 'data' / 'ucs')
graphs = build_canonical_graphs(w, e, lookup)
window_starts = [graph.timestamp for graph in graphs]
adapter = ML1Adapter(config['ml1_interface']['output_dir'], mock_mode=False)
graphs = adapter.inject_novelty_into_graphs(graphs, window_starts)
graphs = graphs[29:]  # warm-up exclusion

# Create minimal splits (use only first 100 for quick test)
graphs_subset = graphs[:100]
splits_cfg = config.get("splits", {})
train_ds, val_ds, test_ds = get_temporal_splits(
    graphs_subset,
    train_frac=0.70,
    val_frac=0.15,
    test_frac=0.15,
)
print(f"Dataset ready: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}", flush=True)

# Create fused model
model = FusedModel(
    node_feature_dim=config['model']['node_feature_dim'],
    graph_hidden_dim=config['model']['hidden_dim'],
    graph_embedding_dim=config['model']['embedding_dim'],
    graph_num_layers=config['model']['num_layers'],
    graph_dropout=config['model']['dropout'],
    z_dim=config['ml1_interface']['z_dim'],
    fusion_hidden_dim=config['model']['fusion_hidden_dim'],
    output_dim=config['model']['node_feature_dim'],
)
print("Model created", flush=True)

# Create trainer
trainer = ML2Trainer(model, config, device='cuda', checkpoint_dir=str(root / 'results' / 'ablation_final' / 'fused_test'))
print(f"Trainer ready on device: {trainer.device}", flush=True)

# Train with minimal epochs
print("Starting training loop...", flush=True)
history = trainer.train(train_ds, val_ds, verbose=True)

print("Training completed!", flush=True)
print(f"Train loss: {history.get('train_loss', [])[:5]}", flush=True)
print(f"Val loss: {history.get('val_loss', [])[:5]}", flush=True)
