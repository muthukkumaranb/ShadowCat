#!/usr/bin/env python
"""Debug ablation setup step by step."""
from pathlib import Path
from ml2.configs import load_config
from ml2.data.canonical_ucs import load_canonical_ucs, build_canonical_graphs
from ml2.data.ml1_interface import ML1Adapter
from ml2.data.dataset import get_temporal_splits, build_paired_sequences

root = Path('ml2')
print("[1] Loading config...")
config = load_config(root / 'configs' / 'ml2_config.yaml')
print("     OK")

print("[2] Loading UCS graphs...")
w, e, lookup = load_canonical_ucs(root / 'data' / 'ucs')
graphs = build_canonical_graphs(w, e, lookup)
print(f"     OK - {len(graphs)} graphs")

print("[3] Injecting novelty...")
window_starts = [graph.timestamp for graph in graphs]
adapter = ML1Adapter(config['ml1_interface']['output_dir'], mock_mode=False)
graphs = adapter.inject_novelty_into_graphs(graphs, window_starts)
print("     OK")

print("[4] Warm-up skip...")
graphs = graphs[29:]
print(f"     OK - {len(graphs)} graphs after warm-up exclusion")

print("[5] Creating splits...")
splits_cfg = config.get("splits", {})
train_ds, val_ds, test_ds = get_temporal_splits(
    graphs,
    train_frac=splits_cfg.get("train_frac", 0.70),
    val_frac=splits_cfg.get("val_frac", 0.15),
    test_frac=splits_cfg.get("test_frac", 0.15),
)
print(f"     train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

print("[6] Building trainer...")
from ml2.models.fusion import FusedModel
import torch
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
print("     Model created")

from ml2.training.trainer import ML2Trainer
trainer = ML2Trainer(model, config, device='cuda', checkpoint_dir=str(root / 'results' / 'ablation_final' / 'fused'))
print("     Trainer created")

print("[7] Testing z_store...")
if hasattr(trainer, 'z_store') and trainer.z_store:
    print(f"     z_store ready with {len(trainer.z_store._z_by_timestamp)} cached z(t)")
else:
    print("     WARNING: z_store is None or not initialized")

print("\nAll setup checks passed. Ready for training.")
