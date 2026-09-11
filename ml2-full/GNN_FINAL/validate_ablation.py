#!/usr/bin/env python
"""Validate ablation results for degeneracies."""
import json
import numpy as np
from pathlib import Path

root = Path('ml2/results/ablation_final')

# Load ablation results
ablation_json = root / 'ablation_results.json'
if ablation_json.exists():
    with open(ablation_json) as f:
        results = json.load(f)
    print("Ablation Results Validation:")
    print("="*70)
    for model_type, model_data in results.items():
        print(f"\n{model_type.upper()}:")
        nse = model_data.get('next_state_error', None)
        print(f"  Next-state error: {nse}")
        
        # Check for degeneracies
        if nse is not None:
            if nse == 0:
                print("  ⚠️  WARNING: Zero error - likely degenerate")
            elif np.isnan(nse) or np.isinf(nse):
                print("  ⚠️  ERROR: Invalid (NaN/Inf)")
            elif nse > 1e9:
                print(f"  ℹ️  Very large MSE - verify this is expected scale")
            else:
                print(f"  ✓ Valid error value")
        
        rollout = model_data.get('rollout_errors', {})
        for K, metrics in rollout.items():
            mse = metrics.get('mean_mse')
            print(f"    K={K}: mean_mse={mse:.2f}")

# Check training histories
print("\n\nTraining History Check:")
print("="*70)
for model_dir in ['fused', 'temporal_only']:
    hist_file = root / model_dir / 'training_history.json'
    if hist_file.exists():
        with open(hist_file) as f:
            hist = json.load(f)
        train_losses = hist.get('train_loss', [])
        val_losses = hist.get('val_loss', [])
        print(f"\n{model_dir.upper()}:")
        print(f"  Epochs trained: {len(train_losses)}")
        if len(train_losses) > 0:
            print(f"  Train loss: start={train_losses[0]:.4f}, end={train_losses[-1]:.4f}")
            print(f"  Val loss:   start={val_losses[0]:.4f}, end={val_losses[-1]:.4f}")
            if train_losses[-1] > 1e10:
                print(f"  ⚠️  WARNING: Very high final loss")
            elif train_losses[-1] == 0:
                print(f"  ⚠️  WARNING: Zero loss - likely degenerate")
            else:
                print(f"  ✓ Valid loss trajectory")
    else:
        print(f"\n{model_dir.upper()}: No history file found")
