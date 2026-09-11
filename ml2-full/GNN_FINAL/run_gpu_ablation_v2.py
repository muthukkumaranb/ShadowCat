#!/usr/bin/env python
"""Run GPU ablation with real z(t) embeddings - version with direct print output."""
import sys
import torch
from pathlib import Path

# Force output flushing
class FlushPrinter:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sys.stdout.flush()
        sys.stderr.flush()

from ml2.configs import load_config
from ml2.data.canonical_ucs import load_canonical_ucs, build_canonical_graphs
from ml2.data.ml1_interface import ML1Adapter
from ml2.experiments.ablation import run_ablation

print("="*80)
print("GPU ABLATION: FusedModel vs TemporalOnlyBaseline")
print("="*80)
print()

root = Path('ml2')

print("[1/5] Loading configuration...", flush=True)
config = load_config(root / 'configs' / 'ml2_config.yaml')
print("      ✓ Config loaded", flush=True)

print("[2/5] Loading canonical UCS graphs...", flush=True)
w, e, lookup = load_canonical_ucs(root / 'data' / 'ucs')
graphs = build_canonical_graphs(w, e, lookup)
print(f"      ✓ {len(graphs)} graphs built", flush=True)

print("[3/5] Injecting ML1 novelty scores...", flush=True)
window_starts = [graph.timestamp for graph in graphs]
adapter = ML1Adapter(config['ml1_interface']['output_dir'], mock_mode=False)
graphs = adapter.inject_novelty_into_graphs(graphs, window_starts)
print("      ✓ Novelty injected", flush=True)

print("[4/5] Preparing device...", flush=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"      ✓ Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})", flush=True)

print("[5/5] Running ablation (may take 30-60 minutes)...", flush=True)
print()

try:
    results = run_ablation(
        graphs, 
        config=config, 
        K_values=[1, 2, 3], 
        output_dir=str(root / 'results' / 'ablation_final'), 
        device=device, 
        verbose=True
    )
    
    print()
    print("="*80)
    print("ABLATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nResults Summary:")
    for model_type, model_results in results.items():
        print(f"\n{model_type.upper()}:")
        print(f"  Next-state error: {model_results['next_state_error']:.6f}")
        print(f"  Rollout errors:")
        for K, re in model_results['rollout_errors'].items():
            print(f"    K={K}: mean_mse={re['mean_mse']:.6f}, mean_mae={re.get('mean_mae', 'N/A')}")
    
    print("\nDetailed results saved to: ml2/results/ablation_final/")
    
except Exception as e:
    print()
    print("ERROR during ablation:")
    print(str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
