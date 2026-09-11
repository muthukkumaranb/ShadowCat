#!/usr/bin/env python
"""Run GPU ablation with real z(t) embeddings."""
from pathlib import Path
from ml2.configs import load_config
from ml2.data.canonical_ucs import load_canonical_ucs, build_canonical_graphs
from ml2.data.ml1_interface import ML1Adapter
from ml2.experiments.ablation import run_ablation

root = Path('ml2')
config = load_config(root / 'configs' / 'ml2_config.yaml')
w, e, lookup = load_canonical_ucs(root / 'data' / 'ucs')
graphs = build_canonical_graphs(w, e, lookup)
window_starts = [graph.timestamp for graph in graphs]
adapter = ML1Adapter(config['ml1_interface']['output_dir'], mock_mode=False)
graphs = adapter.inject_novelty_into_graphs(graphs, window_starts)

print(f"Running ablation with {len(graphs)} graphs on GPU (device='cuda')...")
print(f"FusedModel vs TemporalOnlyBaseline")
print(f"K_values: [1, 2, 3]")
print()

results = run_ablation(
    graphs, 
    config=config, 
    K_values=[1, 2, 3], 
    output_dir=str(root / 'results' / 'ablation_final'), 
    device='cuda', 
    verbose=True
)

print("\n" + "="*80)
print("ABLATION COMPLETED")
print("="*80)
print("\nResults:")
for key, val in results.items():
    print(f"  {key}: {val}")
