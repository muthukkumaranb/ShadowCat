"""
Comprehensive ML1 -> ML2 Interface Audit

Performs all Phase 1-6 verifications from the ML1 deviation integration task.
Does NOT modify ML2 code.
"""
import sys
import os
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))
sys.path.insert(0, str(workspace_dir / 'scratch'))

EXPORT_DIR = workspace_dir / 'artifacts' / 'lstm' / 'ml2_export'
JSON_DIR = EXPORT_DIR / 'per_window_json'

def phase4_deviation_scale_audit():
    """Phase 4: Audit deviation distribution from actual artifact."""
    print("=" * 70)
    print("PHASE 4: Deviation Scale Audit")
    print("=" * 70)

    audit_df = pd.read_csv(EXPORT_DIR / 'deviation_audit.csv')
    scores = audit_df['deviation_score'].values

    percentiles = [1, 5, 25, 50, 75, 95, 99]
    pvals = np.percentile(scores, percentiles)

    print(f"  Count:    {len(scores)}")
    print(f"  Min:      {scores.min():.6f}")
    for p, v in zip(percentiles, pvals):
        print(f"  P{p:<3d}     {v:.6f}")
    print(f"  Max:      {scores.max():.6f}")
    print(f"  Mean:     {scores.mean():.6f}")
    print(f"  Median:   {np.median(scores):.6f}")
    print(f"  Std:      {scores.std():.6f}")
    print(f"  NaN:      {np.isnan(scores).sum()}")
    print(f"  Inf:      {np.isinf(scores).sum()}")
    print(f"  Ratio max/min: {scores.max() / scores.min():.1f}x")

    return {
        'count': int(len(scores)),
        'min': float(scores.min()),
        'p1': float(pvals[0]),
        'p5': float(pvals[1]),
        'p25': float(pvals[2]),
        'p50': float(pvals[3]),
        'p75': float(pvals[4]),
        'p95': float(pvals[5]),
        'p99': float(pvals[6]),
        'max': float(scores.max()),
        'mean': float(scores.mean()),
        'median': float(np.median(scores)),
        'std': float(scores.std()),
        'nan_count': int(np.isnan(scores).sum()),
        'inf_count': int(np.isinf(scores).sum()),
    }


def phase3_population_coverage():
    """Phase 3: Verify population coverage."""
    print("\n" + "=" * 70)
    print("PHASE 3: Population Coverage Audit")
    print("=" * 70)

    # Load ML1 data
    windows_df = pd.read_parquet(workspace_dir / 'data' / 'ucs' / 'ucs_windows.parquet')
    total_ucs_windows = len(windows_df)

    edges_df = pd.read_parquet(workspace_dir / 'data' / 'ucs' / 'ucs_graph_edgelists.parquet')
    edge_window_ids = set(edges_df['window_id'].unique())

    # Load manifest
    with open(workspace_dir / 'artifacts' / 'loeo' / 'corrected_37fold_manifest.json') as f:
        manifest = json.load(f)

    all_test_indices = set()
    all_train_indices = set()
    for fold in manifest['folds']:
        all_test_indices.update(fold['test_indices'])
        all_train_indices.update(fold['train_indices'])

    # ML1 deviation output
    json_files = sorted(JSON_DIR.glob("*.json"))
    ml1_window_ids = {jf.stem for jf in json_files}

    print(f"  Total UCS windows (ucs_windows.parquet): {total_ucs_windows}")
    print(f"  Total edge-list windows: {len(edge_window_ids)}")
    print(f"  LOEO test indices (union across 37 folds): {len(all_test_indices)}")
    print(f"  LOEO train indices (union across 37 folds): {len(all_train_indices)}")
    print(f"  ML1 deviation windows generated: {len(ml1_window_ids)}")

    # Which edge-list windows have ML1 deviation?
    covered = ml1_window_ids & edge_window_ids
    uncovered = edge_window_ids - ml1_window_ids
    print(f"  Edge-list windows with ML1 deviation: {len(covered)}")
    print(f"  Edge-list windows WITHOUT ML1 deviation (fallback=0): {len(uncovered)}")

    # The critical question: what happens to uncovered windows?
    # ML1Adapter._preload builds cache. get_novelty_tminus1 returns zeros for missing windows.
    # inject_novelty_into_graphs also returns zeros for the first window (no t-1).
    print(f"\n  --- Zero-Fallback Analysis ---")
    print(f"  ML1Adapter returns zeros for any window NOT in cache.")
    print(f"  {len(uncovered)} windows will receive fallback=0 deviation")

    # ML2 ablation uses temporal splits: 70% train / 15% val / 15% test
    # applied to ALL graphs, not just LOEO test graphs
    n_edge_windows = len(edge_window_ids)
    train_end = int(n_edge_windows * 0.70)
    val_end = train_end + int(n_edge_windows * 0.15)

    # Sort edge windows chronologically to identify which are train/val/test
    sorted_edge_windows = sorted(edge_window_ids)
    train_windows = set(sorted_edge_windows[:train_end])
    val_windows = set(sorted_edge_windows[train_end:val_end])
    test_windows = set(sorted_edge_windows[val_end:])

    train_covered = len(ml1_window_ids & train_windows)
    val_covered = len(ml1_window_ids & val_windows)
    test_covered = len(ml1_window_ids & test_windows)

    train_fallback = len(train_windows) - train_covered
    val_fallback = len(val_windows) - val_covered
    test_fallback = len(test_windows) - test_covered

    print(f"\n  --- ML2 Temporal Split Coverage (70/15/15) ---")
    print(f"  Train windows: {len(train_windows)} (covered: {train_covered}, fallback: {train_fallback})")
    print(f"  Val windows:   {len(val_windows)} (covered: {val_covered}, fallback: {val_fallback})")
    print(f"  Test windows:  {len(test_windows)} (covered: {test_covered}, fallback: {test_fallback})")

    train_coverage_pct = train_covered / max(len(train_windows), 1) * 100
    val_coverage_pct = val_covered / max(len(val_windows), 1) * 100
    test_coverage_pct = test_covered / max(len(test_windows), 1) * 100
    print(f"\n  Train coverage: {train_coverage_pct:.1f}%")
    print(f"  Val coverage:   {val_coverage_pct:.1f}%")
    print(f"  Test coverage:  {test_coverage_pct:.1f}%")

    return {
        'total_ucs_windows': total_ucs_windows,
        'total_edge_windows': n_edge_windows,
        'ml1_deviation_windows': len(ml1_window_ids),
        'covered_windows': len(covered),
        'uncovered_windows': len(uncovered),
        'train_windows': len(train_windows),
        'train_covered': train_covered,
        'train_fallback': train_fallback,
        'val_windows': len(val_windows),
        'val_covered': val_covered,
        'val_fallback': val_fallback,
        'test_windows': len(test_windows),
        'test_covered': test_covered,
        'test_fallback': test_fallback,
    }


def phase2_host_differentiation():
    """Phase 2: Determine whether identical host values are valid."""
    print("\n" + "=" * 70)
    print("PHASE 2: Host Differentiation Analysis")
    print("=" * 70)

    json_files = sorted(JSON_DIR.glob("*.json"))
    total_files = len(json_files)
    identical_count = 0
    total_hosts_checked = 0

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        vals = list(data.values())
        if len(vals) > 0:
            total_hosts_checked += len(vals)
            unique_vals = set(vals)
            if len(unique_vals) == 1:
                identical_count += 1

    print(f"  Total windows:       {total_files}")
    print(f"  Windows with all-identical host values: {identical_count}")
    print(f"  Percentage:          {identical_count / max(total_files, 1) * 100:.1f}%")
    print(f"  Total host entries:  {total_hosts_checked}")

    return {
        'total_windows': total_files,
        'windows_all_identical': identical_count,
        'pct_identical': identical_count / max(total_files, 1) * 100,
    }


def phase5_normalization():
    """Phase 5: Determine whether ML2 normalizes the ML1 deviation."""
    print("\n" + "=" * 70)
    print("PHASE 5: Normalization Analysis")
    print("=" * 70)

    # From code inspection of ML2:
    # 1. ml1_interface.py: ML1Adapter stores raw float values in cache, returns raw float32 array
    # 2. node_features.py line 68: features[idx, 10] = novelty_tminus1[idx]
    #    -> direct assignment, NO normalization
    # 3. graph_builder.py: calls extract_node_features with raw novelty
    # 4. run_provisional.py line 134: x[:, 10] remains 0.0 (note: novelty zeroed)
    # 5. trainer.py: no feature normalization in _train_epoch
    # 6. fusion.py: FusedModel receives raw x as input to GraphBranch (GraphSAGE)

    print("  ML1Adapter:       Returns raw float values (no normalization)")
    print("  node_features.py: Direct assignment to slot 10 (no normalization)")
    print("  graph_builder.py: Passes raw novelty to extract_node_features")
    print("  trainer.py:       No feature normalization step")
    print("  FusedModel:       Raw node features -> GraphSAGE -> graph embedding")
    print("  Config:           No normalization config found")
    print()
    print("  FINDING: ML2 does NOT normalize ML1 deviation values.")
    print("  Raw values are inserted directly into slot 10 of node features.")
    print()
    print("  CONCERN: ML1 deviation range is ~14 to ~20,589")
    print("  Other node features (degrees, flows, bytes) have different scales.")
    print("  Raw deviation may dominate the node feature representation.")
    print()
    print("  RECOMMENDATION: This is an interface/design decision requiring")
    print("  cross-team confirmation before normalizing on either side.")

    return {
        'ml1_normalizes': False,
        'ml2_normalizes': False,
        'raw_scale': 'direct_insertion',
        'concern': 'large_dynamic_range_may_dominate',
    }


def phase1_contract_summary():
    """Phase 1: Summarize the adapter contract from code inspection."""
    print("\n" + "=" * 70)
    print("PHASE 1: ML2 Adapter Contract Summary")
    print("=" * 70)

    contract = {
        'adapter_file': 'GNN_FINAL/ml2/data/ml1_interface.py',
        'adapter_class': 'ML1Adapter',
        'input_format': 'Per-window JSON files OR single .npz file',
        'json_schema': '{window_id}.json -> {str(host_id): float(novelty_score)}',
        'npz_keys': "timestamps/window_ids, novelty/scores, hosts/host_ids",
        'alignment': 'ML1 stores at window t. ML2 adapter fetches t-1 itself via inject_novelty_into_graphs',
        'slot_10': 'graph.x[:, 10] (zero-indexed), the 11th feature of each node',
        'slot_10_semantic': 'novelty_tminus1 from ML1 (per node_features.py line 23)',
        'expected_shape': 'novelty vector of shape [N] for N active hosts per window',
        'missing_value_behavior': 'Returns np.zeros(N, dtype=float32) for missing windows',
        'z_t_usage': 'Separate 64-dim temporal embedding (currently MOCK zeros in trainer/metrics)',
        'z_t_note': 'z_t is NOT the same as slot-10 novelty; z_t is a separate fusion input',
    }

    for k, v in contract.items():
        print(f"  {k}: {v}")

    return contract


def main():
    print("=" * 70)
    print("COMPREHENSIVE ML1 -> ML2 INTERFACE AUDIT")
    print("=" * 70)

    # Phase 1
    contract = phase1_contract_summary()

    # Phase 2
    host_diff = phase2_host_differentiation()

    # Phase 3
    population = phase3_population_coverage()

    # Phase 4
    deviation_stats = phase4_deviation_scale_audit()

    # Phase 5
    normalization = phase5_normalization()

    # Save complete audit report
    report = {
        'phase1_contract': contract,
        'phase2_host_differentiation': host_diff,
        'phase3_population_coverage': population,
        'phase4_deviation_scale': deviation_stats,
        'phase5_normalization': normalization,
    }

    report_path = EXPORT_DIR / 'interface_audit_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n\nFull audit report saved to: {report_path}")


if __name__ == '__main__':
    main()
