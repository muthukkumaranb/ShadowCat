"""
H-Sweep (Simplified): Run LOEO for H=5, 10, 15 using existing cleaned data
This avoids the pipeline's parquet generation step by loading pre-cleaned intermediate files.
"""
import os
import sys
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from window_aggregator import create_1min_windows
from labeler_and_splits import create_future_attack_labels
from normalizer import normalize_features, fit_and_save_scaler_params
from canonical_mapper import map_canonical_columns
from cleaner import clean_and_normalize_flow_data, impute_missing_flow_values
from ingestion import load_raw_csv
from labeler_and_splits import assign_canonical_labels

def load_config():
    """Load pipeline configuration."""
    with open('configs/pipeline_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_and_process_data(config):
    """Load all cleaned intermediate files and process into merged dataframe."""
    print("[*] Loading pre-cleaned intermediate parquet files...")
    intermediate_dir = config['paths']['intermediate_dir']
    cleaned_files = sorted([f for f in os.listdir(intermediate_dir) if f.startswith('cleaned_') and f.endswith('.parquet')])
    
    if not cleaned_files:
        print("[!] No cleaned intermediate files found. Run full pipeline first.")
        return None
    
    all_dfs = []
    for filename in cleaned_files:
        filepath = os.path.join(intermediate_dir, filename)
        print(f"  -> Loading {filename}")
        df = pd.read_parquet(filepath)
        all_dfs.append(df)
    
    df_merged = pd.concat(all_dfs, ignore_index=True)
    df_merged = df_merged.sort_values('timestamp_utc').reset_index(drop=True)
    print(f"[+] Loaded {len(df_merged):,} cleaned flow records")
    
    return df_merged

def run_h_sweep(config):
    """Run windowing and LOEO for H=5, 10, 15."""
    # Load cleaned data once
    df_merged = load_and_process_data(config)
    if df_merged is None:
        return {}
    
    horizon_values = [5, 10, 15]
    all_results = {}
    
    for h in horizon_values:
        print(f"\n\n{'#'*80}")
        print(f"# HORIZON H = {h}")
        print(f"{'#'*80}\n")
        
        # Stage 4: Windowing
        print(f"[*] Creating 1-minute windows for H={h}...")
        window_df, win_audit = create_1min_windows(
            df_merged,
            interval_sec=config["windowing"]["interval_sec"]
        )
        print(f"[+] Generated {len(window_df):,} windows")
        
        # Stage 5-7: Labels and splits
        print(f"[*] Assigning canonical labels...")
        window_df = assign_canonical_labels(window_df)
        
        print(f"[*] Creating future attack labels with H={h}...")
        window_df = create_future_attack_labels(
            window_df,
            horizon_windows=h,
            target_col="label_binary",
            future_label_col="future_attack_label"
        )
        
        # Normalization
        print(f"[*] Normalizing features...")
        window_df, scaler = normalize_features(
            window_df,
            method=config["normalization"]["method"],
            log1p_cols=config["normalization"]["log1p_cols"]
        )
        
        # Save this H's windowed data
        output_parquet = f"data/ucs/ucs_windows_h{h}.parquet"
        window_df.to_parquet(output_parquet, index=False)
        print(f"[+] Saved H={h} windows to: {output_parquet}")
        
        # Run LOEO
        print(f"[*] Running LOEO analysis for H={h}...")
        import subprocess
        result = subprocess.run(
            ["python", f"src/gate0_protocol4_loeo.py --data data/ucs/ucs_windows_h{h}.parquet --output data/ucs/loeo_fold_results_h{h}.csv"],
            shell=True,
            capture_output=True,
            text=True
        )
        
        # Parse results
        try:
            fold_results = pd.read_csv(f'data/ucs/loeo_fold_results_h{h}.csv')
            
            h_stats = {}
            
            # Per-type stats
            for atk_type in fold_results['attack_type'].unique():
                type_data = fold_results[fold_results['attack_type'] == atk_type]
                h_stats[atk_type] = {
                    'folds': len(type_data),
                    'a_f1_mean': type_data['a_f1'].mean(),
                    'a_f1_std': type_data['a_f1'].std(),
                    'a_prauc_mean': type_data['a_prauc'].mean(),
                    'a_prauc_std': type_data['a_prauc'].std(),
                    'b_f1_mean': type_data['b_f1'].mean(),
                    'b_f1_std': type_data['b_f1'].std(),
                    'b_prauc_mean': type_data['b_prauc'].mean(),
                    'b_prauc_std': type_data['b_prauc'].std(),
                    'a_wins': (type_data['a_f1'] > type_data['b_f1']).sum(),
                    'b_wins': (type_data['b_f1'] > type_data['a_f1']).sum(),
                }
            
            # Overall stats
            h_stats['overall'] = {
                'total_folds': len(fold_results),
                'a_f1_mean': fold_results['a_f1'].mean(),
                'a_f1_std': fold_results['a_f1'].std(),
                'a_prauc_mean': fold_results['a_prauc'].mean(),
                'a_prauc_std': fold_results['a_prauc'].std(),
                'b_f1_mean': fold_results['b_f1'].mean(),
                'b_f1_std': fold_results['b_f1'].std(),
                'b_prauc_mean': fold_results['b_prauc'].mean(),
                'b_prauc_std': fold_results['b_prauc'].std(),
                'a_wins': (fold_results['a_f1'] > fold_results['b_f1']).sum(),
                'b_wins': (fold_results['b_f1'] > fold_results['a_f1']).sum(),
                'ci_overlap': check_ci_overlap(
                    fold_results['a_f1'].mean(),
                    fold_results['a_f1'].std(),
                    fold_results['b_f1'].mean(),
                    fold_results['b_f1'].std()
                ),
                'avg_pre_onset_windows': fold_results.get('test_total', fold_results.get('test_pos', pd.Series())).mean() if 'test_total' in fold_results.columns else 0,
            }
            
            all_results[h] = h_stats
            print(f"[+] Extracted results for H={h}")
            
        except Exception as e:
            print(f"[!] Error parsing results for H={h}: {e}")
            continue
    
    return all_results

def check_ci_overlap(a_mean, a_std, b_mean, b_std):
    """Check if two confidence intervals overlap."""
    a_lower = a_mean - a_std
    a_upper = a_mean + a_std
    b_lower = b_mean - b_std
    b_upper = b_mean + b_std
    return a_lower <= b_upper and b_lower <= a_upper

def print_comparison_table(all_results):
    """Print side-by-side comparison table."""
    if not all_results or len(all_results) < 3:
        print("\n[!] Not enough results to create comparison table")
        return
    
    horizons = sorted(all_results.keys())
    
    print("\n\n" + "="*120)
    print("CONSOLIDATED H-SWEEP COMPARISON (H=5, 10, 15)")
    print("="*120)
    
    # Overall comparison
    print("\n### OVERALL AGGREGATE METRICS ###\n")
    
    print("| Metric | H=5 Set A | H=5 Set B | H=10 Set A | H=10 Set B | H=15 Set A | H=15 Set B |")
    print("|--------|-----------|-----------|------------|------------|------------|------------|")
    
    # F1 comparison
    row = "| F1 Mean ± Std |"
    for h in horizons:
        if 'overall' in all_results[h]:
            overall = all_results[h]['overall']
            a_str = f"{overall['a_f1_mean']:.4f}±{overall['a_f1_std']:.4f}"
            b_str = f"{overall['b_f1_mean']:.4f}±{overall['b_f1_std']:.4f}"
            row += f" {a_str} / {b_str} |"
    print(row)
    
    # PR-AUC comparison
    row = "| PR-AUC Mean ± Std |"
    for h in horizons:
        if 'overall' in all_results[h]:
            overall = all_results[h]['overall']
            a_str = f"{overall['a_prauc_mean']:.4f}±{overall['a_prauc_std']:.4f}"
            b_str = f"{overall['b_prauc_mean']:.4f}±{overall['b_prauc_std']:.4f}"
            row += f" {a_str} / {b_str} |"
    print(row)
    
    # CI overlap
    row = "| CI Overlap (F1) |"
    for h in horizons:
        if 'overall' in all_results[h]:
            overall = all_results[h]['overall']
            row += f" {overall['ci_overlap']} |"
    print(row)
    
    # Head-to-head
    row = "| Head-to-Head (A wins / B wins) |"
    for h in horizons:
        if 'overall' in all_results[h]:
            overall = all_results[h]['overall']
            row += f" {overall['a_wins']}/{overall['b_wins']} |"
    print(row)
    
    # Per-type breakdown
    print("\n### PER-TYPE BREAKDOWN ###\n")
    
    attack_types = set()
    for h_stats in all_results.values():
        attack_types.update([k for k in h_stats.keys() if k != 'overall'])
    
    for atk_type in sorted(attack_types):
        print(f"\n#### {atk_type} ####\n")
        print("| Metric | H=5 Set A | H=5 Set B | H=10 Set A | H=10 Set B | H=15 Set A | H=15 Set B |")
        print("|--------|-----------|-----------|------------|------------|------------|------------|")
        
        # F1
        row = "| F1 Mean ± Std |"
        for h in horizons:
            if atk_type in all_results[h]:
                a_str = f"{all_results[h][atk_type]['a_f1_mean']:.4f}±{all_results[h][atk_type]['a_f1_std']:.4f}"
                b_str = f"{all_results[h][atk_type]['b_f1_mean']:.4f}±{all_results[h][atk_type]['b_f1_std']:.4f}"
                row += f" {a_str} / {b_str} |"
        print(row)
        
        # PR-AUC
        row = "| PR-AUC Mean ± Std |"
        for h in horizons:
            if atk_type in all_results[h]:
                a_str = f"{all_results[h][atk_type]['a_prauc_mean']:.4f}±{all_results[h][atk_type]['a_prauc_std']:.4f}"
                b_str = f"{all_results[h][atk_type]['b_prauc_mean']:.4f}±{all_results[h][atk_type]['b_prauc_std']:.4f}"
                row += f" {a_str} / {b_str} |"
        print(row)
        
        # Head-to-head
        row = "| Head-to-Head (A/B) |"
        for h in horizons:
            if atk_type in all_results[h]:
                row += f" {all_results[h][atk_type]['a_wins']}/{all_results[h][atk_type]['b_wins']} |"
        print(row)

if __name__ == "__main__":
    config = load_config()
    print("[*] Starting H-Sweep (Simplified)")
    print("[*] This will use existing cleaned data and run LOEO for H=5, 10, 15\n")
    
    all_results = run_h_sweep(config)
    
    if all_results:
        print_comparison_table(all_results)
        with open('h_sweep_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print("\n[+] Results saved to h_sweep_results.json")
    
    print("\n[+] H-Sweep complete!")
