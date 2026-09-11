"""
H-Sweep: Run full pipeline + LOEO for H=5, 10, 15
This script runs the entire pipeline (which regenerates ucs_windows.parquet with different horizon_windows)
and the LOEO analysis for three different horizon values, then consolidates results.
"""
import os
import sys
import subprocess
import pandas as pd
import json
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*80}")
    print(f"[*] {description}")
    print(f"{'='*80}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[!] Command failed: {description}")
        return False
    print(f"[+] Completed: {description}")
    return True

def modify_config(h_value):
    """Modify the pipeline config YAML file to set horizon_windows to h_value (text-based)."""
    config_path = "configs/pipeline_config.yaml"
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Use regex to find and replace horizon_windows value
    content = re.sub(
        r'(horizon_windows:)\s*\d+',
        f'\\1 {h_value}',
        content
    )
    
    with open(config_path, 'w') as f:
        f.write(content)
    
    print(f"[+] Updated config: horizon_windows = {h_value}")

def run_h_sweep():
    """Run pipeline and LOEO for H=5, 10, 15."""
    horizon_values = [5, 10, 15]
    all_results = {}
    
    for h in horizon_values:
        print(f"\n\n{'#'*80}")
        print(f"# HORIZON H = {h}")
        print(f"{'#'*80}")
        
        # Update config
        modify_config(h)
        
        # Run pipeline
        success = run_command(
            "python src/pipeline_runner.py",
            f"Running full pipeline for H={h}"
        )
        if not success:
            print(f"[!] Pipeline failed for H={h}, skipping...")
            continue
        
        # Run LOEO
        success = run_command(
            "python src/gate0_protocol4_loeo.py > loeo_h{}_results.txt 2>&1".format(h),
            f"Running LOEO for H={h}"
        )
        if not success:
            print(f"[!] LOEO failed for H={h}, skipping...")
            continue
        
        # Parse results from loeo_fold_results.csv
        try:
            fold_results = pd.read_csv('data/ucs/loeo_fold_results.csv')
            
            # Calculate per-type and overall statistics
            h_stats = {}
            
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
                    'ties': (type_data['a_f1'] == type_data['b_f1']).sum(),
                }
            
            # Overall stats
            overall_a_f1 = fold_results['a_f1'].mean()
            overall_a_f1_std = fold_results['a_f1'].std()
            overall_a_prauc = fold_results['a_prauc'].mean()
            overall_a_prauc_std = fold_results['a_prauc'].std()
            overall_b_f1 = fold_results['b_f1'].mean()
            overall_b_f1_std = fold_results['b_f1'].std()
            overall_b_prauc = fold_results['b_prauc'].mean()
            overall_b_prauc_std = fold_results['b_prauc'].std()
            
            # CI overlap check
            a_lower = overall_a_f1 - overall_a_f1_std
            a_upper = overall_a_f1 + overall_a_f1_std
            b_lower = overall_b_f1 - overall_b_f1_std
            b_upper = overall_b_f1 + overall_b_f1_std
            overlap = a_lower <= b_upper and b_lower <= a_upper
            
            # Pre-onset windows per fold (average)
            avg_pre_onset_windows = fold_results['test_total'].mean()
            
            h_stats['overall'] = {
                'total_folds': len(fold_results),
                'a_f1_mean': overall_a_f1,
                'a_f1_std': overall_a_f1_std,
                'a_prauc_mean': overall_a_prauc,
                'a_prauc_std': overall_a_prauc_std,
                'b_f1_mean': overall_b_f1,
                'b_f1_std': overall_b_f1_std,
                'b_prauc_mean': overall_b_prauc,
                'b_prauc_std': overall_b_prauc_std,
                'ci_overlap': overlap,
                'a_wins': (fold_results['a_f1'] > fold_results['b_f1']).sum(),
                'b_wins': (fold_results['b_f1'] > fold_results['a_f1']).sum(),
                'ties': (fold_results['a_f1'] == fold_results['b_f1']).sum(),
                'avg_pre_onset_windows': avg_pre_onset_windows,
            }
            
            all_results[h] = h_stats
            print(f"\n[+] Extracted results for H={h}")
            
        except Exception as e:
            print(f"[!] Error parsing results for H={h}: {e}")
            continue
    
    # Save consolidated results
    if all_results:
        with open('h_sweep_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n[+] Saved consolidated results to h_sweep_results.json")
    
    return all_results

def print_comparison_table(all_results):
    """Print side-by-side comparison table for H=5, 10, 15."""
    
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
    
    for h in horizons:
        if 'overall' in all_results[h]:
            overall = all_results[h]['overall']
    
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
    
    # Pre-onset windows
    row = "| Avg Pre-Onset Windows/Fold |"
    for h in horizons:
        if 'overall' in all_results[h]:
            overall = all_results[h]['overall']
            row += f" {overall['avg_pre_onset_windows']:.1f} |"
    print(row)
    
    # Head-to-head
    row = "| Head-to-Head (A wins / B wins / Ties) |"
    for h in horizons:
        if 'overall' in all_results[h]:
            overall = all_results[h]['overall']
            row += f" {overall['a_wins']}/{overall['b_wins']}/{overall['ties']} |"
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
        row = "| Head-to-Head (A/B/Ties) |"
        for h in horizons:
            if atk_type in all_results[h]:
                row += f" {all_results[h][atk_type]['a_wins']}/{all_results[h][atk_type]['b_wins']}/{all_results[h][atk_type]['ties']} |"
        print(row)

if __name__ == "__main__":
    print("\n[*] Starting H-Sweep for H=5, 10, 15")
    print("[*] This will regenerate the parquet file and run LOEO for each H value")
    
    all_results = run_h_sweep()
    
    if all_results:
        print_comparison_table(all_results)
    
    print("\n[+] H-Sweep complete!")
