"""
H-Sweep (Minimal): Recompute future_attack_label for H=5,10,15 and run LOEO
"""
import os
import sys
import pandas as pd
import numpy as np
import subprocess
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from labeler_and_splits import generate_future_attack_labels

def run_h_sweep():
    """Run LOEO for H=5, 10, 15 using existing ucs_windows.parquet."""
    print("[*] Loading existing ucs_windows.parquet...")
    df = pd.read_parquet('data/ucs/ucs_windows.parquet')
    df = df.sort_values('window_start_utc').reset_index(drop=True)
    print(f"[+] Loaded {len(df):,} windows\n")
    
    # Drop existing future_attack_label to recreate it
    if 'future_attack_label' in df.columns:
        df = df.drop(columns=['future_attack_label'])
    
    all_results = {}
    horizon_values = [5, 10, 15]
    
    for h in horizon_values:
        print(f"\n{'#'*80}")
        print(f"# HORIZON H = {h}")
        print(f"{'#'*80}\n")
        
        # Regenerate future_attack_label for this H
        print(f"[*] Regenerating future_attack_label with H={h}...")
        df_h = df.copy()
        df_h = generate_future_attack_labels(
            df_h,
            horizon_windows=h,
            target_col="label_binary",
            future_col="future_attack_label"
        )
        
        # Count attack labels for sanity check
        pos_count = (df_h['future_attack_label'] == 1).sum()
        print(f"[+] future_attack_label=1: {pos_count:,} windows ({pos_count/len(df_h)*100:.2f}%)\n")
        
        # Save this H's data
        output_parquet = f'data/ucs/ucs_windows_h{h}.parquet'
        df_h.to_parquet(output_parquet, index=False)
        print(f"[+] Saved H={h} windows to: {output_parquet}")
        
        # Run LOEO for this H
        print(f"[*] Running LOEO for H={h}...")
        
        # Create a temporary Python script that imports and runs LOEO with custom data
        loeo_script = f"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'src')

# Manually run the LOEO logic from gate0_protocol4_loeo.py
df = pd.read_parquet('data/ucs/ucs_windows_h{h}.parquet')
df = df.sort_values('window_start_utc').reset_index(drop=True)

# Episode assignment (from gate0_protocol4_loeo.py)
day_shift = df['source_day'] != df['source_day'].shift(1)
bin_shift = df['label_binary'] != df['label_binary'].shift(1)
atk_shift = df['label_attack_type'] != df['label_attack_type'].shift(1)
df['episode_id'] = (day_shift | bin_shift | atk_shift).cumsum()

# Assign forecasting windows to episodes
df['forecast_episode_id'] = np.nan
attack_mask = df['label_binary'] == 1
df.loc[attack_mask, 'forecast_episode_id'] = df.loc[attack_mask, 'episode_id']
typed_attack_episode = df['episode_id'].where(df['label_attack_type'] != 'Benign')
next_typed_attack_episode = typed_attack_episode.bfill()
pre_onset_mask = (df['future_attack_label'] == 1) & (df['label_binary'] == 0)
df.loc[pre_onset_mask, 'forecast_episode_id'] = next_typed_attack_episode[pre_onset_mask]
not_relevant = (df['future_attack_label'] == 0) & (df['label_binary'] == 0)
df.loc[not_relevant, 'forecast_episode_id'] = np.nan

# Import and run LOEO from gate0_protocol4_loeo module
import sys
sys.path.insert(0, '.')
exec(open('src/gate0_protocol4_loeo.py').read())
"""
        
        # Save and run the script
        script_path = f'run_loeo_h{h}_temp.py'
        with open(script_path, 'w') as f:
            f.write(loeo_script)
        
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True
        )
        
        # Parse results
        try:
            fold_results = pd.read_csv('data/ucs/loeo_fold_results.csv')
            
            h_stats = {}
            for atk_type in fold_results['attack_type'].unique():
                type_data = fold_results[fold_results['attack_type'] == atk_type]
                h_stats[atk_type] = {
                    'folds': len(type_data),
                    'a_f1_mean': float(type_data['a_f1'].mean()),
                    'a_f1_std': float(type_data['a_f1'].std()),
                    'a_prauc_mean': float(type_data['a_prauc'].mean()),
                    'a_prauc_std': float(type_data['a_prauc'].std()),
                    'b_f1_mean': float(type_data['b_f1'].mean()),
                    'b_f1_std': float(type_data['b_f1'].std()),
                    'b_prauc_mean': float(type_data['b_prauc'].mean()),
                    'b_prauc_std': float(type_data['b_prauc'].std()),
                    'a_wins': int((type_data['a_f1'] > type_data['b_f1']).sum()),
                    'b_wins': int((type_data['b_f1'] > type_data['a_f1']).sum()),
                }
            
            # Overall stats
            a_f1_vals = fold_results['a_f1'].values
            b_f1_vals = fold_results['b_f1'].values
            a_f1_mean = a_f1_vals.mean()
            a_f1_std = a_f1_vals.std()
            b_f1_mean = b_f1_vals.mean()
            b_f1_std = b_f1_vals.std()
            
            h_stats['overall'] = {
                'total_folds': len(fold_results),
                'a_f1_mean': float(a_f1_mean),
                'a_f1_std': float(a_f1_std),
                'a_prauc_mean': float(fold_results['a_prauc'].mean()),
                'a_prauc_std': float(fold_results['a_prauc'].std()),
                'b_f1_mean': float(b_f1_mean),
                'b_f1_std': float(b_f1_std),
                'b_prauc_mean': float(fold_results['b_prauc'].mean()),
                'b_prauc_std': float(fold_results['b_prauc'].std()),
                'a_wins': int((fold_results['a_f1'] > fold_results['b_f1']).sum()),
                'b_wins': int((fold_results['b_f1'] > fold_results['a_f1']).sum()),
                'ci_overlap': check_ci_overlap(a_f1_mean, a_f1_std, b_f1_mean, b_f1_std),
            }
            
            all_results[h] = h_stats
            print(f"[+] Extracted results for H={h}\n")
            
        except Exception as e:
            print(f"[!] Error parsing results for H={h}: {e}\n")
        
        # Clean up temp script
        if os.path.exists(script_path):
            os.remove(script_path)
    
    return all_results

def check_ci_overlap(a_mean, a_std, b_mean, b_std):
    """Check if confidence intervals overlap."""
    a_lower = a_mean - a_std
    a_upper = a_mean + a_std
    b_lower = b_mean - b_std
    b_upper = b_mean + b_std
    return a_lower <= b_upper and b_lower <= a_upper

def print_comparison_table(all_results):
    """Print side-by-side H=5, 10, 15 comparison."""
    print("\n" + "="*140)
    print("H-SWEEP COMPARISON: F1 AND PR-AUC ACROSS H=5, 10, 15")
    print("="*140)
    
    horizons = sorted(all_results.keys())
    
    # Overall metrics
    print("\n### OVERALL AGGREGATE METRICS ###\n")
    print("| Metric | H=5 Set A | H=5 Set B | H=10 Set A | H=10 Set B | H=15 Set A | H=15 Set B |")
    print("|--------|-----------|-----------|------------|------------|------------|------------|")
    
    # F1 Mean ± Std
    row = "| F1 Mean ± Std |"
    for h in horizons:
        if 'overall' in all_results[h]:
            o = all_results[h]['overall']
            row += f" {o['a_f1_mean']:.4f}±{o['a_f1_std']:.4f} / {o['b_f1_mean']:.4f}±{o['b_f1_std']:.4f} |"
    print(row)
    
    # PR-AUC Mean ± Std
    row = "| PR-AUC Mean ± Std |"
    for h in horizons:
        if 'overall' in all_results[h]:
            o = all_results[h]['overall']
            row += f" {o['a_prauc_mean']:.4f}±{o['a_prauc_std']:.4f} / {o['b_prauc_mean']:.4f}±{o['b_prauc_std']:.4f} |"
    print(row)
    
    # CI Overlap
    row = "| CI Overlap (F1) |"
    for h in horizons:
        if 'overall' in all_results[h]:
            row += f" {all_results[h]['overall']['ci_overlap']} |"
    print(row)
    
    # Head-to-head
    row = "| A wins / B wins |"
    for h in horizons:
        if 'overall' in all_results[h]:
            o = all_results[h]['overall']
            row += f" {o['a_wins']} / {o['b_wins']} |"
    print(row)
    
    # Per-type breakdown
    print("\n### PER-ATTACK-TYPE BREAKDOWN ###\n")
    
    attack_types = set()
    for h_stats in all_results.values():
        attack_types.update(k for k in h_stats.keys() if k != 'overall')
    
    for atk_type in sorted(attack_types):
        print(f"\n{atk_type}:")
        print("| Metric | H=5 Set A | H=5 Set B | H=10 Set A | H=10 Set B | H=15 Set A | H=15 Set B |")
        print("|--------|-----------|-----------|------------|------------|------------|------------|")
        
        row = "| F1 Mean ± Std |"
        for h in horizons:
            if atk_type in all_results[h]:
                t = all_results[h][atk_type]
                row += f" {t['a_f1_mean']:.4f}±{t['a_f1_std']:.4f} / {t['b_f1_mean']:.4f}±{t['b_f1_std']:.4f} |"
        print(row)
        
        row = "| PR-AUC Mean ± Std |"
        for h in horizons:
            if atk_type in all_results[h]:
                t = all_results[h][atk_type]
                row += f" {t['a_prauc_mean']:.4f}±{t['a_prauc_std']:.4f} / {t['b_prauc_mean']:.4f}±{t['b_prauc_std']:.4f} |"
        print(row)

if __name__ == "__main__":
    print("[*] Starting H-Sweep: H=5, 10, 15")
    print("[*] Regenerating future_attack_label for each H and running LOEO\n")
    
    try:
        all_results = run_h_sweep()
        
        if all_results and len(all_results) > 0:
            with open('h_sweep_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n[+] Saved results to h_sweep_results.json")
            
            print_comparison_table(all_results)
        else:
            print("[!] No results to display")
    except Exception as e:
        print(f"[!] Error during H-sweep: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[+] H-Sweep complete!")
