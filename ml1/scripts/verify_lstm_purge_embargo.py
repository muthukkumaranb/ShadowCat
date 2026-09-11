"""
Verify LSTM Purge/Embargo

For every LOEO fold, calculate actual temporal distances between train
and test windows to ensure no training window is within the required 35-minute protection band.
"""
import pandas as pd
import numpy as np
import sys
import os
import json
from pathlib import Path

def main():
    workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    manifest_path = workspace_dir / 'artifacts' / 'loeo' / 'corrected_37fold_manifest.json'
    data_path = workspace_dir / 'data' / 'ucs' / 'ucs_windows.parquet'
    
    out_dir = workspace_dir / 'artifacts' / 'lstm_diagnostics'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'purge_embargo_audit.csv'

    print("=== Verifying LSTM Purge/Embargo ===")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    df = pd.read_parquet(data_path)
    timestamps = pd.to_datetime(df['window_start_utc'], utc=True)
    
    results = []
    
    for fold in manifest['folds']:
        fold_id = fold['fold_id']
        train_indices = np.array(fold['train_indices'])
        test_indices = np.array(fold['test_indices'])
        
        train_times = timestamps.iloc[train_indices].values
        test_times = timestamps.iloc[test_indices].values
        
        # Calculate min distance
        # We need the absolute time difference between any train window and any test window
        if len(test_times) == 0 or len(train_times) == 0:
            continue
            
        # Using broadcasting
        # This can be large (e.g. 2000 x 50), so we can compute it safely
        train_times_mat = train_times[:, None]
        test_times_mat = test_times[None, :]
        
        diff = np.abs(train_times_mat - test_times_mat)
        min_diff = np.min(diff)
        min_diff_minutes = min_diff / np.timedelta64(1, 'm')
        
        max_diff = np.max(diff)
        max_diff_minutes = max_diff / np.timedelta64(1, 'm')
        
        results.append({
            'fold_id': fold_id,
            'held_out_episode': fold['held_out_episode_id'],
            'min_distance_minutes': min_diff_minutes,
            'max_distance_minutes': max_diff_minutes,
            'passed': min_diff_minutes >= 35.0
        })
        
    res_df = pd.DataFrame(results)
    print(res_df[['fold_id', 'held_out_episode', 'min_distance_minutes', 'passed']].to_string(index=False))
    
    all_passed = res_df['passed'].all()
    print(f"\nAll folds passed 35-minute embargo check: {all_passed}")
    
    res_df.to_csv(out_csv, index=False)
    print(f"Saved audit to {out_csv}")
    
    if not all_passed:
        sys.exit(1)

if __name__ == '__main__':
    main()
