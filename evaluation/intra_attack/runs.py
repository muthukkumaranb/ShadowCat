import pandas as pd
import numpy as np
import json
import logging
import os

logging.basicConfig(level=logging.INFO)

def main():
    parquet_path = 'data-engineering/data/ucs/ucs_windows_models_v1.parquet'
    if not os.path.exists(parquet_path):
        # fall back just in case
        return
        
    df = pd.read_parquet(parquet_path)
    df = df.sort_values('window_start_utc').reset_index(drop=True)
    
    # An attack run is a maximal sequence of consecutive windows (1-minute gaps)
    # with label_binary = 1 AND label_attack_type != "Benign".
    
    runs = []
    current_run = None
    
    for i, row in df.iterrows():
        is_attack = row.get('label_binary', 0) == 1 and row.get('label_attack_type', '') != 'Benign'
        if is_attack:
            if current_run is None:
                current_run = {
                    'episode_id': row.get('episode_id', ''),
                    'family': row.get('label_attack_type', ''),
                    'start': row['window_start_utc'],
                    'end': row['window_start_utc'],
                    'length': 1,
                    'windows': [i],
                    'censored': False
                }
            else:
                gap = row['window_start_utc'] - current_run['end']
                if gap <= pd.Timedelta(minutes=1) and row.get('label_attack_type', '') == current_run['family']:
                    current_run['end'] = row['window_start_utc']
                    current_run['length'] += 1
                    current_run['windows'].append(i)
                else:
                    runs.append(current_run)
                    current_run = {
                        'episode_id': row.get('episode_id', ''),
                        'family': row.get('label_attack_type', ''),
                        'start': row['window_start_utc'],
                        'end': row['window_start_utc'],
                        'length': 1,
                        'windows': [i],
                        'censored': False
                    }
        else:
            if current_run is not None:
                runs.append(current_run)
                current_run = None
                
    if current_run is not None:
        runs.append(current_run)
        
    # Check censorship (if gap > 1 min or day boundary)
    for r in runs:
        last_idx = r['windows'][-1]
        if last_idx + 1 < len(df):
            next_row = df.iloc[last_idx + 1]
            gap = next_row['window_start_utc'] - r['end']
            if gap > pd.Timedelta(minutes=1):
                r['censored'] = True
        else:
            r['censored'] = True
            
    num_runs = len(runs)
    len_ge_6 = sum(1 for r in runs if r['length'] >= 6)
    
    # Per-K count where attack ends within (t, t+K]
    in_attack_windows = []
    for r in runs:
        in_attack_windows.extend(r['windows'])
        
    ends_within_K = {k: 0 for k in range(1, 6)}
    for r in runs:
        last_window = r['windows'][-1]
        for w in r['windows']:
            dist = last_window - w
            for k in range(1, 6):
                if 0 <= dist < k:
                    ends_within_K[k] += 1
                    
    in_attack_windows_ending_within_5 = ends_within_K[5]
    n_run_endings = num_runs
    
    results = {
        'num_runs': num_runs,
        'len_ge_6': len_ge_6,
        'in_attack_windows_ending_within_5': in_attack_windows_ending_within_5,
        'n_run_endings': n_run_endings
    }
    
    with open('evaluation/intra_attack/runs.json', 'w') as f:
        json.dump(results, f)
        
    with open('evaluation/intra_attack/RUNS.md', 'w') as f:
        if len_ge_6 < 20 or in_attack_windows_ending_within_5 < 20:
            f.write(f"intra-attack forecasting not evaluable on this dataset\n")
            f.write(f"Counts: len>=6: {len_ge_6}, in_attack_windows_ending_within_5: {in_attack_windows_ending_within_5}, n_run_endings: {n_run_endings}\n")
            print("STOP")
        else:
            f.write(f"# Runs\nNum runs: {num_runs}\nLen >= 6: {len_ge_6}\nin_attack_windows_ending_within_5: {in_attack_windows_ending_within_5}\nn_run_endings: {n_run_endings}\n")
            print("PROCEED")

if __name__ == '__main__':
    main()
