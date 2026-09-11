import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import sys

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))
from scripts.build_corrected_loeo_protocol import construct_corrected_loeo_folds

data_path = workspace_dir / "data" / "ucs" / "ucs_windows.parquet"
out_dir = workspace_dir / "artifacts" / "purge_embargo_verification"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(data_path)
df['window_start_utc'] = pd.to_datetime(df['window_start_utc'], utc=True)
df = df.sort_values('window_start_utc').reset_index(drop=True)

eligible_attacks = ["SSH-Bruteforce", "DDOS-LOIC-UDP", "Botnet"]
folds = construct_corrected_loeo_folds(df, eligible_attacks)

target_folds = [2, 10, 19]

results_summary = []
benign_summary = []
closest_pairs = []

for fold_id in target_folds:
    f = next(f for f in folds if f["fold_id"] == fold_id)
    
    test_idx = np.array(f["holdout_indices"])
    train_idx = np.array(f["train_indices"])
    
    test_df = df.iloc[test_idx]
    train_df = df.iloc[train_idx]
    
    attack_test = test_df[test_df['label_binary'] == 1]
    pre_onset_test = test_df[(test_df['label_binary'] == 0) & (test_df['future_attack_label'] == 1)]
    benign_test = test_df[(test_df['label_binary'] == 0) & (test_df['future_attack_label'] == 0)]
    
    # Fast distance calculation
    # For every test window, find min distance to train windows
    test_times = test_df['window_start_utc'].values
    train_times = train_df['window_start_utc'].values
    
    distances = []
    min_dist_overall = float('inf')
    max_dist_overall = -1
    violations = 0
    
    closest_pair_for_fold = None
    min_dist_for_fold = float('inf')
    
    fold_dists = []
    
    for i, t_time in enumerate(test_times):
        t_idx = test_idx[i]
        
        diffs = np.abs((train_times - t_time).astype('timedelta64[s]').astype(float))
        min_idx = np.argmin(diffs)
        min_diff_sec = diffs[min_idx]
        min_diff_min = min_diff_sec / 60.0
        
        nearest_train_id = train_idx[min_idx]
        nearest_train_time = train_times[min_idx]
        
        t_type = "attack" if test_df.iloc[i]['label_binary'] == 1 else ("pre-onset" if test_df.iloc[i]['future_attack_label'] == 1 else "benign")
        
        fold_dists.append({
            "test_window_id": t_idx,
            "test_timestamp": pd.Timestamp(t_time).isoformat(),
            "test_window_type": t_type,
            "nearest_train_window_id": nearest_train_id,
            "nearest_train_timestamp": pd.Timestamp(nearest_train_time).isoformat(),
            "distance_minutes": min_diff_min,
            "is_violation": min_diff_min <= 35.0
        })
        
        if min_diff_min < min_dist_overall:
            min_dist_overall = min_diff_min
        if min_diff_min > max_dist_overall:
            max_dist_overall = min_diff_min
            
        if min_diff_min <= 35.0:
            violations += 1
            
        if min_diff_min < min_dist_for_fold:
            min_dist_for_fold = min_diff_min
            closest_pair_for_fold = {
                "fold_id": fold_id,
                "train_window_id": nearest_train_id,
                "train_timestamp": pd.Timestamp(nearest_train_time).isoformat(),
                "test_window_id": t_idx,
                "test_timestamp": pd.Timestamp(t_time).isoformat(),
                "test_window_type": t_type,
                "temporal_distance_minutes": min_diff_min
            }
    
    pd.DataFrame(fold_dists).to_csv(out_dir / f"fold_{fold_id}_train_test_distances.csv", index=False)
    
    if closest_pair_for_fold:
        closest_pairs.append(closest_pair_for_fold)
    
    results_summary.append({
        "fold_id": fold_id,
        "held_out_episode_id": f["held_out_episode_id"],
        "attack_type": f["attack_type"],
        "test_window_count": len(test_idx),
        "attack_test_window_count": len(attack_test),
        "pre_onset_test_window_count": len(pre_onset_test),
        "benign_test_window_count": len(benign_test),
        "original_training_window_count": len(df) - len(test_idx),
        "post_purge_training_window_count": len(train_idx),
        "minimum_train_to_test_distance_minutes": min_dist_overall,
        "maximum_train_to_test_distance_minutes": max_dist_overall,
        "number_of_training_windows_within_35min": violations,
        "purge_embargo_violations": violations
    })
    
    # Benign specific check
    benign_min_dist = float('inf')
    benign_violations = 0
    for b in fold_dists:
        if b["test_window_type"] == "benign":
            if b["distance_minutes"] < benign_min_dist:
                benign_min_dist = b["distance_minutes"]
            if b["is_violation"]:
                benign_violations += 1
                
    benign_summary.append({
        "fold_id": fold_id,
        "benign_test_windows": len(benign_test),
        "nearest_train_distance_min": benign_min_dist if benign_min_dist != float('inf') else None,
        "benign_purge_violations": benign_violations
    })

pd.DataFrame(results_summary).to_csv(out_dir / "purge_embargo_summary.csv", index=False)
pd.DataFrame(benign_summary).to_csv(out_dir / "benign_test_distance_check.csv", index=False)
pd.DataFrame(closest_pairs).to_csv(out_dir / "closest_train_test_pairs.csv", index=False)

with open(out_dir / "purge_embargo_verification.json", "w") as fp:
    json.dump({
        "folds_checked": target_folds,
        "total_violations": sum(r["purge_embargo_violations"] for r in results_summary),
        "status": "FAIL" if any(r["purge_embargo_violations"] > 0 for r in results_summary) else "PASS"
    }, fp, indent=2)

print("Verification complete.")
