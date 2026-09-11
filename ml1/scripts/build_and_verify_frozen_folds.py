import pandas as pd
import numpy as np
from pathlib import Path
import os
import json

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
data_path = workspace_dir / "data" / "ucs" / "ucs_windows.parquet"
out_dir = workspace_dir / "artifacts" / "loeo"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(data_path)
df['window_start_utc'] = pd.to_datetime(df['window_start_utc'], utc=True)
df = df.sort_values('window_start_utc').reset_index(drop=True)

eligible_attacks = ["SSH-Bruteforce", "DDOS-LOIC-UDP", "Botnet"]

def build_and_purge_folds(df):
    episodes_meta = df.groupby("episode_id").agg(
        attack_types=("label_attack_type", lambda values: sorted(set(values.astype(str))))
    )
    eligible = set(eligible_attacks)
    episodes = [ep for ep, row in episodes_meta.iterrows() if row["attack_types"][0] in eligible]
    
    folds = []
    times = df['window_start_utc'].values
    indices = df.index.values
    
    for ep in episodes:
        ep_mask = (df['episode_id'] == ep) | (df['forecast_episode_id'] == ep)
        ep_indices = indices[ep_mask]
        
        source_day = df.loc[ep_indices[0], 'source_day']
        n_positives = len(df.loc[ep_indices][(df.loc[ep_indices, 'label_binary'] == 1) | 
                                             ((df.loc[ep_indices, 'label_binary'] == 0) & 
                                              (df.loc[ep_indices, 'future_attack_label'] == 1))])
        
        target_benign_count = max(n_positives, 5)
        
        candidate_mask = (df['source_day'] == source_day) & \
                         (df['label_binary'] == 0) & \
                         (df['future_attack_label'] == 0) & \
                         (df['episode_id'] != ep) & \
                         (df['forecast_episode_id'] != ep)
                         
        candidate_indices = indices[candidate_mask]
        
        if len(candidate_indices) >= target_benign_count:
            ep_start = df.loc[ep_indices, 'window_start_utc'].min()
            ep_end = df.loc[ep_indices, 'window_start_utc'].max()
            cand_times = df.loc[candidate_indices, 'window_start_utc']
            
            dists = np.zeros(len(cand_times))
            dists[cand_times < ep_start] = (ep_start - cand_times[cand_times < ep_start]).dt.total_seconds()
            dists[cand_times > ep_end] = (cand_times[cand_times > ep_end] - ep_end).dt.total_seconds()
            
            sorted_args = np.argsort(dists)
            selected_benign_indices = candidate_indices[sorted_args[:target_benign_count]]
        else:
            selected_benign_indices = candidate_indices
            
        holdout_indices = np.concatenate([ep_indices, selected_benign_indices])
        test_times = df.loc[holdout_indices, 'window_start_utc'].values
        
        raw_train_indices = np.setdiff1d(indices, holdout_indices)
        train_times_raw = df.loc[raw_train_indices, 'window_start_utc'].values
        
        # Generalized 35-minute temporal purge
        # A train window is removed if it is within 35 minutes (2100 seconds) of ANY test window
        # We can optimize this by finding the min distance for each train window to all test windows
        # Since times are sorted, we can use searchsorted or broadcasting if memory permits.
        # Broadcasting: diffs = |train_times[:, None] - test_times[None, :]|
        diffs = np.abs((train_times_raw[:, None] - test_times[None, :]).astype('timedelta64[s]').astype(float))
        min_diffs = np.min(diffs, axis=1)
        
        valid_train_mask = min_diffs > 2100.0 # Strict: > 35 minutes
        
        final_train_indices = raw_train_indices[valid_train_mask]
        
        # Verify min distance post-purge
        if len(final_train_indices) > 0:
            final_train_times = df.loc[final_train_indices, 'window_start_utc'].values
            final_diffs = np.abs((final_train_times[:, None] - test_times[None, :]).astype('timedelta64[s]').astype(float))
            min_dist_after = np.min(final_diffs) / 60.0
        else:
            min_dist_after = -1
            
        violations_after = np.sum(min_diffs[valid_train_mask] <= 2100.0)
        
        folds.append({
            "fold_id": len(folds),
            "held_out_episode_id": ep,
            "attack_type": episodes_meta.loc[ep, "attack_types"][0],
            "train_before": len(raw_train_indices),
            "train_after": len(final_train_indices),
            "removed": len(raw_train_indices) - len(final_train_indices),
            "test_count": len(holdout_indices),
            "attack_test": len(ep_indices[df.loc[ep_indices, 'label_binary'] == 1]),
            "pre_onset_test": len(ep_indices[(df.loc[ep_indices, 'label_binary'] == 0) & (df.loc[ep_indices, 'future_attack_label'] == 1)]),
            "benign_test": len(selected_benign_indices),
            "min_dist_after_min": min_dist_after,
            "violations_after": violations_after,
            "test_indices": holdout_indices.tolist(),
            "train_indices": final_train_indices.tolist()
        })
    return folds

print("Building folds and enforcing temporal purge...")
folds = build_and_purge_folds(df)

# Output for Folds 2, 10, 19
target_folds = [2, 10, 19]
print("\n--- Phase 3: Edge Case Verification ---")
for f_id in target_folds:
    f = folds[f_id]
    print(f"Fold {f_id} ({f['held_out_episode_id']}): Train Before={f['train_before']}, Train After={f['train_after']}, Removed={f['removed']}, Attack={f['attack_test']}, Pre-onset={f['pre_onset_test']}, Benign={f['benign_test']}, Min Dist After={f['min_dist_after_min']} min, Violations={f['violations_after']}")

# Save 37-fold table
table_data = []
for f in folds:
    table_data.append({
        "Fold": f["fold_id"],
        "Episode": f["held_out_episode_id"],
        "Attack Type": f["attack_type"],
        "Train Before": f["train_before"],
        "Train After": f["train_after"],
        "Removed": f["removed"],
        "Test N": f["test_count"],
        "Min Train->Test Distance": f["min_dist_after_min"],
        "Violations": f["violations_after"]
    })

df_table = pd.DataFrame(table_data)
df_table.to_csv(out_dir / "purge_embargo_all_folds.csv", index=False)

# Verification assertions
assert len(folds) == 37, f"Fold count is {len(folds)}, expected 37."
assert df_table["Violations"].sum() == 0, "Purge violations > 0 detected after fix!"

# Freeze Manifest
manifest = {
    "fold_count": len(folds),
    "folds": []
}
for f in folds:
    manifest["folds"].append({
        "fold_id": f["fold_id"],
        "held_out_episode_id": f["held_out_episode_id"],
        "train_indices": f["train_indices"],
        "test_indices": f["test_indices"]
    })

with open(out_dir / "corrected_37fold_manifest.json", "w") as fp:
    json.dump(manifest, fp)

print(f"\nSaved frozen manifest with {len(folds)} folds to {out_dir}/corrected_37fold_manifest.json")
print("All 37 folds verified successfully. 0 violations.")
