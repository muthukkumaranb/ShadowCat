import pandas as pd
import numpy as np
from pathlib import Path
import os
import json

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
data_path = workspace_dir / "data" / "ucs" / "ucs_windows.parquet"
out_dir = workspace_dir / "artifacts" / "loeo_protocol_audit"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(data_path)
df['window_start_utc'] = pd.to_datetime(df['window_start_utc'], utc=True)
df = df.sort_values('window_start_utc').reset_index(drop=True)

eligible_attacks = ["SSH-Bruteforce", "DDOS-LOIC-UDP", "Botnet"]
episode_metadata = df.groupby("episode_id").agg(
    attack_types=("label_attack_type", lambda values: sorted(set(values.astype(str)))),
)
episodes = [ep for ep, row in episode_metadata.iterrows() if row["attack_types"][0] in eligible_attacks]

records = []
for ep in episodes:
    # Onset evaluation includes both the attack windows (episode_id) and pre-onset windows (forecast_episode_id)
    ep_rows = df[(df['episode_id'] == ep) | (df['forecast_episode_id'] == ep)]
    if ep_rows.empty:
        continue
        
    source_day = ep_rows['source_day'].iloc[0]
    ep_start = ep_rows['window_start_utc'].min()
    ep_end = ep_rows['window_start_utc'].max()
    n_attack = len(ep_rows[ep_rows['label_binary'] == 1])
    n_onset_pos = len(ep_rows[(ep_rows['label_binary'] == 0) & (ep_rows['future_attack_label'] == 1)])
    n_positives = n_attack + n_onset_pos # Total positive units we want to evaluate against
    
    # Target 1:1 ratio with total positive windows for this episode (minimum 5 benign to ensure FPR can be calculated)
    target_benign_count = max(n_positives, 5) 
    
    # Candidate benign windows: same day, not part of the held-out episode (neither episode_id nor forecast_episode_id)
    day_benign = df[(df['source_day'] == source_day) & 
                    (df['label_binary'] == 0) & 
                    (df['episode_id'] != ep) & 
                    (df['forecast_episode_id'] != ep)]
    
    n_day_benign = len(day_benign)
    
    # Compute distances to the episode bounds
    distances = []
    if n_day_benign > 0:
        for idx, row in day_benign.iterrows():
            t = row['window_start_utc']
            if t < ep_start:
                distances.append((ep_start - t).total_seconds())
            elif t > ep_end:
                distances.append((t - ep_end).total_seconds())
            else:
                distances.append(0) # Should not happen if episode is contiguous, but just in case
                
        day_benign = day_benign.copy()
        day_benign['dist'] = distances
        day_benign = day_benign.sort_values('dist')
        
        # Check if we can satisfy the 1:1 ratio
        can_satisfy_1_1 = n_day_benign >= target_benign_count
        closest_n = day_benign.head(target_benign_count)
        max_dist_used = closest_n['dist'].max() if can_satisfy_1_1 else None
    else:
        can_satisfy_1_1 = False
        max_dist_used = None
        
    records.append({
        "held_out_episode_id": ep,
        "source_day": source_day,
        "n_positives": n_positives,
        "target_benign": target_benign_count,
        "n_same_day_benign": n_day_benign,
        "can_satisfy_1_1_same_day": can_satisfy_1_1,
        "max_dist_used_sec": max_dist_used
    })

df_res = pd.DataFrame(records)
df_res.to_csv(out_dir / "benign_sampling_feasibility.csv", index=False)

print(f"Total episodes: {len(episodes)}")
print(f"Episodes that can satisfy 1:1 from same day: {df_res['can_satisfy_1_1_same_day'].sum()}")
print(f"Episodes requiring fallback: {len(episodes) - df_res['can_satisfy_1_1_same_day'].sum()}")
if df_res['can_satisfy_1_1_same_day'].sum() < len(episodes):
    print("Episodes requiring fallback:")
    print(df_res[~df_res['can_satisfy_1_1_same_day']][['held_out_episode_id', 'source_day', 'target_benign', 'n_same_day_benign']])
