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

# 1. Episode vs Forecast Episode Analysis (Phase 4)
records = []
eligible_attacks = [
    "SSH-Bruteforce",
    "DDOS-LOIC-UDP",
    "Botnet"
]

episode_metadata = df.groupby("episode_id").agg(
    attack_types=("label_attack_type", lambda values: sorted(set(values.astype(str)))),
)
episodes = []
for ep, row in episode_metadata.iterrows():
    if row["attack_types"][0] in eligible_attacks:
        episodes.append(ep)

for ep in episodes:
    ep_rows = df[df['episode_id'] == ep]
    fc_rows = df[df['forecast_episode_id'] == ep]
    
    n_ep = len(ep_rows)
    n_fc = len(fc_rows)
    
    n_benign_fc = len(fc_rows[fc_rows['label_binary'] == 0])
    n_attack_ep = len(ep_rows[ep_rows['label_binary'] == 1])
    n_pre_onset_fc = len(fc_rows[(fc_rows['label_binary'] == 0) & (fc_rows['future_attack_label'] == 1)])
    
    records.append({
        "attack_episode_id": ep,
        "N_episode_id": n_ep,
        "N_forecast_episode_id": n_fc,
        "N_attack_episode_id": n_attack_ep,
        "N_benign_forecast_episode_id": n_benign_fc,
        "N_pre_onset_forecast_episode_id": n_pre_onset_fc
    })

df_ep_vs_fc = pd.DataFrame(records)
df_ep_vs_fc.to_csv(out_dir / "episode_vs_forecast_episode.csv", index=False)

# 2. Same-Day Benign Analysis (Phase 14 & 15)
same_day_records = []
temporal_records = []

for ep in episodes:
    ep_rows = df[df['episode_id'] == ep]
    if ep_rows.empty:
        continue
    source_day = ep_rows['source_day'].iloc[0]
    attack_onset_time = ep_rows['window_start_utc'].min()
    
    day_rows = df[df['source_day'] == source_day]
    benign_day_rows = day_rows[day_rows['label_binary'] == 0].copy()
    
    n_attack = len(ep_rows)
    n_benign_day = len(benign_day_rows)
    
    if n_benign_day > 0:
        benign_day_rows['distance_to_onset_sec'] = (attack_onset_time - benign_day_rows['window_start_utc']).dt.total_seconds()
        
        n_benign_before = len(benign_day_rows[benign_day_rows['distance_to_onset_sec'] > 0])
        n_benign_after = len(benign_day_rows[benign_day_rows['distance_to_onset_sec'] < 0])
        
        # Temporal analysis
        dists = benign_day_rows['distance_to_onset_sec'].abs()
        t_0_5 = len(dists[dists <= 300])
        t_5_10 = len(dists[(dists > 300) & (dists <= 600)])
        t_10_30 = len(dists[(dists > 600) & (dists <= 1800)])
        t_30_60 = len(dists[(dists > 1800) & (dists <= 3600)])
        t_gt_60 = len(dists[dists > 3600])
    else:
        n_benign_before = 0
        n_benign_after = 0
        t_0_5 = t_5_10 = t_10_30 = t_30_60 = t_gt_60 = 0
        
    same_day_records.append({
        "attack_episode_id": ep,
        "source_day": source_day,
        "N_attack_windows": n_attack,
        "N_same_day_benign_windows": n_benign_day,
        "N_benign_before_onset": n_benign_before,
        "N_benign_after_onset": n_benign_after
    })
    
    temporal_records.append({
        "attack_episode_id": ep,
        "t_0_5_min": t_0_5,
        "t_5_10_min": t_5_10,
        "t_10_30_min": t_10_30,
        "t_30_60_min": t_30_60,
        "t_gt_60_min": t_gt_60
    })

pd.DataFrame(same_day_records).to_csv(out_dir / "same_day_benign_analysis.csv", index=False)
pd.DataFrame(temporal_records).to_csv(out_dir / "temporal_benign_analysis.csv", index=False)

print("Protocol audit completed.")
