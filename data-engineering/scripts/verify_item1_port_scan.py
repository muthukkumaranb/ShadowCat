import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
scaler_path = repo_root / "data-engineering" / "data" / "ucs" / "scaler_params.yaml"
ucs_path = repo_root / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"

with open(scaler_path, "r", encoding="utf-8") as f:
    scaler = yaml.safe_load(f)

p = scaler["pkt_port_scan_seq_score"]
med = float(p["median"])
scale = float(p["scale"])

df = pd.read_parquet(ucs_path)
df_14 = df[df["source_day"] == "14-02-2018"].sort_values("window_start_utc").reset_index(drop=True)
df_14["raw_port_scan_score"] = df_14["pkt_port_scan_seq_score"] * scale + med

atk = df_14[df_14["label_binary"] == 1]
sample_indices = np.linspace(0, len(atk) - 1, 16, dtype=int)
sample_atk = atk.iloc[sample_indices]

print("=" * 85)
print("ITEM 1 VERIFICATION: 16 SAMPLE ATTACK WINDOWS RAW PKT_PORT_SCAN_SEQ_SCORE")
print("=" * 85)
print(f"Total attack windows on 14-02-2018: {len(atk)}")
print(f"Total unique raw values in attack: {atk['raw_port_scan_score'].nunique()}")
print(f"Raw statistics: Mean={atk['raw_port_scan_score'].mean():.6f}, Std={atk['raw_port_scan_score'].std():.8f}, Min={atk['raw_port_scan_score'].min():.6f}, Max={atk['raw_port_scan_score'].max():.6f}")
print("-" * 85)
print(f"{'Window ID':<36} | {'Attack Type':<16} | {'Scaled Score':<12} | {'Raw Score (8 dec)':<18}")
print("-" * 85)
for _, row in sample_atk.iterrows():
    w_id = row['window_id']
    atk_type = row['label_attack_type']
    s_score = row['pkt_port_scan_seq_score']
    r_score = row['raw_port_scan_score']
    print(f"{w_id:<36} | {atk_type:<16} | {s_score:<12.6f} | {r_score:<18.8f}")

print("\n" + "=" * 85)
print("BENIGN WINDOWS COMPARISON (N=353)")
print("=" * 85)
ben = df_14[df_14["label_binary"] == 0]
print(f"Raw statistics: Mean={ben['raw_port_scan_score'].mean():.6f}, Std={ben['raw_port_scan_score'].std():.6f}, Min={ben['raw_port_scan_score'].min():.6f}, Max={ben['raw_port_scan_score'].max():.6f}")
print(f"Zero values count (score=0.0): {(ben['raw_port_scan_score'] == 0.0).sum()} / {len(ben)}")
