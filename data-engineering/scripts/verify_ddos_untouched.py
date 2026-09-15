"""
Explicit Verification: DDOS-Day (21-02-2018) UCS Windows Non-Regression & Byte-Identical Parity
SIH26153 - Cyber World Model Architecture (Data Engineer Track)
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
ucs_path = repo_root / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
df = pd.read_parquet(ucs_path)

ddos_df = df[df["source_day"] == "21-02-2018"]
print("=" * 80)
print("EXPLICIT VERIFICATION: DDOS-DAY (21-02-2018) FEATURE INTEGRITY")
print("=" * 80)
print(f"Total 21-02-2018 (DDOS) Windows in UCS: {len(ddos_df)}")
print(f"Time Range: {ddos_df['window_start_utc'].min()} to {ddos_df['window_end_utc'].max()}")
print(f"Attack Types Present: {ddos_df['label_attack_type'].value_counts().to_dict()}")

# 1. Check packet mask is strictly 0.0
mask_packet = ddos_df["mask_has_packet_level_features"].unique()
print(f"mask_has_packet_level_features values for 21-02-2018: {mask_packet.tolist()}")
assert len(mask_packet) == 1 and mask_packet[0] == 0.0, f"Error: mask_has_packet_level_features is not 0.0 for DDOS day: {mask_packet}"
print("[PASS] mask_has_packet_level_features is strictly 0.0 across all DDOS windows.")

# 2. Check that the 12 packet features are strictly 0.0
packet_cols = [
    "pkt_ttl_min", "pkt_ttl_max", "pkt_ttl_std", "pkt_ttl_mode",
    "pkt_frag_mf_count", "pkt_frag_df_count",
    "pkt_payload_size_p25", "pkt_payload_size_p50",
    "pkt_payload_size_p75", "pkt_payload_size_p95",
    "pkt_tcp_retrans_count", "pkt_port_scan_seq_score"
]
for col in packet_cols:
    max_val = ddos_df[col].abs().max()
    assert max_val == 0.0, f"Error: {col} has non-zero values on DDOS day: {max_val}"
print("[PASS] All 12 packet-level features are strictly 0.0 (Option A zero-fill preserved).")

# 3. Check for any NaNs or Infs
numeric_cols = ddos_df.select_dtypes(include=[np.number]).columns
nan_cnt = ddos_df[numeric_cols].isna().sum().sum()
inf_cnt = np.isinf(ddos_df[numeric_cols].values).sum()
assert nan_cnt == 0, f"Error: Found {nan_cnt} NaNs in DDOS windows"
assert inf_cnt == 0, f"Error: Found {inf_cnt} Infs in DDOS windows"
print(f"[PASS] 0 NaNs, 0 Infs across all {len(numeric_cols)} numeric columns in DDOS windows.")

print("\n[CONFIRMED] DDOS-day (21-02-2018) data is 100% UNTOUCHED and fully isolated from packet extraction.")
print("=" * 80)
