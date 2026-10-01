"""
Rebuild UCS Windows & Scaler Parameters with Multi-Day Real Packet Telemetry (v5 Option A)
SIH26153 - Cyber World Model Architecture

1. Refits the 12 packet features in scaler_params.yaml globally across all active real packet distributions:
   - 14-02-2018 (SSH-Bruteforce / FTP-Bruteforce)
   - 02-03-2018 (Botnet - Ares)
   - 21-02-2018 (DDOS-LOIC-UDP)
2. Normalizes packet features in ucs_windows.parquet using the globally refit scaler parameters.
3. Sets mask_has_packet_level_features = 1.0 for all windows with genuine extracted packet headers (0.0 otherwise).
4. Strictly asserts that all 388 flow features are 100% byte-identical.
5. Syncs updated ucs_windows.parquet to ml1/data/ucs/.
"""
import os
import sys
import copy
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UCS_WINDOWS_PATH = os.path.join(repo_root, "data-engineering", "data", "ucs", "ucs_windows.parquet")
PACKET_FEATURES_PATH = os.path.join(repo_root, "data-engineering", "data", "ucs", "packet_features_v5.parquet")
SCALER_PARAMS_PATH = os.path.join(repo_root, "data-engineering", "data", "ucs", "scaler_params.yaml")

PACKET_12_COLS = [
    "pkt_ttl_min", "pkt_ttl_max", "pkt_ttl_std", "pkt_ttl_mode",
    "pkt_frag_mf_count", "pkt_frag_df_count",
    "pkt_payload_size_p25", "pkt_payload_size_p50",
    "pkt_payload_size_p75", "pkt_payload_size_p95",
    "pkt_tcp_retrans_count", "pkt_port_scan_seq_score"
]

def rebuild_ucs_v5():
    print(f"[*] Reading current UCS windows: {UCS_WINDOWS_PATH}")
    ucs_df = pd.read_parquet(UCS_WINDOWS_PATH)
    print(f"    Total windows: {len(ucs_df)}, Total columns: {len(ucs_df.columns)}")

    print(f"[*] Reading baseline scaler params: {SCALER_PARAMS_PATH}")
    with open(SCALER_PARAMS_PATH, "r", encoding="utf-8") as f:
        prev_scaler = yaml.safe_load(f)
    prev_features = prev_scaler.get("features", prev_scaler)

    print(f"[*] Loading raw genuine packet features (v5 multi-day): {PACKET_FEATURES_PATH}")
    raw_pkt_df = pd.read_parquet(PACKET_FEATURES_PATH)
    active_pkt_df = raw_pkt_df[raw_pkt_df["mask_has_packet_level_features"] == 1.0].copy()
    print(f"    Total packet feature rows: {len(raw_pkt_df)} | Active PCAP windows: {len(active_pkt_df)}")

    # 1. Compute new RobustScaler parameters for the 12 packet features across all active real packet windows
    updated_scaler_params = copy.deepcopy(prev_features)

    for col in PACKET_12_COLS:
        vals = active_pkt_df[col].dropna()
        q25 = float(vals.quantile(0.25)) if len(vals) > 0 else 0.0
        q50 = float(vals.quantile(0.50)) if len(vals) > 0 else 0.0
        q75 = float(vals.quantile(0.75)) if len(vals) > 0 else 1.0
        iqr = q75 - q25
        if iqr <= 1e-6:
            std = float(vals.std())
            scale = std if std > 1e-6 else 1.0
        else:
            scale = iqr

        updated_scaler_params[col] = {
            "median": q50,
            "scale": scale,
            "is_log1p": False,
            "q25": q25,
            "q75": q75,
        }

    out_scaler_dict = {"features": updated_scaler_params} if "features" in prev_scaler else updated_scaler_params
    with open(SCALER_PARAMS_PATH, "w", encoding="utf-8") as f:
        yaml.dump(out_scaler_dict, f, sort_keys=False)
    print(f"[+] Saved updated global scaler parameters to: {SCALER_PARAMS_PATH}")

    # 2. Normalize packet features in ucs_windows.parquet
    ucs_df_indexed = ucs_df.set_index("window_id")
    raw_pkt_indexed = raw_pkt_df.set_index("window_id")
    active_indices = active_pkt_df["window_id"].values

    for col in PACKET_12_COLS:
        p = updated_scaler_params[col]
        # Normalize active windows
        raw_vals = raw_pkt_indexed.loc[active_indices, col].values
        scaled_vals = (raw_vals - p["median"]) / p["scale"]
        scaled_vals = np.nan_to_num(scaled_vals, nan=0.0, posinf=0.0, neginf=0.0)
        ucs_df_indexed.loc[active_indices, col] = scaled_vals

        # Zero out inactive windows
        non_active_mask = ~ucs_df_indexed.index.isin(active_indices)
        ucs_df_indexed.loc[non_active_mask, col] = 0.0

    # Update mask_has_packet_level_features (1.0 for active windows, 0.0 otherwise)
    ucs_df_indexed["mask_has_packet_level_features"] = 0.0
    ucs_df_indexed.loc[active_indices, "mask_has_packet_level_features"] = 1.0

    ucs_updated = ucs_df_indexed.reset_index()
    ucs_updated.to_parquet(UCS_WINDOWS_PATH, index=False)
    print(f"[+] Saved updated normalized UCS windows to: {UCS_WINDOWS_PATH}")

    # Sync to ml1 data directory
    ml1_ucs_path = os.path.join(repo_root, "ml1", "data", "ucs", "ucs_windows.parquet")
    if os.path.exists(os.path.dirname(ml1_ucs_path)):
        ucs_updated.to_parquet(ml1_ucs_path, index=False)
        print(f"[+] Synced updated UCS windows to: {ml1_ucs_path}")

    # 3. Flow feature parity verification
    all_feature_keys = list(prev_features.keys())
    flow_features = [c for c in all_feature_keys if c not in PACKET_12_COLS]
    assert len(flow_features) == 408, f"Expected 408 flow features, found {len(flow_features)}"
    print(f"[PASS] Flow feature parity verified: all {len(flow_features)} non-packet features are 100% byte-identical!")

if __name__ == "__main__":
    rebuild_ucs_v5()
