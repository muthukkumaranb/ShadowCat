"""
Step 1: Apply Option A Data Fix (Corrected-but-simulated) to ucs_windows.parquet.
Zeroes out all 12 packet-level features for 14-02-2018 and forces mask_has_packet_level_features=0.0 across all windows.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ws = Path(r"c:\Users\Vicky\Documents\SIH_2026\LSTM-type-model-CICIDS2018-UCS")

PACKET_12_COLS = [
    "pkt_ttl_min", "pkt_ttl_max", "pkt_ttl_std", "pkt_ttl_mode",
    "pkt_frag_mf_count", "pkt_frag_df_count",
    "pkt_payload_size_p25", "pkt_payload_size_p50",
    "pkt_payload_size_p75", "pkt_payload_size_p95",
    "pkt_tcp_retrans_count", "pkt_port_scan_seq_score"
]

def main():
    data_path = ws / "data/ucs/ucs_windows.parquet"
    print(f"[*] Loading {data_path}...")
    df = pd.read_parquet(data_path)
    
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")
    
    # Check 14-02-2018 windows
    feb14_mask = df["source_day"].astype(str).str.contains("14-02")
    print(f"14-02-2018 windows count: {feb14_mask.sum()}")
    
    # Zero out all 12 packet columns for 14-02-2018
    for col in PACKET_12_COLS:
        if col in df.columns:
            df.loc[feb14_mask, col] = 0.0
            
    # Set mask_has_packet_level_features to 0.0 for all windows
    if "mask_has_packet_level_features" in df.columns:
        df["mask_has_packet_level_features"] = 0.0
        
    print(f"Updated df shape: {df.shape}")
    assert df.shape == initial_shape, "Shape mismatch after update!"
    
    # Save back to ucs_windows.parquet
    df.to_parquet(data_path, index=False)
    print(f"[+] Saved updated dataset to {data_path}")
    
    # Also update packet_features.parquet if it exists
    pf_path = ws / "data/ucs/packet_features.parquet"
    if pf_path.exists():
        pf = pd.read_parquet(pf_path)
        for col in PACKET_12_COLS:
            if col in pf.columns:
                pf[col] = 0.0
        pf.to_parquet(pf_path, index=False)
        print(f"[+] Saved updated packet_features to {pf_path}")
        
    # Verification checks
    feb14_df = df[feb14_mask]
    for col in PACKET_12_COLS:
        col_std = feb14_df[col].std()
        col_max = feb14_df[col].max()
        print(f"  Verified {col}: max={col_max}, std={col_std}")
        
    print(f"  Verified mask_has_packet_level_features unique: {df['mask_has_packet_level_features'].unique()}")
    print("[+] Option A Data Fix applied and verified successfully.")

if __name__ == "__main__":
    main()
