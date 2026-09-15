"""
Export the exact inference normalization contract v4 for the multi-day real PCAP pipeline.
Produces:
  ml1/artifacts/lstm/inference_scaler_v4.yaml
  ml1/artifacts/lstm/inference_feature_order_v4.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

SCALER_PARAMS = REPO_ROOT.parent / "data-engineering" / "data" / "ucs" / "scaler_params.yaml"
UCS_WINDOWS = REPO_ROOT.parent / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"

OUTPUT_SCALER = REPO_ROOT / "artifacts" / "lstm" / "inference_scaler_v4.yaml"
OUTPUT_FEATURES = REPO_ROOT / "artifacts" / "lstm" / "inference_feature_order_v4.json"

UCS_METADATA_COLUMNS = {
    "window_id", "window_start_utc", "window_end_utc", "source_day", "split",
    "forecast_episode_id", "episode_id"
}
UCS_LABEL_COLUMNS = {
    "label_binary", "label_attack_type", "future_attack_label", "has_malicious_flows", "raw_label_dominant"
}

PACKET_12_FEATURES = {
    "pkt_ttl_min", "pkt_ttl_max", "pkt_ttl_std", "pkt_ttl_mode",
    "pkt_frag_mf_count", "pkt_frag_df_count",
    "pkt_payload_size_p25", "pkt_payload_size_p50",
    "pkt_payload_size_p75", "pkt_payload_size_p95",
    "pkt_tcp_retrans_count", "pkt_port_scan_seq_score"
}

def main():
    print(f"[*] Reading canonical scaler params: {SCALER_PARAMS}")
    with open(SCALER_PARAMS, "r", encoding="utf-8") as f:
        scaler_data = yaml.safe_load(f)
    features_dict = scaler_data.get("features", scaler_data)

    print(f"[*] Reading UCS windows schema: {UCS_WINDOWS}")
    ucs_df = pd.read_parquet(UCS_WINDOWS)
    
    all_cols = [c for c in ucs_df.columns if c not in UCS_METADATA_COLUMNS and c not in UCS_LABEL_COLUMNS]
    mask_cols = [c for c in all_cols if c.startswith("mask_")]
    feat_cols = [c for c in all_cols if not c.startswith("mask_")]

    flow_cols = [c for c in feat_cols if c not in PACKET_12_FEATURES]
    pkt_cols = [c for c in feat_cols if c in PACKET_12_FEATURES]
    
    ordered_model_features = flow_cols + mask_cols + pkt_cols
    print(f"    Ordered features count: {len(ordered_model_features)} (Flow: {len(flow_cols)}, Masks: {len(mask_cols)}, Packet: {len(pkt_cols)})")

    order_payload = {
        "version": "v4.0",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_features": len(ordered_model_features),
        "features": ordered_model_features,
        "feature_segments": {
            "flow_features": {"start": 0, "end": len(flow_cols), "count": len(flow_cols)},
            "presence_masks": {"start": len(flow_cols), "end": len(flow_cols) + len(mask_cols), "count": len(mask_cols)},
            "packet_features": {"start": len(flow_cols) + len(mask_cols), "end": len(ordered_model_features), "count": len(pkt_cols)},
        }
    }

    with open(OUTPUT_FEATURES, "w", encoding="utf-8") as f:
        json.dump(order_payload, f, indent=2)
    print(f"[+] Exported {OUTPUT_FEATURES}")

    scaler_payload = {
        "version": "v4.0",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_features": len(features_dict),
        "features": features_dict
    }

    with open(OUTPUT_SCALER, "w", encoding="utf-8") as f:
        yaml.dump(scaler_payload, f, sort_keys=False)
    print(f"[+] Exported {OUTPUT_SCALER}")

if __name__ == "__main__":
    main()
