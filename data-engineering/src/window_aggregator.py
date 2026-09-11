"""
Stage 4: 1-Minute Temporal Windowing & Feature-Presence Masks Module
Aggregates cleaned flow records into 1-minute time windows, computes comprehensive
statistical summaries (mean/std/sum/min/max), and appends feature-presence mask flags.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def create_1min_windows(
    df: pd.DataFrame,
    feature_groups_config: Optional[Dict[str, List[str]]] = None,
    interval_sec: int = 60,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Groups flows into 1-minute windows (based on timestamp_utc) and calculates:
    1. Aggregations (mean, std, sum, min, max) for numeric traffic, timing, packet, and flag features.
    2. Flow counts per window.
    3. Feature presence mask indicators.
    4. Window metadata (window_id, window_start_utc, window_end_utc, source_day).
    """
    if "timestamp_utc" not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp_utc' column.")

    # Assign window bucket timestamp
    freq_str = f"{interval_sec}s"
    df["window_start_utc"] = df["timestamp_utc"].dt.floor(freq_str)

    # Base grouping
    grouped = df.groupby("window_start_utc")

    # Select numerical feature columns for aggregation
    exclude_cols = {
        "timestamp_utc", "window_start_utc", "raw_timestamp", "raw_label",
        "source_file", "source_day", "destination_port", "protocol"
    }
    numeric_feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    # Compute core aggregate statistics
    agg_dict = {}
    for col in numeric_feature_cols:
        agg_dict[col] = ["mean", "std", "sum", "min", "max"]

    window_df = grouped[numeric_feature_cols].agg(agg_dict)
    
    # Flatten MultiIndex columns (e.g., 'byte_count_fwd_mean')
    window_df.columns = [f"{col}_{stat}" for col, stat in window_df.columns]
    window_df = window_df.reset_index()

    # Flow count per window
    flow_counts = grouped.size().reset_index(name="flow_count")
    window_df = window_df.merge(flow_counts, on="window_start_utc", how="left")

    # Distinct destination ports and protocols seen in the window
    port_counts = grouped["destination_port"].nunique().reset_index(name="unique_dst_ports_count")
    proto_counts = grouped["protocol"].nunique().reset_index(name="unique_protocols_count")
    window_df = window_df.merge(port_counts, on="window_start_utc", how="left")
    window_df = window_df.merge(proto_counts, on="window_start_utc", how="left")

    # Fill NaN stds with 0.0 (occurs when a window has only 1 flow)
    std_cols = [c for c in window_df.columns if c.endswith("_std")]
    window_df[std_cols] = window_df[std_cols].fillna(0.0)

    # Window timestamps & metadata
    source_day = df["source_day"].iloc[0] if "source_day" in df.columns and len(df) > 0 else "unknown"
    window_ids = [
        f"W_{source_day}_{ts.strftime('%Y%m%d_%H%M%S')}"
        for ts in window_df["window_start_utc"]
    ]
    
    meta_df = pd.DataFrame({
        "window_end_utc": window_df["window_start_utc"] + pd.Timedelta(seconds=interval_sec),
        "source_day": source_day,
        "window_id": window_ids,
        "mask_has_traffic_volume_features": 1.0,
        "mask_has_flow_timing_features": 1.0,
        "mask_has_packet_level_features": 1.0,
        "mask_has_tcp_flags": 1.0,
        "mask_has_graph_topology": 1.0,
        "mask_has_identity_auth": 0.0,
    })

    window_df = pd.concat([window_df, meta_df], axis=1)

    # Window dominant label and attack presence
    if "raw_label" in df.columns:
        dominant_labels = grouped["raw_label"].agg(lambda s: s.mode().iloc[0] if not s.empty else "Benign").reset_index(name="raw_label_dominant")
        has_attack = grouped["raw_label"].agg(lambda s: bool((s.astype(str).str.lower() != "benign").any())).reset_index(name="has_malicious_flows")
        window_df = window_df.merge(dominant_labels, on="window_start_utc", how="left")
        window_df = window_df.merge(has_attack, on="window_start_utc", how="left")

    window_df = window_df.copy()

    audit = {
        "source_day": source_day,
        "total_windows_generated": len(window_df),
        "total_window_features": len(window_df.columns),
        "min_flows_per_window": int(window_df["flow_count"].min()) if len(window_df) > 0 else 0,
        "max_flows_per_window": int(window_df["flow_count"].max()) if len(window_df) > 0 else 0,
        "median_flows_per_window": float(window_df["flow_count"].median()) if len(window_df) > 0 else 0,
    }

    return window_df, audit
