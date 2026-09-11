"""
Stage 2: Canonical Field Mapping Module
Translates raw 80 CICFlowMeter column names to canonical UCS feature schema.
"""

import os
import yaml
import pandas as pd
from typing import Dict, List, Tuple, Optional


def load_canonical_mapping(config_path: str = "configs/canonical_mapping.yaml") -> Dict[str, any]:
    """Load canonical mapping yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Comprehensive alias map for common CICFlowMeter versions (CIC-IDS2017, CSE-CIC-IDS2018, CICFlowMeter Java/Python variants)
KNOWN_CIC_ALIASES: Dict[str, str] = {
    "destination_port": "destination_port",
    "dst_port": "destination_port",
    "destination port": "destination_port",
    "total_fwd_packets": "packet_count_fwd",
    "total fwd packets": "packet_count_fwd",
    "total_backward_packets": "packet_count_bwd",
    "total backward packets": "packet_count_bwd",
    "total_length_of_fwd_packets": "byte_count_fwd",
    "total length of fwd packets": "byte_count_fwd",
    "total_length_of_bwd_packets": "byte_count_bwd",
    "total length of bwd packets": "byte_count_bwd",
    "fwd_packet_length_max": "fwd_pkt_len_max",
    "fwd packet length max": "fwd_pkt_len_max",
    "fwd_packet_length_min": "fwd_pkt_len_min",
    "fwd packet length min": "fwd_pkt_len_min",
    "fwd_packet_length_mean": "fwd_pkt_len_mean",
    "fwd packet length mean": "fwd_pkt_len_mean",
    "fwd_packet_length_std": "fwd_pkt_len_std",
    "fwd packet length std": "fwd_pkt_len_std",
    "bwd_packet_length_max": "bwd_pkt_len_max",
    "bwd packet length max": "bwd_pkt_len_max",
    "bwd_packet_length_min": "bwd_pkt_len_min",
    "bwd packet length min": "bwd_pkt_len_min",
    "bwd_packet_length_mean": "bwd_pkt_len_mean",
    "bwd packet length mean": "bwd_pkt_len_mean",
    "bwd_packet_length_std": "bwd_pkt_len_std",
    "bwd packet length std": "bwd_pkt_len_std",
    "flow_bytes_s": "bytes_per_sec",
    "flow bytes/s": "bytes_per_sec",
    "flow_packets_s": "packets_per_sec",
    "flow packets/s": "packets_per_sec",
    "fwd_header_length": "fwd_header_len",
    "fwd header length": "fwd_header_len",
    "bwd_header_length": "bwd_header_len",
    "bwd header length": "bwd_header_len",
    "fwd_packets_s": "fwd_pkts_per_sec",
    "fwd packets/s": "fwd_pkts_per_sec",
    "bwd_packets_s": "bwd_pkts_per_sec",
    "bwd packets/s": "bwd_pkts_per_sec",
    "packet_length_min": "pkt_len_min",
    "packet length min": "pkt_len_min",
    "packet_length_max": "pkt_len_max",
    "packet length max": "pkt_len_max",
    "packet_length_mean": "pkt_len_mean",
    "packet length mean": "pkt_len_mean",
    "packet_length_std": "pkt_len_std",
    "packet length std": "pkt_len_std",
    "packet_length_variance": "pkt_len_var",
    "packet length variance": "pkt_len_var",
    "average_packet_size": "pkt_size_avg",
    "average packet size": "pkt_size_avg",
    "avg_fwd_segment_size": "fwd_seg_size_avg",
    "avg fwd segment size": "fwd_seg_size_avg",
    "avg_bwd_segment_size": "bwd_seg_size_avg",
    "avg bwd segment size": "bwd_seg_size_avg",
    "subflow_fwd_packets": "subflow_fwd_pkts",
    "subflow fwd packets": "subflow_fwd_pkts",
    "subflow_fwd_bytes": "subflow_fwd_byts",
    "subflow fwd bytes": "subflow_fwd_byts",
    "subflow_bwd_packets": "subflow_bwd_pkts",
    "subflow bwd packets": "subflow_bwd_pkts",
    "subflow_bwd_bytes": "subflow_bwd_byts",
    "subflow bwd bytes": "subflow_bwd_byts",
    "init_win_bytes_forward": "init_fwd_win_byts",
    "init win bytes forward": "init_fwd_win_byts",
    "init_win_bytes_backward": "init_bwd_win_byts",
    "init win bytes backward": "init_bwd_win_byts",
    "act_data_pkt_fwd": "fwd_act_data_pkts",
    "min_seg_size_forward": "fwd_seg_size_min",
}


def _normalize_identifier(name: str) -> str:
    """Normalize identifier by stripping, lowercasing, and replacing punctuation with underscores."""
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "")


def map_to_canonical_schema(
    df: pd.DataFrame,
    mapping_config: Optional[Dict[str, any]] = None,
    config_path: str = "configs/canonical_mapping.yaml",
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Rename raw columns to canonical names and cast types where applicable.
    Supports exact matching, canonical passthrough, normalized matching, and alias lookup.
    """
    if mapping_config is None:
        mapping_config = load_canonical_mapping(config_path)

    col_map = mapping_config.get("mapping", {})
    canonical_targets = set(col_map.values())
    norm_col_map = {_normalize_identifier(k): v for k, v in col_map.items()}
    norm_canonical_map = {_normalize_identifier(v): v for v in canonical_targets}

    # Match raw columns against mapping dict (with edge-case fallback)
    rename_dict = {}
    unmapped_raw_cols = []
    
    for raw_col in df.columns:
        stripped = raw_col.strip()
        norm = _normalize_identifier(stripped)

        # 1. Exact raw column match in mapping
        if stripped in col_map:
            rename_dict[raw_col] = col_map[stripped]
        # 2. Pipeline metadata columns preserved as-is
        elif raw_col in ("source_file", "source_day"):
            rename_dict[raw_col] = raw_col
        # 3. Already canonical name
        elif stripped in canonical_targets:
            rename_dict[raw_col] = stripped
        # 4. Normalized match against expected raw columns
        elif norm in norm_col_map:
            rename_dict[raw_col] = norm_col_map[norm]
        # 5. Normalized match against canonical columns
        elif norm in norm_canonical_map:
            rename_dict[raw_col] = norm_canonical_map[norm]
        # 6. Recognized CICFlowMeter naming aliases
        elif norm in KNOWN_CIC_ALIASES:
            rename_dict[raw_col] = KNOWN_CIC_ALIASES[norm]
        else:
            unmapped_raw_cols.append(raw_col)

    df_mapped = df.rename(columns=rename_dict).copy()

    audit = {
        "mapped_columns_count": len(rename_dict),
        "unmapped_columns": unmapped_raw_cols,
        "final_columns": list(df_mapped.columns),
    }

    return df_mapped, audit
