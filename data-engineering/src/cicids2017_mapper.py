import pandas as pd
from typing import Dict, Tuple, Optional
from src.canonical_mapper import load_canonical_mapping, KNOWN_CIC_ALIASES, _normalize_identifier

def map_cicids2017_schema(df: pd.DataFrame, config_path: str = "configs/canonical_mapping.yaml") -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Schema-mapping layer for CIC-IDS2017 dataset.
    Normalizes CIC-IDS2017 into the canonical shape expected by the pipeline.
    Documents columns with no 2018 equivalent.
    """
    mapping_config = load_canonical_mapping(config_path)
    col_map = mapping_config.get("mapping", {})
    canonical_targets = set(col_map.values())
    
    rename_dict = {}
    unmapped_raw_cols = []
    
    # Track 2017 specific columns with no 2018 equivalent
    unmapped_2017_specific = []

    # Explicit 2017 mappings for columns not covered by 2018 canonical mapping
    CIC2017_SPECIFIC = {
        'Fwd IAT Total': 'fwd_iat_tot', 'Bwd IAT Total': 'bwd_iat_tot', 
        'Min Packet Length': 'pkt_len_min', 'Max Packet Length': 'pkt_len_max', 
        'FIN Flag Count': 'fin_flag_cnt', 'SYN Flag Count': 'syn_flag_cnt', 
        'RST Flag Count': 'rst_flag_cnt', 'PSH Flag Count': 'psh_flag_cnt', 
        'ACK Flag Count': 'ack_flag_cnt', 'URG Flag Count': 'urg_flag_cnt', 
        'ECE Flag Count': 'ece_flag_cnt', 'Fwd Avg Bytes/Bulk': 'fwd_byts_b_avg', 
        'Fwd Avg Packets/Bulk': 'fwd_pkts_b_avg', 'Fwd Avg Bulk Rate': 'fwd_blk_rate_avg', 
        'Bwd Avg Bytes/Bulk': 'bwd_byts_b_avg', 'Bwd Avg Packets/Bulk': 'bwd_pkts_b_avg', 
        'Bwd Avg Bulk Rate': 'bwd_blk_rate_avg', 'Label': 'raw_label', 
        'Fwd Header Length.1': 'fwd_header_len_1'
    }

    for raw_col in df.columns:
        stripped = raw_col.strip()
        norm = _normalize_identifier(stripped)
        
        # Exact raw column match in mapping
        if stripped in col_map:
            rename_dict[raw_col] = col_map[stripped]
        # Specific 2017 mapping
        elif stripped in CIC2017_SPECIFIC:
            rename_dict[raw_col] = CIC2017_SPECIFIC[stripped]
        # Already canonical name
        elif stripped in canonical_targets:
            rename_dict[raw_col] = stripped
        # CICFlowMeter naming aliases (covers many 2017 variations)
        elif norm in KNOWN_CIC_ALIASES:
            rename_dict[raw_col] = KNOWN_CIC_ALIASES[norm]
        elif raw_col in ("source_file", "source_day", "source_dataset"):
            rename_dict[raw_col] = raw_col
        else:
            # Document missing equivalent
            unmapped_raw_cols.append(raw_col)
            unmapped_2017_specific.append(raw_col)

    df_mapped = df.rename(columns=rename_dict).copy()
    
    # Remove duplicate columns if they appear
    if df_mapped.columns.duplicated().any():
        df_mapped = df_mapped.loc[:, ~df_mapped.columns.duplicated(keep="first")].copy()
        
    # Synthesize timestamp_utc if missing (MachineLearningCSV lacks it)
    if "raw_timestamp" not in df_mapped.columns and "timestamp_utc" not in df_mapped.columns:
        # Create a sequential timestamp starting from a fixed date (e.g. 2017-07-03)
        # Adding 1 second per row to create meaningful temporal windows (60 flows/min)
        start_time = pd.Timestamp("2017-07-03 00:00:00", tz="UTC")
        df_mapped["timestamp_utc"] = start_time + pd.to_timedelta(df_mapped.index, unit='s')
        
    if "protocol" not in df_mapped.columns:
        df_mapped["protocol"] = 6
    if "destination_port" not in df_mapped.columns:
        df_mapped["destination_port"] = 80
        
    # Clean up 2017 specific label encoding issues and map to canonical
    if "Label" in df_mapped.columns:
        df_mapped["raw_label"] = df_mapped["Label"]
        df_mapped = df_mapped.drop(columns=["Label"])
    
    if "raw_label" in df_mapped.columns:
        # Fix the Windows-1252 / UTF-8 encoding issue for "Web Attack - X"
        df_mapped["raw_label"] = df_mapped["raw_label"].astype(str).str.replace(
            r'Web Attack[\s\S]+Brute Force', 'Web-BruteForce', regex=True
        ).str.replace(
            r'Web Attack[\s\S]+XSS', 'Web-XSS', regex=True
        ).str.replace(
            r'Web Attack[\s\S]+Sql Injection', 'Web-SQLi', regex=True
        ).str.strip()
        
    audit = {
        "mapped_columns_count": len(rename_dict),
        "unmapped_columns": unmapped_raw_cols,
        "unmapped_2017_specific": unmapped_2017_specific,
        "final_columns": list(df_mapped.columns),
    }

    return df_mapped, audit
