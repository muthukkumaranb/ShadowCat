"""
Stage 3: Data Cleaning & Timestamp Normalization Module
Handles duplicate detection, Inf-to-NaN conversions, negative timing audits,
UTC timestamp parsing & sorting, and safe missing value imputation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional


def clean_and_normalize_flow_data(
    df: pd.DataFrame,
    rate_cols: Optional[List[str]] = None,
    timing_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw canonical dataframe:
    1. Replaces string 'Infinity', 'Inf', '-Infinity', '-Inf' and numpy inf with NaN in numeric columns.
    2. Parses 'raw_timestamp' to UTC-aware datetime 'timestamp_utc'.
    3. Filters out corrupted timestamps (e.g., year < 2018).
    4. Detects and logs exact duplicate rows and near-duplicate flows.
    5. Checks and logs negative duration/IAT anomalies.
    6. Sorts chronologically by 'timestamp_utc'.
    """
    audit = {}
    initial_rows = len(df)
    audit["initial_rows"] = initial_rows

    if rate_cols is None:
        rate_cols = ["bytes_per_sec", "packets_per_sec", "fwd_pkts_per_sec", "bwd_pkts_per_sec"]
    if timing_cols is None:
        timing_cols = [
            "duration_microsec", "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
            "fwd_iat_tot", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max", "fwd_iat_min",
            "bwd_iat_tot", "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min",
            "active_mean", "active_std", "active_max", "active_min",
            "idle_mean", "idle_std", "idle_max", "idle_min"
        ]

    # 1. Timestamp parsing to UTC
    if "raw_timestamp" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(
            df["raw_timestamp"],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
            utc=True,
        )
        invalid_ts_count = int(df["timestamp_utc"].isna().sum())
        audit["invalid_timestamps_count"] = invalid_ts_count

        # Filter out 1970 or corrupted years
        valid_ts_mask = df["timestamp_utc"].dt.year >= 2018
        corrupted_year_count = int((~valid_ts_mask).sum())
        audit["corrupted_epoch_timestamps_dropped"] = corrupted_year_count
        df = df[valid_ts_mask].copy()

    candidate_cols = [c for c in df.columns if c not in ["raw_timestamp", "timestamp_utc", "raw_label", "source_file", "source_day"]]
    numeric_cols = df[candidate_cols].columns

    inf_counts = {}
    for col in numeric_cols:
        # First safely coerce to numeric float64
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Check for infinities safely on float array
        col_arr = df[col].to_numpy(dtype=float, na_value=np.nan)
        infs = np.isinf(col_arr)
        n_inf = int(np.sum(infs))
        if n_inf > 0:
            inf_counts[col] = n_inf
            df.loc[infs, col] = np.nan

    audit["infinities_replaced_with_nan"] = inf_counts

    # 3. Investigate negative values in timing columns
    negative_timing_counts = {}
    for col in timing_cols:
        if col in df.columns:
            neg_mask = df[col] < 0
            n_neg = int(neg_mask.sum())
            if n_neg > 0:
                negative_timing_counts[col] = n_neg
                # Replace anomalous negative duration with NaN
                df.loc[neg_mask, col] = np.nan
    audit["negative_timing_values_handled"] = negative_timing_counts

    # Compute duration in seconds for canonical schema
    if "duration_microsec" in df.columns:
        df["duration_sec"] = df["duration_microsec"] / 1e6

    # 4. Exact duplicates check
    feature_cols = [c for c in df.columns if c not in ["source_file", "source_day"]]
    exact_dups = int(df.duplicated(subset=feature_cols).sum())
    audit["exact_duplicate_rows_dropped"] = exact_dups
    df = df.drop_duplicates(subset=feature_cols).copy()

    # 5. Near-duplicate detection (same 5-tuple signature + same rounded timestamp)
    near_dup_keys = [c for c in ["destination_port", "protocol", "byte_count_fwd", "packet_count_fwd"] if c in df.columns]
    if "timestamp_utc" in df.columns and len(near_dup_keys) > 0:
        df["_rounded_sec"] = df["timestamp_utc"].dt.floor("1s")
        near_dups_count = int(df.duplicated(subset=near_dup_keys + ["_rounded_sec"]).sum())
        audit["near_duplicates_detected_and_retained"] = near_dups_count
        df = df.drop(columns=["_rounded_sec"])

    # 6. Chronological sorting
    if "timestamp_utc" in df.columns:
        df = df.sort_values(by="timestamp_utc").reset_index(drop=True)
        is_monotonic = bool(df["timestamp_utc"].is_monotonic_increasing)
        audit["chronological_ordering_verified"] = is_monotonic

    audit["final_cleaned_rows"] = len(df)
    return df, audit


def impute_missing_flow_values(
    df: pd.DataFrame,
    strategy: str = "median",
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Imputes missing values without cross-dataset leakage using training/within-day statistics.
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    imputation_values = {}
    for col in feature_cols:
        nan_count = int(df[col].isna().sum())
        if nan_count > 0:
            if strategy == "median":
                fill_val = float(df[col].median())
                if np.isnan(fill_val):
                    fill_val = 0.0
            else:
                fill_val = 0.0
            df[col] = df[col].fillna(fill_val)
            imputation_values[col] = fill_val

    return df, imputation_values
