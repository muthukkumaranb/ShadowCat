"""
Stage 6: Attack Alignment, Multi-Horizon Future Labels & Chronological Splits
Assigns granular canonical labels (including 2-phase Infiltration), constructs
leakage-free future attack forecast targets, and assigns strict chronological splits.
"""

import os
import yaml
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from typing import Dict, List, Tuple, Optional


def load_attack_schedules(config_path: str = "configs/attack_timelines.yaml") -> Dict[str, any]:
    """Load attack schedule reference config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def assign_window_labels(
    window_df: pd.DataFrame,
    schedules_config: Optional[Dict[str, any]] = None,
    config_path: str = "configs/attack_timelines.yaml",
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Assigns:
    1. label_binary (0 = Benign, 1 = Attack)
    2. label_attack_type (Canonical name with 2-phase Infiltration breakdown)
    """
    if schedules_config is None:
        schedules_config = load_attack_schedules(config_path)

    label_norm_map = schedules_config.get("label_normalization_map", {})

    window_df = window_df.copy()
    
    # Default labels
    binary_series = pd.Series(0, index=window_df.index)
    attack_type_series = pd.Series("Benign", index=window_df.index)

    # 1. Base mapping from CSV flow dominant label / has_malicious_flows
    if "has_malicious_flows" in window_df.columns:
        binary_series = window_df["has_malicious_flows"].astype(int)

    if "raw_label_dominant" in window_df.columns:
        attack_type_series = window_df["raw_label_dominant"].map(
            lambda lbl: label_norm_map.get(str(lbl).strip(), str(lbl).strip())
        )

    # 2. Preserve 2-phase Infiltration for 01-03-2018 (and 28-02-2018)
    is_infil_day = window_df["source_day"].astype(str).str.contains("01-03-2018", na=False)
    is_attack = binary_series == 1
    infil_compromise_mask = is_infil_day & is_attack & (window_df["window_start_utc"].dt.hour <= 5)
    infil_portscan_mask = is_infil_day & is_attack & (window_df["window_start_utc"].dt.hour > 5)

    attack_type_series.loc[infil_compromise_mask] = "Infiltration-Compromise"
    attack_type_series.loc[infil_portscan_mask] = "Infiltration-Portscan"

    window_df["label_binary"] = binary_series
    window_df["label_attack_type"] = attack_type_series

    # Ensure label_binary is 1 whenever attack_type is not Benign
    attack_mask = window_df["label_attack_type"].astype(str).str.lower() != "benign"
    window_df.loc[attack_mask, "label_binary"] = 1

    audit = {
        "binary_distribution": window_df["label_binary"].value_counts().to_dict(),
        "attack_type_distribution": window_df["label_attack_type"].value_counts().to_dict(),
    }

    return window_df, audit


def generate_future_attack_labels(
    window_df: pd.DataFrame,
    horizon_windows: int = 5,
    target_col: str = "label_binary",
    future_col: str = "future_attack_label",
) -> pd.DataFrame:
    """
    Derives future_attack_label: Does an attack occur within the next H windows?
    P(event in [t+1, t+H]).
    Computed via backward rolling maximum over chronologically sorted windows.
    Zero future information leaks into current-window features.
    """
    # Ensure dataframe is chronologically sorted
    window_df = window_df.sort_values("window_start_utc").reset_index(drop=True)

    # Forward-looking rolling max: shift backward by 1, then take rolling max of size H looking ahead
    # In pandas: reverse series, rolling max of size H, reverse back, then shift by 1
    rev_target = window_df[target_col].iloc[::-1]
    rev_future = rev_target.rolling(window=horizon_windows, min_periods=1).max()
    future_series = rev_future.iloc[::-1].shift(-1).fillna(0).astype(int)

    window_df[future_col] = future_series
    return window_df


def assign_chronological_splits(
    window_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Partitions windows into train, val, and test splits strictly by chronological order.
    Never randomly shuffles time series data.
    """
    window_df = window_df.sort_values("window_start_utc").reset_index(drop=True)
    n = len(window_df)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    window_df["split"] = splits

    train_df = window_df[window_df["split"] == "train"]
    val_df = window_df[window_df["split"] == "val"]
    test_df = window_df[window_df["split"] == "test"]

    split_audit = {
        "train_count": len(train_df),
        "train_start": str(train_df["window_start_utc"].min()) if len(train_df) > 0 else None,
        "train_end": str(train_df["window_start_utc"].max()) if len(train_df) > 0 else None,
        "val_count": len(val_df),
        "val_start": str(val_df["window_start_utc"].min()) if len(val_df) > 0 else None,
        "val_end": str(val_df["window_start_utc"].max()) if len(val_df) > 0 else None,
        "test_count": len(test_df),
        "test_start": str(test_df["window_start_utc"].min()) if len(test_df) > 0 else None,
        "test_end": str(test_df["window_start_utc"].max()) if len(test_df) > 0 else None,
    }

    return window_df, split_audit


def apply_purge_embargo(
    window_df: pd.DataFrame,
    lookback_windows: int,
    horizon_windows: int,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Applies purge + embargo at both split boundaries (train→val, val→test).

    At each boundary, drops `width` windows from BOTH sides:
      - Last `width` windows of the earlier partition
      - First `width` windows of the later partition

    Width = lookback_windows + horizon_windows (computed, never hardcoded).
    This ensures no window's LSTM lookback OR forecast horizon can reach
    across a split boundary.

    Parameters
    ----------
    window_df : pd.DataFrame
        Must already have 'split' column from assign_chronological_splits().
    lookback_windows : int
        LSTM lookback length (L). Read from config, not hardcoded.
    horizon_windows : int
        Forecast horizon (H). Read from config, not hardcoded.

    Returns
    -------
    purged_df : pd.DataFrame
        DataFrame with embargo-zone windows removed (split column preserved).
    audit : dict
        Detailed accounting of windows dropped at each boundary.
    """
    width = lookback_windows + horizon_windows

    window_df = window_df.sort_values("window_start_utc").reset_index(drop=True)

    # Identify partition index ranges
    train_idx = window_df.index[window_df["split"] == "train"]
    val_idx = window_df.index[window_df["split"] == "val"]
    test_idx = window_df.index[window_df["split"] == "test"]

    original_counts = {
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx),
        "total": len(window_df),
    }

    # Compute indices to drop at train→val boundary
    train_drop = set(train_idx[-width:]) if len(train_idx) >= width else set(train_idx)
    val_drop_front = set(val_idx[:width]) if len(val_idx) >= width else set(val_idx)

    # Compute indices to drop at val→test boundary
    val_drop_back = set(val_idx[-width:]) if len(val_idx) >= width else set(val_idx)
    test_drop = set(test_idx[:width]) if len(test_idx) >= width else set(test_idx)

    # Combined val drops (front from train→val boundary + back from val→test boundary)
    val_drop = val_drop_front | val_drop_back

    all_dropped_idx = train_drop | val_drop | test_drop

    purged_df = window_df.drop(index=all_dropped_idx).reset_index(drop=True)

    purged_counts = {
        "train": int((purged_df["split"] == "train").sum()),
        "val": int((purged_df["split"] == "val").sum()),
        "test": int((purged_df["split"] == "test").sum()),
        "total": len(purged_df),
    }

    audit = {
        "purge_embargo_width": width,
        "lookback_windows": lookback_windows,
        "horizon_windows": horizon_windows,
        "original_counts": original_counts,
        "dropped_at_train_val_boundary": {
            "train_tail_dropped": len(train_drop),
            "val_head_dropped": len(val_drop_front),
        },
        "dropped_at_val_test_boundary": {
            "val_tail_dropped": len(val_drop_back),
            "test_head_dropped": len(test_drop),
        },
        "total_val_dropped": len(val_drop),
        "total_windows_dropped": len(all_dropped_idx),
        "purged_counts": purged_counts,
    }

    return purged_df, audit


def assign_episode_ids(window_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns granular episode_id to every window based on unbroken contiguous runs.
    An episode is an unbroken contiguous run of windows sharing the same source_day
    and label_attack_type.

    Format: {source_day}_{attack_type}_{run_index}
    where run_index is 0-indexed per (source_day, label_attack_type).

    Parameters
    ----------
    window_df : pd.DataFrame
        Window dataframe containing 'source_day', 'label_attack_type', and 'window_start_utc'.

    Returns
    -------
    pd.DataFrame
        Dataframe with added 'episode_id' column.
    """
    window_df = window_df.sort_values("window_start_utc").reset_index(drop=True)
    day_shift = window_df["source_day"] != window_df["source_day"].shift(1)
    atk_shift = window_df["label_attack_type"] != window_df["label_attack_type"].shift(1)
    block_id = (day_shift | atk_shift).cumsum()

    block_meta = window_df.groupby(block_id, sort=False).agg(
        source_day=("source_day", "first"),
        label_attack_type=("label_attack_type", "first")
    ).reset_index()

    block_meta["run_index"] = block_meta.groupby(["source_day", "label_attack_type"]).cumcount()
    block_meta["episode_id"] = (
        block_meta["source_day"].astype(str) + "_" +
        block_meta["label_attack_type"].astype(str) + "_" +
        block_meta["run_index"].astype(str)
    )

    block_to_ep = dict(zip(block_meta.iloc[:, 0], block_meta["episode_id"]))
    window_df["episode_id"] = block_id.map(block_to_ep)
    return window_df
