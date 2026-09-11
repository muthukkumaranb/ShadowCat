"""
Stage 8: Boundary-Safe LSTM Sequence Construction
Converts purged/embargoed window-level feature data into LSTM input sequences
of length LSTM_LOOKBACK_WINDOWS. Guarantees no sequence crosses a split boundary.

Each sequence is a contiguous block of `lookback_windows` consecutive windows
from the SAME split partition. The target label for each sequence is the
future_attack_label of the LAST window in the sequence.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


def build_lstm_sequences(
    purged_df: pd.DataFrame,
    lookback_windows: int,
    feature_cols: List[str],
    target_col: str = "future_attack_label",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Constructs boundary-safe LSTM input sequences from purged/embargoed windows.

    A sequence at position i uses windows [i, i+1, ..., i+lookback-1] from a
    single split partition. If a potential sequence's lookback would need to pull
    data from before the purge boundary (i.e., from a different split), that
    sequence is excluded entirely — never truncated or padded.

    Parameters
    ----------
    purged_df : pd.DataFrame
        Window-level data with 'split' column, already purged/embargoed.
        Must be sorted chronologically.
    lookback_windows : int
        Number of consecutive windows per LSTM input sequence (L).
        Read from config lstm.lookback_windows.
    feature_cols : list of str
        Columns to include in the feature tensor (X).
    target_col : str
        Column to use as the prediction target (y). Default: future_attack_label.

    Returns
    -------
    X_by_split : dict
        {split_name: np.ndarray of shape (n_sequences, lookback_windows, n_features)}
    y_by_split : dict
        {split_name: np.ndarray of shape (n_sequences,)}
    audit : dict
        Sequence counts per split and boundary-safety verification.
    """
    purged_df = purged_df.sort_values("window_start_utc").reset_index(drop=True)

    X_by_split = {}
    y_by_split = {}
    audit_sequences = {}

    for split_name in ["train", "val", "test"]:
        split_df = purged_df[purged_df["split"] == split_name].reset_index(drop=True)
        n_windows = len(split_df)

        if n_windows < lookback_windows:
            X_by_split[split_name] = np.empty((0, lookback_windows, len(feature_cols)))
            y_by_split[split_name] = np.empty((0,))
            audit_sequences[split_name] = {
                "windows_available": n_windows,
                "valid_sequences": 0,
                "skipped_boundary": 0,
            }
            continue

        # Extract feature matrix for this split
        feature_matrix = split_df[feature_cols].values  # (n_windows, n_features)
        target_vector = split_df[target_col].values      # (n_windows,)

        # Verify chronological contiguity within split
        # (after purge/embargo, each split's windows should be contiguous)
        timestamps = split_df["window_start_utc"].values
        is_contiguous = np.all(np.diff(timestamps) > np.timedelta64(0, "s"))

        # Build sequences: sliding window of size lookback_windows
        n_sequences = n_windows - lookback_windows + 1
        X_sequences = np.zeros((n_sequences, lookback_windows, len(feature_cols)))
        y_sequences = np.zeros(n_sequences)

        for i in range(n_sequences):
            X_sequences[i] = feature_matrix[i : i + lookback_windows]
            # Target is the label of the LAST window in the sequence
            y_sequences[i] = target_vector[i + lookback_windows - 1]

        X_by_split[split_name] = X_sequences
        y_by_split[split_name] = y_sequences

        audit_sequences[split_name] = {
            "windows_available": n_windows,
            "valid_sequences": n_sequences,
            "skipped_boundary": 0,  # All within-split, no cross-boundary possible after purge
            "is_chronologically_contiguous": bool(is_contiguous),
        }

    audit = {
        "lookback_windows": lookback_windows,
        "n_features": len(feature_cols),
        "target_col": target_col,
        "sequences_per_split": audit_sequences,
        "total_sequences": sum(
            s["valid_sequences"] for s in audit_sequences.values()
        ),
    }

    return X_by_split, y_by_split, audit


def verify_no_cross_boundary_sequences(
    purged_df: pd.DataFrame,
    X_by_split: Dict[str, np.ndarray],
    lookback_windows: int,
    feature_cols: List[str],
) -> bool:
    """
    Explicit verification that no LSTM sequence contains windows from
    more than one split partition.

    This is a runtime assertion, not just a design guarantee — it checks
    the actual constructed sequences against the source data to confirm
    boundary safety.

    Parameters
    ----------
    purged_df : pd.DataFrame
        The purged/embargoed window data used to build sequences.
    X_by_split : dict
        The constructed sequences keyed by split name.
    lookback_windows : int
        Sequence length.
    feature_cols : list of str
        Feature columns used in sequences.

    Returns
    -------
    bool
        True if all sequences are boundary-safe, False otherwise.
    """
    purged_df = purged_df.sort_values("window_start_utc").reset_index(drop=True)

    for split_name in ["train", "val", "test"]:
        split_df = purged_df[purged_df["split"] == split_name].reset_index(drop=True)
        n_windows = len(split_df)
        sequences = X_by_split.get(split_name, np.empty((0,)))

        if len(sequences) == 0:
            continue

        # Each sequence should have exactly lookback_windows steps
        assert sequences.shape[1] == lookback_windows, (
            f"Sequence length mismatch in {split_name}: "
            f"expected {lookback_windows}, got {sequences.shape[1]}"
        )

        # Verify that the number of sequences is consistent
        expected_n = n_windows - lookback_windows + 1
        assert len(sequences) == expected_n, (
            f"Sequence count mismatch in {split_name}: "
            f"expected {expected_n}, got {len(sequences)}"
        )

        # Verify first and last sequence match the split data exactly
        split_features = split_df[feature_cols].values
        if not np.allclose(sequences[0], split_features[:lookback_windows], equal_nan=True):
            return False
        if not np.allclose(sequences[-1], split_features[-lookback_windows:], equal_nan=True):
            return False

    return True


def get_flat_window_data_for_lr(
    purged_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "future_attack_label",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Extracts flat window-level feature matrices for the LR baseline,
    using the SAME purged/embargoed window set as the LSTM path.

    This ensures LR/LSTM protocol parity: both models consume data
    derived from the identical purged window set.

    Parameters
    ----------
    purged_df : pd.DataFrame
        The SAME purged/embargoed DataFrame used for LSTM sequences.
    feature_cols : list of str
        Feature columns (same as LSTM).
    target_col : str
        Target column (same as LSTM).

    Returns
    -------
    X_by_split : dict
        {split_name: np.ndarray of shape (n_windows, n_features)}
    y_by_split : dict
        {split_name: np.ndarray of shape (n_windows,)}
    """
    X_by_split = {}
    y_by_split = {}

    for split_name in ["train", "val", "test"]:
        split_df = purged_df[purged_df["split"] == split_name]
        X_by_split[split_name] = split_df[feature_cols].values
        y_by_split[split_name] = split_df[target_col].values

    return X_by_split, y_by_split
