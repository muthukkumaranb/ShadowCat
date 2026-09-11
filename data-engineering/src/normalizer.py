"""
Stage 7: Leakage-Safe Feature Normalization Module
Implements Log1p skew reduction and RobustScaler ((x - Q50) / (Q75 - Q25)).
Fits exclusively on training split windows and transforms val/test splits without leakage.
"""

import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any


class LeakageSafeRobustScaler:
    """
    Fits RobustScaler parameters on training split and transforms any dataset.
    Formula: x_norm = (x - median) / IQR
    where IQR = max(Q75 - Q25, 1e-6)
    """
    def __init__(self, log1p_cols: Optional[List[str]] = None):
        self.log1p_cols = log1p_cols or []
        self.params: Dict[str, Dict[str, float]] = {}
        self.feature_cols: List[str] = []

    def fit(self, train_df: pd.DataFrame, feature_cols: List[str]) -> "LeakageSafeRobustScaler":
        """Fit scaler parameters solely on the training partition."""
        self.feature_cols = feature_cols
        self.params = {}

        for col in feature_cols:
            vals = train_df[col].dropna()
            
            # Apply log1p if specified
            is_log1p = any(sub in col for sub in self.log1p_cols)
            if is_log1p:
                vals = np.log1p(np.maximum(vals, 0.0))

            q25 = float(vals.quantile(0.25)) if len(vals) > 0 else 0.0
            q50 = float(vals.quantile(0.50)) if len(vals) > 0 else 0.0
            q75 = float(vals.quantile(0.75)) if len(vals) > 0 else 1.0
            
            iqr = q75 - q25
            if iqr <= 1e-6:
                # If IQR is 0, fallback to standard deviation or 1.0
                std = float(vals.std())
                scale = std if std > 1e-6 else 1.0
            else:
                scale = iqr

            self.params[col] = {
                "median": q50,
                "scale": scale,
                "is_log1p": is_log1p,
                "q25": q25,
                "q75": q75,
            }

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies fitted transformations to a dataframe."""
        df_out = df.copy()
        for col in self.feature_cols:
            if col not in df_out.columns:
                continue

            param = self.params.get(col, {"median": 0.0, "scale": 1.0, "is_log1p": False})
            vals = df_out[col].fillna(param["median"]).values

            if param["is_log1p"]:
                vals = np.log1p(np.maximum(vals, 0.0))

            scaled_vals = (vals - param["median"]) / param["scale"]
            df_out[col] = np.nan_to_num(scaled_vals, nan=0.0, posinf=0.0, neginf=0.0)

        return df_out

    def save_parameters(self, filepath: str):
        """Save fitted scaler parameters to YAML/JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.params, f, sort_keys=False)


def normalize_window_features(
    window_df: pd.DataFrame,
    log1p_sub_keys: Optional[List[str]] = None,
    save_params_path: Optional[str] = "data/ucs/scaler_params.yaml",
) -> Tuple[pd.DataFrame, LeakageSafeRobustScaler, Dict[str, Any]]:
    """
    Fits scaler exclusively on train split and normalizes the entire dataset safely.
    """
    if log1p_sub_keys is None:
        log1p_sub_keys = ["byte", "packet", "pkts", "byts", "duration", "count", "rate"]

    metadata_cols = {
        "window_id", "window_start_utc", "window_end_utc", "source_day",
        "split", "label_binary", "label_attack_type", "future_attack_label",
        "raw_label_dominant", "has_malicious_flows", "episode_id",
        "mask_has_traffic_volume_features", "mask_has_flow_timing_features",
        "mask_has_packet_level_features", "mask_has_tcp_flags",
        "mask_has_graph_topology", "mask_has_identity_auth"
    }

    feature_cols = [c for c in window_df.select_dtypes(include=[np.number]).columns if c not in metadata_cols]

    # Split verification
    train_mask = window_df["split"] == "train"
    train_df = window_df[train_mask].copy()

    if len(train_df) == 0:
        raise ValueError("Training split is empty; cannot fit normalization parameters.")

    scaler = LeakageSafeRobustScaler(log1p_cols=log1p_sub_keys)
    scaler.fit(train_df, feature_cols)

    # Transform full dataset using training-fitted params
    normalized_df = scaler.transform(window_df)

    if save_params_path:
        scaler.save_parameters(save_params_path)

    audit = {
        "features_normalized_count": len(feature_cols),
        "train_rows_fitted_on": int(train_mask.sum()),
        "val_rows_transformed": int((window_df["split"] == "val").sum()),
        "test_rows_transformed": int((window_df["split"] == "test").sum()),
        "scaler_method": "RobustScaler (IQR) + Log1p skew correction",
    }

    return normalized_df, scaler, audit
