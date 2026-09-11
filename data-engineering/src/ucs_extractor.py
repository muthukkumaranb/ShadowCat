"""
UCSExtractor: Reusable Runtime Feature Extraction Module
SIH 2026 - Unified Cyber State (UCS) Ingestion Pipeline

Packages the batch ingestion pipeline into a reusable runtime extractor that converts
streaming or batched flow/PCAP records into standardized 410-column UCS window tensors.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import yaml

from src.canonical_mapper import map_to_canonical_schema
from src.cleaner import clean_and_normalize_flow_data
from src.window_aggregator import create_1min_windows
from src.normalizer import LeakageSafeRobustScaler


class SchemaValidationError(Exception):
    """Raised when extraction output violates the 410-column schema contract."""
    pass


def _load_class_schema_constants() -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Derives schema constants using the exact same metadata exclusion logic
    as pipeline_runner.py and sequence_builder.py.
    """
    # 4 Window identifiers
    window_id_cols = [
        "window_id",
        "window_start_utc",
        "window_end_utc",
        "source_day",
    ]

    # 6 Masks in authoritative order
    mask_cols = [
        "mask_has_traffic_volume_features",
        "mask_has_flow_timing_features",
        "mask_has_packet_level_features",
        "mask_has_tcp_flags",
        "mask_has_graph_topology",
        "mask_has_identity_auth",
    ]

    metadata_cols = {
        "window_id", "window_start_utc", "window_end_utc", "source_day",
        "split", "label_binary", "label_attack_type", "future_attack_label",
        "raw_label_dominant", "has_malicious_flows", "episode_id",
        "forecast_episode_id", "temporal_gap_flag", "future_attack_type",
        "mask_has_traffic_volume_features", "mask_has_flow_timing_features",
        "mask_has_packet_level_features", "mask_has_tcp_flags",
        "mask_has_graph_topology", "mask_has_identity_auth"
    }

    base_dir = Path(__file__).resolve().parent.parent
    parquet_path = base_dir / "data" / "ucs" / "ucs_windows.parquet"
    scaler_yaml_path = base_dir / "data" / "ucs" / "scaler_params.yaml"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in metadata_cols]
    elif scaler_yaml_path.exists():
        with open(scaler_yaml_path, "r", encoding="utf-8") as f:
            scaler_data = yaml.safe_load(f)
        feature_cols = list(scaler_data.keys())
    else:
        raise FileNotFoundError("Neither ucs_windows.parquet nor scaler_params.yaml found to derive schema.")

    # 410 output columns: 4 window IDs + 6 masks + 400 features
    output_cols = window_id_cols + mask_cols + feature_cols

    # PCAP packet-level features (12 features at end of feature_cols)
    pcap_packet_cols = feature_cols[388:]

    # Authoritative 406 model-facing order matching ML1's inference_feature_order_v1.json
    # Exactly: 388 flow features + 6 masks + 12 packet features
    model_input_cols = feature_cols[:388] + mask_cols + feature_cols[388:]

    return window_id_cols, mask_cols, feature_cols, output_cols, model_input_cols, pcap_packet_cols


class UCSExtractor:
    """
    Reusable runtime extraction engine.
    Standardizes raw network flows into 410-column Unified Cyber State tensors.
    """
    SUPPORTED_SCHEMA_VERSIONS = {"v3.0"}

    # Class-level schema constants
    (
        WINDOW_ID_COLUMNS,
        MASK_COLUMNS,
        MODEL_FEATURE_COLUMNS,
        OUTPUT_COLUMNS,
        MODEL_INPUT_COLUMNS,
        PCAP_PACKET_COLUMNS,
    ) = _load_class_schema_constants()

    def __init__(
        self,
        schema_version: str = "v3.0",
        config_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
    ):
        if schema_version not in self.SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported schema_version '{schema_version}'. Supported: {self.SUPPORTED_SCHEMA_VERSIONS}"
            )
        self.schema_version = schema_version

        base_dir = Path(__file__).resolve().parent.parent
        self.config_dir = Path(config_dir) if config_dir else base_dir / "configs"
        self.data_dir = Path(data_dir) if data_dir else base_dir / "data" / "ucs"

        # 1. Load canonical mapping config
        canonical_map_file = self.config_dir / "canonical_mapping.yaml"
        if not canonical_map_file.exists():
            raise FileNotFoundError(f"Canonical mapping config missing: {canonical_map_file}")
        with open(canonical_map_file, "r", encoding="utf-8") as f:
            self.canonical_mapping = yaml.safe_load(f)

        # 2. Load scaler parameters
        scaler_params_file = self.data_dir / "scaler_params.yaml"
        if not scaler_params_file.exists():
            raise FileNotFoundError(f"Scaler parameters missing: {scaler_params_file}")
        with open(scaler_params_file, "r", encoding="utf-8") as f:
            self.scaler_params = yaml.safe_load(f)

        # 3. Load imputation parameters (raw-scale medians)
        imputation_params_file = self.data_dir / "imputation_params.yaml"
        if not imputation_params_file.exists():
            raise FileNotFoundError(f"Imputation parameters missing: {imputation_params_file}")
        with open(imputation_params_file, "r", encoding="utf-8") as f:
            self.imputation_params = yaml.safe_load(f)

        # 4. Load pipeline config for log1p columns and interval_sec
        pipeline_config_file = self.config_dir / "pipeline_config.yaml"
        if pipeline_config_file.exists():
            with open(pipeline_config_file, "r", encoding="utf-8") as f:
                self.pipeline_config = yaml.safe_load(f)
        else:
            self.pipeline_config = {
                "windowing": {"interval_sec": 60},
                "normalization": {"log1p_cols": ["count", "bytes", "duration", "packets"]},
            }

        # 5. Initialize LeakageSafeRobustScaler in transform-only mode
        log1p_cols = self.pipeline_config.get("normalization", {}).get("log1p_cols", ["count", "bytes", "duration", "packets"])
        self.scaler = LeakageSafeRobustScaler(log1p_cols=log1p_cols)
        self.scaler.params = self.scaler_params
        self.scaler.feature_cols = list(self.scaler_params.keys())

    def extract(self, raw_input: pd.DataFrame, source_type: str = "csv") -> pd.DataFrame:
        """
        Extracts standardized 410-column UCS window DataFrame from raw inputs.

        Parameters
        ----------
        raw_input : pd.DataFrame
            Input flow or window records.
        source_type : str
            - 'csv': Standard CICFlowMeter CSV flow data. Full pipeline:
                     mapping -> cleaning -> windowing -> packet fill -> imputation -> scaling.
            - 'pcap': Pre-extracted flow records with packet features present.
            - 'flows': Cleaned canonical flow records directly.

        Returns
        -------
        pd.DataFrame
            DataFrame with exactly 410 columns matching OUTPUT_COLUMNS.
        """
        if raw_input is None or len(raw_input) == 0:
            raise ValueError("Input DataFrame is empty or None.")

        if source_type not in ("csv", "pcap", "flows"):
            raise ValueError(f"Invalid source_type '{source_type}'. Supported: 'csv', 'pcap', 'flows'")

        df = raw_input.copy()

        # Step 1: Mapping to canonical schema
        if source_type == "csv":
            df, _ = map_to_canonical_schema(df, mapping_config=self.canonical_mapping)

        # Step 2: Cleaning & timestamp normalization
        if source_type in ("csv", "pcap"):
            df, _ = clean_and_normalize_flow_data(df)

            # Flow-level imputation using frozen medians
            flow_medians = self.imputation_params.get("flow_medians_global", {})
            if "source_day" in df.columns and len(df["source_day"]) > 0:
                s_day = str(df["source_day"].iloc[0])
                by_day = self.imputation_params.get("flow_medians_by_day", {})
                if s_day in by_day:
                    flow_medians = by_day[s_day]
            for col, fill_val in flow_medians.items():
                if col in df.columns and df[col].isna().any():
                    df[col] = df[col].fillna(fill_val)

        # Step 3: Window aggregation
        interval_sec = self.pipeline_config.get("windowing", {}).get("interval_sec", 60)
        # If input is already windowed (has window_start_utc and window features)
        if "window_start_utc" in df.columns and "flow_count" in df.columns:
            window_df = df.copy()
        else:
            window_df, _ = create_1min_windows(df, interval_sec=interval_sec)

        if len(window_df) == 0:
            raise ValueError("Windowing resulted in 0 windows. Check timestamp ranges in input.")

        # Step 4: Add & validate window identifier columns
        if "window_start_utc" not in window_df.columns:
            raise SchemaValidationError("Missing required window_start_utc column after windowing.")

        new_cols: Dict[str, Any] = {}

        if "window_end_utc" not in window_df.columns:
            new_cols["window_end_utc"] = window_df["window_start_utc"] + pd.Timedelta(seconds=interval_sec)

        if "source_day" not in window_df.columns:
            new_cols["source_day"] = window_df["window_start_utc"].dt.strftime("%d-%m-%Y")

        if "window_id" not in window_df.columns:
            src_day = new_cols["source_day"] if "source_day" in new_cols else window_df["source_day"]
            new_cols["window_id"] = [
                f"W_{d}_{ts.strftime('%Y%m%d%H%M%S')}"
                for d, ts in zip(src_day, window_df["window_start_utc"])
            ]

        # Step 5: Handle packet-level features & presence masks
        has_packet_data = False
        if source_type == "pcap":
            present_pkt = [c for c in self.PCAP_PACKET_COLUMNS if c in window_df.columns]
            if len(present_pkt) == len(self.PCAP_PACKET_COLUMNS):
                has_packet_data = bool((window_df[present_pkt] != 0.0).any().any())

        for pkt_col in self.PCAP_PACKET_COLUMNS:
            if pkt_col not in window_df.columns:
                new_cols[pkt_col] = np.zeros(len(window_df), dtype=np.float64)

        # Step 6: Frozen imputation on missing feature columns
        win_impute = self.imputation_params.get("window_feature_medians", self.imputation_params)
        for feat_col in self.MODEL_FEATURE_COLUMNS:
            impute_val = win_impute.get(feat_col, 0.0)
            if feat_col not in window_df.columns and feat_col not in new_cols:
                new_cols[feat_col] = np.full(len(window_df), impute_val, dtype=np.float64)

        # Assign new columns in a single concatenated block to avoid DataFrame fragmentation
        if new_cols:
            window_df = pd.concat([window_df, pd.DataFrame(new_cols, index=window_df.index)], axis=1)

        # Update masks in-place (already present in window_df from window aggregator)
        window_df["mask_has_traffic_volume_features"] = 1.0
        window_df["mask_has_flow_timing_features"] = 1.0
        window_df["mask_has_packet_level_features"] = 1.0 if has_packet_data else 0.0
        window_df["mask_has_tcp_flags"] = 1.0
        window_df["mask_has_graph_topology"] = 1.0
        window_df["mask_has_identity_auth"] = 0.0

        # Fill any remaining NaNs in feature columns with frozen medians
        for feat_col in self.MODEL_FEATURE_COLUMNS:
            impute_val = win_impute.get(feat_col, 0.0)
            if window_df[feat_col].isna().any():
                window_df[feat_col] = window_df[feat_col].fillna(impute_val)

        # Step 7: Leakage-safe scaling (transform-only)
        scaled_df = self.scaler.transform(window_df)

        # Step 8: Strict schema contract validation
        missing_cols = [c for c in self.OUTPUT_COLUMNS if c not in scaled_df.columns]
        if missing_cols:
            raise SchemaValidationError(
                f"Output missing {len(missing_cols)} required columns: {missing_cols[:10]}"
            )

        # Select exactly the 410 columns in authoritative order
        final_df = scaled_df[self.OUTPUT_COLUMNS].copy()

        # Final sanity checks
        if final_df.shape[1] != 410:
            raise SchemaValidationError(f"Expected 410 columns, but got {final_df.shape[1]}")

        # Check for any remaining NaNs or Infs in numeric output
        numeric_output = final_df[self.MASK_COLUMNS + self.MODEL_FEATURE_COLUMNS]
        if numeric_output.isna().any().any():
            nan_cols = numeric_output.columns[numeric_output.isna().any()].tolist()
            raise SchemaValidationError(f"NaN values detected in feature columns: {nan_cols[:10]}")

        return final_df

    def extract_model_tensor(
        self,
        raw_input: pd.DataFrame,
        source_type: str = "csv",
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        """
        Authoritative input-side entry point for Backend's predict().

        Extracts standardized UCS windows and returns a 2D NumPy tensor in the exact
        positional order required by the LSTM world model (MODEL_INPUT_COLUMNS, 406 dims):
        - Indices 0..387: 388 scaled flow aggregation features
        - Indices 388..393: 6 presence masks (1.0 or 0.0)
        - Indices 394..405: 12 scaled PCAP packet features

        Parameters
        ----------
        raw_input : pd.DataFrame
            Input flow or window records.
        source_type : str
            Source format: 'csv', 'pcap', or 'flows'.
        dtype : np.dtype
            Target numpy data type, defaults to np.float32 for PyTorch tensors.

        Returns
        -------
        np.ndarray
            Array of shape (N_windows, 406) in exact LSTM input feature order.
        """
        extracted_df = self.extract(raw_input, source_type=source_type)
        return extracted_df[self.MODEL_INPUT_COLUMNS].to_numpy(dtype=dtype)
