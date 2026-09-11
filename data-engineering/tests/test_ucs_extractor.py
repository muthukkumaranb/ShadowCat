"""
Unit & Regression Tests for UCSExtractor Runtime Module
SIH 2026 - Unified Cyber State (UCS) Ingestion Pipeline
Compatible with standard python unittest and pytest.
"""

import os
import sys
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import yaml

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ucs_extractor import UCSExtractor, SchemaValidationError


class TestUCSExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.extractor = UCSExtractor()
        cls.parquet_path = Path("data/ucs/ucs_windows.parquet")
        if cls.parquet_path.exists():
            cls.parquet_df = pd.read_parquet(cls.parquet_path)
        else:
            cls.parquet_df = None

    def test_schema_constants_guardrail(self):
        """
        Test 2: Schema Constants Guardrail
        Assert UCSExtractor.MODEL_FEATURE_COLUMNS == sequence_builder/pipeline_runner feature_cols
        list exactly (order and names).
        """
        if self.parquet_df is None:
            self.skipTest("ucs_windows.parquet not found")

        metadata_cols = {
            "window_id", "window_start_utc", "window_end_utc", "source_day",
            "split", "label_binary", "label_attack_type", "future_attack_label",
            "raw_label_dominant", "has_malicious_flows", "episode_id",
            "mask_has_traffic_volume_features", "mask_has_flow_timing_features",
            "mask_has_packet_level_features", "mask_has_tcp_flags",
            "mask_has_graph_topology", "mask_has_identity_auth"
        }
        derived_feature_cols = [
            c for c in self.parquet_df.select_dtypes(include=[np.number]).columns
            if c not in metadata_cols
        ]

        self.assertEqual(len(self.extractor.MODEL_FEATURE_COLUMNS), 400)
        self.assertEqual(
            self.extractor.MODEL_FEATURE_COLUMNS,
            derived_feature_cols,
            "UCSExtractor.MODEL_FEATURE_COLUMNS does not match sequence_builder derivation exactly!"
        )
        self.assertEqual(len(self.extractor.MASK_COLUMNS), 6)
        self.assertEqual(len(self.extractor.WINDOW_ID_COLUMNS), 4)
        self.assertEqual(len(self.extractor.OUTPUT_COLUMNS), 410)

    def test_column_order_stability(self):
        """
        Test 4: Column Order Stability
        Assert OUTPUT_COLUMNS == WINDOW_ID_COLUMNS + MASK_COLUMNS + MODEL_FEATURE_COLUMNS.
        """
        expected_order = (
            self.extractor.WINDOW_ID_COLUMNS +
            self.extractor.MASK_COLUMNS +
            self.extractor.MODEL_FEATURE_COLUMNS
        )
        self.assertEqual(self.extractor.OUTPUT_COLUMNS, expected_order)
        self.assertEqual(len(self.extractor.OUTPUT_COLUMNS), 410)

    def test_ml1_ground_truth_alignment(self):
        """
        Assert UCSExtractor feature lists align with ML1's inference_feature_order_v1.json
        and scaler_params.yaml matches inference_scaler_v1.yaml.
        """
        ml1_order_path = Path("scratch/ml1_repo/artifacts/lstm/inference_feature_order_v1.json")
        if not ml1_order_path.exists():
            self.skipTest("ML1 repo artifacts not found in scratch/ml1_repo")

        with open(ml1_order_path, "r", encoding="utf-8") as f:
            ml1_order_data = json.load(f)

        ml1_features = ml1_order_data["features"]
        self.assertEqual(len(ml1_features), 406)

        # Verify our 406 MODEL_INPUT_COLUMNS matches ML1 order exactly
        self.assertEqual(self.extractor.MODEL_INPUT_COLUMNS, ml1_features)

        # Verify non-mask features (400) match MODEL_FEATURE_COLUMNS exactly
        ml1_non_masks = [c for c in ml1_features if not c.startswith("mask_")]
        self.assertEqual(self.extractor.MODEL_FEATURE_COLUMNS, ml1_non_masks)

        # Verify masks match MASK_COLUMNS exactly
        ml1_masks = [c for c in ml1_features if c.startswith("mask_")]
        self.assertEqual(self.extractor.MASK_COLUMNS, ml1_masks)

    def test_imputation_params_integrity(self):
        """
        Assert imputation_params.yaml has exactly 400 keys matching scaler_params.yaml
        and when transformed, results in approximately 0.0.
        """
        with open("data/ucs/scaler_params.yaml", "r", encoding="utf-8") as f:
            scaler_data = yaml.safe_load(f)
        with open("data/ucs/imputation_params.yaml", "r", encoding="utf-8") as f:
            impute_data = yaml.safe_load(f)

        win_medians = impute_data.get("window_feature_medians", impute_data)
        self.assertEqual(len(win_medians), 400)
        self.assertEqual(list(win_medians.keys()), list(scaler_data.keys()))

        sample_df = pd.DataFrame([win_medians])
        scaled = self.extractor.scaler.transform(sample_df)
        max_dev = np.abs(scaled[self.extractor.MODEL_FEATURE_COLUMNS].values).max()
        self.assertLess(max_dev, 1e-6, f"Imputed medians did not center near 0.0, max deviation: {max_dev}")

    def test_regression_exact_reproduction(self):
        """
        Test 1: Regression Test
        Pick a known window from Wednesday-21-02-2018 (non-PCAP day).
        Extract from intermediate cleaned flows and assert bit-exact match for all 400 features.
        """
        if self.parquet_df is None:
            self.skipTest("ucs_windows.parquet not found")

        cleaned_path = Path("data/intermediate/cleaned_Wednesday-21-02-2018_TrafficForML_CICFlowMeter.parquet")
        if not cleaned_path.exists():
            self.skipTest("Intermediate cleaned data not available")

        df_cleaned = pd.read_parquet(cleaned_path)
        t0 = df_cleaned["timestamp_utc"].min().floor("1min")
        t1 = t0 + pd.Timedelta(minutes=1)
        flows_window = df_cleaned[(df_cleaned["timestamp_utc"] >= t0) & (df_cleaned["timestamp_utc"] < t1)].copy()

        extracted_df = self.extractor.extract(flows_window, source_type="flows")
        self.assertEqual(extracted_df.shape, (1, 410))
        self.assertEqual(list(extracted_df.columns), self.extractor.OUTPUT_COLUMNS)

        # Compare with ucs_windows.parquet
        target_row = self.parquet_df[self.parquet_df["window_start_utc"] == t0].iloc[0]

        # Verify all 6 masks match exactly
        for mask_col in self.extractor.MASK_COLUMNS:
            self.assertEqual(
                extracted_df[mask_col].iloc[0],
                target_row[mask_col],
                f"Mask mismatch on {mask_col}"
            )

        # Verify all 400 features match identically
        mismatches = []
        for feat_col in self.extractor.MODEL_FEATURE_COLUMNS:
            ext_val = extracted_df[feat_col].iloc[0]
            tgt_val = target_row[feat_col]
            if ext_val != tgt_val:
                mismatches.append((feat_col, ext_val, tgt_val))

        self.assertEqual(len(mismatches), 0, f"Bit-exact mismatch on {len(mismatches)} features: {mismatches[:5]}")

    def test_synthetic_csv_extraction(self):
        """
        Test 3: Synthetic / No-PCAP extraction
        Pass minimal valid CICFlowMeter CSV data and verify extraction contract.
        """
        synthetic_flows = pd.DataFrame({
            "Dst Port": [80, 443, 80],
            "Protocol": [6, 6, 6],
            "Timestamp": ["14/02/2018 09:00:00", "14/02/2018 09:00:20", "14/02/2018 09:00:45"],
            "Flow Duration": [1000000, 2000000, 1500000],
            "Tot Fwd Pkts": [10, 20, 15],
            "Tot Bwd Pkts": [8, 15, 12],
            "TotLen Fwd Pkts": [1000, 2000, 1500],
            "TotLen Bwd Pkts": [800, 1500, 1200],
        })

        out_df = self.extractor.extract(synthetic_flows, source_type="csv")
        self.assertEqual(out_df.shape, (1, 410))
        self.assertEqual(list(out_df.columns), self.extractor.OUTPUT_COLUMNS)
        self.assertEqual(out_df["mask_has_packet_level_features"].iloc[0], 0.0)
        self.assertEqual(out_df["mask_has_identity_auth"].iloc[0], 0.0)
        self.assertEqual(out_df["mask_has_traffic_volume_features"].iloc[0], 1.0)

        # Assert no NaNs or Infs anywhere in features
        numeric_part = out_df[self.extractor.MASK_COLUMNS + self.extractor.MODEL_FEATURE_COLUMNS]
        self.assertFalse(numeric_part.isna().any().any())
        self.assertTrue(np.isfinite(numeric_part.values).all())

    def test_extract_model_tensor_entry_point(self):
        """
        Verify extract_model_tensor returns 2D NumPy float32 tensor of shape (N, 406)
        in exact MODEL_INPUT_COLUMNS positional order for backend predict().
        """
        synthetic_flows = pd.DataFrame({
            "Dst Port": [80, 443],
            "Protocol": [6, 6],
            "Timestamp": ["14/02/2018 09:00:00", "14/02/2018 09:00:30"],
            "Flow Duration": [1000000, 2000000],
            "Tot Fwd Pkts": [10, 20],
            "Tot Bwd Pkts": [8, 15],
            "TotLen Fwd Pkts": [1000, 2000],
            "TotLen Bwd Pkts": [800, 1500],
        })
        tensor = self.extractor.extract_model_tensor(synthetic_flows, source_type="csv")
        self.assertIsInstance(tensor, np.ndarray)
        self.assertEqual(tensor.shape, (1, 406))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.isfinite(tensor).all())

    def test_programmatic_diff_engine(self):
        """
        Runs the authoritative programmatic contract diff between UCSExtractor and ML1.
        Asserts 100% positional and parameter parity.
        """
        from scripts.diff_ucs_ml1_contract import run_contract_diff
        result = run_contract_diff(version="v2")
        self.assertTrue(result.passed, f"Contract diff failed: {result.positional_mismatches}")
        self.assertEqual(result.ucs_model_input_count, 406)
        self.assertEqual(result.ucs_feature_count, 400)
        self.assertEqual(result.ucs_mask_count, 6)
        self.assertEqual(len(result.positional_mismatches), 0)
        self.assertEqual(len(result.scaler_mismatches), 0)


if __name__ == "__main__":
    unittest.main()
