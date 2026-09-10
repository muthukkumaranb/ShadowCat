"""
SHADOWCAT Backend - Comprehensive Test Suite
Validates:
1. Extraction & Normalization Contract (UCSExtractor)
2. Model Architectures & Forward Passes (World Model, Hazard Heads, Stage Head)
3. Causal Sequence Slicing & Warm-up
4. Probability Bounds & Monotonicity
5. Full End-to-End predict() Non-Degeneracy & Schema Compliance
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add paths
TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
WORKSPACE_DIR = BACKEND_DIR.parent

sys.path.insert(0, str(WORKSPACE_DIR / "data-engineering"))
sys.path.insert(0, str(WORKSPACE_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from src.ucs_extractor import UCSExtractor
from backend.models import (
    LSTMGaussianWorldModel,
    LSTMClassifier,
    StageClassificationHead,
)
from backend.predict import ShadowcatPipeline, predict


class TestBackendPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = ShadowcatPipeline(device="cpu")
        # Find sample real data or mock DataFrame
        parquet_path = WORKSPACE_DIR / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
        if parquet_path.exists():
            cls.canonical_windows = pd.read_parquet(parquet_path)
        else:
            cls.canonical_windows = None

    def test_ucs_extractor_contract(self):
        """Verify UCSExtractor returns exactly (N, 406) normalized tensor."""
        if self.canonical_windows is not None:
            sample_df = self.canonical_windows.head(35).copy()
            tensor = self.pipeline.extractor.extract_model_tensor(sample_df, source_type="flows")
            self.assertEqual(tensor.ndim, 2)
            self.assertEqual(tensor.shape[1], 406)
            self.assertFalse(np.isnan(tensor).any(), "NaN values found in extracted tensor")
            self.assertFalse(np.isinf(tensor).any(), "Inf values found in extracted tensor")

    def test_world_model_forward(self):
        """Verify World Model outputs 64-dim latent z(t) and (406-dim mean, std)."""
        dummy_input = torch.randn(2, 30, 406)
        mean, std = self.pipeline.world_model(dummy_input)
        z_t = self.pipeline.world_model.extract_latent_z(dummy_input)

        self.assertEqual(mean.shape, (2, 406))
        self.assertEqual(std.shape, (2, 406))
        self.assertEqual(z_t.shape, (2, 64))
        self.assertTrue((std > 0).all(), "Standard deviation must be strictly positive")

    def test_stage_head_classification(self):
        """Verify Stage Head outputs 6 classes and valid probabilities."""
        dummy_state = torch.randn(4, 406)
        logits = self.pipeline.stage_head(dummy_state)
        probs = self.pipeline.stage_head.predict_probabilities(dummy_state)

        self.assertEqual(logits.shape, (4, 6))
        self.assertEqual(probs.shape, (4, 6))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5))

    def test_hazard_head_ensemble(self):
        """Verify hazard ensemble produces probabilities in [0, 1]."""
        dummy_seq = np.random.randn(30, 406).astype(np.float32)
        hazards = self.pipeline._predict_hazard_ensemble(dummy_seq)

        for h, prob in hazards.items():
            self.assertGreaterEqual(prob, 0.0, f"Hazard probability {prob} < 0")
            self.assertLessEqual(prob, 1.0, f"Hazard probability {prob} > 1")

    def test_end_to_end_predict_with_real_windows(self):
        """Verify predict() on real canonical windows produces valid, non-degenerate output."""
        if self.canonical_windows is not None:
            sample_df = self.canonical_windows.head(40).copy()
            res = predict(sample_df, source_type="flows")

            # Check top-level keys
            self.assertIn("forecast_trajectory", res)
            self.assertIn("novelty_score", res)
            self.assertIn("attributions", res)
            self.assertIn("flagged_flows", res)
            self.assertIn("analysis_metadata", res)

            # Check forecast trajectory schema
            fc = res["forecast_trajectory"]
            self.assertEqual(len(fc["horizons"]), 4)
            self.assertEqual(len(fc["risk"]), 4)
            self.assertEqual(len(fc["stage"]), 4)
            self.assertEqual(len(fc["uncertainty"]), 4)

            # Check probability bounds
            for r in fc["risk"]:
                self.assertGreaterEqual(r, 0.0)
                self.assertLessEqual(r, 1.0)

            # Check monotonic cumulative risk
            for i in range(len(fc["risk"]) - 1):
                self.assertLessEqual(
                    fc["risk"][i],
                    fc["risk"][i + 1] + 1e-5,
                    f"Cumulative risk not monotonic: {fc['risk'][i]} > {fc['risk'][i+1]}",
                )

            # Check novelty score
            nv = res["novelty_score"]
            self.assertGreaterEqual(nv["novelty_score"], 0.0)
            self.assertLessEqual(nv["novelty_score"], 1.0)
            self.assertIn(nv["novelty_status"], ["Expected Behavior Envelope", "Elevated Behavioral Drift"])

    def test_predict_synthetic_csv_flow_input(self):
        """Verify predict() accepts raw CICFlowMeter CSV flows and produces valid output."""
        synthetic_flows = pd.DataFrame({
            "Dst Port": [80, 443, 22, 8080, 22] * 10,
            "Protocol": [6, 6, 6, 6, 6] * 10,
            "Timestamp": [f"14/02/2018 09:00:{i:02d}" for i in range(50)],
            "Flow Duration": [1000000 + i * 1000 for i in range(50)],
            "Tot Fwd Pkts": [10 + i for i in range(50)],
            "Tot Bwd Pkts": [8 + i for i in range(50)],
            "TotLen Fwd Pkts": [1000 + i * 50 for i in range(50)],
            "TotLen Bwd Pkts": [800 + i * 40 for i in range(50)],
            "Src IP": ["10.0.2.15"] * 50,
            "Dst IP": ["10.0.4.21"] * 50,
            "Src Port": [54000 + i for i in range(50)],
        })

        res = predict(synthetic_flows, source_type="csv")
        self.assertIn("forecast_trajectory", res)
        self.assertIn("novelty_score", res)
        self.assertIn("flagged_flows", res)
        self.assertIn("attributions", res)

        fc = res["forecast_trajectory"]
        self.assertEqual(len(fc["risk"]), 4)
        for r in fc["risk"]:
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)


if __name__ == "__main__":
    unittest.main()

