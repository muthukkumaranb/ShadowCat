"""
Phase B-t3 Verification Test: Current-Window ATT&CK Stage Classification
Validates:
1. One real benign window outputs 'No attack detected' when detection_prob < 5% FPR threshold.
2. One real SSH-Bruteforce window outputs 'Credential Access' when detection_prob >= 5% FPR threshold.
3. current_stage_source contains 'family classifier (current window, LOEO macro-F1 = 0.6402)'.
4. rollout_stage_k and rollout_stage_k_note ('world-model rollout, not validated') are present.
"""

from pathlib import Path
import unittest
import pandas as pd

from backend.predict import predict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PARQUET_PATH = REPO_ROOT / "data-engineering" / "data" / "ucs" / "ucs_windows_models_v1.parquet"


class TestCurrentStage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        assert PARQUET_PATH.exists(), f"Parquet file not found at {PARQUET_PATH}"
        cls.df = pd.read_parquet(PARQUET_PATH)

    def test_benign_window_current_stage(self):
        # First 35 windows are benign
        benign_sample = self.df.iloc[0:35].copy()
        res = predict(benign_sample, source_type="windows")

        # Gated by detection prob < 5% FPR threshold
        self.assertEqual(res.get("current_stage"), "No attack detected")
        self.assertIsNone(res.get("current_stage_confidence"))
        self.assertIn("family classifier (current window, LOEO macro-F1 = 0.6402)", res.get("current_stage_source", ""))

        # Verify rollout_stage_k note is present
        self.assertIn("rollout_stage_k", res)
        self.assertEqual(res.get("rollout_stage_k_note"), "world-model rollout, not validated")

        # Also verify in forecast_trajectory
        fc = res.get("forecast_trajectory", {})
        self.assertEqual(fc.get("current_stage"), "No attack detected")
        self.assertEqual(fc.get("rollout_stage_k_note"), "world-model rollout, not validated")

    def test_ssh_bruteforce_window_current_stage(self):
        # Windows 35:67 ends on window 66 which is SSH-Bruteforce
        ssh_sample = self.df.iloc[35:67].copy()
        self.assertEqual(ssh_sample.iloc[-1]["label_attack_type"], "SSH-Bruteforce")

        res = predict(ssh_sample, source_type="windows")

        # Detection prob should exceed 5% FPR threshold (0.4014)
        det_p = res.get("detection_probability")
        self.assertIsNotNone(det_p)
        self.assertGreaterEqual(det_p, 0.4014)

        # Stage should be Credential Access
        self.assertEqual(res.get("current_stage"), "Credential Access")
        conf = res.get("current_stage_confidence")
        self.assertIsNotNone(conf)
        self.assertGreater(conf, 0.5)
        self.assertIn("family classifier (current window, LOEO macro-F1 = 0.6402)", res.get("current_stage_source", ""))

        # Rollout stage note
        self.assertIn("rollout_stage_k", res)
        self.assertEqual(res.get("rollout_stage_k_note"), "world-model rollout, not validated")


if __name__ == "__main__":
    unittest.main()
