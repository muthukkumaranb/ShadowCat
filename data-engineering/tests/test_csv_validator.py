"""
Unit Tests for CICFlowMeter CSV Validator Module
SIH 2026 - Unified Cyber State (UCS) Ingestion Pipeline
"""

import os
import sys
from pathlib import Path
import unittest
import pandas as pd

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.csv_validator import CICFlowMeterValidator, ValidationStatus


class TestCICFlowMeterValidator(unittest.TestCase):

    def setUp(self):
        self.validator = CICFlowMeterValidator()

    def test_exact_match(self):
        """Test A: Exact-match CSV containing all raw 80 CICFlowMeter columns passes cleanly."""
        raw_cols = list(self.validator.expected_raw_columns)
        dummy_row = {col: 0.0 for col in raw_cols}
        dummy_row["Timestamp"] = "14/02/2018 08:30:00"
        dummy_row["Dst Port"] = 80
        dummy_row["Protocol"] = 6
        dummy_df = pd.DataFrame([dummy_row])

        res = self.validator.validate(dummy_df)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.status, ValidationStatus.EXACT_MATCH)
        self.assertEqual(len(res.errors), 0)

    def test_mapped_variant(self):
        """Test B: Plausible variant CSV (lowercase, underscores, canonical names) maps correctly."""
        variant_df = pd.DataFrame({
            "dst_port": [80, 443],
            "protocol": [6, 6],
            "timestamp": ["14/02/2018 08:30:00", "14/02/2018 08:31:00"],
            "flow_duration": [1000, 2000],
            "tot_fwd_pkts": [5, 10],
            "tot_bwd_pkts": [4, 8],
            "totlen_fwd_pkts": [500, 1000],
            "totlen_bwd_pkts": [400, 800],
        })

        res = self.validator.validate(variant_df)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.status, ValidationStatus.MAPPED_VARIANT)
        self.assertGreater(len(res.variant_mappings), 0)
        self.assertIn("dst_port", res.variant_mappings)
        self.assertEqual(res.variant_mappings["dst_port"], "Dst Port")
        self.assertIsNotNone(res.mapped_df)
        self.assertIn("Dst Port", res.mapped_df.columns)

    def test_rejected_incompatible(self):
        """Test C: Incompatible CSV missing critical fields is rejected with explicit errors."""
        bad_df = pd.DataFrame({
            "random_field_1": [1, 2],
            "random_field_2": ["a", "b"],
        })

        res = self.validator.validate(bad_df)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.status, ValidationStatus.REJECTED)
        self.assertGreater(len(res.missing_required_columns), 0)
        self.assertIn("destination_port", res.missing_required_columns)
        self.assertIn("duration_microsec", res.missing_required_columns)
        self.assertTrue(any("missing critical flow fields" in err for err in res.errors))

    def test_corrupted_data_rejection(self):
        """Test D: Non-numeric corrupted fields are caught and rejected."""
        corrupt_df = pd.DataFrame({
            "Dst Port": ["corrupted_not_a_port", "bad_port"],
            "Protocol": [6, 6],
            "Timestamp": ["14/02/2018 08:30:00", "14/02/2018 08:31:00"],
            "Flow Duration": ["bad_duration", "worse_duration"],
            "Tot Fwd Pkts": [5, 10],
            "Tot Bwd Pkts": [4, 8],
            "TotLen Fwd Pkts": [500, 1000],
            "TotLen Bwd Pkts": [400, 800],
        })

        res = self.validator.validate(corrupt_df)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.status, ValidationStatus.REJECTED)
        self.assertTrue(any("non-numeric" in err for err in res.errors))

    def test_empty_dataframe_rejection(self):
        """Test E: Empty or None DataFrame is rejected."""
        res_none = self.validator.validate(None)
        self.assertFalse(res_none.is_valid)
        self.assertEqual(res_none.status, ValidationStatus.REJECTED)

        res_empty = self.validator.validate(pd.DataFrame())
        self.assertFalse(res_empty.is_valid)
        self.assertEqual(res_empty.status, ValidationStatus.REJECTED)

    def test_canonical_mapper_edge_cases(self):
        """Test F: map_to_canonical_schema handles edge-case aliases and spacing."""
        from src.canonical_mapper import map_to_canonical_schema
        edge_df = pd.DataFrame({
            " Destination Port ": [80, 443],
            "Flow Duration ": [1000, 2000],
            "Total Fwd Packets": [5, 10],
            "Total Backward Packets": [4, 8],
            "Total Length of Fwd Packets": [500, 1000],
            "Total Length of Bwd Packets": [400, 800],
            "Flow Bytes/s": [100.0, 200.0],
            "Flow Packets/s": [10.0, 20.0],
        })
        mapped_df, audit = map_to_canonical_schema(edge_df)
        self.assertIn("destination_port", mapped_df.columns)
        self.assertIn("duration_microsec", mapped_df.columns)
        self.assertIn("packet_count_fwd", mapped_df.columns)
        self.assertIn("packet_count_bwd", mapped_df.columns)
        self.assertIn("byte_count_fwd", mapped_df.columns)
        self.assertIn("byte_count_bwd", mapped_df.columns)
        self.assertIn("bytes_per_sec", mapped_df.columns)
        self.assertIn("packets_per_sec", mapped_df.columns)
        self.assertEqual(len(audit["unmapped_columns"]), 0)


if __name__ == "__main__":
    unittest.main()
