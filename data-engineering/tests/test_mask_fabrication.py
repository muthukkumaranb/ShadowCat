import os
import sys
import unittest
import pandas as pd
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ucs_extractor import UCSExtractor

class TestMaskFabrication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = UCSExtractor()

    def test_incomplete_csv_masks(self):
        """
        Feed a deliberately incomplete CSV (missing entire feature categories like TCP flags and timing)
        and assert that the corresponding masks come back 0.0, not 1.0.
        """
        # Create a mock CSV input that ONLY has traffic volume (bytes/packets), but NO timing or TCP flags.
        mock_data = {
            "Timestamp": [datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S")],
            "Protocol": [6],
            "Dst Port": [80],
            "Tot Fwd Pkts": [10],
            "Tot Bwd Pkts": [10],
            "TotLen Fwd Pkts": [500],
            "TotLen Bwd Pkts": [500],
            # Notice we omit 'Flow Duration', 'Fwd PSH Flags', 'Flow IAT Mean', etc.
        }
        
        df_incomplete = pd.DataFrame(mock_data)
        
        # Extract features
        result_df = self.extractor.extract(df_incomplete, source_type="csv")
        
        # We expect traffic volume to be 1.0, and timing / tcp flags to be 0.0
        self.assertEqual(result_df["mask_has_traffic_volume_features"].iloc[0], 1.0, "Traffic volume mask should be 1.0")
        self.assertEqual(result_df["mask_has_flow_timing_features"].iloc[0], 0.0, "Flow timing mask should be 0.0")
        self.assertEqual(result_df["mask_has_tcp_flags"].iloc[0], 0.0, "TCP flags mask should be 0.0")
        self.assertEqual(result_df["mask_has_graph_topology"].iloc[0], 0.0, "Graph topology mask should be 0.0")
        self.assertEqual(result_df["mask_has_identity_auth"].iloc[0], 0.0, "Identity auth mask should be 0.0")
        
        # Ensure imputation still filled the missing features (e.g., flow_iat_mean_mean should not be NaN)
        # Using a known feature from missing categories
        self.assertFalse(result_df["flow_iat_mean_mean"].isna().any(), "Imputation should fill flow_iat_mean_mean")
        self.assertFalse(result_df["fwd_psh_flags_mean"].isna().any(), "Imputation should fill fwd_psh_flags_mean")

if __name__ == "__main__":
    unittest.main()
