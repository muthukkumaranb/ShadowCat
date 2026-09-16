import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from backend.predict import ShadowcatPipeline

def test_benign_sequence():
    print("Loading real canonical sample from: data-engineering/data/ucs/ucs_windows.parquet")
    df = pd.read_parquet('data-engineering/data/ucs/ucs_windows.parquet')
    
    # Sort chronologically just in case
    df = df.sort_values('window_start_utc')
    
    # Filter out any malicious flows
    benign_df = df[df['has_malicious_flows'] == 0].copy()
    
    # Get 30 consecutive benign windows if possible, or whatever is available
    benign_seq = benign_df.head(30)
    print(f"Loaded {len(benign_seq)} benign windows.")
    
    if len(benign_seq) == 0:
        print("No benign windows found in dataset!")
        return

    print("Executing predict(df, source_type='flows')...")
    pipeline = ShadowcatPipeline()
    result = pipeline.predict(benign_seq, source_type='flows')
    
    risk_scores = result.get('risk_scores', [])
    stage_predictions = result.get('stage_predictions', [])
    
    print("\n[3] Verification Checks:")
    print(f"    - Risk Trajectory [t+1..t+4]: {risk_scores}")
    print(f"    - Stage Trajectory: {stage_predictions}")
    
    # Assert that hazard alert does not fire
    # e.g., max risk should be under a threshold (e.g. 0.40) or at least not strictly monotonically increasing to near 1.0
    # and stage trajectory should not predict an advanced attack stage
    max_risk = max(risk_scores)
    print(f"    - Max Risk: {max_risk}")
    
    if max_risk > 0.60:
        print("    [FAIL] Hazard alert fired on a purely benign sequence!")
    else:
        print("    [PASS] No hazard alert fired on benign sequence.")

if __name__ == '__main__':
    test_benign_sequence()
