import os
import sys
from pathlib import Path
import pandas as pd
import traceback

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "data-engineering"))
sys.path.insert(0, str(ROOT_DIR))

from src.ucs_extractor import UCSExtractor
from backend.predict import ShadowcatPipeline

def test_live_csv_path():
    print("============================================================")
    print("LIVE CSV DEMO PATH VERIFICATION")
    print("============================================================")
    
    csv_path = ROOT_DIR / "backend" / "tests" / "fixtures" / "judge_demo_sample.csv"
    if not csv_path.exists():
        print(f"[FAIL] Fixture not found: {csv_path}")
        sys.exit(1)
        
    print(f"\n[1] Loading raw CSV from: {csv_path.relative_to(ROOT_DIR)}")
    try:
        raw_df = pd.read_csv(csv_path)
        print(f"    -> Raw DataFrame shape: {raw_df.shape}")
        print(f"    -> Columns sample: {list(raw_df.columns)[:5]}...")
    except Exception as e:
        print(f"    [FAIL] Could not load CSV: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n[2] Explicitly running UCSExtractor (front-door extraction)...")
    try:
        extractor = UCSExtractor(schema_version="v3.0")
        model_tensor = extractor.extract_model_tensor(raw_df, source_type="csv")
        print(f"    -> Extracted Tensor shape: {model_tensor.shape}")
        print(f"    -> Extracted Tensor slice (first 5 features): {model_tensor[0, :5]}")
    except Exception as e:
        print(f"    [FAIL] UCSExtractor extraction failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n[3] Initializing ShadowcatPipeline...")
    try:
        pipeline = ShadowcatPipeline()
    except Exception as e:
        print(f"    [FAIL] Pipeline init failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n[4] Running predict() on raw DataFrame...")
    try:
        results = pipeline.predict(raw_df, source_type="csv")
        
        print("\n[5] Pipeline output results:")
        print(f"    -> Analysis Source: {results.get('analysis_metadata', {}).get('source')}")
        print(f"    -> Novelty Score: {results.get('novelty_score', {}).get('novelty_score')}")
        print(f"    -> Hazard Alert: {results.get('forecast_trajectory', {}).get('hazard_alert')}")
        print(f"    -> Risk Scores: {results.get('forecast_trajectory', {}).get('risk')}")
        
        print("\n============================================================")
        print("VERIFICATION COMPLETED SUCCESSFULLY")
        print("============================================================")
        
    except Exception as e:
        print(f"    [FAIL] predict() failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_live_csv_path()
