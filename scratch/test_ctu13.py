import sys
import os
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path("d:/sih2026")))
sys.path.insert(0, str(Path("d:/sih2026/backend")))
sys.path.insert(0, str(Path("d:/sih2026/data-engineering")))

from backend.predict import ShadowcatPipeline

def test_ctu13():
    print("Loading CTU-13 dummy slice...")
    df = pd.read_csv("d:/sih2026/data/ctu13_sample.csv")
    print(f"Columns: {list(df.columns)}")
    
    print("Initializing ShadowcatPipeline...")
    pipeline = ShadowcatPipeline()
    
    print("Running predict()...")
    try:
        result = pipeline.predict(df, source_type="csv")
        print("SUCCESS! Output summary:")
        print(f"Hazard Alert: {result.get('forecast', {}).get('hazard_alert', 'N/A')}")
        print(f"Risk Trajectory: {result.get('forecast', {}).get('risk', [])}")
        print(f"Stage Trajectory: {result.get('forecast', {}).get('stage', [])}")
    except Exception as e:
        import traceback
        print("CRASHED!")
        traceback.print_exc()

if __name__ == "__main__":
    test_ctu13()
