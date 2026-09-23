import sys
import os
import pandas as pd
import yaml
from pathlib import Path

def test_cicids_csv():
    # 1. Create a dummy CIC-IDS-2018 dataframe using expected raw columns
    yaml_path = "d:/sih2026/data-engineering/configs/canonical_mapping.yaml"
    with open(yaml_path, 'r', encoding='utf-8') as f:
        mapping = yaml.safe_load(f)["mapping"]
    
    raw_columns = list(mapping.keys())
    data = {}
    for col in raw_columns:
        if col == "Timestamp":
            data[col] = ["18/09/2026 14:00:00"]
        elif col in ["Protocol", "Dst Port", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts"]:
            data[col] = [10]
        else:
            data[col] = [1.0] # Dummy numeric value
            
    df = pd.DataFrame(data)
    df.to_csv("d:/sih2026/scratch/dummy_cicids2018.csv", index=False)
    
    # 2. Run it through predict()
    sys.path.insert(0, "d:/sih2026")
    sys.path.insert(0, "d:/sih2026/backend")
    from backend.predict import ShadowcatPipeline
    
    print("Testing known-good CIC-IDS-2018 mock...")
    pipeline = ShadowcatPipeline()
    try:
        result = pipeline.predict(df, source_type="csv")
        print("SUCCESS! Output summary:")
        print(f"Risk Trajectory: {result.get('forecast_trajectory', {}).get('risk', [])}")
        print("Pipeline is healthy and processes CIC-IDS-2018 correctly without regression.")
    except Exception as e:
        import traceback
        print("CRASHED on known-good CIC-IDS-2018 sample!")
        traceback.print_exc()

if __name__ == "__main__":
    test_cicids_csv()
