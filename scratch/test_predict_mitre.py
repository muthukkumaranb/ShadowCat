import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from backend.predict import predict

parquet_path = "data-engineering/data/ucs/ucs_windows.parquet"
df = pd.read_parquet(parquet_path).head(45).copy()

output = predict(df, source_type="flows")
fc = output["forecast_trajectory"]

print("FORECAST TRAJECTORY MITRE FIELDS:")
print("tactic_id:", fc.get("tactic_id"))
print("stage:", fc.get("stage"))
print("technique_id:", fc.get("technique_id"))
print("technique_name:", fc.get("technique_name"))
print("technique_full_name:", fc.get("technique_full_name"))
print("technique_url:", fc.get("technique_url"))
print("is_heuristic_progression:", fc.get("is_heuristic_progression"))
print("stage_attribution_source:", fc.get("stage_attribution_source"))

print("\nSAMPLE RAW STEP (Step 1):")
print(json.dumps(fc["raw_steps"][0], indent=2))
