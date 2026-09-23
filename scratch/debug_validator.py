import sys
import pandas as pd
sys.path.insert(0, "d:/sih2026")
sys.path.insert(0, "d:/sih2026/data-engineering")

from src.csv_validator import CICFlowMeterValidator
from src.canonical_mapper import map_to_canonical_schema

df = pd.read_csv("d:/sih2026/scratch/dummy_cicids2018.csv")
print("Input cols:", list(df.columns)[:5])

validator = CICFlowMeterValidator()
val_result = validator.validate(df)
print("Validator Status:", val_result.status)

mapped_df = val_result.mapped_df
print("Validator mapped_df cols:", list(mapped_df.columns)[:5])

canonical_df, _ = map_to_canonical_schema(mapped_df)
print("Canonical df cols:", list(canonical_df.columns)[:5])
if "raw_timestamp" in canonical_df.columns:
    print("Has raw_timestamp!")
else:
    print("MISSING raw_timestamp")
