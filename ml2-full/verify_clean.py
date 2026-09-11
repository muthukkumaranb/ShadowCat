import pyarrow.parquet as pq
import numpy as np

FILE_PATH = "data/intermediate/ugr16/cleaned_sorted.parquet"

parquet_file = pq.ParquetFile(FILE_PATH)
table = parquet_file.read(columns=["timestamp", "scope_id", "dataset"])
df = table.to_pandas()

print(f"Total rows: {len(df):,}")
print("-" * 50)

# 1. Confirm sort worked
ts_numeric = df["timestamp"].astype("int64")  # nanoseconds since epoch, for comparison
is_sorted = np.all(np.diff(ts_numeric) >= 0)
print(f"Is timestamp sorted ascending? {is_sorted}")

# 2. Confirm scope_id and dataset are correctly set on every row
print(f"Unique scope_id values: {df['scope_id'].unique()}")
print(f"Unique dataset values: {df['dataset'].unique()}")

# 3. Show first and last timestamp to confirm range still makes sense
print(f"First timestamp: {df['timestamp'].iloc[0]}")
print(f"Last timestamp:  {df['timestamp'].iloc[-1]}")