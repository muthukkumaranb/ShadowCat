import pyarrow.parquet as pq
import pandas as pd
import numpy as np

FILE_PATH = r"E:\Data Set folders\UGR 16\august_week5.parquet"

parquet_file = pq.ParquetFile(FILE_PATH)

columns_to_read = ["ts_start"]
table = parquet_file.read(columns=columns_to_read)
df = table.to_pandas()

total = len(df)
print(f"Total rows: {total:,}")
print("-" * 60)

ts = df["ts_start"].to_numpy()

# 1. Is the column already sorted in increasing order?
is_sorted = np.all(np.diff(ts) >= 0)
print(f"Is ts_start already sorted (ascending)? {is_sorted}")

# 2. How many rows are "out of order" (i.e. earlier than the row before them)?
out_of_order_count = np.sum(np.diff(ts) < 0)
print(f"Number of out-of-order rows: {out_of_order_count:,}")
print("-" * 60)

# 3. Duplicate exact timestamps (multiple flows starting at the same instant)
duplicate_ts_count = total - df["ts_start"].nunique()
print(f"Rows with a duplicate ts_start value: {duplicate_ts_count:,}")
print("-" * 60)

# 4. Gap analysis - sort first (in memory, but only 1 column, so it's light),
#    then look at the biggest gaps between consecutive timestamps
sorted_ts = np.sort(ts)
gaps = np.diff(sorted_ts)

print(f"Smallest gap between consecutive timestamps: {gaps.min():.6f} seconds")
print(f"Largest gap between consecutive timestamps:  {gaps.max():.6f} seconds")
print(f"Average gap: {gaps.mean():.6f} seconds")

# Show the 5 biggest gaps - these could indicate capture outages
top_gap_indices = np.argsort(gaps)[-5:][::-1]
print("\nTop 5 largest gaps (in seconds), with surrounding timestamps:")
for idx in top_gap_indices:
    print(f"  Gap of {gaps[idx]:.2f}s between {sorted_ts[idx]:.2f} and {sorted_ts[idx+1]:.2f}")