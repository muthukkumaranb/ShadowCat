import pandas as pd
import numpy as np
import yaml

with open("configs/dataset_config.yaml", "r") as f:
    dataset_config = yaml.safe_load(f)
with open("configs/feature_config.yaml", "r") as f:
    feature_config = yaml.safe_load(f)

WINDOW_HOURS = feature_config["baseline"]["window_hours"]
METRICS = feature_config["baseline"]["metrics"]
INPUT_FILE = dataset_config["paths"]["windows_file"]
OUTPUT_FILE = dataset_config["paths"]["baseline_output"]

print("Loading UCS windows...")
df = pd.read_parquet(INPUT_FILE)
df = df.sort_values("timestamp").reset_index(drop=True)
total = len(df)
print(f"Loaded {total:,} windows.")

first_timestamp = df["timestamp"].min()
lookback = pd.Timedelta(hours=WINDOW_HOURS)

# We need a time-indexed copy to use Pandas' time-based rolling window
df_indexed = df.set_index("timestamp")

for zscore_name, source_column in METRICS.items():
    print(f"\nComputing {zscore_name} from '{source_column}'...")

    # closed="left" means: window covers [t - 24h, t), NOT including t itself.
    # This guarantees the baseline never includes the current (or future) observation.
    rolling = df_indexed[source_column].rolling(f"{WINDOW_HOURS}h", closed="left")

    mean_past = rolling.mean()
    std_past = rolling.std()
    count_past = rolling.count()

    eps = 1e-6
    raw_zscore = (df_indexed[source_column] - mean_past) / (std_past + eps)

    # ELIGIBILITY RULE (this is the core of the frozen decision):
    # A row only gets a real z-score if:
    #   (a) at least 24 real hours have elapsed since the first timestamp in the file, AND
    #   (b) there are at least 2 historical points to compute a meaningful std dev
    hours_elapsed = (df_indexed.index - first_timestamp).total_seconds() / 3600.0
    has_full_window = hours_elapsed >= WINDOW_HOURS
    has_enough_points = count_past >= 2
    eligible = has_full_window & has_enough_points

    missing_col = f"{zscore_name}_missing"

    df_indexed[zscore_name] = np.where(eligible, raw_zscore, np.nan)
    df_indexed[missing_col] = np.where(eligible, 0, 1)

    eligible_count = eligible.sum()
    missing_count = total - eligible_count
    print(f"  Eligible (real z-score computed): {eligible_count:,} ({eligible_count/total*100:.4f}%)")
    print(f"  Missing (insufficient 24h history): {missing_count:,} ({missing_count/total*100:.4f}%)")

df = df_indexed.reset_index()

# --- IMPUTATION NOTE ---
# The frozen decision requires: feature = training-derived imputed value, when missing.
# A training-derived median can only be calculated from rows that HAVE a real
# z-score. If 100% of rows are missing (as expected for this file), there is
# no valid training data to derive a median from yet. We leave the value as
# NaN in that case and document this explicitly, rather than inventing a number.
for zscore_name in METRICS.keys():
    eligible_values = df.loc[df[f"{zscore_name}_missing"] == 0, zscore_name]
    if len(eligible_values) > 0:
        median_value = eligible_values.median()
        df.loc[df[f"{zscore_name}_missing"] == 1, zscore_name] = median_value
        print(f"\n{zscore_name}: imputed missing rows with training-derived median = {median_value:.6f}")
    else:
        print(f"\n{zscore_name}: NO eligible rows exist in this file — "
              f"no training-derived median can be calculated. "
              f"Left as NaN pending a dataset with sufficient history.")

print(f"\nFinal columns: {list(df.columns)}")
df.to_parquet(OUTPUT_FILE, index=False)
print(f"\nSaved to: {OUTPUT_FILE}")