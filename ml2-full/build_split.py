import pandas as pd
import numpy as np
import yaml
import os

with open("configs/dataset_config.yaml", "r") as f:
    dataset_config = yaml.safe_load(f)
with open("configs/feature_configs.yaml", "r") as f:
    feature_config = yaml.safe_load(f)
with open("configs/split_config.yaml", "r") as f:
    split_config = yaml.safe_load(f)

LOOKBACK_SEC = feature_config["temporal"]["lookback_minutes"] * 60
HORIZON_SEC = feature_config["forecast"]["horizon_sec"]

TRAIN_FRAC = split_config["split"]["train_fraction"]
VAL_FRAC = split_config["split"]["val_fraction"]

# NOTE: adjust this key if your dataset_config.yaml uses a different name
# for the labels.parquet path.
LABELS_FILE = dataset_config["paths"]["labels_output"]
OUTPUT_FILE = os.path.join(dataset_config["paths"]["split_dir"], "split_assignments.parquet")
REPORT_FILE = "reports/split_boundaries.md"

print("Loading labeled windows...")
df = pd.read_parquet(LABELS_FILE)
df = df.sort_values("timestamp").reset_index(drop=True)
n = len(df)
print(f"Loaded {n:,} windows.")

# --- Step 1: chronological cut points, by row count ---
train_end_idx = int(np.floor(n * TRAIN_FRAC))
val_end_idx = int(np.floor(n * (TRAIN_FRAC + VAL_FRAC)))

df["split"] = "test"
df.loc[:train_end_idx - 1, "split"] = "train"
df.loc[train_end_idx:val_end_idx - 1, "split"] = "val"

val_start_time = df.loc[df["split"] == "val", "timestamp"].min()
test_start_time = df.loc[df["split"] == "test", "timestamp"].min()

print(f"\nInitial chronological split (before purge/embargo):")
print(f"  train: {(df['split']=='train').sum():,} rows, before {val_start_time}")
print(f"  val:   {(df['split']=='val').sum():,} rows, {val_start_time} to before {test_start_time}")
print(f"  test:  {(df['split']=='test').sum():,} rows, from {test_start_time}")

df["purged"] = False
df["embargoed"] = False

# --- Step 2: PURGE @ train->val boundary ---
purge_mask_1 = (df["split"] == "train") & (df["timestamp"] + pd.Timedelta(seconds=HORIZON_SEC) > val_start_time)
df.loc[purge_mask_1, "purged"] = True
print(f"\nPurge @ train->val: {purge_mask_1.sum():,} train rows (label horizon crosses into val)")

# --- Step 3: PURGE @ val->test boundary ---
purge_mask_2 = (df["split"] == "val") & (df["timestamp"] + pd.Timedelta(seconds=HORIZON_SEC) > test_start_time)
df.loc[purge_mask_2, "purged"] = True
print(f"Purge @ val->test:   {purge_mask_2.sum():,} val rows (label horizon crosses into test)")

# --- Step 4: EMBARGO @ train->val boundary ---
embargo_cutoff_val = val_start_time + pd.Timedelta(seconds=LOOKBACK_SEC)
embargo_mask_1 = (df["split"] == "val") & (df["timestamp"] < embargo_cutoff_val)
df.loc[embargo_mask_1, "embargoed"] = True
print(f"\nEmbargo @ train->val: {embargo_mask_1.sum():,} val rows (lookback reaches into train, embargoed until {embargo_cutoff_val})")

# --- Step 5: EMBARGO @ val->test boundary ---
embargo_cutoff_test = test_start_time + pd.Timedelta(seconds=LOOKBACK_SEC)
embargo_mask_2 = (df["split"] == "test") & (df["timestamp"] < embargo_cutoff_test)
df.loc[embargo_mask_2, "embargoed"] = True
print(f"Embargo @ val->test:  {embargo_mask_2.sum():,} test rows (lookback reaches into val, embargoed until {embargo_cutoff_test})")

# --- Step 6: final usable rows ---
df["usable"] = ~(df["purged"] | df["embargoed"])

print("\nFinal usable counts per split:")
for s in ["train", "val", "test"]:
    total_s = (df["split"] == s).sum()
    usable_s = ((df["split"] == s) & df["usable"]).sum()
    print(f"  {s}: {usable_s:,} usable / {total_s:,} total")

# --- Step 7: save ---
os.makedirs(dataset_config["paths"]["split_dir"], exist_ok=True)
df.to_parquet(OUTPUT_FILE, index=False)
print(f"\nSaved split assignments to: {OUTPUT_FILE}")

os.makedirs("reports", exist_ok=True)
with open(REPORT_FILE, "w") as f:
    f.write("# Temporal Split Boundaries (Section 30-32)\n\n")
    f.write(f"Total windows: {n}\n\n")
    f.write("## Scope note\n")
    f.write(
        "This dataset spans only ~9.33 hours. This split validates the "
        "MECHANICS of chronological split + purge + embargo, per team lead "
        "direction — it is not a production-scale train/val/test split.\n\n"
    )
    f.write("## Boundaries\n")
    f.write(f"- Train: {df.loc[df['split']=='train','timestamp'].min()} to {df.loc[df['split']=='train','timestamp'].max()}\n")
    f.write(f"- Val start: {val_start_time}\n")
    f.write(f"- Val: {df.loc[df['split']=='val','timestamp'].min()} to {df.loc[df['split']=='val','timestamp'].max()}\n")
    f.write(f"- Test start: {test_start_time}\n")
    f.write(f"- Test: {df.loc[df['split']=='test','timestamp'].min()} to {df.loc[df['split']=='test','timestamp'].max()}\n\n")
    f.write("## Purge (Section 31)\n")
    f.write(f"- Train rows purged: {purge_mask_1.sum()}\n")
    f.write(f"- Val rows purged: {purge_mask_2.sum()}\n\n")
    f.write("## Embargo (Section 32)\n")
    f.write(f"- Val rows embargoed (until {embargo_cutoff_val}): {embargo_mask_1.sum()}\n")
    f.write(f"- Test rows embargoed (until {embargo_cutoff_test}): {embargo_mask_2.sum()}\n\n")
    f.write("## Final usable counts\n")
    for s in ["train", "val", "test"]:
        total_s = (df["split"] == s).sum()
        usable_s = ((df["split"] == s) & df["usable"]).sum()
        f.write(f"- {s}: {usable_s} usable / {total_s} total\n")

print(f"Saved boundary report to: {REPORT_FILE}")