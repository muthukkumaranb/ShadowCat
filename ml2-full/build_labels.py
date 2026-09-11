import pandas as pd
import yaml

with open("configs/dataset_config.yaml", "r") as f:
    dataset_config = yaml.safe_load(f)
with open("configs/feature_configs.yaml", "r") as f:
    feature_config = yaml.safe_load(f)

HORIZON_SEC = feature_config["forecast"]["horizon_sec"]  # 600
WINDOWS_FILE = dataset_config["paths"]["windows_file"]
FLOWS_FILE = dataset_config["paths"]["intermediate_dir"] + "../ugr16/validated_flows.parquet"
OUTPUT_FILE = dataset_config["paths"]["labels_output"]
DATASET_NAME = dataset_config["dataset"]["name"]
SCOPE_ID = dataset_config["scope"]["id"]

print("Loading UCS windows (defines one row per prediction timestamp t)...")
windows = pd.read_parquet(WINDOWS_FILE, columns=["timestamp"])
windows = windows.sort_values("timestamp").reset_index(drop=True)

print("Loading raw flow labels (to find WHEN the verified attack occurred)...")
flows = pd.read_parquet("data/intermediate/ugr16/validated_flows.parquet",
                         columns=["timestamp", "label_binary", "label_attack_type"])

attack_flows = flows[flows["label_binary"] == "malicious"].copy()
print(f"Found {len(attack_flows):,} attack flow(s):")
print(attack_flows)

horizon = pd.Timedelta(seconds=HORIZON_SEC)

labels = []
for t in windows["timestamp"]:
    window_end = t + horizon
    # future_attack(t, H) = 1 if an attack timestamp falls within [t, t+H]
    matches = attack_flows[(attack_flows["timestamp"] >= t) & (attack_flows["timestamp"] <= window_end)]
    if len(matches) > 0:
        future_attack = True
        attack_type = matches.iloc[0]["label_attack_type"]
        attack_onset = matches.iloc[0]["timestamp"]
    else:
        future_attack = False
        attack_type = None
        attack_onset = pd.NaT

    labels.append({
        "timestamp": t,
        "dataset": DATASET_NAME,
        "scope_id": SCOPE_ID,
        "future_attack": future_attack,
        "attack_type": attack_type,
        "attack_onset": attack_onset,
        "horizon_sec": HORIZON_SEC,
    })

labels_df = pd.DataFrame(labels)

positive_count = labels_df["future_attack"].sum()
total = len(labels_df)
print(f"\nTotal label rows: {total:,}")
print(f"future_attack = True: {positive_count:,} ({positive_count/total*100:.4f}%)")
print(f"future_attack = False: {total - positive_count:,} ({(total-positive_count)/total*100:.4f}%)")

labels_df.to_parquet(OUTPUT_FILE, index=False)
print(f"\nSaved to: {OUTPUT_FILE}")