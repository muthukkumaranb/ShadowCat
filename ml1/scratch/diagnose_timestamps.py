import json
import pandas as pd

df = pd.read_parquet("data/ucs/ucs_windows.parquet")
df["window_start_utc"] = pd.to_datetime(df["window_start_utc"], utc=True)
df = df.sort_values("window_start_utc").reset_index(drop=True)

ssh_df = df[df["label_attack_type"] == "SSH-Bruteforce"]
print("SSH-Bruteforce Episodes Timestamps:")
for ep, group in ssh_df.groupby("episode_id"):
    start = group["window_start_utc"].min()
    end = group["window_start_utc"].max()
    print(f"  {ep}: count={len(group)}, start={start}, end={end}")

with open("artifacts/loeo/corrected_37fold_manifest.json", "r") as f:
    manifest = json.load(f)

fold16 = manifest["folds"][16]
print("\nFold 16 held out episode:", fold16["held_out_episode_id"])
print("Fold 16 test indices range:", min(fold16["test_indices"]), "to", max(fold16["test_indices"]))

test_rows = df.iloc[fold16["test_indices"]]
print("Fold 16 test start:", test_rows["window_start_utc"].min(), "end:", test_rows["window_start_utc"].max())

train_rows = df.iloc[fold16["train_indices"]]
print("Fold 16 train start:", train_rows["window_start_utc"].min(), "end:", train_rows["window_start_utc"].max())
print("Fold 16 train attack types present in training set:")
print(train_rows[train_rows["label_binary"] == 1]["label_attack_type"].value_counts())
