import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

workspace_dir = Path(".")
df = pd.read_parquet("data/ucs/ucs_windows.parquet")
df["window_start_utc"] = pd.to_datetime(df["window_start_utc"], utc=True)
df = df.sort_values("window_start_utc").reset_index(drop=True)

with open("artifacts/lr_set_a_corrected/set_a_features.json", "r") as f:
    set_a = json.load(f)
features = set_a["ordered_feature_names"]

with open("artifacts/loeo/corrected_37fold_manifest.json", "r") as f:
    manifest = json.load(f)

fold16 = manifest["folds"][16]
train_idx = fold16["train_indices"]
test_idx = fold16["test_indices"]

print("Fold 16 held out episode:", fold16["held_out_episode_id"])
fold16_attack_type = df.iloc[test_idx]["label_attack_type"].dropna().unique()[0] if "label_attack_type" in df.columns else fold16["held_out_episode_id"].split("_")[1]
print("Fold 16 attack_type:", fold16_attack_type)

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[features].to_numpy(dtype=float))
y_train = train_df["label_binary"].to_numpy(dtype=int)

X_test = scaler.transform(test_df[features].to_numpy(dtype=float))
y_test = test_df["label_binary"].to_numpy(dtype=int)

clf = LogisticRegression(solver="liblinear", max_iter=2000, random_state=42)
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)[:, 1]
test_df_copy = test_df.copy()
test_df_copy["prob"] = probs
test_df_copy["pred"] = (probs >= 0.5).astype(int)

print("\n--- Test Set Probability Summary by Class (Fold 16) ---")
print(test_df_copy.groupby("label_binary")["prob"].describe())

# Find top positive coefficients
coefs = clf.coef_[0]
top_pos_idx = np.argsort(coefs)[::-1][:10]
print("\n--- Top 10 Positive Coefficients (Fold 16 Model) ---")
for idx in top_pos_idx:
    print(f"{features[idx]}: coef = {coefs[idx]:.4f}")

# Let's inspect features for:
# 1) Fold 16 test attack windows (n=33)
# 2) Fold 16 train attack windows (all other SSH-Bruteforce attack windows in train)
# 3) Other SSH-Bruteforce test attack windows across other folds

other_ssh_test_attack_rows = []
for f in manifest["folds"]:
    if "SSH-Bruteforce" in f["held_out_episode_id"] and f["fold_id"] != 16:
        t_df = df.iloc[f["test_indices"]]
        att = t_df[t_df["label_binary"] == 1]
        other_ssh_test_attack_rows.append(att)
other_ssh_test_attacks = pd.concat(other_ssh_test_attack_rows)

train_ssh_attacks = train_df[(train_df["label_binary"] == 1) & (train_df["label_attack_type"] == "SSH-Bruteforce")]

fold16_test_attacks = test_df[test_df["label_binary"] == 1]

print(f"\nFold 16 test attack count: {len(fold16_test_attacks)}")
print(f"Train SSH attack count (in Fold 16 train set): {len(train_ssh_attacks)}")
print(f"Other SSH folds test attack count: {len(other_ssh_test_attacks)}")

# Check top feature comparisons across these groups
print("\n--- Feature Comparisons across SSH-Bruteforce Groups ---")
top_feats = [features[idx] for idx in top_pos_idx[:5]]
for tf in top_feats:
    print(f"\nFeature: {tf}")
    print(f"  Fold 16 Test Attacks (n={len(fold16_test_attacks)}): mean={fold16_test_attacks[tf].mean():.4f}, median={fold16_test_attacks[tf].median():.4f}, std={fold16_test_attacks[tf].std():.4f}")
    print(f"  Train SSH Attacks (n={len(train_ssh_attacks)}):    mean={train_ssh_attacks[tf].mean():.4f}, median={train_ssh_attacks[tf].median():.4f}, std={train_ssh_attacks[tf].std():.4f}")
    print(f"  Other SSH Test Attacks (n={len(other_ssh_test_attacks)}): mean={other_ssh_test_attacks[tf].mean():.4f}, median={other_ssh_test_attacks[tf].median():.4f}, std={other_ssh_test_attacks[tf].std():.4f}")
    benign_train = train_df[train_df["label_binary"] == 0]
    print(f"  Train Benign (n={len(benign_train)}):          mean={benign_train[tf].mean():.4f}, median={benign_train[tf].median():.4f}, std={benign_train[tf].std():.4f}")
