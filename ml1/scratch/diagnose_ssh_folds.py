import json
import pandas as pd
import numpy as np

df = pd.read_parquet("data/ucs/ucs_windows.parquet")
print("Total dataset rows:", len(df))
print("\nAttack types and labels breakdown:")
print(df.groupby(["label_attack_type", "label_binary"]).size())

print("\nSSH-Bruteforce breakdown by episode:")
ssh_df = df[df["label_attack_type"] == "SSH-Bruteforce"]
print(ssh_df.groupby(["episode_id", "label_binary"]).size())

with open("artifacts/loeo/corrected_37fold_manifest.json", "r") as f:
    manifest = json.load(f)

print("\nSSH-Bruteforce Folds in Manifest:")
for f in manifest["folds"]:
    if "SSH-Bruteforce" in f["held_out_episode_id"]:
        t_df = df.iloc[f["test_indices"]]
        tr_df = df.iloc[f["train_indices"]]
        test_att = len(t_df[t_df["label_binary"] == 1])
        test_ben = len(t_df[t_df["label_binary"] == 0])
        train_att_ssh = len(tr_df[(tr_df["label_binary"] == 1) & (tr_df["label_attack_type"] == "SSH-Bruteforce")])
        train_att_total = len(tr_df[tr_df["label_binary"] == 1])
        print(f"Fold {f['fold_id']} ({f['held_out_episode_id']}): test_att={test_att}, test_ben={test_ben}, train_att_ssh={train_att_ssh}, train_att_total={train_att_total}")
