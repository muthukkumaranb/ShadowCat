import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
csv_path = repo_root / "ml1" / "artifacts" / "lstm" / "hazard_head_v3" / "hazard_head_all_horizons_loeo.csv"

df = pd.read_csv(csv_path)

def get_attack_category(fold_id):
    if 10 <= fold_id <= 18:
        return "SSH-Bruteforce"
    elif 0 <= fold_id <= 9:
        return "Botnet"
    elif 19 <= fold_id <= 36:
        return "DDOS-LOIC-UDP"
    return "Other"

df["attack_type"] = df["fold_id"].apply(get_attack_category)

attack_types = ["SSH-Bruteforce", "Botnet", "DDOS-LOIC-UDP"]
fold_counts = {"SSH-Bruteforce": 9, "Botnet": 10, "DDOS-LOIC-UDP": 18}

print("=" * 95)
print("HAZARD HEAD (v3) EVALUATION BREAKDOWN BY ATTACK TYPE ACROSS 37 LOEO FOLDS")
print("=" * 95)

results = []

for h in (1, 2, 5):
    df_h = df[df["horizon"] == h]
    
    # Blended
    b_f1 = df_h["f1"].mean()
    b_prec = df_h["precision"].mean()
    b_rec = df_h["recall"].mean()
    b_fpr = df_h["fpr"].mean()
    b_roc = df_h["roc_auc"].mean()
    b_pr = df_h["pr_auc"].mean()
    
    results.append({
        "horizon": h,
        "subset": "Blended Total (All Folds)",
        "fold_count": 37,
        "f1": b_f1,
        "precision": b_prec,
        "recall": b_rec,
        "fpr": b_fpr,
        "roc_auc": b_roc,
        "pr_auc": b_pr,
    })
    
    for atype in attack_types:
        sub = df_h[df_h["attack_type"] == atype]
        results.append({
            "horizon": h,
            "subset": atype,
            "fold_count": len(sub),
            "f1": sub["f1"].mean(),
            "precision": sub["precision"].mean(),
            "recall": sub["recall"].mean(),
            "fpr": sub["fpr"].mean(),
            "roc_auc": sub["roc_auc"].mean(),
            "pr_auc": sub["pr_auc"].mean(),
        })

res_df = pd.DataFrame(results)

for h in (1, 2, 5):
    print(f"\n--- Horizon H={h} ---")
    h_sub = res_df[res_df["horizon"] == h]
    print(f"{'Subset':<28} | {'Folds':<5} | {'F1 Mean':<8} | {'Precision':<9} | {'Recall':<8} | {'FPR Mean':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8}")
    print("-" * 95)
    for _, r in h_sub.iterrows():
        print(f"{r['subset']:<28} | {r['fold_count']:<5} | {r['f1']:<8.4f} | {r['precision']:<9.4f} | {r['recall']:<8.4f} | {r['fpr']:<8.4f} | {r['roc_auc']:<8.4f} | {r['pr_auc']:<8.4f}")

# Also generate markdown table snippet
print("\n" + "=" * 95)
print("MARKDOWN TABLE SNIPPET FOR REPORT")
print("=" * 95)
md_lines = []
md_lines.append("## Onset Forecasting Performance Broken Down by Attack Type\n")
md_lines.append("| Horizon | Attack Type Subset | Fold Count | F1 Mean | Precision Mean | Recall Mean | FPR Mean | ROC-AUC Mean | PR-AUC Mean |")
md_lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
for _, r in res_df.iterrows():
    md_lines.append(f"| **H={r['horizon']}** | {r['subset']} | {r['fold_count']} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['fpr']:.4f} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} |")

print("\n".join(md_lines))
