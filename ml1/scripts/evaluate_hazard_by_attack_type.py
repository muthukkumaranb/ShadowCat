"""
Hazard Head Evaluation by Attack Type across 37 LOEO Folds (Supports v3 and v4 artifacts)
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent

def evaluate_hazard_dir(target_dir: Path, tag: str = "v4"):
    csv_path = target_dir / "hazard_head_all_horizons_loeo.csv"
    if not csv_path.exists():
        print(f"[-] CSV not found at {csv_path}")
        return None

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

    print("=" * 95)
    print(f"HAZARD HEAD ({tag}) EVALUATION BREAKDOWN BY ATTACK TYPE ACROSS 37 LOEO FOLDS")
    print("=" * 95)

    results = []
    for h in (1, 2, 5):
        df_h = df[df["horizon"] == h]
        
        # Blended Total
        results.append({
            "horizon": h,
            "subset": "Blended Total (All Folds)",
            "fold_count": len(df_h),
            "f1": df_h["f1"].mean(),
            "precision": df_h["precision"].mean(),
            "recall": df_h["recall"].mean(),
            "fpr": df_h["fpr"].mean(),
            "roc_auc": df_h["roc_auc"].mean(),
            "pr_auc": df_h["pr_auc"].mean(),
        })
        
        # Per attack type
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

    return res_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None, help="Directory containing hazard head results")
    parser.add_argument("--tag", type=str, default="v4")
    parser.add_argument("--compare", action="store_true", help="Compare v3 vs v4")
    args = parser.parse_args()

    if args.compare:
        v3_dir = repo_root / "ml1" / "artifacts" / "lstm" / "hazard_head_v3"
        v4_dir = repo_root / "ml1" / "artifacts" / "lstm" / "hazard_head_v4"
        df_v3 = evaluate_hazard_dir(v3_dir, tag="v3")
        df_v4 = evaluate_hazard_dir(v4_dir, tag="v4")
        
        if df_v3 is not None and df_v4 is not None:
            print("\n" + "=" * 95)
            print("SIDE-BY-SIDE COMPARISON: HAZARD HEAD v3 (14-02 only) vs v4 (14-02 + 02-03)")
            print("=" * 95)
            merged = pd.merge(df_v3, df_v4, on=["horizon", "subset", "fold_count"], suffixes=("_v3", "_v4"))
            for h in (1, 2, 5):
                print(f"\n--- Horizon H={h} ---")
                h_sub = merged[merged["horizon"] == h]
                print(f"{'Subset':<24} | {'v3 F1':<7} -> {'v4 F1':<7} | {'v3 ROC':<7} -> {'v4 ROC':<7} | {'v3 PR':<7} -> {'v4 PR':<7}")
                print("-" * 80)
                for _, r in h_sub.iterrows():
                    print(f"{r['subset']:<24} | {r['f1_v3']:<7.4f} -> {r['f1_v4']:<7.4f} | {r['roc_auc_v3']:<7.4f} -> {r['roc_auc_v4']:<7.4f} | {r['pr_auc_v3']:<7.4f} -> {r['pr_auc_v4']:<7.4f}")
    else:
        v5_default = repo_root / "ml1" / "artifacts" / "lstm" / "hazard_head_v5"
        v4_default = repo_root / "ml1" / "artifacts" / "lstm" / "hazard_head_v4"
        v3_default = repo_root / "ml1" / "artifacts" / "lstm" / "hazard_head_v3"
        target_dir = Path(args.dir) if args.dir else (v5_default if v5_default.exists() else (v4_default if v4_default.exists() else v3_default))
        tag = args.tag or target_dir.name.split("_")[-1]
        evaluate_hazard_dir(target_dir, tag=tag)

if __name__ == "__main__":
    main()
