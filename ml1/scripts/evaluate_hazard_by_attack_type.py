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

def evaluate_hazard_dir(target_dir: Path, tag: str = "v4", csv_name: str = "hazard_head_all_horizons_loeo.csv"):
    csv_path = target_dir / csv_name
    if not csv_path.exists():
        # Fall back if calibrated requested but default exists
        alt_path = target_dir / "hazard_head_all_horizons_loeo.csv"
        if alt_path.exists():
            print(f"[*] {csv_name} not found, falling back to {alt_path.name}")
            csv_path = alt_path
        else:
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

    if "attack_type" not in df.columns:
        df["attack_type"] = df["fold_id"].apply(get_attack_category)
    attack_types = ["SSH-Bruteforce", "Botnet", "DDOS-LOIC-UDP"]

    mode_label = "VALIDATION-CALIBRATED HONEST" if "val_calibrated" in csv_name else "DEFAULT (tau=0.50)"
    print("=" * 95)
    print(f"HAZARD HEAD ({tag} - {mode_label}) EVALUATION BREAKDOWN BY ATTACK TYPE (37 LOEO FOLDS)")
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
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--calibrated", action="store_true", help="Evaluate validation-calibrated test results")
    parser.add_argument("--compare", action="store_true", help="Compare v3 vs v4")
    args = parser.parse_args()

    v5_default = repo_root / "ml1" / "artifacts" / "lstm" / "hazard_head_v5"
    v4_default = repo_root / "ml1" / "artifacts" / "lstm" / "hazard_head_v4"
    v3_default = repo_root / "ml1" / "artifacts" / "lstm" / "hazard_head_v3"
    target_dir = Path(args.dir) if args.dir else (v5_default if v5_default.exists() else (v4_default if v4_default.exists() else v3_default))
    tag = args.tag or target_dir.name.split("_")[-1]

    csv_name = "hazard_head_all_horizons_loeo_val_calibrated.csv" if args.calibrated else "hazard_head_all_horizons_loeo.csv"
    evaluate_hazard_dir(target_dir, tag=tag, csv_name=csv_name)

if __name__ == "__main__":
    main()
