import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path("ml1/artifacts/lstm/hazard_head_v4/hazard_threshold_sweep_all_types.csv")
df = pd.read_csv(csv_path)

print("=" * 115)
print("COMPREHENSIVE AUDIT: UNCONSTRAINED VS FPR <= 10% VS FPR <= 5% VS DEFAULT tau=0.50")
print("=" * 115)

summary_rows = []

for h in (1, 2, 5):
    sub_h = df[df["horizon"] == h]
    for grp in ("SSH-Bruteforce", "DDOS-LOIC-UDP", "Botnet", "Blended Total"):
        sub_g = sub_h[sub_h["attack_group"] == grp]
        
        # 1. Unconstrained Max F1
        best_uncon = sub_g.loc[sub_g["f1_mean"].idxmax()]
        
        # 2. FPR <= 10%
        low_fpr = sub_g[sub_g["fpr_mean"] <= 0.105]
        best_10 = low_fpr.loc[low_fpr["f1_mean"].idxmax()] if len(low_fpr) > 0 else None
        
        # 3. FPR <= 5%
        vlow_fpr = sub_g[sub_g["fpr_mean"] <= 0.055]
        best_5 = vlow_fpr.loc[vlow_fpr["f1_mean"].idxmax()] if len(vlow_fpr) > 0 else None
        
        # 4. Default tau = 0.50
        def_row = sub_g[sub_g["threshold"] == 0.50].iloc[0]
        
        # 5. tau = 0.40
        row_040 = sub_g[sub_g["threshold"] == 0.40].iloc[0] if len(sub_g[sub_g["threshold"] == 0.40]) > 0 else None
        # 6. tau = 0.45
        row_045 = sub_g[sub_g["threshold"] == 0.45].iloc[0] if len(sub_g[sub_g["threshold"] == 0.45]) > 0 else None

        summary_rows.append({
            "horizon": h,
            "group": grp,
            "uncon_tau": best_uncon["threshold"],
            "uncon_f1": best_uncon["f1_mean"],
            "uncon_rec": best_uncon["recall_mean"],
            "uncon_fpr": best_uncon["fpr_mean"],
            "c10_tau": best_10["threshold"] if best_10 is not None else np.nan,
            "c10_f1": best_10["f1_mean"] if best_10 is not None else np.nan,
            "c10_rec": best_10["recall_mean"] if best_10 is not None else np.nan,
            "c10_fpr": best_10["fpr_mean"] if best_10 is not None else np.nan,
            "def_f1": def_row["f1_mean"],
            "def_rec": def_row["recall_mean"],
            "def_fpr": def_row["fpr_mean"],
            "roc_auc": best_uncon["roc_auc_mean"]
        })

sum_df = pd.DataFrame(summary_rows)

for h in (1, 2, 5):
    print(f"\n--- Horizon H={h} ---")
    h_sub = sum_df[sum_df["horizon"] == h]
    print(f"{'Group':<16} | {'Uncon (Max F1)':<24} | {'FPR <= 10% Ceiling':<25} | {'Default (tau=0.50)':<22} | {'ROC-AUC':<8}")
    print(f"{'':<16} | {'tau':<4} {'F1':<7} {'Rec':<5} {'FPR':<5} | {'tau':<4} {'F1':<7} {'Rec':<5} {'FPR':<5} | {'F1':<7} {'Rec':<5} {'FPR':<5} |")
    print("-" * 115)
    for _, r in h_sub.iterrows():
        uncon_str = f"{r['uncon_tau']:<4.2f} {r['uncon_f1']:<7.4f} {r['uncon_rec']:<5.2f} {r['uncon_fpr']:<5.2f}"
        c10_str = f"{r['c10_tau']:<4.2f} {r['c10_f1']:<7.4f} {r['c10_rec']:<5.2f} {r['c10_fpr']:<5.2f}" if pd.notna(r['c10_tau']) else "N/A (FPR > 10% everywhere)"
        def_str = f"{r['def_f1']:<7.4f} {r['def_rec']:<5.2f} {r['def_fpr']:<5.2f}"
        print(f"{r['group']:<16} | {uncon_str:<24} | {c10_str:<25} | {def_str:<22} | {r['roc_auc']:<8.4f}")
