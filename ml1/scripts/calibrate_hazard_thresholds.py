"""
Calibrate and Sweep Hazard Head Thresholds Across All Attack Types (v4 Telemetry)
Supports H=1, H=2, H=5 for:
  - SSH-Bruteforce (14-02-2018, Folds 10-18)
  - DDOS-LOIC-UDP (21-02-2018, Folds 19-36)
  - Botnet (02-03-2018, Folds 0-9)
  - Blended Total (All 37 Folds)
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, average_precision_score

repo_root = Path(__file__).resolve().parent.parent.parent
ml1_dir = repo_root / "ml1"
sys.path.insert(0, str(ml1_dir))

from lstm.model import LSTMClassifier
from lstm.ucs import UCSConfig, apply_training_only_pca, validate_ucs_windows
from lstm.utils import get_device, set_seed

def get_attack_category(fold_id: int) -> str:
    if 10 <= fold_id <= 18:
        return "SSH-Bruteforce"
    elif 0 <= fold_id <= 9:
        return "Botnet"
    elif 19 <= fold_id <= 36:
        return "DDOS-LOIC-UDP"
    return "Other"

def run_calibration(models_dir: Path, out_dir: Path):
    set_seed(42)
    device = get_device()

    data_path = repo_root / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
    if not data_path.exists():
        data_path = ml1_dir / "data" / "ucs" / "ucs_windows.parquet"

    manifest_path = ml1_dir / "artifacts" / "loeo" / "corrected_37fold_manifest.json"
    
    windows = pd.read_parquet(data_path).sort_values("window_start_utc").reset_index(drop=True)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    config = UCSConfig()
    features = validate_ucs_windows(windows, config=config)

    windows["target_h1"] = windows.groupby("source_day")["label_binary"].shift(-1).fillna(0)
    windows["target_h2"] = windows.groupby("source_day")["label_binary"].shift(-2).fillna(0)
    windows["target_h5"] = windows["future_attack_label"]

    horizons = {
        1: "target_h1",
        2: "target_h2",
        5: "target_h5"
    }

    folds = manifest["folds"]
    attack_groups = {
        "SSH-Bruteforce": [f for f in folds if 10 <= f["fold_id"] <= 18],
        "DDOS-LOIC-UDP": [f for f in folds if 19 <= f["fold_id"] <= 36],
        "Botnet": [f for f in folds if 0 <= f["fold_id"] <= 9],
        "Blended Total": folds
    }

    # Threshold range: fine grid
    thresholds = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    all_sweep_records = []
    
    # Precompute raw predictions across all 37 folds for each horizon
    # fold_predictions[h_val][fold_id] = {'y_true': ..., 'probs': ..., 'roc': ..., 'pr': ...}
    fold_predictions = {1: {}, 2: {}, 5: {}}

    for h_val, target_col in horizons.items():
        print(f"\n[*] Precomputing predictions for Horizon H={h_val} across all 37 LOEO folds...")
        h_dir = models_dir / f"H{h_val}"
        
        for fold in folds:
            fold_id = fold["fold_id"]
            train_indices = np.array(fold["train_indices"])
            test_indices = np.array(fold["test_indices"])

            fold_frame = windows.copy()
            fold_frame["split"] = "none"
            fold_frame.loc[test_indices, "split"] = "test"

            train_df_sorted = fold_frame.loc[train_indices].sort_values("window_start_utc")
            val_start = int(len(train_df_sorted) * 0.8)
            fold_frame.loc[train_df_sorted.index[:val_start], "split"] = "train"
            fold_frame.loc[train_df_sorted.index[val_start:], "split"] = "val"

            purged = fold_frame[fold_frame["split"].isin(["train", "val", "test"])].copy()
            transformed, pca = apply_training_only_pca(purged, features, 32, random_state=42)
            pca_features = [f"pca_{i}" for i in range(32)]

            full_transformed = pca.transform(fold_frame)
            full_transformed_np = full_transformed[pca_features].to_numpy(dtype=np.float32)
            target_np = fold_frame[target_col].to_numpy(dtype=np.float32)

            X_test, y_test = [], []
            idx_map = {idx: pos for pos, idx in enumerate(fold_frame.index)}
            for idx in test_indices:
                t_pos = idx_map[idx]
                start = t_pos - config.lookback_windows + 1
                if start < 0 or t_pos >= len(fold_frame):
                    continue
                hist = full_transformed_np[start:t_pos+1]
                if len(hist) == config.lookback_windows:
                    X_test.append(hist)
                    y_test.append(target_np[t_pos])

            X_test = np.array(X_test, dtype=np.float32)
            y_test = np.array(y_test, dtype=np.float32)

            if len(X_test) == 0:
                continue

            model = LSTMClassifier(input_size=32, hidden_size=64, num_layers=1, dropout=0.2)
            model.to(device)
            ckpt = h_dir / f"model_fold_{fold_id}.pt"
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model.eval()

            with torch.no_grad():
                probs = model.predict_proba(torch.as_tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

            y_test_int = y_test.astype(int)
            roc = roc_auc_score(y_test_int, probs) if len(np.unique(y_test_int)) > 1 else np.nan
            pr = average_precision_score(y_test_int, probs) if len(np.unique(y_test_int)) > 1 else np.nan

            fold_predictions[h_val][fold_id] = {
                "fold_id": fold_id,
                "attack_type": get_attack_category(fold_id),
                "y_true": y_test_int,
                "probs": probs,
                "roc_auc": roc,
                "pr_auc": pr
            }

    # Now compute sweeps for each Attack Type x Horizon
    summary_optimal = []

    print("\n" + "=" * 105)
    print("FULL THRESHOLD SWEEP PER ATTACK TYPE & HORIZON")
    print("=" * 105)

    for group_name, group_folds in attack_groups.items():
        group_fold_ids = [f["fold_id"] for f in group_folds]
        
        for h_val in (1, 2, 5):
            eval_items = [fold_predictions[h_val][fid] for fid in group_fold_ids if fid in fold_predictions[h_val]]
            
            sweep_rows = []
            for thresh in thresholds:
                fold_metrics = []
                for item in eval_items:
                    y_true = item["y_true"]
                    probs = item["probs"]
                    preds = (probs >= thresh).astype(int)

                    f1 = f1_score(y_true, preds, zero_division=0)
                    prec = precision_score(y_true, preds, zero_division=0)
                    rec = recall_score(y_true, preds, zero_division=0)

                    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                    fold_metrics.append({
                        "f1": f1,
                        "precision": prec,
                        "recall": rec,
                        "fpr": fpr,
                        "tp": tp, "fp": fp, "tn": tn, "fn": fn
                    })

                f_df = pd.DataFrame(fold_metrics)
                mean_roc = np.nanmean([item["roc_auc"] for item in eval_items])
                mean_pr = np.nanmean([item["pr_auc"] for item in eval_items])

                row_data = {
                    "attack_group": group_name,
                    "horizon": h_val,
                    "fold_count": len(eval_items),
                    "threshold": thresh,
                    "f1_mean": f_df["f1"].mean(),
                    "precision_mean": f_df["precision"].mean(),
                    "recall_mean": f_df["recall"].mean(),
                    "fpr_mean": f_df["fpr"].mean(),
                    "roc_auc_mean": mean_roc,
                    "pr_auc_mean": mean_pr,
                    "total_tp": int(f_df["tp"].sum()),
                    "total_fp": int(f_df["fp"].sum()),
                    "total_fn": int(f_df["fn"].sum()),
                    "total_tn": int(f_df["tn"].sum())
                }
                sweep_rows.append(row_data)
                all_sweep_records.append(row_data)

            df_sweep = pd.DataFrame(sweep_rows)
            best_idx = df_sweep["f1_mean"].idxmax()
            best_row = df_sweep.loc[best_idx]
            
            # Row at default 0.50
            row_050 = df_sweep.loc[df_sweep["threshold"] == 0.50].iloc[0]

            summary_optimal.append({
                "group": group_name,
                "horizon": h_val,
                "opt_thresh": best_row["threshold"],
                "opt_f1": best_row["f1_mean"],
                "opt_prec": best_row["precision_mean"],
                "opt_rec": best_row["recall_mean"],
                "opt_fpr": best_row["fpr_mean"],
                "def_f1": row_050["f1_mean"],
                "def_prec": row_050["precision_mean"],
                "def_rec": row_050["recall_mean"],
                "def_fpr": row_050["fpr_mean"],
                "roc_auc": best_row["roc_auc_mean"],
                "pr_auc": best_row["pr_auc_mean"],
            })

            print(f"\n--- {group_name} | Horizon H={h_val} (Folds: {len(eval_items)}) ---")
            print(f"Optimal Threshold: tau={best_row['threshold']:.2f} => F1={best_row['f1_mean']:.4f}, Prec={best_row['precision_mean']:.4f}, Rec={best_row['recall_mean']:.4f}, FPR={best_row['fpr_mean']:.4f}")
            print(f"Default Threshold: tau=0.50 => F1={row_050['f1_mean']:.4f}, Prec={row_050['precision_mean']:.4f}, Rec={row_050['recall_mean']:.4f}, FPR={row_050['fpr_mean']:.4f}")
            print(f"{'Thresh':<7} | {'F1':<7} | {'Prec':<7} | {'Rec':<7} | {'FPR':<7} | {'ROC':<7} | {'PR':<7} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'TN':<4}")
            print("-" * 80)
            for _, r in df_sweep.iterrows():
                is_opt = "*" if r["threshold"] == best_row["threshold"] else (" " if r["threshold"] != 0.50 else "!")
                print(f"{r['threshold']:<6.2f}{is_opt} | {r['f1_mean']:<7.4f} | {r['precision_mean']:<7.4f} | {r['recall_mean']:<7.4f} | {r['fpr_mean']:<7.4f} | {r['roc_auc_mean']:<7.4f} | {r['pr_auc_mean']:<7.4f} | {r['total_tp']:<4} | {r['total_fp']:<4} | {r['total_fn']:<4} | {r['total_tn']:<4}")

    # Save full sweep results to CSV
    sweep_df = pd.DataFrame(all_sweep_records)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv_path = out_dir / "hazard_threshold_sweep_all_types.csv"
    sweep_df.to_csv(sweep_csv_path, index=False)
    print(f"\n[+] Saved full threshold sweep dataset to {sweep_csv_path}")

    # Summary table
    sum_df = pd.DataFrame(summary_optimal)
    print("\n" + "=" * 110)
    print("CALIBRATION SUMMARY: DEFAULT (tau=0.50) VS OPTIMAL PER SUBSET")
    print("=" * 110)
    print(f"{'Attack Group':<18} | {'H':<3} | {'Opt Thresh':<10} | {'Opt F1':<8} | {'Opt Prec':<8} | {'Opt Rec':<8} | {'Def F1':<8} | {'Def Prec':<8} | {'Def Rec':<8} | {'ROC-AUC':<8}")
    print("-" * 110)
    for _, r in sum_df.iterrows():
        print(f"{r['group']:<18} | H={r['horizon']} | tau={r['opt_thresh']:<6.2f} | {r['opt_f1']:<8.4f} | {r['opt_prec']:<8.4f} | {r['opt_rec']:<8.4f} | {r['def_f1']:<8.4f} | {r['def_prec']:<8.4f} | {r['def_rec']:<8.4f} | {r['roc_auc']:<8.4f}")

    return sweep_df, sum_df, fold_predictions

if __name__ == "__main__":
    v4_models = ml1_dir / "artifacts" / "lstm" / "hazard_head_v4"
    out_dir = ml1_dir / "artifacts" / "lstm" / "hazard_head_v4"
    run_calibration(v4_models, out_dir)
