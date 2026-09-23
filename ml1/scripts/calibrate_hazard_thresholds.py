"""
Calibrate Hazard Head Thresholds Exclusively on Validation Split (Leakage-Free Protocol)
Supports H=1, H=2, H=5 across:
  - SSH-Bruteforce (14-02-2018, Folds 10-18)
  - DDOS-LOIC-UDP (21-02-2018, Folds 19-36)
  - Botnet (02-03-2018, Folds 0-9)
  - Blended Total (All 37 Folds)

STRICT INTEGRITY ENFORCEMENT:
- Threshold sweeps run exclusively on the chronological validation split (X_val / y_val).
- Zero threshold tuning or threshold selection on held-out test folds (X_test / y_test).
- Group deployable thresholds selected via median of per-fold validation optima.
- Held-out test performance evaluated strictly using fixed, validation-derived thresholds.
"""
import os
import sys
import json
import argparse
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

    # Check if standardized PCA should be used (e.g. for v5 models)
    pca_file = repo_root / "models" / "pca_32.pkl"
    use_standardized_pca = False
    pca_scaler = None
    pca_obj = None
    if ("v5" in models_dir.name or "v5" in str(models_dir)) and pca_file.exists():
        import pickle
        with open(pca_file, "rb") as f:
            pca_data = pickle.load(f)
        pca_obj = pca_data["pca"]
        pca_scaler = pca_data.get("scaler")
        use_standardized_pca = True
        print(f"[*] Detected v5 models: using standardized PCA artifact from {pca_file} (scaler present: {pca_scaler is not None})")
    else:
        print("[*] Using per-fold training-only PCA fit (Option B multi-day cascade protocol)")

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

    # Threshold range: fine grid identical to original protocol
    thresholds = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    # Precompute raw validation and test predictions across all 37 folds for each horizon
    fold_predictions = {1: {}, 2: {}, 5: {}}

    for h_val, target_col in horizons.items():
        print(f"\n[*] Precomputing validation & test predictions for Horizon H={h_val} across all 37 LOEO folds...")
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
            val_indices = np.array(train_df_sorted.index[val_start:])
            train_sub_indices = np.array(train_df_sorted.index[:val_start])
            fold_frame.loc[train_sub_indices, "split"] = "train"
            fold_frame.loc[val_indices, "split"] = "val"

            pca_features = [f"pca_{i}" for i in range(32)]
            if use_standardized_pca:
                raw_vals = fold_frame[features].to_numpy(dtype=np.float64)
                scaled_vals = pca_scaler.transform(raw_vals) if pca_scaler is not None else raw_vals
                scaled_frame = pd.DataFrame(scaled_vals, columns=features, index=fold_frame.index)
                full_transformed = pca_obj.transform(scaled_frame)
                full_transformed_np = full_transformed[pca_features].to_numpy(dtype=np.float32)
            else:
                purged = fold_frame[fold_frame["split"].isin(["train", "val", "test"])].copy()
                transformed, pca = apply_training_only_pca(purged, features, 32, random_state=42)
                full_transformed = pca.transform(fold_frame)
                full_transformed_np = full_transformed[pca_features].to_numpy(dtype=np.float32)

            target_np = fold_frame[target_col].to_numpy(dtype=np.float32)

            idx_map = {idx: pos for pos, idx in enumerate(fold_frame.index)}
            def extract_seqs(indices_arr):
                X, y = [], []
                for idx in indices_arr:
                    t_pos = idx_map[idx]
                    start = t_pos - config.lookback_windows + 1
                    if start < 0 or t_pos >= len(fold_frame):
                        continue
                    hist = full_transformed_np[start:t_pos+1]
                    if len(hist) == config.lookback_windows:
                        X.append(hist)
                        y.append(target_np[t_pos])
                return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

            X_val, y_val = extract_seqs(val_indices)
            X_test, y_test = extract_seqs(test_indices)

            if len(X_val) == 0 or len(X_test) == 0:
                print(f"    [!] Skipping fold {fold_id}: missing val ({len(X_val)}) or test ({len(X_test)}) samples.")
                continue

            model = LSTMClassifier(input_size=32, hidden_size=64, num_layers=1, dropout=0.2)
            model.to(device)
            ckpt = h_dir / f"model_fold_{fold_id}.pt"
            if not ckpt.exists():
                print(f"    [-] Checkpoint not found: {ckpt}")
                continue
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model.eval()

            with torch.no_grad():
                probs_val = model.predict_proba(torch.as_tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()
                probs_test = model.predict_proba(torch.as_tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

            y_val_int = y_val.astype(int)
            y_test_int = y_test.astype(int)

            roc_val = roc_auc_score(y_val_int, probs_val) if len(np.unique(y_val_int)) > 1 else np.nan
            pr_val = average_precision_score(y_val_int, probs_val) if len(np.unique(y_val_int)) > 1 else np.nan

            roc_test = roc_auc_score(y_test_int, probs_test) if len(np.unique(y_test_int)) > 1 else np.nan
            pr_test = average_precision_score(y_test_int, probs_test) if len(np.unique(y_test_int)) > 1 else np.nan

            fold_predictions[h_val][fold_id] = {
                "fold_id": fold_id,
                "attack_type": get_attack_category(fold_id),
                "val": {
                    "y_true": y_val_int,
                    "probs": probs_val,
                    "roc_auc": roc_val,
                    "pr_auc": pr_val,
                },
                "test": {
                    "y_true": y_test_int,
                    "probs": probs_test,
                    "roc_auc": roc_test,
                    "pr_auc": pr_test,
                }
            }

    # Run validation-only calibration and unbiased test evaluation
    all_sweep_records = []
    summary_calibration = []
    deployed_group_thresholds = {}

    print("\n" + "=" * 115)
    print("LEAKAGE-FREE VALIDATION-ONLY THRESHOLD SWEEP & HELD-OUT TEST EVALUATION")
    print("=" * 115)

    for group_name, group_folds in attack_groups.items():
        group_fold_ids = [f["fold_id"] for f in group_folds]
        deployed_group_thresholds[group_name] = {}
        
        for h_val in (1, 2, 5):
            eval_items = [fold_predictions[h_val][fid] for fid in group_fold_ids if fid in fold_predictions[h_val]]
            if len(eval_items) == 0:
                continue

            # Step 1: Per-fold validation optimal threshold analysis
            fold_val_opt_taus = []
            for item in eval_items:
                f_y_val = item["val"]["y_true"]
                f_p_val = item["val"]["probs"]
                best_f_f1 = -1.0
                best_f_tau = 0.50
                for thresh in thresholds:
                    f_pred = (f_p_val >= thresh).astype(int)
                    sc = f1_score(f_y_val, f_pred, zero_division=0)
                    if sc > best_f_f1:
                        best_f_f1 = sc
                        best_f_tau = thresh
                    elif sc == best_f_f1 and abs(thresh - 0.50) < abs(best_f_tau - 0.50):
                        best_f_tau = thresh
                fold_val_opt_taus.append(best_f_tau)

            fold_taus_arr = np.array(fold_val_opt_taus)
            tau_median = float(np.median(fold_taus_arr))
            tau_mean = float(np.mean(fold_taus_arr))
            tau_min = float(np.min(fold_taus_arr))
            tau_max = float(np.max(fold_taus_arr))
            tau_iqr = float(np.percentile(fold_taus_arr, 75) - np.percentile(fold_taus_arr, 25))

            # Step 2: Group-level validation sweep (evaluate entire grid on validation data only)
            val_sweep_rows = []
            test_metrics_by_thresh = {}

            for thresh in thresholds:
                # Validation fold metrics
                v_metrics = []
                for item in eval_items:
                    y_v = item["val"]["y_true"]
                    p_v = item["val"]["probs"]
                    preds_v = (p_v >= thresh).astype(int)
                    f1_v = f1_score(y_v, preds_v, zero_division=0)
                    prec_v = precision_score(y_v, preds_v, zero_division=0)
                    rec_v = recall_score(y_v, preds_v, zero_division=0)
                    tn, fp, fn, tp = confusion_matrix(y_v, preds_v, labels=[0, 1]).ravel()
                    fpr_v = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    v_metrics.append({"f1": f1_v, "precision": prec_v, "recall": rec_v, "fpr": fpr_v, "tp": tp, "fp": fp, "tn": tn, "fn": fn})
                v_df = pd.DataFrame(v_metrics)

                # Held-out test fold metrics at this threshold
                t_metrics = []
                for item in eval_items:
                    y_t = item["test"]["y_true"]
                    p_t = item["test"]["probs"]
                    preds_t = (p_t >= thresh).astype(int)
                    f1_t = f1_score(y_t, preds_t, zero_division=0)
                    prec_t = precision_score(y_t, preds_t, zero_division=0)
                    rec_t = recall_score(y_t, preds_t, zero_division=0)
                    tn, fp, fn, tp = confusion_matrix(y_t, preds_t, labels=[0, 1]).ravel()
                    fpr_t = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    t_metrics.append({"f1": f1_t, "precision": prec_t, "recall": rec_t, "fpr": fpr_t, "tp": tp, "fp": fp, "tn": tn, "fn": fn})
                t_df = pd.DataFrame(t_metrics)

                mean_roc_val = np.nanmean([item["val"]["roc_auc"] for item in eval_items])
                mean_pr_val = np.nanmean([item["val"]["pr_auc"] for item in eval_items])
                mean_roc_test = np.nanmean([item["test"]["roc_auc"] for item in eval_items])
                mean_pr_test = np.nanmean([item["test"]["pr_auc"] for item in eval_items])

                row_data = {
                    "attack_group": group_name,
                    "horizon": h_val,
                    "fold_count": len(eval_items),
                    "threshold": thresh,
                    "val_f1_mean": v_df["f1"].mean(),
                    "val_precision_mean": v_df["precision"].mean(),
                    "val_recall_mean": v_df["recall"].mean(),
                    "val_fpr_mean": v_df["fpr"].mean(),
                    "val_roc_auc_mean": mean_roc_val,
                    "val_pr_auc_mean": mean_pr_val,
                    "test_f1_mean": t_df["f1"].mean(),
                    "test_precision_mean": t_df["precision"].mean(),
                    "test_recall_mean": t_df["recall"].mean(),
                    "test_fpr_mean": t_df["fpr"].mean(),
                    "test_roc_auc_mean": mean_roc_test,
                    "test_pr_auc_mean": mean_pr_test,
                    "test_tp_total": int(t_df["tp"].sum()),
                    "test_fp_total": int(t_df["fp"].sum()),
                    "test_fn_total": int(t_df["fn"].sum()),
                    "test_tn_total": int(t_df["tn"].sum()),
                }
                val_sweep_rows.append(row_data)
                all_sweep_records.append(row_data)
                test_metrics_by_thresh[thresh] = row_data

            df_sweep = pd.DataFrame(val_sweep_rows)
            # Pick best threshold solely from VALIDATION metrics
            best_idx = df_sweep["val_f1_mean"].idxmax()
            best_val_row = df_sweep.loc[best_idx]
            tau_val_opt = float(best_val_row["threshold"])

            # Deployable threshold choice: median across folds (with validation sweep best as cross-check)
            # If median is in grid, choose median, else nearest grid threshold
            nearest_median_idx = int(np.argmin([abs(t - tau_median) for t in thresholds]))
            tau_deployed = thresholds[nearest_median_idx]

            deployed_group_thresholds[group_name][h_val] = {
                "tau_deployed": tau_deployed,
                "tau_val_group_opt": tau_val_opt,
                "tau_median": tau_median,
                "tau_mean": tau_mean,
                "tau_spread": f"[{tau_min:.2f} .. {tau_max:.2f}] (IQR={tau_iqr:.2f})"
            }

            # Out-of-sample held-out test evaluation at the chosen validation-derived threshold
            test_row_deployed = test_metrics_by_thresh[tau_deployed]
            test_row_050 = test_metrics_by_thresh[0.50]

            summary_calibration.append({
                "group": group_name,
                "horizon": h_val,
                "fold_count": len(eval_items),
                "val_opt_tau": tau_val_opt,
                "median_tau": tau_median,
                "spread_str": f"[{tau_min:.2f}..{tau_max:.2f}] (IQR={tau_iqr:.2f})",
                "deployed_tau": tau_deployed,
                "val_f1": best_val_row["val_f1_mean"],
                "val_prec": best_val_row["val_precision_mean"],
                "val_rec": best_val_row["val_recall_mean"],
                "val_fpr": best_val_row["val_fpr_mean"],
                "test_f1": test_row_deployed["test_f1_mean"],
                "test_prec": test_row_deployed["test_precision_mean"],
                "test_rec": test_row_deployed["test_recall_mean"],
                "test_fpr": test_row_deployed["test_fpr_mean"],
                "test_f1_050": test_row_050["test_f1_mean"],
                "test_prec_050": test_row_050["test_precision_mean"],
                "test_rec_050": test_row_050["test_recall_mean"],
                "test_fpr_050": test_row_050["test_fpr_mean"],
                "test_roc": test_row_deployed["test_roc_auc_mean"],
                "test_pr": test_row_deployed["test_pr_auc_mean"],
            })

            print(f"\n--- {group_name} | Horizon H={h_val} (Folds: {len(eval_items)}) ---")
            print(f"Validation Per-Fold Tau Distribution: Median={tau_median:.2f}, Mean={tau_mean:.2f}, Range=[{tau_min:.2f}..{tau_max:.2f}], IQR={tau_iqr:.2f}")
            print(f"Group Val-Optimal Tau: tau={tau_val_opt:.2f} (Val F1={best_val_row['val_f1_mean']:.4f}, Val FPR={best_val_row['val_fpr_mean']:.4f})")
            print(f"Selected Deployable Tau: tau={tau_deployed:.2f} (Median across validation folds)")
            print(f" -> Real Held-Out Test Evaluation at tau={tau_deployed:.2f}: F1={test_row_deployed['test_f1_mean']:.4f}, Prec={test_row_deployed['test_precision_mean']:.4f}, Rec={test_row_deployed['test_recall_mean']:.4f}, FPR={test_row_deployed['test_fpr_mean']:.4f}")
            print(f" -> Real Held-Out Test Evaluation at tau=0.50 (Default) : F1={test_row_050['test_f1_mean']:.4f}, Prec={test_row_050['test_precision_mean']:.4f}, Rec={test_row_050['test_recall_mean']:.4f}, FPR={test_row_050['test_fpr_mean']:.4f}")

    # Generate per-fold honest test metrics at group-deployed tau for all 37 folds
    per_fold_calibrated_rows = []
    for h_val in (1, 2, 5):
        for fold in folds:
            fid = fold["fold_id"]
            if fid not in fold_predictions[h_val]:
                continue
            item = fold_predictions[h_val][fid]
            grp = item["attack_type"]
            tau_grp = deployed_group_thresholds.get(grp, {}).get(h_val, {}).get("tau_deployed", 0.50)
            
            y_t = item["test"]["y_true"]
            p_t = item["test"]["probs"]
            p_bin = (p_t >= tau_grp).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_t, p_bin, labels=[0, 1]).ravel()
            
            per_fold_calibrated_rows.append({
                "horizon": h_val,
                "fold_id": fid,
                "attack_type": grp,
                "calibrated_tau": tau_grp,
                "f1": f1_score(y_t, p_bin, zero_division=0),
                "precision": precision_score(y_t, p_bin, zero_division=0),
                "recall": recall_score(y_t, p_bin, zero_division=0),
                "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
                "roc_auc": item["test"]["roc_auc"],
                "pr_auc": item["test"]["pr_auc"],
                "tp": tp, "fp": fp, "tn": tn, "fn": fn
            })

    per_fold_df = pd.DataFrame(per_fold_calibrated_rows)
    per_fold_path = out_dir / "hazard_head_all_horizons_loeo_val_calibrated.csv"
    per_fold_df.to_csv(per_fold_path, index=False)
    print(f"\n[+] Saved per-fold calibrated evaluation dataset to: {per_fold_path}")

    # Save validation-only sweep results to versioned CSV (does not overwrite old CSV to preserve audit chain)
    sweep_df = pd.DataFrame(all_sweep_records)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv_path = out_dir / "hazard_threshold_sweep_validation_only.csv"
    sweep_df.to_csv(sweep_csv_path, index=False)
    print(f"\n[+] Saved validation-only threshold sweep artifact to: {sweep_csv_path}")

    # Summary table
    sum_df = pd.DataFrame(summary_calibration)
    print("\n" + "=" * 125)
    print("CALIBRATION SUMMARY: HONEST VALIDATION-DERIVED THRESHOLD VS DEFAULT 0.50 (HELD-OUT TEST EVALUATION)")
    print("=" * 125)
    print(f"{'Attack Group':<18} | {'H':<3} | {'Val Tau':<7} | {'Median Tau':<10} | {'Spread [Min..Max]':<20} | {'Test F1':<8} | {'Test Prec':<9} | {'Test Rec':<8} | {'Test FPR':<8} | {'Def0.5 F1':<9}")
    print("-" * 125)
    for _, r in sum_df.iterrows():
        print(f"{r['group']:<18} | H={r['horizon']} | {r['val_opt_tau']:<7.2f} | {r['median_tau']:<10.2f} | {r['spread_str']:<20} | {r['test_f1']:<8.4f} | {r['test_prec']:<9.4f} | {r['test_rec']:<8.4f} | {r['test_fpr']:<8.4f} | {r['test_f1_050']:<9.4f}")

    # Generate versioned markdown report side-by-side with original
    report_path = out_dir / "hazard_head_report_validation_calibrated.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Hazard Head Validation-Only Threshold Calibration Report (Leakage-Free Protocol)\n\n")
        f.write("## 1. Executive Summary & Calibration Protocol\n\n")
        f.write("In previous iterations (cf999c9), threshold sweeps were evaluated directly against held-out LOEO test folds (`X_test`/`y_test`), resulting in an overfitted operational threshold recommendation ($\\tau=0.45$ global, $\\tau=0.40$ for SSH/DDOS).\n\n")
        f.write("This report documents the **honest, leakage-free calibration protocol**:\n")
        f.write("1. Threshold sweeps are computed **strictly on the chronological validation split** ($X_{val} / y_{val}$) computed during LOEO fold preparation (`val_start = int(len(train_df_sorted) * 0.8)`).\n")
        f.write("2. Per-fold validation-optimal thresholds are identified, and the deployable threshold per group is selected via the **median across folds** (consistent with the Botnet detection/onset calibration in `9fae9ab`).\n")
        f.write("3. Held-out test performance is reported strictly out-of-sample on $X_{test} / y_{test}$ using the fixed, validation-derived threshold.\n\n")
        f.write("## 2. Validation-Derived Thresholds & Held-Out Test Evaluation Matrix\n\n")
        f.write("| Attack Group | Horizon | Validation Optimal $\\tau$ | Per-Fold Median $\\tau$ | Per-Fold Spread [Min..Max] | Deployed $\\tau$ | Held-Out Test F1 | Test Precision | Test Recall | Test FPR | Test ROC-AUC | Default $\\tau=0.50$ Test F1 |\n")
        f.write("|---|:---:|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for _, r in sum_df.iterrows():
            f.write(f"| **{r['group']}** | **H={r['horizon']}** | {r['val_opt_tau']:.2f} | {r['median_tau']:.2f} | `{r['spread_str']}` | **{r['deployed_tau']:.2f}** | **{r['test_f1']:.4f}** | {r['test_prec']:.4f} | {r['test_rec']:.4f} | {r['test_fpr']:.4f} | {r['test_roc']:.4f} | {r['test_f1_050']:.4f} |\n")
        f.write("\n---\n\n")
        f.write("## 3. Recommended Deployed Thresholds for Backend Inference\n\n")
        f.write("Summarizing across horizons using the median validation threshold:\n\n")
        
        group_medians = {}
        for g in attack_groups.keys():
            g_rows = sum_df[sum_df["group"] == g]
            g_med = float(np.median(g_rows["deployed_tau"]))
            group_medians[g] = g_med
            f.write(f"- **{g}**: $\\tau = {g_med:.2f}$ (Median across validated horizons H=1, H=2, H=5)\n")

    print(f"\n[+] Saved comprehensive calibration report to: {report_path}")

    return sweep_df, sum_df, deployed_group_thresholds

def main():
    parser = argparse.ArgumentParser(description="Calibrate hazard head thresholds on validation data.")
    parser.add_argument("--models-dir", type=str, default=None, help="Directory containing hazard head fold models")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for reports and artifacts")
    args = parser.parse_args()

    v5_dir = ml1_dir / "artifacts" / "lstm" / "hazard_head_v5"
    v4_dir = ml1_dir / "artifacts" / "lstm" / "hazard_head_v4"
    
    if args.models_dir:
        models_dir = Path(args.models_dir)
    else:
        # Default to hazard_head_v4 as requested in prompt to resolve the cf999c9 artifact, or hazard_head_v5 if selected
        models_dir = v4_dir

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = models_dir

    print(f"[*] Running calibration on models in: {models_dir}")
    print(f"[*] Output directory: {out_dir}")
    run_calibration(models_dir, out_dir)

if __name__ == "__main__":
    main()
