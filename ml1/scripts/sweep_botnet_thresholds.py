"""
Threshold Sweep for Botnet-Only LOEO Folds (Folds 0-9) across Horizons H=1, H=2, H=5
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

def main():
    set_seed(42)
    device = get_device()
    
    data_path = repo_root / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
    if not data_path.exists():
        data_path = ml1_dir / "data" / "ucs" / "ucs_windows.parquet"
        
    manifest_path = ml1_dir / "artifacts" / "loeo" / "corrected_37fold_manifest.json"
    models_dir = ml1_dir / "artifacts" / "lstm" / "hazard_head_v3"
    
    windows = pd.read_parquet(data_path).sort_values("window_start_utc").reset_index(drop=True)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    config = UCSConfig()
    features = validate_ucs_windows(windows, config=config)
    
    # Create Hazard targets
    windows["target_h1"] = windows.groupby("source_day")["label_binary"].shift(-1).fillna(0)
    windows["target_h2"] = windows.groupby("source_day")["label_binary"].shift(-2).fillna(0)
    windows["target_h5"] = windows["future_attack_label"]
    
    horizons = {
        1: "target_h1",
        2: "target_h2",
        5: "target_h5"
    }
    
    # Botnet folds are fold_id 0 to 9
    botnet_folds = [f for f in manifest["folds"] if 0 <= f["fold_id"] <= 9]
    print(f"Loaded {len(botnet_folds)} Botnet folds (0-9).")
    
    thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]
    
    sweep_results = {}
    
    for h_val, target_col in horizons.items():
        print(f"\nEvaluating Horizon H={h_val} across {len(botnet_folds)} Botnet folds...")
        h_dir = models_dir / f"H{h_val}"
        
        # Store per-fold ground truth and probs
        fold_eval_data = []
        
        for fold in botnet_folds:
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
            
            # Extract test sequences
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
            
            fold_eval_data.append({
                "fold_id": fold_id,
                "y_true": y_test_int,
                "probs": probs,
                "roc_auc": roc,
                "pr_auc": pr
            })
            
        # Perform threshold sweep across all folds
        h_table = []
        for thresh in thresholds:
            fold_metrics = []
            for item in fold_eval_data:
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
            mean_roc = np.nanmean([item["roc_auc"] for item in fold_eval_data])
            mean_pr = np.nanmean([item["pr_auc"] for item in fold_eval_data])
            
            h_table.append({
                "threshold": thresh,
                "f1_mean": f_df["f1"].mean(),
                "precision_mean": f_df["precision"].mean(),
                "recall_mean": f_df["recall"].mean(),
                "fpr_mean": f_df["fpr"].mean(),
                "roc_auc_mean": mean_roc,
                "pr_auc_mean": mean_pr,
                "total_tp": f_df["tp"].sum(),
                "total_fp": f_df["fp"].sum(),
                "total_fn": f_df["fn"].sum(),
                "total_tn": f_df["tn"].sum()
            })
            
        sweep_results[h_val] = pd.DataFrame(h_table)
        
    print("\n" + "=" * 90)
    print("BOTNET (02-03-2018) THRESHOLD SWEEP RESULTS ACROSS 10 FOLDS (0-9)")
    print("=" * 90)
    
    for h_val in (1, 2, 5):
        df_res = sweep_results[h_val]
        best_row = df_res.loc[df_res["f1_mean"].idxmax()]
        print(f"\n### Horizon H={h_val} (Best Threshold = {best_row['threshold']:.2f} -> F1={best_row['f1_mean']:.4f}, Prec={best_row['precision_mean']:.4f}, Rec={best_row['recall_mean']:.4f})")
        print(f"{'Threshold':<10} | {'F1 Mean':<8} | {'Precision':<10} | {'Recall':<8} | {'FPR Mean':<9} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'TP':<4} | {'FP':<5}")
        print("-" * 90)
        for _, r in df_res.iterrows():
            print(f"{r['threshold']:<10.2f} | {r['f1_mean']:<8.4f} | {r['precision_mean']:<10.4f} | {r['recall_mean']:<8.4f} | {r['fpr_mean']:<9.4f} | {r['roc_auc_mean']:<8.4f} | {r['pr_auc_mean']:<8.4f} | {int(r['total_tp']):<4} | {int(r['total_fp']):<5}")
            
    # Also save to CSV
    out_csv = models_dir / "botnet_threshold_sweep.csv"
    all_sw = []
    for h_val, df_res in sweep_results.items():
        df_c = df_res.copy()
        df_c["horizon"] = h_val
        all_sw.append(df_c)
    pd.concat(all_sw, ignore_index=True).to_csv(out_csv, index=False)
    print(f"\nSaved threshold sweep results to {out_csv}")

if __name__ == "__main__":
    main()
