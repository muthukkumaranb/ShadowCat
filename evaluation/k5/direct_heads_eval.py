import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score
import joblib

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "evaluation" / "k5"
DATA_PATH = REPO / "data-engineering/data/ucs/ucs_windows_models_v1.parquet"
MANIFEST_PATH = REPO / "ml1/artifacts/loeo/corrected_37fold_manifest.json"
FEATURES_PATH = REPO / "ml1/artifacts/lr_set_a/set_a_features.json"

def compute_metrics_at_5pct_fpr(y_true, y_score, target_fpr=0.05):
    if len(np.unique(y_true)) < 2: return {"roc_auc": None, "pr_auc": None, "threshold_5pct_fpr": None, "recall_5pct_fpr": None, "precision_5pct_fpr": None, "fpr_actual": None}
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    thresholds = np.sort(np.unique(y_score))[::-1]
    best_thresh = float(thresholds[0])
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        fp = int(((preds == 1) & (y_true == 0)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        if fpr <= target_fpr:
            best_thresh = float(t)
        else:
            break
    preds = (y_score >= best_thresh).astype(int)
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fpr_actual = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "threshold_5pct_fpr": round(best_thresh, 4),
        "recall_5pct_fpr": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "precision_5pct_fpr": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "fpr_actual": round(float(fpr_actual), 4),
    }

def run():
    df = pd.read_parquet(DATA_PATH).sort_values("window_start_utc").reset_index(drop=True)
    with open(FEATURES_PATH) as f: features = json.load(f)["ordered_feature_names"]
    with open(MANIFEST_PATH) as f: folds = json.load(f)["folds"]
    
    rollout_cache_path = OUT_DIR / "rollout_predictions_cache.joblib"
    if rollout_cache_path.exists():
        rollout_cache = joblib.load(rollout_cache_path)
    else:
        raise FileNotFoundError(f"Missing {rollout_cache_path}")

    times = pd.to_datetime(df["window_start_utc"], utc=True)
    time_diffs = times.diff().dt.total_seconds().values
    is_step_60 = (time_diffs == 60.0).astype(int)
    cumsum = np.cumsum(np.insert(is_step_60, 0, 0))
    N = len(df)
    labels_bin = df["label_binary"].to_numpy(dtype=int)

    targets_per_k = {}
    valid_mask_per_k = {}
    for k in range(1, 6):
        y_k = np.zeros(N, dtype=int)
        valid_k = np.zeros(N, dtype=bool)
        for t in range(N - k):
            if (cumsum[t + k] - cumsum[t + 1]) == (k - 1):
                y_k[t] = int(np.max(labels_bin[t + 1 : t + k + 1]) == 1)
                valid_k[t] = True
        targets_per_k[k] = y_k
        valid_mask_per_k[k] = valid_k

    X_matrix = df[features].to_numpy(dtype=np.float64)
    horizon_results = {}

    for k in range(1, 6):
        y_k_all = targets_per_k[k]
        valid_k_all = valid_mask_per_k[k]
        direct_preds = []
        direct_targets = []
        pers_preds = []
        rollout_preds = []
        labels_t = []

        k_cache = rollout_cache[rollout_cache["k"] == k].set_index(["fold_id", "window_index"])

        for fold in folds:
            fold_id = fold["fold_id"]
            tr_idx = np.array(fold["train_indices"])
            te_idx = np.array(fold["test_indices"])
            tr_valid = [t for t in tr_idx if valid_k_all[t]]
            te_valid = [t for t in te_idx if valid_k_all[t]]
            
            valid_te_in_cache = []
            for t in te_valid:
                if (fold_id, t) in k_cache.index:
                    valid_te_in_cache.append(t)
            
            if len(valid_te_in_cache) == 0: continue
            te_valid = valid_te_in_cache

            X_tr, y_tr = X_matrix[tr_valid], y_k_all[tr_valid]
            X_te, y_te = X_matrix[te_valid], y_k_all[te_valid]
            
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)
            
            clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            clf.fit(X_tr_sc, y_tr)
            
            if len(clf.classes_) == 1:
                p_te = np.zeros(len(X_te)) if clf.classes_[0] == 0 else np.ones(len(X_te))
            else:
                p_te = clf.predict_proba(X_te_sc)[:, list(clf.classes_).index(1)]
                
            direct_preds.extend(p_te.tolist())
            direct_targets.extend(y_te.tolist())
            pers_preds.extend(labels_bin[te_valid].tolist())
            labels_t.extend(labels_bin[te_valid].tolist())

            valid_tr_base = []
            for t in tr_idx:
                if t < len(labels_bin) - 1:
                    valid_tr_base.append(t)
            y_tr_base = labels_bin[np.array(valid_tr_base) + 1]
            X_tr_base = X_matrix[valid_tr_base]
            
            base_scaler = StandardScaler()
            X_tr_base_sc = base_scaler.fit_transform(X_tr_base)
            
            base_clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            base_clf.fit(X_tr_base_sc, y_tr_base)
            
            rolled_states = np.stack(k_cache.loc[[(fold_id, t) for t in te_valid]]["pred_lstm_state"].values)
            rolled_states_sc = base_scaler.transform(rolled_states)
            
            if len(base_clf.classes_) == 1:
                p_rollout = np.zeros(len(rolled_states)) if base_clf.classes_[0] == 0 else np.ones(len(rolled_states))
            else:
                p_rollout = base_clf.predict_proba(rolled_states_sc)[:, list(base_clf.classes_).index(1)]
            
            rollout_preds.extend(p_rollout.tolist())


        y_true_arr = np.array(direct_targets, dtype=int)
        p_direct_arr = np.array(direct_preds, dtype=float)
        p_rollout_arr = np.array(rollout_preds, dtype=float)
        p_pers_arr = np.array(pers_preds, dtype=float)
        p_inv_pers_arr = 1.0 - p_pers_arr
        label_t_arr = np.array(labels_t, dtype=int)

        direct_metrics = compute_metrics_at_5pct_fpr(y_true_arr, p_direct_arr)
        rollout_metrics = compute_metrics_at_5pct_fpr(y_true_arr, p_rollout_arr)
        pers_metrics = compute_metrics_at_5pct_fpr(y_true_arr, p_pers_arr)
        inv_pers_metrics = compute_metrics_at_5pct_fpr(y_true_arr, p_inv_pers_arr)
        
        precursor_mask = (label_t_arr == 0)
        y_true_prec = y_true_arr[precursor_mask]
        p_direct_prec = p_direct_arr[precursor_mask]
        p_rollout_prec = p_rollout_arr[precursor_mask]
        p_pers_prec = p_pers_arr[precursor_mask]
        p_inv_pers_prec = p_inv_pers_arr[precursor_mask]
        
        direct_prec_metrics = compute_metrics_at_5pct_fpr(y_true_prec, p_direct_prec)
        rollout_prec_metrics = compute_metrics_at_5pct_fpr(y_true_prec, p_rollout_prec)
        pers_prec_metrics = compute_metrics_at_5pct_fpr(y_true_prec, p_pers_prec)
        inv_pers_prec_metrics = compute_metrics_at_5pct_fpr(y_true_prec, p_inv_pers_prec)

        onset_vs_benign_mask = ((label_t_arr == 0) & (y_true_arr == 1)) | ((label_t_arr == 0) & (y_true_arr == 0))
        y_true_ob = y_true_arr[onset_vs_benign_mask]
        if len(y_true_ob) > 0 and len(np.unique(y_true_ob)) > 1:
            direct_ob = compute_metrics_at_5pct_fpr(y_true_ob, p_direct_arr[onset_vs_benign_mask])
            rollout_ob = compute_metrics_at_5pct_fpr(y_true_ob, p_rollout_arr[onset_vs_benign_mask])
            pers_ob = compute_metrics_at_5pct_fpr(y_true_ob, p_pers_arr[onset_vs_benign_mask])
            inv_pers_ob = compute_metrics_at_5pct_fpr(y_true_ob, p_inv_pers_arr[onset_vs_benign_mask])
        else:
            direct_ob = {"roc_auc": None, "pr_auc": None, "recall_5pct_fpr": None}
            rollout_ob = {"roc_auc": None, "pr_auc": None, "recall_5pct_fpr": None}
            pers_ob = {"roc_auc": None, "pr_auc": None, "recall_5pct_fpr": None}
            inv_pers_ob = {"roc_auc": None, "pr_auc": None, "recall_5pct_fpr": None}
        
        already_pos = int(np.sum((y_true_arr == 1) & (label_t_arr == 1)))

        horizon_results[str(k)] = {
            "n_test_windows": len(y_true_arr),
            "positives": int(y_true_arr.sum()),
            "negatives": int(len(y_true_arr) - y_true_arr.sum()),
            "already_in_attack": already_pos,
            "direct_lr_head": direct_metrics,
            "rollout_onset_probability": rollout_metrics,
            "persistence": pers_metrics,
            "inverse_persistence": inv_pers_metrics,
            "precursor_only": {
                "n_test_windows": int(precursor_mask.sum()),
                "positives": int(y_true_prec.sum()),
                "direct_lr_head": direct_prec_metrics,
                "rollout_onset_probability": rollout_prec_metrics,
                "persistence": pers_prec_metrics,
                "inverse_persistence": inv_pers_prec_metrics
            },
            "onset_vs_benign": {
                "n_test_windows": int(onset_vs_benign_mask.sum()),
                "positives": int(y_true_ob.sum()),
                "direct_lr_head": direct_ob,
                "rollout_onset_probability": rollout_ob,
                "persistence": pers_ob,
                "inverse_persistence": inv_pers_ob
            }
        }
    final_output = {
        "generated_by": "evaluation/k5/direct_heads_eval.py",
        "dataset": "data-engineering/data/ucs/ucs_windows_models_v1.parquet",
        "manifest": "ml1/artifacts/loeo/corrected_37fold_manifest.json",
        "features": "ml1/artifacts/lr_set_a/set_a_features.json",
        "horizons": horizon_results,
    }
    with open(OUT_DIR / "direct_heads_results.json", "w") as f: json.dump(final_output, f, indent=2)

    md_lines = [
        "# Direct Multi-Horizon Onset Heads vs Rollout-Based Probability", "",
        f"Generated by `evaluation/k5/direct_heads_eval.py` on under 37-fold LOEO.", "",
        "## Comparison Table", "",
        "| Horizon K | Pos/Total | Already Pos | Direct AUC | Rollout AUC | Pers AUC | Inv-Pers AUC | Direct (Onset/Benign) AUC | Rollout (Onset/Benign) AUC | Pers (Onset/Benign) AUC | Inv-Pers (Onset/Benign) AUC |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in range(1, 6):
        res = horizon_results[str(k)]
        pos, tot, already = res["positives"], res["n_test_windows"], res["already_in_attack"]
        d_roc = f"{res['direct_lr_head']['roc_auc']:.4f}" if res['direct_lr_head']['roc_auc'] else "N/A"
        r_roc = f"{res['rollout_onset_probability']['roc_auc']:.4f}" if res['rollout_onset_probability']['roc_auc'] else "N/A"
        p_roc = f"{res['persistence']['roc_auc']:.4f}" if res['persistence']['roc_auc'] else "N/A"
        ip_roc = f"{res['inverse_persistence']['roc_auc']:.4f}" if res['inverse_persistence']['roc_auc'] else "N/A"
        dob_roc = f"{res['onset_vs_benign']['direct_lr_head']['roc_auc']:.4f}" if res['onset_vs_benign']['direct_lr_head']['roc_auc'] else "N/A"
        rob_roc = f"{res['onset_vs_benign']['rollout_onset_probability']['roc_auc']:.4f}" if res['onset_vs_benign']['rollout_onset_probability']['roc_auc'] else "N/A"
        pob_roc = f"{res['onset_vs_benign']['persistence']['roc_auc']:.4f}" if res['onset_vs_benign']['persistence']['roc_auc'] else "N/A"
        ipob_roc = f"{res['onset_vs_benign']['inverse_persistence']['roc_auc']:.4f}" if res['onset_vs_benign']['inverse_persistence']['roc_auc'] else "N/A"
        dp_roc = f"{res['precursor_only']['direct_lr_head']['roc_auc']:.4f}" if res['precursor_only']['direct_lr_head']['roc_auc'] else "N/A"
        rp_roc = f"{res['precursor_only']['rollout_onset_probability']['roc_auc']:.4f}" if res['precursor_only']['rollout_onset_probability']['roc_auc'] else "N/A"
        pp_roc = f"{res['precursor_only']['persistence']['roc_auc']:.4f}" if res['precursor_only']['persistence']['roc_auc'] else "N/A"
        md_lines.append(f"| K={k} | {pos}/{tot} | {already} | {d_roc} | {r_roc} | {p_roc} | {ip_roc} | {dob_roc} | {rob_roc} | {pob_roc} | {ipob_roc} |")

    with open(OUT_DIR / "DIRECT_HEADS.md", "w") as f: f.write("\n".join(md_lines) + "\n")

if __name__ == '__main__':
    run()
