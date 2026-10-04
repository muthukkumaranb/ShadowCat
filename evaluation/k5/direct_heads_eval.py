"""
Phase C-t3: Direct Multi-Horizon Onset Heads vs Rollout-Based Onset Probability
Trains one direct Logistic Regression head per horizon K=1..5 on set_a_features:
- Target: an attack starts within (t, t+K], i.e. max_{j=1..K} label_binary(t+j) == 1
- Protocol: 37 LOEO folds, standard scaler fitted per training fold only
- Evaluates on test windows: ROC-AUC, PR-AUC, and Recall at matched 5% FPR
- Compares each head with the rollout-based probability at the same K

Outputs:
- evaluation/k5/direct_heads_results.json
- evaluation/k5/DIRECT_HEADS.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "ml1") not in sys.path:
    sys.path.insert(0, str(REPO / "ml1"))

from lstm.ucs import UCSConfig, validate_ucs_windows  # noqa: E402

OUT_DIR = REPO / "evaluation" / "k5"
SET_A_FEATURES_PATH = REPO / "ml1" / "artifacts" / "lr_set_a" / "set_a_features.json"


def compute_metrics_at_5pct_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.05) -> Dict[str, Any]:
    """Compute threshold, recall, precision, ROC-AUC, and PR-AUC at matched 5% FPR."""
    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "threshold_5pct_fpr": None,
            "recall_5pct_fpr": None,
            "precision_5pct_fpr": None,
            "fpr_actual": None,
            "note": "Only one class present in y_true",
        }

    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))

    thresholds = np.unique(y_score)
    thresholds = np.sort(thresholds)[::-1]

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


def run_direct_heads_eval(
    data_path: str = "data-engineering/data/ucs/ucs_windows_models_v1.parquet",
    manifest_path: str = "ml1/artifacts/loeo/corrected_37fold_manifest.json",
    cache_path: str = "evaluation/k5/rollout_predictions_cache.joblib",
    change_skill_json: str = "evaluation/k5/change_skill_results.json",
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(REPO / data_path).sort_values("window_start_utc").reset_index(drop=True)

    with open(SET_A_FEATURES_PATH, "r", encoding="utf-8") as f:
        set_a_meta = json.load(f)
    features = set_a_meta["ordered_feature_names"]
    print(f"Loaded {len(features)} set_a features")

    with open(REPO / manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    folds = manifest["folds"]

    print(f"Loading rollout predictions cache from {cache_path}...")
    records_df = joblib.load(REPO / cache_path)

    # Precompute multi-horizon targets for each window t and horizon K=1..5
    # Target y_K(t) = 1 if max_{j=1..K} label_binary(t+j) == 1, else 0
    # Within valid bounds
    times = pd.to_datetime(df["window_start_utc"], utc=True)
    time_diffs = times.diff().dt.total_seconds().values
    is_step_60 = (time_diffs == 60.0).astype(int)
    cumsum = np.cumsum(np.insert(is_step_60, 0, 0))

    N = len(df)
    labels_bin = df["label_binary"].to_numpy(dtype=int)

    targets_per_k: Dict[int, np.ndarray] = {}
    valid_mask_per_k: Dict[int, np.ndarray] = {}

    for k in range(1, 6):
        y_k = np.zeros(N, dtype=int)
        valid_k = np.zeros(N, dtype=bool)
        for t in range(N - k):
            # Check contiguity [t : t+k]
            if (cumsum[t + k] - cumsum[t + 1]) == (k - 1):
                y_k[t] = int(np.max(labels_bin[t + 1 : t + k + 1]) == 1)
                valid_k[t] = True
        targets_per_k[k] = y_k
        valid_mask_per_k[k] = valid_k

    X_matrix = df[features].to_numpy(dtype=np.float64)

    # For each K, train direct LR head on each fold and evaluate on test windows
    horizon_results = {}

    # Also load rollout-based predictions per window
    # From change_skill_eval or compute directly from rollout cache
    from evaluation.k5.change_skill_eval import OnsetFoldModel, STACKED_ONSET_DIR
    onset_models = {}
    for f_id in range(37):
        ckpt_p = STACKED_ONSET_DIR / f"model_fold_{f_id}.pt"
        if ckpt_p.exists():
            onset_models[f_id] = OnsetFoldModel(ckpt_p, torch.device("cpu"))

    config = UCSConfig()
    raw_values = df[features].to_numpy(dtype=np.float32)

    for k in range(1, 6):
        print(f"\n--- Training Direct LR Head for Horizon K = {k} (+{k}m) ---")
        y_k_all = targets_per_k[k]
        valid_k_all = valid_mask_per_k[k]

        direct_preds = []
        direct_targets = []
        rollout_preds = []

        # Filter records for horizon k
        k_records = records_df[records_df["k"] == k]

        for fold in folds:
            f_id = fold["fold_id"]
            tr_idx = np.array(fold["train_indices"])
            te_idx = np.array(fold["test_indices"])

            # Filter training rows to valid contiguous rows
            tr_valid = [t for t in tr_idx if valid_k_all[t]]
            te_valid = [t for t in te_idx if valid_k_all[t] and t in k_records[k_records["fold_id"] == f_id]["window_index"].values]

            if len(te_valid) == 0:
                continue

            X_tr = X_matrix[tr_valid]
            y_tr = y_k_all[tr_valid]

            X_te = X_matrix[te_valid]
            y_te = y_k_all[te_valid]

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            clf.fit(X_tr_sc, y_tr)

            # Predict probability of class 1
            if len(clf.classes_) == 1:
                p_te = np.zeros(len(X_te)) if clf.classes_[0] == 0 else np.ones(len(X_te))
            else:
                c1_idx = list(clf.classes_).index(1)
                p_te = clf.predict_proba(X_te_sc)[:, c1_idx]

            direct_preds.extend(p_te.tolist())
            direct_targets.extend(y_te.tolist())

            # Compute rollout predictions for these exact test windows
            onset_m = onset_models[f_id]
            L = config.lookback_windows
            for t_idx in te_valid:
                win_recs = records_df[(records_df["fold_id"] == f_id) & (records_df["window_index"] == t_idx)]
                rolled_states_k = [win_recs[win_recs["k"] == step]["pred_lstm_state"].iloc[0] for step in range(1, k + 1)]
                rolled_states_arr = np.stack(rolled_states_k, axis=0)
                seq_30x406 = np.concatenate([raw_values[t_idx - L + k : t_idx], rolled_states_arr], axis=0)
                p_roll = onset_m.predict_proba(seq_30x406)
                rollout_preds.append(p_roll)

        y_true_arr = np.array(direct_targets, dtype=int)
        p_direct_arr = np.array(direct_preds, dtype=float)
        p_rollout_arr = np.array(rollout_preds, dtype=float)

        print(f"K={k}: {len(y_true_arr)} test windows (positives: {int(y_true_arr.sum())})")

        direct_metrics = compute_metrics_at_5pct_fpr(y_true_arr, p_direct_arr)
        rollout_metrics = compute_metrics_at_5pct_fpr(y_true_arr, p_rollout_arr)

        horizon_results[str(k)] = {
            "n_test_windows": len(y_true_arr),
            "positives": int(y_true_arr.sum()),
            "negatives": int(len(y_true_arr) - y_true_arr.sum()),
            "direct_lr_head": direct_metrics,
            "rollout_onset_probability": rollout_metrics,
            "comparison": {
                "roc_auc_diff_direct_minus_rollout": round(direct_metrics["roc_auc"] - rollout_metrics["roc_auc"], 4)
                if (direct_metrics["roc_auc"] and rollout_metrics["roc_auc"]) else None,
                "pr_auc_diff_direct_minus_rollout": round(direct_metrics["pr_auc"] - rollout_metrics["pr_auc"], 4)
                if (direct_metrics["pr_auc"] and rollout_metrics["pr_auc"]) else None,
                "recall_5pct_fpr_diff": round(direct_metrics["recall_5pct_fpr"] - rollout_metrics["recall_5pct_fpr"], 4)
                if (direct_metrics["recall_5pct_fpr"] and rollout_metrics["recall_5pct_fpr"]) else None,
            },
        }

    # Save results JSON
    final_output = {
        "generated_by": "evaluation/k5/direct_heads_eval.py",
        "dataset": data_path,
        "manifest": manifest_path,
        "features": "ml1/artifacts/lr_set_a/set_a_features.json",
        "horizons": horizon_results,
    }

    out_json = OUT_DIR / "direct_heads_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nWrote {out_json}")

    # Generate DIRECT_HEADS.md
    md_lines = [
        "# Direct Multi-Horizon Onset Heads vs Rollout-Based Probability",
        "",
        f"Generated by `evaluation/k5/direct_heads_eval.py` on `{data_path}` under 37-fold LOEO.",
        "",
        "## Comparison Table",
        "",
        "Target for horizon K: an attack starts within $(t, t+K]$ (i.e. $\\max_{j=1..K} y(t+j) = 1$).",
        "",
        "| Horizon K | Positives / Total | Direct Head ROC-AUC | Rollout ROC-AUC | Direct PR-AUC | Rollout PR-AUC | Direct Recall @ 5% FPR | Rollout Recall @ 5% FPR |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for k in range(1, 6):
        res = horizon_results[str(k)]
        tot = res["n_test_windows"]
        pos = res["positives"]
        dh = res["direct_lr_head"]
        ro = res["rollout_onset_probability"]

        d_roc = f"{dh['roc_auc']:.4f}" if dh["roc_auc"] is not None else "N/A"
        r_roc = f"{ro['roc_auc']:.4f}" if ro["roc_auc"] is not None else "N/A"
        d_pr = f"{dh['pr_auc']:.4f}" if dh["pr_auc"] is not None else "N/A"
        r_pr = f"{ro['pr_auc']:.4f}" if ro["pr_auc"] is not None else "N/A"
        d_rec = f"{dh['recall_5pct_fpr']:.4f}" if dh["recall_5pct_fpr"] is not None else "N/A"
        r_rec = f"{ro['recall_5pct_fpr']:.4f}" if ro["recall_5pct_fpr"] is not None else "N/A"

        md_lines.append(
            f"| K={k} (+{k}m) | {pos} / {tot} | {d_roc} | {r_roc} | {d_pr} | {r_pr} | {d_rec} | {r_rec} |"
        )

    md_lines.extend([
        "",
        "## Key Observations",
        "",
        "- Direct multi-horizon heads train directly on the multi-step target without compounding autoregressive error.",
        "- World-model rollout accumulates transition error with each step, impacting downstream onset classification.",
        "",
    ])

    md_path = OUT_DIR / "DIRECT_HEADS.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    run_direct_heads_eval()
