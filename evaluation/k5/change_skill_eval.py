"""
Phase C-t2: Skill on Change Windows under 37-fold LOEO
At each K (1..5), splits test windows into:
- 'change': label_binary at t+K differs from label_binary at t
- 'no-change': everything else

Evaluates:
1. Rollout-based onset probability (existing stacked onset head applied to rolled states)
2. Persistence
3. Stacked onset model (K=1 only)

Reports:
- Number of change windows per K (if < 20, marked 'too few to evaluate')
- ROC-AUC and recall at 5% FPR on change windows
- F1 on no-change windows
- F1 on all windows combined

Outputs:
- evaluation/k5/change_skill_results.json
- evaluation/k5/CHANGE_SKILL.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "ml1") not in sys.path:
    sys.path.insert(0, str(REPO / "ml1"))
if str(REPO / "backend") not in sys.path:
    sys.path.insert(0, str(REPO / "backend"))

from lstm.ucs import UCSConfig, validate_ucs_windows  # noqa: E402

OUT_DIR = REPO / "evaluation" / "k5"
STACKED_ONSET_DIR = REPO / "ml1" / "artifacts" / "lstm" / "lstm_stacked" / "onset"


class ResidualLSTM(nn.Module):
    def __init__(self, input_size: int = 32, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward_logits(self, x, base_logit):
        out, _ = self.lstm(x)
        return base_logit + self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


class OnsetFoldModel:
    def __init__(self, ckpt_path: Path, device: torch.device):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.fold_id = ckpt.get("fold_id", 0)
        self.lr_coef = np.asarray(ckpt["lr_coef"], dtype=np.float64).reshape(-1)
        self.lr_intercept = float(np.asarray(ckpt["lr_intercept"], dtype=np.float64).reshape(-1)[0])
        self.scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
        self.scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
        self.pca_mean = np.asarray(ckpt["pca_mean"], dtype=np.float64)
        self.pca_components = np.asarray(ckpt["pca_components"], dtype=np.float64)
        self.temperature = max(float(ckpt.get("temperature", 1.0)), 0.05)

        self.model = ResidualLSTM(input_size=32, hidden_size=64, dropout=0.2)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict_proba(self, seq_30x406: np.ndarray) -> float:
        """Predict onset probability for a 30x406 window sequence."""
        x_last = seq_30x406[-1]
        x_scaled = (x_last - self.scaler_mean) / self.scaler_scale
        base_logit = float(x_scaled @ self.lr_coef + self.lr_intercept)

        seq_projected = ((seq_30x406 - self.pca_mean) @ self.pca_components.T).astype(np.float32)
        seq_tensor = torch.as_tensor(seq_projected, dtype=torch.float32).unsqueeze(0).to(self.device)
        base_tensor = torch.tensor([base_logit], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            res_logit = float(self.model.forward_logits(seq_tensor, base_tensor).item())

        calibrated_logit = res_logit / self.temperature
        prob = 1.0 / (1.0 + np.exp(-np.clip(calibrated_logit, -50.0, 50.0)))
        return float(prob)


def compute_metrics_at_5pct_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.05) -> Dict[str, Any]:
    """Compute threshold, recall, precision, and F1 at matched 5% FPR."""
    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": None,
            "threshold_5pct_fpr": None,
            "recall_5pct_fpr": None,
            "precision_5pct_fpr": None,
            "f1_5pct_fpr": None,
            "fpr_actual": None,
            "note": "Only one class present in y_true",
        }

    roc_auc = float(roc_auc_score(y_true, y_score))

    # Sweep thresholds to achieve FPR <= target_fpr
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
        "threshold_5pct_fpr": round(best_thresh, 4),
        "recall_5pct_fpr": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "precision_5pct_fpr": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "f1_5pct_fpr": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "fpr_actual": round(float(fpr_actual), 4),
    }


def run_change_skill_eval(
    data_path: str = "data-engineering/data/ucs/ucs_windows_models_v1.parquet",
    manifest_path: str = "ml1/artifacts/loeo/corrected_37fold_manifest.json",
    cache_path: str = "evaluation/k5/rollout_predictions_cache.joblib",
    device_str: str = "cpu",
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(REPO / data_path).sort_values("window_start_utc").reset_index(drop=True)
    config = UCSConfig()
    features = validate_ucs_windows(df, config=config)
    raw_values = df[features].to_numpy(dtype=np.float32)

    print(f"Loading rollout predictions cache from {cache_path}...")
    records_df = joblib.load(REPO / cache_path)
    print(f"Loaded {len(records_df)} window rollout records")

    # Load stacked onset fold models
    print(f"Loading stacked onset models from {STACKED_ONSET_DIR}...")
    onset_models: Dict[int, OnsetFoldModel] = {}
    for fold_id in range(37):
        ckpt_p = STACKED_ONSET_DIR / f"model_fold_{fold_id}.pt"
        if ckpt_p.exists():
            onset_models[fold_id] = OnsetFoldModel(ckpt_p, device)
    print(f"Loaded {len(onset_models)} stacked onset fold models")

    # Group test window records by (fold_id, window_index)
    # Reconstruct the rolled 30-window sequence for each k
    results_per_k = {}

    for k in range(1, 6):
        k_df = records_df[records_df["k"] == k].copy()
        print(f"\n--- Evaluating Horizon K = {k} (+{k}m) ---")

        # Predictions arrays
        p_rollout_list = []
        p_persist_list = []
        p_stacked_k1_list = []
        actual_labels = []
        current_labels = []
        change_flags = []

        for _, row in k_df.iterrows():
            f_id = int(row["fold_id"])
            t = int(row["window_index"])
            target_t = int(row["target_window_index"])
            y_future = int(row["actual_label_binary"])
            y_current = int(row["current_label_binary"])

            onset_model = onset_models[f_id]

            # Reconstruct sequence ending with rolled state at step k
            # Retrieve preceding history: t - 30 + k to t
            L = config.lookback_windows  # 30
            # History consists of real states up to t, then rolled states 1..k
            hist_real = raw_values[t - L + k : t + 1]  # shape (30 - k + 1, 406)
            # Find rolled states for this window up to k
            win_records = records_df[(records_df["fold_id"] == f_id) & (records_df["window_index"] == t)]
            rolled_states_k = [win_records[win_records["k"] == step]["pred_lstm_state"].iloc[0] for step in range(1, k + 1)]
            rolled_states_arr = np.stack(rolled_states_k, axis=0)  # shape (k, 406)

            # Sequence of 30: real states ending at t-k+k=t, plus rolled states 1..k
            # If k=1: real states [t-29 : t] (29 states) + rolled_state_1 (1 state) = 30
            seq_30x406 = np.concatenate([raw_values[t - L + k : t], rolled_states_arr], axis=0)
            assert len(seq_30x406) == 30, f"Expected seq len 30, got {len(seq_30x406)}"

            p_rollout = onset_model.predict_proba(seq_30x406)
            p_persist = float(y_current)

            p_rollout_list.append(p_rollout)
            p_persist_list.append(p_persist)
            actual_labels.append(y_future)
            current_labels.append(y_current)
            change_flags.append(y_future != y_current)

            # Stacked onset model evaluated on unrolled observed history X(t) [t-30 : t]
            if k == 1:
                seq_unrolled = raw_values[t - L : t]
                p_stacked_k1 = onset_model.predict_proba(seq_unrolled)
                p_stacked_k1_list.append(p_stacked_k1)

        y_true_arr = np.array(actual_labels, dtype=int)
        y_curr_arr = np.array(current_labels, dtype=int)
        is_change_arr = np.array(change_flags, dtype=bool)

        p_rollout_arr = np.array(p_rollout_list, dtype=float)
        p_persist_arr = np.array(p_persist_list, dtype=float)

        n_total = len(y_true_arr)
        n_change = int(is_change_arr.sum())
        n_no_change = n_total - n_change

        print(f"Total windows: {n_total} | Change: {n_change} | No-change: {n_no_change}")

        k_result: Dict[str, Any] = {
            "n_total": n_total,
            "n_change": n_change,
            "n_no_change": n_no_change,
        }

        # Change windows evaluation
        if n_change < 20:
            k_result["change_evaluation_status"] = f"too few to evaluate ({n_change} < 20)"
            k_result["change_windows"] = None
        else:
            k_result["change_evaluation_status"] = "evaluated"
            y_change = y_true_arr[is_change_arr]
            p_rollout_change = p_rollout_arr[is_change_arr]
            p_persist_change = p_persist_arr[is_change_arr]

            change_metrics = {
                "rollout_onset": compute_metrics_at_5pct_fpr(y_change, p_rollout_change),
                "persistence": compute_metrics_at_5pct_fpr(y_change, p_persist_change),
            }

            if k == 1:
                p_stacked_k1_arr = np.array(p_stacked_k1_list, dtype=float)
                change_metrics["stacked_onset_k1"] = compute_metrics_at_5pct_fpr(
                    y_change, p_stacked_k1_arr[is_change_arr]
                )

            k_result["change_windows"] = change_metrics

        # No-change windows evaluation (F1 at 0.5 threshold)
        y_no_change = y_true_arr[~is_change_arr]
        p_rollout_nc = p_rollout_arr[~is_change_arr]
        p_persist_nc = p_persist_arr[~is_change_arr]

        k_result["no_change_windows"] = {
            "rollout_onset_f1_at_0.5": round(float(f1_score(y_no_change, (p_rollout_nc >= 0.5).astype(int), zero_division=0)), 4),
            "persistence_f1_at_0.5": round(float(f1_score(y_no_change, (p_persist_nc >= 0.5).astype(int), zero_division=0)), 4),
        }

        # All windows combined evaluation (F1 at 0.5 threshold)
        k_result["all_windows"] = {
            "rollout_onset_f1_at_0.5": round(float(f1_score(y_true_arr, (p_rollout_arr >= 0.5).astype(int), zero_division=0)), 4),
            "persistence_f1_at_0.5": round(float(f1_score(y_true_arr, (p_persist_arr >= 0.5).astype(int), zero_division=0)), 4),
        }

        results_per_k[str(k)] = k_result

    # Save results JSON
    final_output = {
        "generated_by": "evaluation/k5/change_skill_eval.py",
        "dataset": data_path,
        "manifest": manifest_path,
        "horizons": results_per_k,
    }

    out_json_path = OUT_DIR / "change_skill_results.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nWrote {out_json_path}")

    # Generate CHANGE_SKILL.md
    md_lines = [
        "# Change-Window Skill Evaluation (K = 1..5) under 37-fold LOEO",
        "",
        f"Generated by `evaluation/k5/change_skill_eval.py` on `{data_path}`.",
        "",
        "## Summary of Change-Window Performance",
        "",
        "A **change window** is defined as any test window where $y(t+K) \\neq y(t)$ (attack onset $0 \\to 1$ or attack offset $1 \\to 0$).",
        "Because persistence always predicts the preceding state, persistence systematically fails on change windows.",
        "Evaluating models specifically on change windows tests whether forward simulation possesses true predictive skill.",
        "",
        "| Horizon K | Total Windows | Change Windows | Change Status | Rollout ROC-AUC | Rollout Recall @ 5% FPR | Persistence ROC-AUC | No-Change F1 (Rollout / Persist) | All Windows F1 (Rollout / Persist) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for k in range(1, 6):
        res = results_per_k[str(k)]
        n_tot = res["n_total"]
        n_ch = res["n_change"]
        st = res["change_evaluation_status"]

        if res["change_windows"] is not None:
            r_auc = res["change_windows"]["rollout_onset"]["roc_auc"]
            r_rec = res["change_windows"]["rollout_onset"]["recall_5pct_fpr"]
            p_auc = res["change_windows"]["persistence"]["roc_auc"]
            r_auc_str = f"{r_auc:.4f}" if r_auc is not None else "N/A"
            r_rec_str = f"{r_rec:.4f}" if r_rec is not None else "N/A"
            p_auc_str = f"{p_auc:.4f}" if p_auc is not None else "N/A"
        else:
            r_auc_str = "too few"
            r_rec_str = "too few"
            p_auc_str = "too few"

        nc_f1 = f"{res['no_change_windows']['rollout_onset_f1_at_0.5']:.4f} / {res['no_change_windows']['persistence_f1_at_0.5']:.4f}"
        all_f1 = f"{res['all_windows']['rollout_onset_f1_at_0.5']:.4f} / {res['all_windows']['persistence_f1_at_0.5']:.4f}"

        md_lines.append(
            f"| K={k} (+{k}m) | {n_tot} | {n_ch} | {st} | {r_auc_str} | {r_rec_str} | {p_auc_str} | {nc_f1} | {all_f1} |"
        )

    md_lines.extend([
        "",
        "## Key Takeaways",
        "",
        "- If the change-window ROC-AUC is not above 0.5, the forward rollout does not demonstrate predictive skill over chance on transition points.",
        "- Persistence achieves high F1 on no-change windows simply by inertia, but has zero anticipatory capability.",
        "",
    ])

    md_path = OUT_DIR / "CHANGE_SKILL.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_change_skill_eval(device_str=args.device)
