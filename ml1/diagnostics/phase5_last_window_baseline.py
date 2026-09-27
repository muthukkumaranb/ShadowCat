"""
Phase 5 — Single-Window Signal Diagnostic

Tests whether the LSTM's temporal lookback provides value over a single window.

Extracts the FINAL window from each 30-window LSTM sequence (using the same
purged splits, same PCA transform, same normalization), then trains a simple
Logistic Regression on that single-window feature vector.

Compares:
  - Full-history LR baseline (0.889 on 406-D features)
  - Single-window LR on PCA-32 features
  - Single-window LR on FULL 406-D features (no PCA)
  - LSTM on PCA-32 sequences (0.750 baseline / 0.913 stacked)

Outputs: diagnostics/phase5_last_window_report.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

SCRIPT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
ML1_DIR = SCRIPT_DIR.parent
REPO_ROOT = ML1_DIR.parent
sys.path.insert(0, str(ML1_DIR))

from lstm.ucs import (
    UCSConfig,
    validate_ucs_windows,
    apply_training_only_pca,
    build_lstm_sequences,
)
from lstm.utils import set_seed


def sweep_threshold(y_true, probs):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_f1 = -1
    best_t = 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def main():
    set_seed(42)

    # Load data
    data_paths = [
        REPO_ROOT / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet",
        ML1_DIR / "data" / "ucs" / "ucs_windows.parquet",
    ]
    data_path = None
    for p in data_paths:
        if p.exists():
            data_path = p
            break
    if data_path is None:
        raise FileNotFoundError("Cannot find ucs_windows.parquet")

    manifest_path = ML1_DIR / "artifacts" / "loeo" / "corrected_37fold_manifest.json"
    df = pd.read_parquet(data_path).sort_values("window_start_utc").reset_index(drop=True)
    with open(manifest_path) as f:
        manifest = json.load(f)

    config = UCSConfig()
    target_col = "future_attack_label"

    print("=" * 60)
    print("Phase 5: Last-Window-Only Signal Diagnostic")
    print("=" * 60)

    # Get base features (full 406-D)
    base_features = validate_ucs_windows(df, config=config)
    print(f"  Full feature count: {len(base_features)}")

    results_pca32 = []  # LR on last window of PCA-32 sequences
    results_full = []   # LR on last window of full features

    for fold in manifest["folds"]:
        fold_id = fold["fold_id"]
        train_indices = fold["train_indices"]
        test_indices = fold["test_indices"]

        # --- Full-feature LR (matching run_lr_frozen.py approach) ---
        train_df = df.iloc[train_indices]
        test_df = df.iloc[test_indices]

        x_train_full = train_df[base_features].to_numpy(dtype=float)
        x_test_full = test_df[base_features].to_numpy(dtype=float)
        y_train = train_df[target_col].to_numpy(dtype=int)
        y_test = test_df[target_col].to_numpy(dtype=int)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  Fold {fold_id}: skipped (single class)")
            continue

        # Full-feature last-window LR (same as the original LR baseline, per-window)
        scaler_full = StandardScaler()
        x_train_scaled = scaler_full.fit_transform(x_train_full)
        x_test_scaled = scaler_full.transform(x_test_full)

        lr_full = LogisticRegression(
            solver="liblinear", max_iter=2000, class_weight=None, random_state=42
        )
        lr_full.fit(x_train_scaled, y_train)
        y_prob_full = lr_full.predict_proba(x_test_scaled)[:, 1]

        # Use 80/20 chronological train/val split for threshold sweep
        val_start = int(len(x_train_scaled) * 0.8)
        x_val_scaled = x_train_scaled[val_start:]
        y_val = y_train[val_start:]
        y_val_prob = lr_full.predict_proba(scaler_full.transform(
            train_df.iloc[val_start:][base_features].to_numpy(dtype=float)
        ))[:, 1]

        if len(np.unique(y_val)) >= 2:
            best_t_full, _ = sweep_threshold(y_val, y_val_prob)
        else:
            best_t_full = 0.5

        test_f1_full = f1_score(y_test, (y_prob_full >= best_t_full).astype(int), zero_division=0)
        test_prec_full = precision_score(y_test, (y_prob_full >= best_t_full).astype(int), zero_division=0)
        test_rec_full = recall_score(y_test, (y_prob_full >= best_t_full).astype(int), zero_division=0)

        results_full.append({
            "fold_id": fold_id,
            "threshold": round(float(best_t_full), 3),
            "test_f1": round(float(test_f1_full), 4),
            "test_precision": round(float(test_prec_full), 4),
            "test_recall": round(float(test_rec_full), 4),
        })

        # --- PCA-32 last-window LR ---
        # Build LOEO fold frame
        fold_frame = df.copy()
        fold_frame["split"] = "none"
        fold_frame.loc[train_indices, "split"] = "train"
        fold_frame.loc[test_indices, "split"] = "test"

        train_indices_arr = np.array(train_indices)
        train_df_sorted = fold_frame.loc[train_indices_arr].sort_values("window_start_utc")
        n_train = len(train_df_sorted)
        val_start_idx = int(n_train * 0.8)
        actual_train_indices = train_df_sorted.index[:val_start_idx]
        actual_val_indices = train_df_sorted.index[val_start_idx:]

        fold_frame.loc[actual_train_indices, "split"] = "train"
        fold_frame.loc[actual_val_indices, "split"] = "val"
        fold_frame.loc[test_indices, "split"] = "test"

        active_mask = fold_frame["split"].isin(["train", "val", "test"])
        purged = fold_frame[active_mask].copy()

        try:
            feat_cols = validate_ucs_windows(purged, config=config)
            transformed, pca = apply_training_only_pca(
                purged, feat_cols, 32, random_state=42
            )
            pca_features = [f"pca_{i}" for i in range(32)]

            sequences = build_lstm_sequences(
                transformed, features=pca_features, config=config, target_col=target_col
            )

            # Extract last window from each 30-window sequence
            train_last = sequences["train"].X[:, -1, :]  # (N, 32)
            train_labels = sequences["train"].y
            val_last = sequences["val"].X[:, -1, :]
            val_labels = sequences["val"].y

            # Build test sequences (same as in other scripts)
            full_transformed = pca.transform(fold_frame)
            full_transformed_np = full_transformed[pca_features].to_numpy(dtype=np.float32)
            targets_array = fold_frame[target_col].values

            idx_to_pos = {idx: pos for pos, idx in enumerate(fold_frame.index)}
            test_pos_indices = [idx_to_pos[idx] for idx in test_indices]

            test_last_windows = []
            test_labels_list = []
            for t_pos in test_pos_indices:
                if t_pos >= 0:
                    test_last_windows.append(full_transformed_np[t_pos])
                    test_labels_list.append(targets_array[t_pos])

            if len(test_last_windows) == 0:
                continue

            test_last = np.asarray(test_last_windows, dtype=np.float32)
            test_labels_pca = np.asarray(test_labels_list, dtype=np.float32)

            if len(np.unique(train_labels)) < 2:
                continue

            lr_pca = LogisticRegression(
                solver="liblinear", max_iter=2000, class_weight=None, random_state=42
            )
            lr_pca.fit(train_last, train_labels.astype(int))

            val_prob_pca = lr_pca.predict_proba(val_last)[:, 1]
            test_prob_pca = lr_pca.predict_proba(test_last)[:, 1]

            if len(np.unique(val_labels)) >= 2:
                best_t_pca, _ = sweep_threshold(val_labels, val_prob_pca)
            else:
                best_t_pca = 0.5

            test_f1_pca = f1_score(
                test_labels_pca.astype(int),
                (test_prob_pca >= best_t_pca).astype(int),
                zero_division=0,
            )
            test_prec_pca = precision_score(
                test_labels_pca.astype(int),
                (test_prob_pca >= best_t_pca).astype(int),
                zero_division=0,
            )
            test_rec_pca = recall_score(
                test_labels_pca.astype(int),
                (test_prob_pca >= best_t_pca).astype(int),
                zero_division=0,
            )

            results_pca32.append({
                "fold_id": fold_id,
                "threshold": round(float(best_t_pca), 3),
                "test_f1": round(float(test_f1_pca), 4),
                "test_precision": round(float(test_prec_pca), 4),
                "test_recall": round(float(test_rec_pca), 4),
            })

        except Exception as e:
            print(f"  Fold {fold_id}: PCA-32 failed ({e})")
            continue

        print(
            f"  Fold {fold_id:2d}: "
            f"full_F1={test_f1_full:.3f}  "
            f"pca32_F1={test_f1_pca:.3f}"
        )

    # Summary
    df_full = pd.DataFrame(results_full)
    df_pca32 = pd.DataFrame(results_pca32)

    summary = {
        "last_window_lr_full_406d": {
            "n_folds": len(results_full),
            "mean_test_f1": round(float(df_full["test_f1"].mean()), 4) if len(results_full) else None,
            "mean_test_precision": round(float(df_full["test_precision"].mean()), 4) if len(results_full) else None,
            "mean_test_recall": round(float(df_full["test_recall"].mean()), 4) if len(results_full) else None,
            "per_fold": results_full,
        },
        "last_window_lr_pca32": {
            "n_folds": len(results_pca32),
            "mean_test_f1": round(float(df_pca32["test_f1"].mean()), 4) if len(results_pca32) else None,
            "mean_test_precision": round(float(df_pca32["test_precision"].mean()), 4) if len(results_pca32) else None,
            "mean_test_recall": round(float(df_pca32["test_recall"].mean()), 4) if len(results_pca32) else None,
            "per_fold": results_pca32,
        },
        "reference_numbers": {
            "lr_baseline_onset_f1": 0.888,
            "lstm_baseline_onset_f1": 0.750,
            "lstm_stacked_onset_f1": 0.913,
        },
    }

    print(f"\n  Summary:")
    if len(results_full):
        print(f"    Last-window LR (406-D):  mean F1 = {df_full['test_f1'].mean():.4f}")
    if len(results_pca32):
        print(f"    Last-window LR (PCA-32): mean F1 = {df_pca32['test_f1'].mean():.4f}")
    print(f"    Reference LR baseline:   mean F1 = 0.888")
    print(f"    Reference LSTM baseline: mean F1 = 0.750")

    output_path = ML1_DIR / "diagnostics" / "phase5_last_window_report.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
