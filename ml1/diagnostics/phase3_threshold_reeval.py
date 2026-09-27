"""
Phase 3 — Threshold Re-evaluation Diagnostic

Loads existing LSTM LOEO fold checkpoints, sweeps thresholds on validation,
freezes the best threshold, and re-evaluates once on test.

Outputs: diagnostics/threshold_reeval_report.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

# Setup paths
SCRIPT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
ML1_DIR = SCRIPT_DIR.parent
REPO_ROOT = ML1_DIR.parent
sys.path.insert(0, str(ML1_DIR))

from lstm.model import LSTMClassifier
from lstm.ucs import (
    UCSConfig,
    validate_ucs_windows,
    apply_training_only_pca,
    build_lstm_sequences,
)
from lstm.utils import set_seed


def build_fold_data(df, fold, config, target_col):
    """Build train/val/test data for a single LOEO fold, matching run_lstm_frozen.py."""
    fold_frame = df.copy()
    fold_frame["split"] = "none"
    train_indices = fold["train_indices"]
    test_indices = fold["test_indices"]

    fold_frame.loc[train_indices, "split"] = "train"
    fold_frame.loc[test_indices, "split"] = "test"

    # 80/20 chronological train/val split
    train_indices_arr = np.array(train_indices)
    train_df_sorted = fold_frame.loc[train_indices_arr].sort_values("window_start_utc")
    n_train = len(train_df_sorted)
    val_start = int(n_train * 0.8)
    actual_train_indices = train_df_sorted.index[:val_start]
    actual_val_indices = train_df_sorted.index[val_start:]

    fold_frame.loc[actual_train_indices, "split"] = "train"
    fold_frame.loc[actual_val_indices, "split"] = "val"
    fold_frame.loc[test_indices, "split"] = "test"

    active_mask = fold_frame["split"].isin(["train", "val", "test"])
    purged = fold_frame[active_mask].copy()

    base_features = validate_ucs_windows(purged, config=config)

    transformed, pca = apply_training_only_pca(
        purged, base_features, 32, random_state=42
    )
    features = [f"pca_{i}" for i in range(32)]

    sequences = build_lstm_sequences(
        transformed, features=features, config=config, target_col=target_col
    )

    # Build test sequences (matching run_lstm_frozen.py logic)
    full_transformed = pca.transform(fold_frame)
    full_transformed_np = full_transformed[features].to_numpy(dtype=np.float32)
    targets_array = fold_frame[target_col].values

    idx_to_pos = {idx: pos for pos, idx in enumerate(fold_frame.index)}
    test_pos_indices = [idx_to_pos[idx] for idx in test_indices]

    histories = []
    labels = []
    for t_pos in test_pos_indices:
        start = t_pos - config.lookback_windows + 1
        if start < 0:
            continue
        hist = full_transformed_np[start : t_pos + 1]
        if len(hist) == config.lookback_windows:
            histories.append(hist)
            labels.append(targets_array[t_pos])

    if len(histories) == 0:
        return sequences, None, None

    test_X = np.asarray(histories, dtype=np.float32)
    test_y = np.asarray(labels, dtype=np.float32)
    return sequences, test_X, test_y


def get_probabilities(model, X, device):
    """Get probabilities from model."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model(X_tensor).cpu().numpy()
    return probs


def sweep_threshold(y_true, probs, thresholds=None):
    """Sweep thresholds and return the one maximizing F1."""
    if thresholds is None:
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
    device = torch.device("cpu")

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

    # Checkpoint directories for different models
    # detection_v5 = latest LOEO fold checkpoints (state dicts directly)
    # lstm_diagnostics/checkpoints = older checkpoints (trainer format)
    ckpt_dir_detection = ML1_DIR / "artifacts" / "lstm" / "detection_v5"
    ckpt_dir_diagnostics = ML1_DIR / "artifacts" / "lstm_diagnostics" / "checkpoints"

    results = {"phases": {}}

    # === Phase 3: Threshold re-evaluation for onset (future_attack_label) ===
    print("=" * 60)
    print("Phase 3: Threshold Re-evaluation (onset/future_attack_label)")
    print("=" * 60)

    fold_results = []

    for fold in manifest["folds"]:
        fold_id = fold["fold_id"]

        # Use diagnostics checkpoint (onset model)
        ckpt_path = ckpt_dir_diagnostics / f"fold_{fold_id}" / "classifier_best.pt"
        if not ckpt_path.exists():
            print(f"  Fold {fold_id}: checkpoint not found, skipping")
            continue

        try:
            sequences, test_X, test_y = build_fold_data(
                df, fold, config, "future_attack_label"
            )
        except Exception as e:
            print(f"  Fold {fold_id}: data build failed ({e}), skipping")
            continue

        if test_X is None or len(test_X) == 0:
            print(f"  Fold {fold_id}: no test sequences, skipping")
            continue

        # Load model
        model = LSTMClassifier(input_size=32, hidden_size=64, num_layers=1, dropout=0.20)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        # Get validation probabilities
        val_X = sequences["val"].X
        val_y = sequences["val"].y

        val_probs = get_probabilities(model, val_X, device)
        test_probs = get_probabilities(model, test_X, device)

        # Sweep threshold on validation
        best_threshold, val_f1 = sweep_threshold(val_y, val_probs)

        # Default 0.5 test F1
        test_preds_050 = (test_probs >= 0.5).astype(int)
        test_f1_050 = f1_score(test_y.astype(int), test_preds_050, zero_division=0)

        # Default 0.38 test F1
        test_preds_038 = (test_probs >= 0.38).astype(int)
        test_f1_038 = f1_score(test_y.astype(int), test_preds_038, zero_division=0)

        # Frozen threshold test F1
        test_preds_frozen = (test_probs >= best_threshold).astype(int)
        test_f1_frozen = f1_score(test_y.astype(int), test_preds_frozen, zero_division=0)

        fold_results.append({
            "fold_id": fold_id,
            "episode": fold["held_out_episode_id"],
            "val_best_threshold": round(float(best_threshold), 3),
            "val_f1_at_best_threshold": round(float(val_f1), 4),
            "test_f1_at_050": round(float(test_f1_050), 4),
            "test_f1_at_038": round(float(test_f1_038), 4),
            "test_f1_frozen_threshold": round(float(test_f1_frozen), 4),
            "test_n": int(len(test_y)),
            "test_pos_rate": round(float(test_y.mean()), 4),
            "prob_min": round(float(test_probs.min()), 4),
            "prob_max": round(float(test_probs.max()), 4),
            "prob_mean": round(float(test_probs.mean()), 4),
        })

        print(
            f"  Fold {fold_id:2d}: "
            f"val_best_t={best_threshold:.3f}  "
            f"test_F1@0.5={test_f1_050:.3f}  "
            f"test_F1@frozen={test_f1_frozen:.3f}"
        )

    if fold_results:
        df_results = pd.DataFrame(fold_results)
        mean_f1_050 = df_results["test_f1_at_050"].mean()
        mean_f1_038 = df_results["test_f1_at_038"].mean()
        mean_f1_frozen = df_results["test_f1_frozen_threshold"].mean()

        summary = {
            "target": "future_attack_label (onset)",
            "n_folds_evaluated": len(fold_results),
            "mean_test_f1_at_050": round(float(mean_f1_050), 4),
            "mean_test_f1_at_038": round(float(mean_f1_038), 4),
            "mean_test_f1_frozen_threshold": round(float(mean_f1_frozen), 4),
            "per_fold_results": fold_results,
        }

        print(f"\n  Summary:")
        print(f"    Mean test F1 @ 0.50:  {mean_f1_050:.4f}")
        print(f"    Mean test F1 @ 0.38:  {mean_f1_038:.4f}")
        print(f"    Mean test F1 @ frozen: {mean_f1_frozen:.4f}")
    else:
        summary = {"error": "No folds could be evaluated"}

    results["phases"]["phase3_threshold_reeval"] = summary

    # Save results
    output_path = ML1_DIR / "diagnostics" / "threshold_reeval_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
