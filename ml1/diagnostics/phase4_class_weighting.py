"""
Phase 4 — Class Weighting Diagnostic

1. Confirms whether the current loss applies pos_weight.
2. Runs one controlled experiment with class weighting.
3. Saves checkpoint as: diagnostics/checkpoints/lstm_class_weighted_experiment.pt

Outputs: diagnostics/phase4_class_weighting_report.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

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
from lstm.trainer import train
from lstm.utils import set_seed


def build_fold_data(df, fold, config, target_col):
    """Build train/val/test data for a single LOEO fold."""
    fold_frame = df.copy()
    fold_frame["split"] = "none"
    train_indices = fold["train_indices"]
    test_indices = fold["test_indices"]

    fold_frame.loc[train_indices, "split"] = "train"
    fold_frame.loc[test_indices, "split"] = "test"

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

    # Build test sequences
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


def get_probabilities(model, X, device):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model(X_tensor).cpu().numpy()
    return probs


def main():
    set_seed(42)
    device_str = "cpu"

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

    # =============================================
    # Part 1: Confirm current class weighting status
    # =============================================
    print("=" * 60)
    print("Phase 4: Class Weighting Diagnostic")
    print("=" * 60)

    # Read model_config.yaml
    config_path = ML1_DIR / "configs" / "model_config.yaml"
    import yaml
    with open(config_path) as f:
        model_cfg = yaml.safe_load(f)

    class_weighting_config = model_cfg.get("training", {}).get("class_weighting", None)
    loss_config = model_cfg.get("training", {}).get("loss", None)

    print(f"  model_config.yaml training.class_weighting: {class_weighting_config}")
    print(f"  model_config.yaml training.loss: {loss_config}")

    # Check trainer.py: BCEWithLogitsLoss uses pos_weight only when pos_weight is passed.
    # The train() function accepts pos_weight parameter.
    # run_lstm_frozen.py does NOT pass pos_weight, so class_weighting is NOT applied.
    print(f"\n  Trainer: pos_weight is only applied when explicitly passed to train().")
    print(f"  run_lstm_frozen.py: does NOT pass pos_weight. Class weighting is NOT applied.")
    print(f"  Config says class_weighting: true, but the actual training scripts DON'T use it.")

    status = {
        "config_says_class_weighting": class_weighting_config,
        "config_loss": loss_config,
        "trainer_supports_pos_weight": True,
        "run_lstm_frozen_uses_pos_weight": False,
        "actual_class_weighting_applied": False,
        "diagnosis": (
            "model_config.yaml says class_weighting: true, "
            "but run_lstm_frozen.py does NOT pass pos_weight to train(). "
            "Therefore class weighting was NOT applied during training."
        ),
    }

    # =============================================
    # Part 2: Run class-weighted experiment on 3 representative folds
    # =============================================
    print(f"\n  Running class-weighted experiment on 3 representative folds...")

    # Pick 3 folds: one Botnet, one SSH, one DDOS
    representative_folds = []
    seen_types = set()
    for fold in manifest["folds"]:
        attack_type = df.loc[
            df["episode_id"] == fold["held_out_episode_id"], "label_attack_type"
        ].iloc[0]
        if attack_type not in seen_types:
            seen_types.add(attack_type)
            representative_folds.append((fold, attack_type))
        if len(representative_folds) == 3:
            break

    weighted_results = []
    unweighted_results = []

    ckpt_dir = ML1_DIR / "diagnostics" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for fold, attack_type in representative_folds:
        fold_id = fold["fold_id"]
        print(f"\n  Fold {fold_id} ({attack_type}):")

        try:
            sequences, test_X, test_y = build_fold_data(df, fold, config, target_col)
        except Exception as e:
            print(f"    Data build failed: {e}")
            continue

        if test_X is None:
            print(f"    No test sequences")
            continue

        train_y = sequences["train"].y
        n_pos = (train_y == 1).sum()
        n_neg = (train_y == 0).sum()
        pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"    Train class balance: {n_neg} neg / {n_pos} pos (pos_weight={pos_weight_val:.2f})")

        # --- Unweighted ---
        set_seed(42)
        model_uw = LSTMClassifier(input_size=32, hidden_size=64, num_layers=1, dropout=0.20)
        uw_result = train(
            model_uw,
            sequences["train"].X, sequences["train"].y,
            sequences["val"].X, sequences["val"].y,
            epochs=30, batch_size=64, learning_rate=0.001,
            seed=42, device=device_str, weight_decay=0.0001,
            early_stopping_patience=5, early_stopping_min_delta=0.001,
        )

        val_probs_uw = get_probabilities(uw_result["model"], sequences["val"].X, torch.device(device_str))
        test_probs_uw = get_probabilities(uw_result["model"], test_X, torch.device(device_str))
        best_t_uw, val_f1_uw = sweep_threshold(sequences["val"].y, val_probs_uw)
        test_f1_uw = f1_score(test_y.astype(int), (test_probs_uw >= best_t_uw).astype(int), zero_division=0)

        unweighted_results.append({
            "fold_id": fold_id,
            "attack_type": attack_type,
            "val_threshold": round(float(best_t_uw), 3),
            "val_f1": round(float(val_f1_uw), 4),
            "test_f1": round(float(test_f1_uw), 4),
            "best_epoch": uw_result["best_epoch"],
        })

        # --- Weighted ---
        set_seed(42)
        model_w = LSTMClassifier(input_size=32, hidden_size=64, num_layers=1, dropout=0.20)
        ckpt_path = ckpt_dir / f"lstm_class_weighted_fold{fold_id}.pt"
        w_result = train(
            model_w,
            sequences["train"].X, sequences["train"].y,
            sequences["val"].X, sequences["val"].y,
            epochs=30, batch_size=64, learning_rate=0.001,
            seed=42, device=device_str, weight_decay=0.0001,
            pos_weight=pos_weight_val,
            early_stopping_patience=5, early_stopping_min_delta=0.001,
            checkpoint_path=str(ckpt_path),
        )

        val_probs_w = get_probabilities(w_result["model"], sequences["val"].X, torch.device(device_str))
        test_probs_w = get_probabilities(w_result["model"], test_X, torch.device(device_str))
        best_t_w, val_f1_w = sweep_threshold(sequences["val"].y, val_probs_w)
        test_f1_w = f1_score(test_y.astype(int), (test_probs_w >= best_t_w).astype(int), zero_division=0)

        weighted_results.append({
            "fold_id": fold_id,
            "attack_type": attack_type,
            "pos_weight": round(float(pos_weight_val), 3),
            "val_threshold": round(float(best_t_w), 3),
            "val_f1": round(float(val_f1_w), 4),
            "test_f1": round(float(test_f1_w), 4),
            "best_epoch": w_result["best_epoch"],
            "checkpoint": str(ckpt_path),
        })

        print(f"    Unweighted: val_t={best_t_uw:.3f} val_F1={val_f1_uw:.4f} test_F1={test_f1_uw:.4f}")
        print(f"    Weighted:   val_t={best_t_w:.3f} val_F1={val_f1_w:.4f} test_F1={test_f1_w:.4f}")

    # Save results
    report = {
        "class_weighting_status": status,
        "unweighted_experiment": unweighted_results,
        "weighted_experiment": weighted_results,
    }

    output_path = ML1_DIR / "diagnostics" / "phase4_class_weighting_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
