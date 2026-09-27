"""
Phase 6 — Architecture Swap Experiment: GRU vs LSTM

Swaps the LSTM cell for a GRU cell, keeping every other hyperparameter identical.
Trains on 3 representative LOEO folds (one per attack type).

Outputs:
  - diagnostics/checkpoints/gru_experiment_fold{N}.pt
  - diagnostics/phase6_gru_experiment_report.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

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
from lstm.trainer import train
from lstm.utils import set_seed


class GRUClassifier(nn.Module):
    """Drop-in GRU replacement for LSTMClassifier, identical hyperparameters."""

    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_logits(self, x):
        output, _ = self.gru(x)
        last_output = self.dropout(output[:, -1, :])
        return self.fc(last_output).squeeze(-1)

    def forward(self, x):
        return self.sigmoid(self.forward_logits(x))

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


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
    transformed, pca = apply_training_only_pca(purged, base_features, 32, random_state=42)
    features = [f"pca_{i}" for i in range(32)]
    sequences = build_lstm_sequences(transformed, features=features, config=config, target_col=target_col)

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
    device = torch.device(device_str)

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
    print("Phase 6: GRU Architecture Swap Experiment")
    print("=" * 60)

    # Pick 3 representative folds
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

    ckpt_dir = ML1_DIR / "diagnostics" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    lstm_results = []
    gru_results = []

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

        from lstm.model import LSTMClassifier

        # --- LSTM ---
        set_seed(42)
        model_lstm = LSTMClassifier(input_size=32, hidden_size=64, num_layers=1, dropout=0.20)
        lstm_result = train(
            model_lstm,
            sequences["train"].X, sequences["train"].y,
            sequences["val"].X, sequences["val"].y,
            epochs=30, batch_size=64, learning_rate=0.001,
            seed=42, device=device_str, weight_decay=0.0001,
            early_stopping_patience=5, early_stopping_min_delta=0.001,
        )
        val_probs_lstm = get_probabilities(lstm_result["model"], sequences["val"].X, device)
        test_probs_lstm = get_probabilities(lstm_result["model"], test_X, device)
        best_t_lstm, val_f1_lstm = sweep_threshold(sequences["val"].y, val_probs_lstm)
        test_f1_lstm = f1_score(test_y.astype(int), (test_probs_lstm >= best_t_lstm).astype(int), zero_division=0)

        lstm_results.append({
            "fold_id": fold_id, "attack_type": attack_type,
            "val_threshold": round(float(best_t_lstm), 3),
            "val_f1": round(float(val_f1_lstm), 4),
            "test_f1": round(float(test_f1_lstm), 4),
            "best_epoch": lstm_result["best_epoch"],
        })

        # --- GRU ---
        set_seed(42)
        model_gru = GRUClassifier(input_size=32, hidden_size=64, num_layers=1, dropout=0.20)
        gru_ckpt_path = ckpt_dir / f"gru_experiment_fold{fold_id}.pt"
        gru_result = train(
            model_gru,
            sequences["train"].X, sequences["train"].y,
            sequences["val"].X, sequences["val"].y,
            epochs=30, batch_size=64, learning_rate=0.001,
            seed=42, device=device_str, weight_decay=0.0001,
            early_stopping_patience=5, early_stopping_min_delta=0.001,
            checkpoint_path=str(gru_ckpt_path),
        )
        val_probs_gru = get_probabilities(gru_result["model"], sequences["val"].X, device)
        test_probs_gru = get_probabilities(gru_result["model"], test_X, device)
        best_t_gru, val_f1_gru = sweep_threshold(sequences["val"].y, val_probs_gru)
        test_f1_gru = f1_score(test_y.astype(int), (test_probs_gru >= best_t_gru).astype(int), zero_division=0)

        gru_results.append({
            "fold_id": fold_id, "attack_type": attack_type,
            "val_threshold": round(float(best_t_gru), 3),
            "val_f1": round(float(val_f1_gru), 4),
            "test_f1": round(float(test_f1_gru), 4),
            "best_epoch": gru_result["best_epoch"],
            "checkpoint": str(gru_ckpt_path),
        })

        print(f"    LSTM: val_t={best_t_lstm:.3f} val_F1={val_f1_lstm:.4f} test_F1={test_f1_lstm:.4f} (epoch {lstm_result['best_epoch']})")
        print(f"    GRU:  val_t={best_t_gru:.3f} val_F1={val_f1_gru:.4f} test_F1={test_f1_gru:.4f} (epoch {gru_result['best_epoch']})")

    report = {
        "lstm_results": lstm_results,
        "gru_results": gru_results,
    }

    output_path = ML1_DIR / "diagnostics" / "phase6_gru_experiment_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
