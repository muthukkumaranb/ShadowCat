from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lstm.evaluator import evaluate
from lstm.model import LSTMClassifier
from lstm.trainer import train
from lstm.ucs import (
    UCSConfig,
    apply_training_only_pca,
    build_lstm_sequences,
    build_loeo_episode_folds,
    feature_columns,
    load_ucs_windows,
    purge_and_embargo,
    validate_ucs_windows,
)
from lstm.utils import get_device, set_seed


def _build_holdout_sequences(frame, transformed, held_out, features, config, target_col):
    ordered = transformed.sort_values("window_start_utc").reset_index(drop=True)
    times = pd.DatetimeIndex(pd.to_datetime(ordered["window_start_utc"], utc=True))
    values = ordered[features].to_numpy(dtype=np.float32)
    targets = ordered[target_col].to_numpy(dtype=np.float32)
    episodes = ordered["episode_id"].to_numpy()
    delta = pd.Timedelta(seconds=config.window_seconds)
    histories, labels, timestamps = [], [], []
    for target_index in np.flatnonzero(episodes == held_out):
        start = target_index - config.lookback_windows
        if start < 0:
            continue
        history_times = times[start:target_index]
        if len(history_times) != config.lookback_windows:
            continue
        if (history_times[-1] - history_times[0]) != delta * (config.lookback_windows - 1):
            continue
        if not (history_times.to_series().diff().dropna() == delta).all():
            continue
        histories.append(values[start:target_index])
        labels.append(targets[target_index])
        timestamps.append(times[target_index])
    return (
        np.asarray(histories, dtype=np.float32).reshape(-1, config.lookback_windows, len(features)),
        np.asarray(labels, dtype=np.float32),
        pd.DatetimeIndex(timestamps, tz="UTC"),
    )

def run(input_path: Path, output_dir: Path, pca_components: int = 32, target_col: str = "future_attack_label") -> dict:
    set_seed(42)
    windows = load_ucs_windows(input_path)
    if "episode_id" not in windows.columns:
        raise ValueError("LOEO evaluation requires corrected episode_id metadata; source_day is not an episode identifier.")
    config = UCSConfig()
    if target_col not in {"label_binary", "future_attack_label"}:
        raise ValueError("target_col must be label_binary or future_attack_label.")
    base_features = validate_ucs_windows(windows, config=config)
    folds = build_loeo_episode_folds(windows, episode_col="episode_id")
    device = get_device()
    fold_metrics = []
    skipped_episodes = []
    pooled_labels = []
    pooled_probabilities = []
    prediction_rows = []

    for fold in folds:
        held_out = fold["held_out_episode"]
        fold_frame = windows.copy()
        holdout_mask = fold_frame["episode_id"] == held_out
        fold_frame.loc[holdout_mask, "split"] = "test"
        non_holdout_mask = ~holdout_mask
        fold_frame.loc[non_holdout_mask & (fold_frame["split"] != "val"), "split"] = "train"
        purged = purge_and_embargo(fold_frame, config=config)
        transformed, pca = apply_training_only_pca(
            purged,
            base_features,
            pca_components,
            random_state=42,
        )
        features = [f"pca_{index}" for index in range(pca_components)]
        sequences = build_lstm_sequences(
            transformed,
            features=features,
            config=config,
            target_col=target_col,
        )
        full_transformed = pca.transform(fold_frame)
        test_X, test_y, test_timestamps = _build_holdout_sequences(
            fold_frame, full_transformed, held_out, features, config, target_col
        )
        if not len(sequences["train"].y) or not len(sequences["val"].y) or not len(test_y):
            skipped_episodes.append({
                "held_out_episode": held_out,
                "reason": "No train, validation, or held-out sequence remained after purge and embargo.",
            })
            continue

        model = LSTMClassifier(input_size=pca_components, hidden_size=64, num_layers=1, dropout=0.20)
        checkpoint = output_dir / "folds" / str(held_out) / "classifier_best.pt"
        result = train(
            model,
            sequences["train"].X,
            sequences["train"].y,
            sequences["val"].X,
            sequences["val"].y,
            epochs=30,
            batch_size=64,
            learning_rate=0.001,
            seed=42,
            device=str(device),
            checkpoint_path=checkpoint,
            weight_decay=0.0001,
            early_stopping_patience=5,
            early_stopping_min_delta=0.001,
        )
        evaluation = evaluate(
            model,
            test_X,
            test_y,
            threshold=0.38,
            device=device,
        )
        tn, fp, _, _ = evaluation["confusion_matrix"].ravel()
        fpr = float(fp / (tn + fp)) if tn + fp else 0.0
        fold_result = {
            "held_out_episode": held_out,
            "attack_type": fold["attack_type"],
            "target": target_col,
            "protocol": "LOEO (retrained per fold)",
            "train_sequences": len(sequences["train"].y),
            "validation_sequences": len(sequences["val"].y),
            "test_sequences": len(test_y),
            "best_epoch": result["best_epoch"],
            "f1": float(evaluation["f1"]),
            "pr_auc": None if evaluation["pr_auc"] is None else float(evaluation["pr_auc"]),
            "precision": float(evaluation["precision"]),
            "recall": float(evaluation["recall"]),
            "fpr": fpr,
            "roc_auc": None if evaluation["roc_auc"] is None else float(evaluation["roc_auc"]),
            "threshold": 0.38,
        }
        fold_metrics.append(fold_result)
        pooled_labels.extend(test_y.astype(int).tolist())
        pooled_probabilities.extend(evaluation["probabilities"].tolist())
        prediction_rows.extend({
            "fold_id": fold_id,
            "held_out_episode_id": held_out,
            "attack_type": fold["attack_type"],
            "target": target_col,
            "timestamp": str(timestamp),
            "y_true": int(y_true),
            "y_probability": float(probability),
            "y_pred": int(prediction),
        } for fold_id, (timestamp, y_true, probability, prediction) in enumerate(zip(
            test_timestamps,
            test_y,
            evaluation["probabilities"],
            evaluation["predictions"],
        )))

    if not fold_metrics:
        raise RuntimeError("No LOEO fold produced train, validation, and held-out sequences.")
    pooled_labels = np.asarray(pooled_labels, dtype=np.float32)
    pooled_probabilities = np.asarray(pooled_probabilities, dtype=np.float32)
    pooled_predictions = (pooled_probabilities >= 0.38).astype(int)
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    aggregate = {
        "protocol": "LOEO (retrained per fold)",
        "target": target_col,
        "available_episodes": int(len(folds)),
        "completed_folds": int(len(fold_metrics)),
        "skipped_episodes": skipped_episodes,
        "f1_macro": float(np.mean([row["f1"] for row in fold_metrics])),
        "pr_auc_macro": float(np.nanmean([row["pr_auc"] if row["pr_auc"] is not None else np.nan for row in fold_metrics])),
        "precision_macro": float(np.mean([row["precision"] for row in fold_metrics])),
        "recall_macro": float(np.mean([row["recall"] for row in fold_metrics])),
        "roc_auc_pooled": None if len(np.unique(pooled_labels)) < 2 else float(roc_auc_score(pooled_labels, pooled_probabilities)),
        "f1_pooled": float(f1_score(pooled_labels, pooled_predictions, zero_division=0)),
        "pr_auc_pooled": float(average_precision_score(pooled_labels, pooled_probabilities)),
        "precision_pooled": float(precision_score(pooled_labels, pooled_predictions, zero_division=0)),
        "recall_pooled": float(recall_score(pooled_labels, pooled_predictions, zero_division=0)),
        "threshold": 0.38,
        "validated_fold_count": int(len(folds)),
        "note": "Eligibility is restricted to SSH-Bruteforce, DDOS-LOIC-UDP, and Botnet episode_id values; evaluation may skip folds that cannot form boundary-safe sequences after purge and embargo.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate["fpr_macro"] = float(np.mean([
        row["fpr"] for row in fold_metrics
    ])) if fold_metrics else None
    (output_dir / "metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (output_dir / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain and evaluate the direct UCS classifier under episode-wise LOEO.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/world_model_20260904/direct_head"))
    parser.add_argument("--target", choices=("label_binary", "future_attack_label"), default="future_attack_label")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir, target_col=args.target), indent=2))


if __name__ == "__main__":
    main()
