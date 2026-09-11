from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lstm.model import LSTMWorldModel
from lstm.trainer import train
from lstm.ucs import (
    UCSConfig,
    build_next_state_sequences,
    fit_lagged_linear_baseline,
    load_ucs_windows,
    purge_and_embargo,
    persistence_baseline,
    rollout_predictions,
    rollout_targets,
    validate_ucs_windows,
)
from lstm.utils import get_device, set_seed


def run(input_path: Path, output_dir: Path, config_path: Path | None = None) -> dict:
    set_seed(42)
    windows = load_ucs_windows(input_path)
    config = UCSConfig()
    features = validate_ucs_windows(windows, config=config)
    purged = purge_and_embargo(windows, config=config)
    sequences = build_next_state_sequences(purged, features=features, config=config)

    train_X = sequences["train"].X
    train_y = sequences["train"].y
    val_X = sequences["val"].X
    val_y = sequences["val"].y

    model = LSTMWorldModel(input_size=train_X.shape[-1], hidden_size=64, state_dim=train_y.shape[-1], dropout=0.2)
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = train(
        model,
        train_X,
        train_y,
        val_X,
        val_y,
        epochs=30,
        batch_size=64,
        learning_rate=0.001,
        seed=42,
        device=str(device),
        checkpoint_path=output_dir / "next_state_best.pt",
        weight_decay=0.0001,
        early_stopping_patience=5,
        early_stopping_min_delta=0.001,
    )

    def mse(pred, target):
        return float(np.mean((pred - target) ** 2))

    test_X = sequences["test"].X
    test_y = sequences["test"].y
    with torch.no_grad():
        model.eval()
        preds = model(torch.as_tensor(test_X, dtype=torch.float32)).cpu().numpy()
    with torch.no_grad():
        train_preds = model(torch.as_tensor(train_X, dtype=torch.float32)).cpu().numpy()
        val_preds = model(torch.as_tensor(val_X, dtype=torch.float32)).cpu().numpy()

    lagged_linear = fit_lagged_linear_baseline(sequences["train"])
    persistence_preds = persistence_baseline(test_X)
    lagged_preds = lagged_linear.predict(test_X[:, -1, :]).astype(np.float32)

    def regression_metrics(pred, target):
        error = np.asarray(pred) - np.asarray(target)
        return {
            "mse": float(np.mean(error ** 2)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
        }

    one_step_metrics = {
        "persistence": regression_metrics(persistence_preds, test_y),
        "lagged_linear": regression_metrics(lagged_preds, test_y),
        "lstm_world_model": regression_metrics(preds, test_y),
        "protocol": "chronological split",
        "representation": "UCS feature space supplied to the LSTM",
        "target": "S(t+1)",
    }
    metrics = {
        "task": "next-state prediction",
        "protocol": "chronological split",
        "train_mse": float(mse(train_preds, train_y)),
        "val_mse": float(mse(val_preds, val_y)),
        "test_mse": float(mse(preds, test_y)),
        "test_mae": float(np.mean(np.abs(preds - test_y))),
        "test_rmse": float(np.sqrt(np.mean((preds - test_y) ** 2))),
        "train_validation_gap": float(mse(train_preds, train_y) - mse(val_preds, val_y)),
        "feature_count": int(train_X.shape[-1]),
        "state_dim": int(train_y.shape[-1]),
        "one_step_comparison": one_step_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(result["history"], indent=2), encoding="utf-8")

    rollout_rows = []
    for k in (1, 2, 3):
        histories, targets, timestamps = rollout_targets(
            purged,
            features=features,
            config=config,
            k=k,
            split="test",
        )
        with torch.no_grad():
            model.eval()

            def model_predictor(history):
                return model(torch.as_tensor(history, dtype=torch.float32)).detach().cpu().numpy()

        learned = rollout_predictions(histories, model_predictor, k=k)
        persistence = np.repeat(histories[:, -1:, :], k, axis=1)
        linear = np.empty_like(learned)
        current = histories.copy()
        for step in range(k):
            prediction = lagged_linear.predict(current[:, -1, :]).astype(np.float32)
            linear[:, step, :] = prediction
            if step < k - 1:
                current = np.concatenate([current[:, 1:, :], prediction[:, None, :]], axis=1)
        for name, prediction in (("persistence", persistence), ("lagged_linear", linear), ("lstm_world_model", learned)):
            values = regression_metrics(prediction, targets)
            rollout_rows.append({
                "k": k,
                "model": name,
                "protocol": "chronological split",
                "representation": "UCS feature space supplied to the LSTM",
                "samples": int(len(histories)),
                **values,
            })
        if k == 3:
            prediction_rows = []
            for sample_index in range(len(histories)):
                for step in range(k):
                    prediction_rows.append({
                        "sample_index": sample_index,
                        "step": step + 1,
                        "target_timestamp": str(timestamps[sample_index][step]),
                        "lstm_prediction": learned[sample_index, step].tolist(),
                        "actual_state": targets[sample_index, step].tolist(),
                    })
            (output_dir / "rollout_k3_predictions.json").write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")
    (output_dir / "rollout_metrics.json").write_text(json.dumps({
        "protocol": "chronological split",
        "target": "S(t+1), ..., S(t+K)",
        "rollout": rollout_rows,
    }, indent=2), encoding="utf-8")
    metadata = {
        "model_type": "LSTMWorldModel",
        "encoder_configuration": {"lookback_windows": config.lookback_windows, "hidden_size": 64, "num_layers": 1, "dropout": 0.2},
        "input_dimensions": int(train_X.shape[-1]),
        "state_dimensions": int(train_y.shape[-1]),
        "lookback": config.lookback_windows,
        "protocol": "chronological split",
        "target": "S(t+1)",
        "task": "next-state prediction",
        "seed": 42,
        "loss": "MSE",
        "baselines": ["persistence", "lagged_linear"],
        "rollout_horizons": [1, 2, 3],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a next-state UCS world-model head.")
    parser.add_argument("input", type=Path, help="Path to ucs_windows.parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/world_model_20260904/next_state"))
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
