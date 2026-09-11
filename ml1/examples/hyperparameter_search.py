from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from lstm.model import LSTMClassifier
from lstm.trainer import train
from lstm.ucs import UCSConfig, build_lstm_sequences, load_ucs_windows, purge_and_embargo, validate_ucs_windows
from lstm.utils import get_device, set_seed


SEARCH_SPACE = {
    "hidden_size": (32, 64, 128),
    "learning_rate": (0.001, 0.0005),
    "dropout": (0.0, 0.2),
    "weight_decay": (0.0, 0.0001),
}


def validation_metrics(model, sequence_set, device):
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(
            model.forward_logits(torch.tensor(sequence_set.X, dtype=torch.float32, device=device))
        ).cpu().numpy()
    labels = sequence_set.y.astype(np.int64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "validation_pr_auc": float(average_precision_score(labels, probabilities)),
        "validation_roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "validation_f1_at_0_5": float(f1_score(labels, predictions, zero_division=0)),
    }


def run_search(input_path: Path, output_dir: Path, config_path: Path, epochs: int) -> dict:
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ucs_settings = settings.get("ucs", {})
    base_model = settings.get("model", {})
    base_training = settings.get("training", {})
    seed = int(settings.get("reproducibility", {}).get("seed", 42))
    config = UCSConfig(
        window_seconds=int(ucs_settings.get("window_seconds", 60)),
        lookback_windows=int(ucs_settings.get("lookback_windows", 30)),
        horizon_windows=int(ucs_settings.get("horizon_windows", 5)),
    )
    windows = load_ucs_windows(input_path)
    features = validate_ucs_windows(windows, config=config)
    purged = purge_and_embargo(windows, config=config)
    sequences = build_lstm_sequences(purged, features=features, config=config)
    train_set = sequences["train"]
    validation_set = sequences["val"]
    positive = int(train_set.y.sum())
    pos_weight = (len(train_set.y) - positive) / positive
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    combinations = itertools.product(*SEARCH_SPACE.values())
    for index, values in enumerate(combinations, start=1):
        params = dict(zip(SEARCH_SPACE, values))
        set_seed(seed)
        model = LSTMClassifier(
            input_size=len(features),
            hidden_size=params["hidden_size"],
            num_layers=int(base_model.get("num_layers", 1)),
            dropout=params["dropout"],
        )
        result = train(
            model,
            train_set.X,
            train_set.y,
            validation_set.X,
            validation_set.y,
            epochs=epochs,
            batch_size=int(base_training.get("batch_size", 64)),
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            pos_weight=pos_weight,
            early_stopping_patience=int(base_training.get("early_stopping", {}).get("patience", 5)),
            early_stopping_min_delta=float(base_training.get("early_stopping", {}).get("min_delta", 0.001)),
            seed=seed,
            device=str(device),
        )
        metrics = validation_metrics(model, validation_set, device)
        record = {
            "run": index,
            "parameters": params,
            **metrics,
            "best_epoch": result["best_epoch"],
            "stopped_epoch": result["stopped_epoch"],
            "best_validation_pr_auc": result["best_validation_metric"],
            "pos_weight": pos_weight,
        }
        results.append(record)
        print(json.dumps(record))

    results.sort(key=lambda item: (item["validation_pr_auc"], item["validation_f1_at_0_5"]), reverse=True)
    summary = {
        "search_space": SEARCH_SPACE,
        "runs": len(results),
        "selection": "validation_pr_auc_then_validation_f1_at_0_5",
        "feature_count": len(features),
        "train_sequences": len(train_set.y),
        "validation_sequences": len(validation_set.y),
        "results": results,
        "winner": results[0],
    }
    (output_dir / "search_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only LSTM UCS hyperparameter search.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/hyperparameter_search"))
    parser.add_argument("--config", type=Path, default=Path("configs/model_config.yaml"))
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    summary = run_search(args.input, args.output_dir, args.config, args.epochs)
    print(json.dumps(summary["winner"], indent=2))


if __name__ == "__main__":
    main()
