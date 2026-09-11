from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
import numpy as np
import torch
import joblib

from lstm.evaluator import evaluate
from lstm.model import LSTMClassifier
from lstm.trainer import train
from lstm.ucs import (
    UCSConfig,
    build_lstm_sequences,
    load_ucs_windows,
    prepare_lstm_inputs,
    purge_and_embargo,
    validate_ucs_windows,
    apply_training_only_pca,
)
from lstm.utils import get_device, save_model, set_seed


def run(input_path: Path, output_dir: Path, config_path: Path, pca_components: int | None = None) -> dict:
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ucs_settings = settings.get("ucs", {})
    model_settings = settings.get("model", {})
    training_settings = settings.get("training", {})
    evaluation_settings = settings.get("evaluation", {})
    config = UCSConfig(
        window_seconds=int(ucs_settings.get("window_seconds", 60)),
        lookback_windows=int(ucs_settings.get("lookback_windows", 30)),
        horizon_windows=int(ucs_settings.get("horizon_windows", 5)),
        train_ratio=float(ucs_settings.get("train_ratio", 0.70)),
        val_ratio=float(ucs_settings.get("val_ratio", 0.15)),
        test_ratio=float(ucs_settings.get("test_ratio", 0.15)),
    )
    seed = int(settings.get("reproducibility", {}).get("seed", 42))
    set_seed(seed)

    windows = load_ucs_windows(input_path)
    raw_window_counts = windows["split"].value_counts().to_dict()
    pca = None
    scaler_params_path = input_path.parent / "scaler_params.yaml"
    if scaler_params_path.exists():
        # The canonical UCS producer has already applied these parameters.
        features = validate_ucs_windows(windows, config=config)
        purged = purge_and_embargo(windows, config=config)
        if pca_components is not None:
            purged, pca = apply_training_only_pca(purged, features, pca_components, random_state=seed)
            pca_metadata = pca.get_metadata()
            features = [f"pca_{index}" for index in range(pca_components)]
        else:
            pca_metadata = None
        sets = build_lstm_sequences(purged, features=features, config=config)
        scaler = None
        scaler_metadata = yaml.safe_load(scaler_params_path.read_text(encoding="utf-8"))
    else:
        sets, features, scaler, _ = prepare_lstm_inputs(windows, config=config)
        scaler_metadata = scaler.get_params()
        purged = None
        pca_metadata = None
    model = LSTMClassifier(
        input_size=len(features),
        hidden_size=int(model_settings.get("hidden_size", 64)),
        num_layers=int(model_settings.get("num_layers", 1)),
        dropout=float(model_settings.get("dropout", 0.2)),
    )
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    if pca_components is not None:
        joblib.dump(pca, output_dir / "preprocessing_pca.joblib")
    checkpoint = output_dir / "lstm_ucs_best.pt"
    train_positive = int(sets["train"].y.sum())
    train_negative = len(sets["train"].y) - train_positive
    use_class_weight = bool(training_settings.get("class_weighting", False))
    pos_weight = train_negative / train_positive if use_class_weight and train_positive else None
    early_settings = training_settings.get("early_stopping", {})
    patience = int(early_settings.get("patience", 5)) if early_settings.get("enabled", False) else None
    result = train(
        model,
        sets["train"].X,
        sets["train"].y,
        sets["val"].X,
        sets["val"].y,
        epochs=int(training_settings.get("epochs", 30)),
        batch_size=int(training_settings.get("batch_size", 64)),
        learning_rate=float(training_settings.get("learning_rate", 0.001)),
        seed=seed,
        device=str(device),
        checkpoint_path=checkpoint,
        weight_decay=float(training_settings.get("weight_decay", 0.0)),
        pos_weight=pos_weight,
        early_stopping_patience=patience,
        early_stopping_min_delta=float(early_settings.get("min_delta", 0.001)),
    )
    threshold_mode = evaluation_settings.get("threshold", {}).get("mode", "fixed")
    threshold = float(ucs_settings.get("threshold", 0.5))
    validation_scores = {}
    if threshold_mode == "validation_f1":
        for candidate in np.linspace(0.05, 0.95, 91):
            score = evaluate(model, sets["val"].X, sets["val"].y, threshold=float(candidate), device=device)["f1"]
            validation_scores[float(candidate)] = score
        threshold = max(validation_scores, key=validation_scores.get)
    elif threshold_mode != "fixed":
        raise ValueError("threshold mode must be fixed or validation_f1")
    evaluation = evaluate(model, sets["test"].X, sets["test"].y, threshold=threshold, device=device)
    save_model(model, checkpoint, metadata={
        "dataset": "CSE-CIC-IDS2018",
        "pipeline": "UCS",
        "pca": pca_metadata,
        "features": features,
        "sequence_length": config.lookback_windows,
        "forecast_horizon": config.horizon_windows,
        "threshold": threshold,
        "pos_weight": pos_weight,
        "model_config": model.get_config(),
        "training_config": training_settings,
        "validation_threshold_scores": validation_scores,
        "seed": seed,
        "device": str(device),
        "scaler": scaler_metadata,
    })

    predictions = pd.DataFrame({
        "timestamp": sets["test"].timestamps,
        "true_label": sets["test"].y.astype(int),
        "probability": evaluation["probabilities"],
        "predicted_label": evaluation["predictions"],
    })
    predictions.to_csv(output_dir / "lstm_predictions.csv", index=False)
    metrics = {key: value for key, value in evaluation.items() if key in {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold"}}
    metrics["confusion_matrix"] = evaluation["confusion_matrix"].tolist()
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = evaluation["confusion_matrix"].ravel().tolist()
    (output_dir / "lstm_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(result["history"], indent=2), encoding="utf-8")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        epochs = range(1, len(result["history"]["train_loss"]) + 1)
        for name, series, ylabel in (
            ("loss", (result["history"]["train_loss"], result["history"]["validation_loss"]), "Loss"),
            ("f1", (result["history"]["validation_f1"],), "Validation F1"),
            ("pr_auc", (result["history"]["validation_pr_auc"],), "Validation PR-AUC"),
            ("recall", (result["history"]["validation_recall"],), "Validation Recall"),
        ):
            plt.figure()
            for values, label in zip(series, ("train", "validation")):
                plt.plot(epochs, values, label=label)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / f"{name}.png")
            plt.close()
    except ImportError:
        pass
    probabilities = evaluation["probabilities"]
    diagnostics = {
        "test_probability_min": float(probabilities.min()),
        "test_probability_max": float(probabilities.max()),
        "test_probability_mean": float(probabilities.mean()),
        "test_probability_median": float(np.median(probabilities)),
        "test_positive_predictions": int(evaluation["predictions"].sum()),
        "test_negative_predictions": int(len(evaluation["predictions"]) - evaluation["predictions"].sum()),
        "test_roc_auc": evaluation["roc_auc"],
        "test_inverted_roc_auc": None if evaluation["roc_auc"] is None else float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(sets["test"].y, 1.0 - probabilities)),
        "train_positive": train_positive,
        "train_negative": train_negative,
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    metadata = {
        "dataset": "CSE-CIC-IDS2018",
        "model": "LSTM",
        "pipeline": "UCS",
        "window_seconds": config.window_seconds,
        "lookback_windows": config.lookback_windows,
        "forecast_horizon": config.horizon_windows,
        "feature_count": len(features),
        "pca_enabled": pca_components is not None,
        "pca_components": pca_components,
        "pca": pca_metadata,
        "feature_names": features,
        "target": "future_attack_label",
        "train_sequences": len(sets["train"].y),
        "validation_sequences": len(sets["val"].y),
        "test_sequences": len(sets["test"].y),
        "raw_window_counts": {key: int(value) for key, value in raw_window_counts.items()},
        "purged_window_counts": {
            split: int((purged["split"] == split).sum())
            for split in ("train", "val", "test")
        } if purged is not None else None,
        "sequence_label_counts": {
            split: {str(int(label)): int(count) for label, count in zip(*np.unique(sets[split].y, return_counts=True))}
            for split in ("train", "val", "test")
        },
        "best_epoch": result["best_epoch"],
        "stopped_epoch": result["stopped_epoch"],
        "best_validation_pr_auc": result["best_validation_metric"],
        "best_validation_loss": result["best_validation_loss"],
        "pos_weight": pos_weight,
        "weight_decay": float(training_settings.get("weight_decay", 0.0)),
        "threshold_mode": threshold_mode,
        "validation_f1_at_selected_threshold": validation_scores.get(threshold),
        "python_version": __import__("platform").python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "seed": seed,
        "device": str(device),
        "threshold": threshold,
        "checkpoint": str(checkpoint),
    }
    (output_dir / "lstm_ucs_experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM on canonical UCS windows.")
    parser.add_argument("input", type=Path, help="Canonical ucs_windows.parquet or CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ucs_experiment"))
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/model_config.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir, args.config, args.pca_components), indent=2))


if __name__ == "__main__":
    main()
