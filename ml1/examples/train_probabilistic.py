from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import os
from pathlib import Path

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from lstm.model import LSTMGaussianWorldModel
from lstm.probabilistic import (
    ablate_history,
    deviation_scores,
    feature_groups,
    gaussian_nll,
    gaussian_nll_per_sample,
    interval_coverage,
    prediction_interval,
    regression_metrics,
    sigma_diagnostics,
    summarize_nll,
    top_nll_outliers,
    train_gaussian,
)
from lstm.ucs import UCSConfig, apply_training_only_pca, build_next_state_sequences, fit_pca_lagged_linear_baseline, load_ucs_windows, pca_rollout_predictions, persistence_baseline, purge_and_embargo, rollout_targets, validate_ucs_windows
from lstm.utils import get_device, set_seed


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        mean, std = model(torch.as_tensor(X, dtype=torch.float32).to(device))
    return mean.cpu().numpy(), std.cpu().numpy()


def score_split(model, X, y, device):
    mean, std = predict(model, X, device)
    with torch.no_grad():
        nll = float(gaussian_nll(torch.as_tensor(mean), torch.as_tensor(std), torch.as_tensor(y)))
    result = regression_metrics(mean, y)
    result.update({
        "nll": nll,
        "mean_predicted_std": float(np.mean(std)),
        "interval_width_80": float(np.mean(2.0 * 1.2815515655446004 * std)),
        "interval_width_95": float(np.mean(2.0 * 1.959963984540054 * std)),
        "coverage_80_nominal": 0.80,
        "coverage_80_empirical": interval_coverage(mean, std, y, 0.80),
        "coverage_95_nominal": 0.95,
        "coverage_95_empirical": interval_coverage(mean, std, y, 0.95),
    })
    return result, mean, std


def write_nll_diagnostics(output_dir, predictions):
    diagnostic_summary = {}
    sigma_rows = []
    outlier_rows = []
    for split, values in predictions.items():
        nll = gaussian_nll_per_sample(values["mean"], values["std"], values["target"])
        summary = summarize_nll(nll)
        diagnostic_summary[split] = summary
        rows = []
        for index, timestamp in enumerate(values["timestamps"]):
            rows.append({
                "split": split,
                "index": index,
                "timestamp": str(timestamp),
                "nll": float(nll[index]),
                "prediction_error_mae": float(np.mean(np.abs(values["mean"][index] - values["target"][index]))),
                "target_magnitude": float(np.linalg.norm(values["target"][index])),
                "predicted_mean": json.dumps(values["mean"][index].tolist()),
                "predicted_std": json.dumps(values["std"][index].tolist()),
                "target": json.dumps(values["target"][index].tolist()),
            })
            sigma = values["std"][index]
            sigma_rows.append({
                "split": split,
                "index": index,
                "timestamp": str(timestamp),
                "sigma_min": float(np.min(sigma)),
                "sigma_median": float(np.median(sigma)),
                "sigma_mean": float(np.mean(sigma)),
                "sigma_max": float(np.max(sigma)),
                "count_below_1e-6": int(np.sum(sigma < 1e-6)),
                "count_below_1e-5": int(np.sum(sigma < 1e-5)),
                "count_below_1e-4": int(np.sum(sigma < 1e-4)),
            })
        pd.DataFrame(rows).to_csv(output_dir / f"{split}_nll_distribution.csv", index=False)
        if split == "train":
            for top_n in (10, 25, 50):
                for row in top_nll_outliers(nll, values["mean"], values["std"], values["target"], values["timestamps"], top_n):
                    row["top_n"] = top_n
                    row["predicted_mean"] = json.dumps(row["predicted_mean"])
                    row["predicted_std"] = json.dumps(row["predicted_std"])
                    row["target"] = json.dumps(row["target"])
                    outlier_rows.append(row)
    pd.DataFrame(sigma_rows).to_csv(output_dir / "sigma_diagnostics.csv", index=False)
    pd.DataFrame(outlier_rows).to_csv(output_dir / "nll_outliers.csv", index=False)
    (output_dir / "nll_diagnostics.json").write_text(json.dumps(diagnostic_summary, indent=2), encoding="utf-8")
    return diagnostic_summary


def write_rollout_plot(output_dir, rollout_rows):
    """Write the chronological rollout RMSE figure from saved metric rows."""
    import matplotlib.pyplot as plt

    plt.figure()
    for model_name in ("persistence", "lagged_linear_pca", "lstm_gaussian"):
        rows = [row for row in rollout_rows if row.get("model") == model_name]
        plt.plot([row["k"] for row in rows], [row["rmse"] for row in rows], marker="o", label=model_name)
    plt.xlabel("Rollout horizon K")
    plt.ylabel("RMSE")
    plt.title("UCS rollout error: chronological split; PCA lagged-LR")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "rollout_error_vs_k.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def run(input_path: Path, output_dir: Path, config_path: Path | None = None) -> dict:
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path and config_path.exists() else {}
    world_settings = settings.get("world_model", {})
    model_settings = settings.get("model", {})
    training_settings = settings.get("training", {})
    early_settings = training_settings.get("early_stopping", {})
    rollout_horizons = [int(value) for value in world_settings.get("rollout", {}).get("horizons", [1, 2, 3])]
    interval_levels = [float(value) for value in world_settings.get("intervals", {}).get("levels", [0.80, 0.95])]
    seed = int(world_settings.get("seed", 42))
    set_seed(seed)
    config = UCSConfig()
    windows = load_ucs_windows(input_path)
    features = validate_ucs_windows(windows, config=config)
    purged = purge_and_embargo(windows, config=config)
    sequences = build_next_state_sequences(purged, features=features, config=config)
    train_set, val_set, test_set = sequences["train"], sequences["val"], sequences["test"]
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = LSTMGaussianWorldModel(input_size=train_set.X.shape[-1], hidden_size=int(model_settings.get("hidden_size", 64)), state_dim=train_set.y.shape[-1], num_layers=int(model_settings.get("num_layers", 1)), dropout=float(model_settings.get("dropout", 0.2)))
    training = train_gaussian(
        model,
        train_set.X,
        train_set.y,
        val_set.X,
        val_set.y,
        epochs=int(training_settings.get("epochs", 30)),
        batch_size=int(training_settings.get("batch_size", 64)),
        learning_rate=float(training_settings.get("learning_rate", 0.001)),
        weight_decay=float(training_settings.get("weight_decay", 0.0001)),
        patience=int(early_settings.get("patience", 5)),
        min_delta=float(early_settings.get("min_delta", 0.001)),
        seed=seed,
        device=str(device),
        checkpoint_path=output_dir / "gaussian_next_state_best.pt",
    )
    split_results = {}
    predictions = {}
    for name, sequence_set in (("train", train_set), ("validation", val_set), ("test", test_set)):
        split_results[name], mean, std = score_split(model, sequence_set.X, sequence_set.y, device)
        predictions[name] = {"mean": mean, "std": std, "target": sequence_set.y, "timestamps": sequence_set.timestamps}
    nll_diagnostics = write_nll_diagnostics(output_dir, predictions)
    test_mean, test_std = predictions["test"]["mean"], predictions["test"]["std"]
    pca_components = 128
    pca_frame, pca = apply_training_only_pca(purged, features, pca_components, random_state=seed)
    pca_features = [f"pca_{index}" for index in range(pca_components)]
    pca_sequences = build_next_state_sequences(pca_frame, features=pca_features, config=config)
    lagged_linear = fit_pca_lagged_linear_baseline(pca_sequences["train"], pca)
    pca_test_last = pca.pca.transform(test_set.X[:, -1, :])
    pca_one_step = pca.pca.inverse_transform(lagged_linear.predict(pca_test_last)).astype(np.float32)
    one_step_baselines = {
        "persistence": regression_metrics(persistence_baseline(test_set.X), test_set.y),
        "lagged_linear_pca": regression_metrics(pca_one_step, test_set.y),
        "lstm_gaussian_mean": regression_metrics(test_mean, test_set.y),
        "lagged_linear_representation": "PCA-reduced UCS",
        "pca_components": pca_components,
        "pca_fit": "purged training windows only",
        "protocol": "chronological split",
    }
    groups = feature_groups(features)
    attribution = []
    deletion = []
    full_test = split_results["test"]
    base_mean, _ = predict(model, test_set.X, device)
    for group, indices in sorted(groups.items()):
        masked_X = ablate_history(test_set.X, indices)
        masked_mean, masked_std = predict(model, masked_X, device)
        attribution.append({
            "feature_group": group,
            "feature_count": len(indices),
            "mean_output_change": float(np.mean(np.abs(base_mean - masked_mean))),
            "mean_uncertainty_change": float(np.mean(np.abs(test_std - masked_std))),
            "protocol": "chronological split",
            "masking": "zero standardized/input representation features",
        })
        masked_metrics = regression_metrics(masked_mean, test_set.y)
        deletion.append({
            "feature_group": group,
            "mae": masked_metrics["mae"],
            "rmse": masked_metrics["rmse"],
            "delta_mae": masked_metrics["mae"] - full_test["mae"],
            "delta_rmse": masked_metrics["rmse"] - full_test["rmse"],
            "protocol": "chronological split",
        })
    histories, targets, timestamps = rollout_targets(purged, features=features, config=config, k=max(rollout_horizons), split="test")
    def predictor(history):
        mean, _ = predict(model, history, device)
        return mean
    rollout_rows = []
    rollout_predictions = []
    current = histories.copy()
    persistence_history = histories.copy()
    linear_rollout = pca_rollout_predictions(histories, lagged_linear, pca, k=max(rollout_horizons))
    for step in range(max(rollout_horizons)):
        mean, std = predict(model, current, device)
        actual = targets[:, step, :]
        metrics = regression_metrics(mean, actual)
        metrics.update({"k": step + 1, "model": "lstm_gaussian", "protocol": "chronological split", "mean_predicted_std": float(np.mean(std)), "coverage_80_empirical": interval_coverage(mean, std, actual, 0.80), "coverage_95_empirical": interval_coverage(mean, std, actual, 0.95)})
        rollout_rows.append(metrics)
        persistence_metrics = regression_metrics(persistence_history[:, -1, :], actual)
        rollout_rows.append({"k": step + 1, "model": "persistence", "protocol": "chronological split", **persistence_metrics})
        linear_mean = linear_rollout[:, step, :]
        linear_metrics = regression_metrics(linear_mean, actual)
        rollout_rows.append({"k": step + 1, "model": "lagged_linear_pca", "protocol": "chronological split", **linear_metrics})
        rollout_predictions.append({"step": step + 1, "mean": mean, "std": std, "actual": actual, "timestamps": [str(value[step]) for value in timestamps]})
        current = np.concatenate([current[:, 1:, :], mean[:, None, :]], axis=1)
        persistence_history = np.concatenate([persistence_history[:, 1:, :], persistence_history[:, -1:, :]], axis=1)
    (output_dir / "rollout_error_curve.csv").write_text(
        "k,model,rmse,mae,mse,protocol\n" + "\n".join(
            f"{row['k']},{row.get('model', 'lstm_gaussian')},{row['rmse']},{row['mae']},{row['mse']},chronological split"
            for row in rollout_rows
        ) + "\n",
        encoding="utf-8",
    )
    write_rollout_plot(output_dir, rollout_rows)
    test_lower80, test_upper80 = prediction_interval(test_mean, test_std, 0.80)
    test_lower95, test_upper95 = prediction_interval(test_mean, test_std, 0.95)
    one_step_rows = []
    for index in range(len(test_set.y)):
        one_step_rows.append({
            "index": index,
            "timestamp": str(test_set.timestamps[index]),
            "actual_state": test_set.y[index].tolist(),
            "predicted_mean": test_mean[index].tolist(),
            "predicted_std": test_std[index].tolist(),
            "lower_80": test_lower80[index].tolist(),
            "upper_80": test_upper80[index].tolist(),
            "lower_95": test_lower95[index].tolist(),
            "upper_95": test_upper95[index].tolist(),
            "error": (test_mean[index] - test_set.y[index]).tolist(),
            "deviation_mae": float(deviation_scores(test_mean[index:index + 1], test_set.y[index:index + 1])[0]),
        })
    deviation = deviation_scores(test_mean, test_set.y)
    test_labels = (
        purged.loc[purged["split"] == "test", ["window_start_utc", "future_attack_label"]]
        .assign(window_start_utc=lambda frame: pd.to_datetime(frame["window_start_utc"], utc=True))
        .set_index("window_start_utc")
        .reindex(test_set.timestamps)["future_attack_label"]
        .to_numpy(dtype=np.float32)
    )
    if np.isnan(test_labels).any():
        raise ValueError("Chronological deviation labels are not aligned to test predictions.")
    deviation_rows = pd.DataFrame({
        "timestamp": [str(value) for value in test_set.timestamps],
        "deviation_score": deviation,
        "future_attack_label": test_labels.astype(int),
        "protocol": "chronological split",
    })
    deviation_rows.to_csv(output_dir / "deviation_predictions.csv", index=False)
    report = {
        "protocol": "chronological split",
        "representation": "UCS feature space supplied to the LSTM",
        "target": "S(t+1)",
        "model": model.get_config(),
        "train_count": len(train_set.y),
        "validation_count": len(val_set.y),
        "test_count": len(test_set.y),
        "best_epoch": training["best_epoch"],
        "best_validation_nll": training["best_validation_nll"],
        "train_validation_nll_gap": split_results["train"]["nll"] - split_results["validation"]["nll"],
        "nll_diagnostics": nll_diagnostics,
        "splits": split_results,
        "one_step_baselines": one_step_baselines,
        "rollout": [row for row in rollout_rows if row["k"] in rollout_horizons],
        "deviation": {"score": "mean absolute error of S(t+1) versus predicted mean", "mean": float(np.mean(deviation)), "median": float(np.median(deviation)), "p95": float(np.percentile(deviation, 95)), "roc_auc": float(roc_auc_score(test_labels, deviation)), "pr_auc": float(average_precision_score(test_labels, deviation)), "protocol": "chronological split"},
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(training["history"], indent=2), encoding="utf-8")
    (output_dir / "one_step_predictions.json").write_text(json.dumps(one_step_rows, indent=2), encoding="utf-8")
    (output_dir / "rollout_predictions.json").write_text(json.dumps([{**row, "mean": row["mean"].tolist(), "std": row["std"].tolist(), "actual": row["actual"].tolist()} for row in rollout_predictions], indent=2), encoding="utf-8")
    (output_dir / "attribution.json").write_text(json.dumps(attribution, indent=2), encoding="utf-8")
    (output_dir / "deletion.json").write_text(json.dumps(deletion, indent=2), encoding="utf-8")
    try:
        repository_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        repository_commit = None
    metadata = {"experiment": "probabilistic_ucs_transition", "protocol": "chronological split", "seed": seed, "device": str(device), "optimizer": training_settings.get("optimizer", "Adam"), "learning_rate": float(training_settings.get("learning_rate", 0.001)), "weight_decay": float(training_settings.get("weight_decay", 0.0001)), "dropout": float(model_settings.get("dropout", 0.2)), "batch_size": int(training_settings.get("batch_size", 64)), "epochs": int(training_settings.get("epochs", 30)), "early_stopping": {"patience": int(early_settings.get("patience", 5)), "min_delta": float(early_settings.get("min_delta", 0.001))}, "lookback": config.lookback_windows, "state_dimension": int(train_set.y.shape[-1]), "input_dimension": int(train_set.X.shape[-1]), "features": features, "pca": pca.get_metadata(), "pca_fit": "purged training windows only", "lagged_linear": {"representation": "PCA-reduced UCS", "components": pca_components, "recursive": True}, "normalization": "as supplied by canonical UCS artifact", "target": "S(t+1)", "rollout_horizons": rollout_horizons, "interval_levels": interval_levels, "future_episode_id": "unavailable; source_day not used as episode ID", "repository_commit": repository_commit, "software": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": __import__("sklearn").__version__, "matplotlib": __import__("matplotlib").__version__}}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "reproducibility_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if config_path and config_path.exists():
        (output_dir / "config_snapshot.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "attribution_deletion.json").write_text(json.dumps({"protocol": "chronological split", "attribution": attribution, "deletion": deletion}, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a probabilistic UCS transition model.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/world_model_20260904/probabilistic"))
    parser.add_argument("--config", type=Path, default=Path("configs/model_config.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir, args.config), indent=2))


if __name__ == "__main__":
    main()
