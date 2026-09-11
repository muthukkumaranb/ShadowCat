from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_DIR))

from lstm.model import LSTMClassifier
from lstm.trainer import train
from lstm.ucs import UCSConfig, SequenceSet, apply_training_only_pca, build_lstm_sequences, purge_and_embargo, validate_ucs_windows
from lstm.utils import get_device


THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.38, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
PCA_COMPONENTS = 32
EPOCHS = 30
SEED = 42


def metric_row(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, _, _ = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "fpr": float(fp / (tn + fp)) if tn + fp else 0.0,
        "predicted_positive": int(predictions.sum()),
        "predicted_negative": int(len(predictions) - predictions.sum()),
    }


def safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float | None, float | None]:
    if len(np.unique(y_true)) != 2:
        return None, None
    return float(roc_auc_score(y_true, probabilities)), float(average_precision_score(y_true, probabilities))


def checkpoint_matches_model(model: torch.nn.Module, checkpoint_path: Path) -> bool:
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        return False
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    checkpoint_state = checkpoint.get("model_state_dict")
    if not isinstance(checkpoint_state, dict):
        return False
    model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return model_state.keys() == checkpoint_state.keys() and all(
        torch.equal(model_state[key], checkpoint_state[key].detach().cpu()) for key in model_state
    )


def write_artifacts(out_dir: Path, tables: dict[str, list[dict]]) -> None:
    names = {
        "training": "training_completion_37fold.csv",
        "probability": "probability_diagnostics_37fold.csv",
        "auc": "auc_prauc_37fold.csv",
        "threshold": "threshold_diagnostics_37fold.csv",
        "loss": "train_validation_loss_37fold.csv",
        "balance": "class_balance_37fold.csv",
        "pca": "pca_diagnostics_37fold.csv",
        "pca_numeric": "pca_numerical_diagnostics_37fold.csv",
        "predictions": "lstm_loeo_probability_diagnostics.csv",
        "summary": "diagnostic_summary_37fold.csv",
    }
    for key, filename in names.items():
        pd.DataFrame(tables[key]).to_csv(out_dir / filename, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-preserving 37-fold LSTM+UCS LOEO diagnosis.")
    parser.add_argument("--data", type=Path, default=WORKSPACE_DIR / "data/ucs/ucs_windows.parquet")
    parser.add_argument("--manifest", type=Path, default=WORKSPACE_DIR / "artifacts/loeo/corrected_37fold_manifest.json")
    parser.add_argument("--feature-manifest", type=Path, default=WORKSPACE_DIR / "artifacts/lr_set_a/set_a_features.json")
    parser.add_argument("--out-dir", type=Path, default=WORKSPACE_DIR / "artifacts/lstm_diagnostics")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--folds", type=int, nargs="*", help="Optional fold ids for a smoke run; default is all folds.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.data)
    df["window_start_utc"] = pd.to_datetime(df["window_start_utc"], utc=True)
    df = df.sort_values("window_start_utc").reset_index(drop=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    feature_manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    base_features = list(feature_manifest["ordered_feature_names"])
    config = UCSConfig()
    validate_ucs_windows(df, features=base_features, config=config)
    device = get_device()
    folds = [fold for fold in manifest["folds"] if args.folds is None or fold["fold_id"] in args.folds]
    tables: dict[str, list[dict]] = {key: [] for key in ("training", "probability", "auc", "threshold", "loss", "balance", "pca", "pca_numeric", "predictions", "summary")}

    for position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        held_out_episode = fold["held_out_episode_id"]
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        test_indices = np.asarray(fold["test_indices"], dtype=int)
        checkpoint_path = args.out_dir / "checkpoints" / f"fold_{fold_id}" / "classifier_best.pt"
        completion = {
            "fold_id": fold_id, "training_started": False, "training_completed": False,
            "epochs_requested": args.epochs, "epochs_completed": 0, "early_stopping_triggered": False,
            "training_exception": "", "diagnostic_exception": "", "checkpoint_created": False, "best_checkpoint_available": False,
            "best_epoch": None, "final_epoch": None,
        }
        print(f"Fold {position}/{len(folds)}: {fold_id} {held_out_episode}", flush=True)

        try:
            fold_frame = df.copy()
            fold_frame["split"] = "none"
            fold_frame.loc[train_indices, "split"] = "train"
            fold_frame.loc[test_indices, "split"] = "test"
            train_rows = fold_frame.loc[train_indices].sort_values("window_start_utc")
            val_start = int(len(train_rows) * 0.8)
            fold_frame.loc[train_rows.index[val_start:], "split"] = "val"
            active_mask = fold_frame["split"].isin(["train", "val", "test"])
            purged = purge_and_embargo(fold_frame[active_mask].copy(), config=config)
            transformed, pca = apply_training_only_pca(purged, base_features, PCA_COMPONENTS, random_state=SEED)
            pca_features = [f"pca_{index}" for index in range(PCA_COMPONENTS)]
            sequences = build_lstm_sequences(transformed, features=pca_features, config=config, target_col="label_binary")
            train_set, val_set = sequences["train"], sequences["val"]
            full_transformed = pca.transform(fold_frame)
            full_values = full_transformed[pca_features].to_numpy(dtype=np.float32)
            positions = {index: position for position, index in enumerate(fold_frame.index)}
            test_histories, test_labels, test_timestamps = [], [], []
            for index in test_indices:
                position = positions[index]
                start = position - config.lookback_windows + 1
                if start < 0:
                    continue
                history = full_values[start:position + 1]
                if len(history) == config.lookback_windows:
                    test_histories.append(history)
                    test_labels.append(int(fold_frame.loc[index, "label_binary"]))
                    test_timestamps.append(fold_frame.loc[index, "window_start_utc"])
            test_set = SequenceSet(
                X=np.asarray(test_histories, dtype=np.float32),
                y=np.asarray(test_labels, dtype=np.float32),
                timestamps=pd.DatetimeIndex(test_timestamps, tz="UTC"),
            )

            transformed_values = transformed[pca_features].to_numpy(dtype=np.float64)
            pca_metadata = pca.get_metadata()
            tables["pca"].append({
                "fold_id": fold_id, "input_feature_count": len(base_features), "pca_component_count": PCA_COMPONENTS,
                "explained_variance_ratio": json.dumps(pca_metadata["explained_variance_ratio"]),
                "explained_variance_ratio_sum": float(np.sum(pca.pca.explained_variance_ratio_)),
                "cumulative_explained_variance": float(np.sum(pca.pca.explained_variance_ratio_)),
                "component_mean": json.dumps(transformed_values.mean(axis=0).tolist()),
                "component_std": json.dumps(transformed_values.std(axis=0).tolist()),
            })
            tables["pca_numeric"].append({
                "fold_id": fold_id, "min": float(np.min(transformed_values)), "max": float(np.max(transformed_values)),
                "mean": float(np.mean(transformed_values)), "std": float(np.std(transformed_values)),
                "nan_count": int(np.isnan(transformed_values).sum()), "inf_count": int(np.isinf(transformed_values).sum()),
                "near_zero_variance_components": int(np.sum(np.var(transformed_values, axis=0) < 1e-12)),
            })
            for split, sequence_set in (("train", train_set), ("validation", val_set), ("test", test_set)):
                labels = sequence_set.y.astype(int)
                tables["balance"].append({
                    "fold_id": fold_id, "split": split, "positive_count": int(labels.sum()),
                    "negative_count": int(len(labels) - labels.sum()), "sequence_count": int(len(labels)),
                    "positive_prevalence": float(labels.mean()) if len(labels) else None,
                })

            model = LSTMClassifier(input_size=PCA_COMPONENTS, hidden_size=64, num_layers=1, dropout=0.20)
            completion["training_started"] = True
            result = train(
                model, train_set.X, train_set.y, val_set.X, val_set.y,
                epochs=args.epochs, batch_size=64, learning_rate=0.001, seed=SEED,
                device=str(device), checkpoint_path=checkpoint_path, weight_decay=0.0001,
                pos_weight=None, early_stopping_patience=5, early_stopping_min_delta=0.001,
            )
            history = result["history"]
            epochs_completed = len(history["train_loss"])
            checkpoint_valid = checkpoint_matches_model(result["model"], checkpoint_path)
            completion.update({
                "training_completed": epochs_completed == result["stopped_epoch"] and result["best_epoch"] > 0 and checkpoint_valid,
                "epochs_completed": epochs_completed, "early_stopping_triggered": result["stopped_epoch"] < args.epochs,
                "checkpoint_created": checkpoint_path.is_file(), "best_checkpoint_available": checkpoint_valid,
                "best_epoch": int(result["best_epoch"]), "final_epoch": int(result["stopped_epoch"]),
            })
            best_index = result["best_epoch"] - 1
            tables["loss"].append({
                "fold_id": fold_id, "initial_train_loss": float(history["train_loss"][0]),
                "best_train_loss": float(np.min(history["train_loss"])), "final_train_loss": float(history["train_loss"][-1]),
                "initial_validation_loss": float(history["validation_loss"][0]), "best_validation_loss": float(np.min(history["validation_loss"])),
                "final_validation_loss": float(history["validation_loss"][-1]), "best_epoch": int(result["best_epoch"]),
                "epochs_completed": epochs_completed,
                "train_validation_gap_at_best": float(history["validation_loss"][best_index] - history["train_loss"][best_index]),
                "pos_weight_runtime": result["pos_weight"],
            })

            model = result["model"].eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model.forward_logits(torch.as_tensor(val_set.X, dtype=torch.float32, device=device))).cpu().numpy().reshape(-1)
                test_probs = torch.sigmoid(model.forward_logits(torch.as_tensor(test_set.X, dtype=torch.float32, device=device))).cpu().numpy().reshape(-1)
            val_y, test_y = val_set.y.astype(int), test_set.y.astype(int)
            threshold_rows = []
            for threshold in THRESHOLDS:
                validation_metrics = metric_row(val_y, val_probs, threshold)
                test_metrics = metric_row(test_y, test_probs, threshold)
                threshold_rows.append({
                    "fold_id": fold_id, "threshold": threshold, "validation_f1": validation_metrics["f1"],
                    "test_f1": test_metrics["f1"], "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"], "test_fpr": test_metrics["fpr"],
                    "selected_on_validation": False,
                })
            selected_index = max(range(len(threshold_rows)), key=lambda index: (threshold_rows[index]["validation_f1"], -index))
            threshold_rows[selected_index]["selected_on_validation"] = True
            selected_threshold = float(threshold_rows[selected_index]["threshold"])
            tables["threshold"].extend(threshold_rows)
            selected_metrics = metric_row(test_y, test_probs, selected_threshold)
            roc_auc, pr_auc = safe_auc(test_y, test_probs)
            tables["auc"].append({
                "fold_id": fold_id, "completed": completion["training_completed"], "roc_auc": roc_auc, "pr_auc": pr_auc,
                "f1": selected_metrics["f1"], "precision": selected_metrics["precision"], "recall": selected_metrics["recall"],
                "fpr": selected_metrics["fpr"], "validation_selected_threshold": selected_threshold,
                "fixed_038_f1": metric_row(test_y, test_probs, 0.38)["f1"],
            })
            positive_probs, negative_probs = test_probs[test_y == 1], test_probs[test_y == 0]
            tables["probability"].append({
                "fold_id": fold_id, "min_probability": float(test_probs.min()), "max_probability": float(test_probs.max()),
                "mean_probability": float(test_probs.mean()), "median_probability": float(np.median(test_probs)),
                "std_probability": float(test_probs.std()), "positive_probability_mean": float(positive_probs.mean()) if len(positive_probs) else None,
                "negative_probability_mean": float(negative_probs.mean()) if len(negative_probs) else None,
                "positive_probability_median": float(np.median(positive_probs)) if len(positive_probs) else None,
                "negative_probability_median": float(np.median(negative_probs)) if len(negative_probs) else None,
                "number_predicted_positive_at_threshold_0.38": int((test_probs >= 0.38).sum()),
                "number_predicted_negative_at_threshold_0.38": int((test_probs < 0.38).sum()),
            })
            for sequence_index, (timestamp, label, probability) in enumerate(zip(test_set.timestamps, test_y, test_probs)):
                tables["predictions"].append({
                    "fold_id": fold_id, "sample_id": f"{fold_id}:{sequence_index}", "timestamp": pd.Timestamp(timestamp).isoformat(),
                    "true_label": int(label), "predicted_probability": float(probability), "threshold_prediction": int(probability >= 0.38),
                })
            print(f"  completed={completion['training_completed']} epochs={epochs_completed} best_epoch={result['best_epoch']} checkpoint_valid={checkpoint_valid} auc={roc_auc}", flush=True)
        except Exception as error:
            error_field = "training_exception" if completion["training_started"] and completion["epochs_completed"] == 0 else "diagnostic_exception"
            completion[error_field] = f"{type(error).__name__}: {error}"
            print(f"  ERROR: {completion[error_field]}", flush=True)
            traceback.print_exc()
        tables["training"].append(completion)
        write_artifacts(args.out_dir, tables)

    training_frame = pd.DataFrame(tables["training"])
    total = len(folds)
    completed_count = int(training_frame["training_completed"].sum()) if not training_frame.empty else 0
    failed_count = int(training_frame["training_exception"].ne("").sum()) if not training_frame.empty else 0
    tables["summary"] = [
        {"summary": "all-run folds", "total_folds": total, "fully_completed_folds": completed_count, "incomplete_folds": total - completed_count, "failed_folds": failed_count},
        {"summary": "fully completed folds", "total_folds": completed_count, "fully_completed_folds": completed_count, "incomplete_folds": 0, "failed_folds": 0},
    ]
    write_artifacts(args.out_dir, tables)
    metadata = {
        "command": " ".join(sys.argv), "data": str(args.data), "manifest": str(args.manifest), "feature_manifest": str(args.feature_manifest),
        "fold_count_requested": total, "epochs_requested": args.epochs, "seed": SEED, "device": str(device),
        "class_weighting": False, "pos_weight_runtime": None, "pca_components": PCA_COMPONENTS,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"SUMMARY total={total} fully_completed={completed_count} incomplete={total - completed_count} failed={failed_count}", flush=True)


if __name__ == "__main__":
    main()