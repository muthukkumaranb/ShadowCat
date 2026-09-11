"""Real Set-A Logistic Regression Baseline under Canonical 37-Fold LOEO Protocol."""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Ensure we can import lstm package
workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from lstm.ucs import (
    LOEO_ELIGIBLE_ATTACK_TYPES,
    LOEO_EXCLUDED_ATTACK_TYPES,
    LOEO_GROUPING_COLUMN,
    UCS_LABEL_COLUMNS,
    UCS_METADATA_COLUMNS,
    load_ucs_windows,
    purge_and_embargo,
    UCSConfig
)


FORBIDDEN_FEATURES = {
    "episode_id",
    "forecast_episode_id",
    "source_day",
    "window_start_utc",
    "window_end_utc",
    "future_attack_label",
    "label_binary",
    "label_attack_type",
    "raw_label_dominant",
    "has_malicious_flows",
    "split",
    "fold_id",
}


def get_set_a_features(df: pd.DataFrame) -> List[str]:
    excluded = UCS_METADATA_COLUMNS | UCS_LABEL_COLUMNS | FORBIDDEN_FEATURES
    features = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    if not features:
        raise ValueError("Set A has no numeric features.")
    leaked = FORBIDDEN_FEATURES.intersection(features)
    if leaked:
        raise AssertionError(f"Forbidden Set-A features found: {sorted(leaked)}")
    values = df[features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Set-A features contain NaN or infinite values.")
    return features


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    has_both_classes = len(np.unique(y_true)) > 1
    roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else None
    pr_auc = float(average_precision_score(y_true, y_prob)) if has_both_classes else None

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def aggregate_metrics(fold_results: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, Any]:
    metric_names = ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"]
    summary: Dict[str, Any] = {
        "protocol": "37-fold LOEO (retrained per fold)",
        "fold_count": int(len(fold_results)),
        "aggregation": "mean_across_folds and pooled predictions across held-out folds",
        "mean_across_folds": {},
        "std_across_folds": {},
        "per_attack_type_means": {},
    }
    for name in metric_names:
        values = pd.to_numeric(fold_results[name], errors="coerce").dropna()
        summary["mean_across_folds"][name] = float(values.mean()) if len(values) > 0 else None
        summary["std_across_folds"][name] = float(values.std()) if len(values) > 1 else 0.0

    if "attack_type" in fold_results.columns:
        for attack_type, group in fold_results.groupby("attack_type"):
            summary["per_attack_type_means"][attack_type] = {
                name: float(pd.to_numeric(group[name], errors="coerce").dropna().mean())
                for name in metric_names
            }

    pooled = compute_metrics(
        predictions["y_true"].to_numpy(dtype=int),
        predictions["y_probability"].to_numpy(dtype=float),
        threshold=0.5,
    )
    summary["pooled_metrics"] = pooled
    return summary


def run_target(
    df: pd.DataFrame,
    folds: List[Dict[str, Any]],
    feature_columns: List[str],
    target: str,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    target_col = "label_binary" if target == "detection" else "future_attack_label"
    predictions: List[pd.DataFrame] = []
    fold_records: List[Dict[str, Any]] = []

    config = UCSConfig()

    for fold_id, fold in enumerate(folds):
        held_out_episode = fold["held_out_episode_id"]
        train_indices = fold["train_indices"]
        holdout_indices = fold["test_indices"]

        # Re-apply the temporal embargo logic per fold to properly drop training windows within 35-min
        fold_frame = df.copy()
        fold_frame["split"] = "none"
        fold_frame.loc[train_indices, "split"] = "train"
        fold_frame.loc[holdout_indices, "split"] = "test"
        
        train_rows = fold_frame.loc[train_indices].sort_values("window_start_utc")
        val_start = int(len(train_rows) * 0.8)
        fold_frame.loc[train_rows.index[val_start:], "split"] = "val"
        
        # Purge train sequences within 35 minutes of test sequences
        active_mask = fold_frame["split"].isin(["train", "val", "test"])
        purged = purge_and_embargo(fold_frame[active_mask].copy(), config=config)
        
        train_df = pd.concat([purged[purged["split"] == "train"], purged[purged["split"] == "val"]])
        test_df = fold_frame[fold_frame["split"] == "test"]
        
        # Attack type logic
        attack_type = df.loc[df["episode_id"] == held_out_episode, "label_attack_type"].iloc[0]

        x_train = train_df[feature_columns].to_numpy(dtype=float)
        x_test = test_df[feature_columns].to_numpy(dtype=float)
        y_train = train_df[target_col].to_numpy(dtype=int)
        y_test = test_df[target_col].to_numpy(dtype=int)

        if len(np.unique(y_train)) < 2:
            print(f"Fold {fold_id} (episode {held_out_episode}) train set lacks both binary classes, skipping.")
            continue
        if len(y_test) == 0:
            print(f"Fold {fold_id} (episode {held_out_episode}) has empty test set, skipping.")
            continue

        # Fold-local scaler fit ONLY on training data
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight=None,
            random_state=42,
        )
        model.fit(x_train_scaled, y_train)
        y_prob = model.predict_proba(x_test_scaled)[:, 1]

        fold_m = compute_metrics(y_test, y_prob, threshold=0.5)
        
        # Save coefficients
        coefs = pd.DataFrame({"feature": feature_columns, "coef": model.coef_[0]})
        coefs = coefs.sort_values(by="coef", key=abs, ascending=False)
        coefs.to_csv(output_dir / f"lr_coefficient_audit_{target}_fold{fold_id}.csv", index=False)
            
        fold_records.append({
            "fold_id": fold_id,
            "held_out_episode_id": held_out_episode,
            "attack_type": attack_type,
            "target": target,
            "target_col": target_col,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "test_attack_windows": int((y_test == 1).sum()),
            "test_benign_windows": int((y_test == 0).sum()),
            **fold_m,
        })

        pred_df = pd.DataFrame({
            "fold_id": fold_id,
            "held_out_episode_id": held_out_episode,
            "attack_type": attack_type,
            "target": target,
            "window_id": test_df["window_id"].to_numpy() if "window_id" in test_df.columns else np.arange(len(test_df)),
            "timestamp": test_df["window_start_utc"].astype(str).to_numpy(),
            "y_true": y_test,
            "y_probability": y_prob,
            "y_pred": (y_prob >= 0.5).astype(int),
        })
        predictions.append(pred_df)

    fold_frame = pd.DataFrame(fold_records)
    prediction_frame = pd.concat(predictions, ignore_index=True)

    fold_frame.to_csv(output_dir / f"lr_{target}_loeo_per_fold.csv", index=False)
    prediction_frame.to_csv(output_dir / f"lr_{target}_loeo_predictions.csv", index=False)

    summary = aggregate_metrics(fold_frame, prediction_frame)
    summary["target"] = target
    summary["target_col"] = target_col
    summary["feature_set"] = "Set A"
    summary["feature_count"] = len(feature_columns)
    (output_dir / f"lr_{target}_loeo_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return fold_frame, prediction_frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Real Set-A Logistic Regression under canonical 37-fold LOEO.")
    parser.add_argument("--data", type=Path, default=Path("data/ucs/ucs_windows.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/loeo/corrected_37fold_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lr_set_a_corrected"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_ucs_windows(args.data).sort_values("window_start_utc").reset_index(drop=True)
    if LOEO_GROUPING_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain {LOEO_GROUPING_COLUMN} metadata.")

    feature_columns = get_set_a_features(df)
    
    with open(args.manifest, "r") as f:
        manifest_data = json.load(f)
        folds = manifest_data["folds"]

    if len(folds) != 37:
        raise ValueError(f"Expected 37 eligible LOEO folds, but found {len(folds)}.")

    fold_manifest = pd.DataFrame([
        {
            "fold_id": fold_id,
            "held_out_episode_id": fold["held_out_episode_id"],
            "n_train": len(fold["train_indices"]), 
            "n_test": len(fold["test_indices"]),
        }
        for fold_id, fold in enumerate(folds)
    ])
    fold_manifest.to_csv(args.output_dir / "lr_37fold_loeo_fold_manifest.csv", index=False)

    manifest = {
        "feature_set": "Set A",
        "feature_count": len(feature_columns),
        "ordered_feature_names": feature_columns,
        "dataset": str(args.data),
        "protocol": "37-fold LOEO",
        "grouping_column": LOEO_GROUPING_COLUMN,
        "rejected_grouping_columns": ["forecast_episode_id", "source_day"],
        "preprocessing": "StandardScaler fitted independently on each training fold; no imputation required",
        "model": {
            "class": "sklearn.linear_model.LogisticRegression",
            "solver": "liblinear",
            "max_iter": 2000,
            "class_weight": None,
            "random_state": 42,
            "threshold": 0.5,
        },
        "eligible_attack_types": list(LOEO_ELIGIBLE_ATTACK_TYPES),
        "excluded_attack_types": list(LOEO_EXCLUDED_ATTACK_TYPES),
    }
    (args.output_dir / "set_a_features.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    det_folds, _, det_summary = run_target(df, folds, feature_columns, "detection", args.output_dir)
    onset_folds, _, onset_summary = run_target(df, folds, feature_columns, "onset", args.output_dir)

    print(f"Set A Logistic Regression baseline completed across {len(folds)} LOEO folds.")
    print(f"Set A feature count: {len(feature_columns)}")
    print("Detection results (label_binary):")
    print(json.dumps(det_summary["mean_across_folds"], indent=2))
    print("Onset results (future_attack_label):")
    print(json.dumps(onset_summary["mean_across_folds"], indent=2))


if __name__ == "__main__":
    main()
