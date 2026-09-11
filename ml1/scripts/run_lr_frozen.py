import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    n_classes = len(np.unique(y_true))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if n_classes > 1 else None
    pr_auc = float(average_precision_score(y_true, y_prob)) if n_classes > 1 else None
    
    if n_classes > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    else:
        tn, fp, fn, tp = (len(y_true), 0, 0, 0) if y_true[0] == 0 else (0, 0, 0, len(y_true))
    
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

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

def run_target(
    df: pd.DataFrame,
    manifest_folds: List[Dict[str, Any]],
    feature_columns: List[str],
    target: str,
    output_dir: Path,
) -> pd.DataFrame:
    target_col = "label_binary" if target == "detection" else "future_attack_label"
    fold_records = []

    for fold in manifest_folds:
        fold_id = fold["fold_id"]
        held_out_episode = fold["held_out_episode_id"]
        train_indices = fold["train_indices"]
        test_indices = fold["test_indices"]

        train_df = df.iloc[train_indices]
        test_df = df.iloc[test_indices]
        
        attack_type = df.loc[df['episode_id'] == held_out_episode, 'label_attack_type'].iloc[0]

        x_train = train_df[feature_columns].to_numpy(dtype=float)
        x_test = test_df[feature_columns].to_numpy(dtype=float)
        y_train = train_df[target_col].to_numpy(dtype=int)
        y_test = test_df[target_col].to_numpy(dtype=int)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight=None,
            random_state=42,
        )
        
        # If train is unlabelled (e.g., onset all 0s sometimes?), skip or fit dummy?
        if len(np.unique(y_train)) < 2:
            print(f"Skipping fold {fold_id} train set lacks both binary classes.")
            continue
            
        model.fit(x_train_scaled, y_train)
        y_prob = model.predict_proba(x_test_scaled)[:, 1]

        fold_m = compute_metrics(y_test, y_prob, threshold=0.5)
        
        # Check coefficients if perfect score
        if fold_m["f1"] == 1.0 and fold_m["fpr"] == 0.0:
            coefs = pd.DataFrame({"feature": feature_columns, "coef": model.coef_[0]})
            coefs = coefs.sort_values(by="coef", key=abs, ascending=False)
            coefs.to_csv(output_dir / f"lr_coefficient_audit_{target}_fold{fold_id}.csv", index=False)
            
        fold_records.append({
            "fold_id": fold_id,
            "held_out_episode_id": held_out_episode,
            "attack_type": attack_type,
            "target": target,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "test_attack_windows": int((y_test == 1).sum()),
            "test_benign_windows": int((y_test == 0).sum()),
            **fold_m,
        })

    fold_frame = pd.DataFrame(fold_records)
    fold_frame.to_csv(output_dir / f"lr_{target}_loeo_folds.csv", index=False)
    return fold_frame

def main():
    data_path = Path("data/ucs/ucs_windows.parquet")
    manifest_path = Path("artifacts/loeo/corrected_37fold_manifest.json")
    feature_path = Path("artifacts/lr_set_a/set_a_features.json")
    output_dir = Path("artifacts/lr_frozen_verified")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path).sort_values("window_start_utc").reset_index(drop=True)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    with open(feature_path, "r") as f:
        features_meta = json.load(f)
        feature_columns = features_meta["ordered_feature_names"]
        
    # Security check: Prohibited features
    prohibited = ["window_start_utc", "window_end_utc", "source_day", "episode_id", "forecast_episode_id"]
    for p in prohibited:
        if p in feature_columns:
            raise ValueError(f"Prohibited feature found: {p}")

    det_folds = run_target(df, manifest["folds"], feature_columns, "detection", output_dir)
    onset_folds = run_target(df, manifest["folds"], feature_columns, "onset", output_dir)
    print("LR Eval Done")

if __name__ == "__main__":
    main()
