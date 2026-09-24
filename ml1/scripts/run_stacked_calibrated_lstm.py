"""
Run Stacked & Temperature-Calibrated Residual LSTM under 37-Fold LOEO Protocol
Strict Leakage-Free Discipline:
- Test fold episode is completely held out.
- LR out-of-fold predictions on training data generated via 5-fold CV across training episodes.
- Chronological 80/20 train/validation split.
- PCA fitted strictly on training split.
- Residual LSTM predicts additive correction on top of LR base logit: z = z_base + z_lstm.
- Single temperature parameter T fitted exclusively on validation split via NLL minimization.
- Calibrated test evaluation: p = sigmoid((z_base + z_lstm) / T) >= 0.5.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from scipy.optimize import minimize_scalar

repo_root = Path(__file__).resolve().parent.parent.parent
ml1_dir = repo_root / "ml1"
sys.path.insert(0, str(ml1_dir))

from lstm.ucs import UCSConfig, validate_ucs_windows, apply_training_only_pca
from lstm.utils import set_seed, get_device

class ResidualLSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.20):
        super().__init__()
        # Dropout applied strictly to non-recurrent connections (dropout=0 on 1-layer LSTM)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        # Small weight initialization so initial residual starts near zero
        nn.init.zeros_(self.fc.bias)
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward_logits(self, x: torch.Tensor, base_logit: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_step = self.dropout(output[:, -1, :])
        residual = self.fc(last_step).squeeze(-1)
        return base_logit + residual

    def forward(self, x: torch.Tensor, base_logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x, base_logit))

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error across n_bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def fit_temperature_on_val(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a single temperature parameter T > 0 on the validation split via NLL minimization."""
    def nll(t):
        scaled = logits / t
        # Numerically stable BCE loss
        loss = np.maximum(scaled, 0) - scaled * labels + np.log(1 + np.exp(-np.abs(scaled)))
        return np.mean(loss)

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)

def run_stacked_target(
    df: pd.DataFrame,
    manifest_folds: List[Dict[str, Any]],
    lr_features: List[str],
    target_col: str,
    output_dir: Path,
    target_name: str,
    epochs: int = 12,
    batch_size: int = 64,
    learning_rate: float = 0.001,
) -> pd.DataFrame:
    set_seed(42)
    device = get_device()
    config = UCSConfig()
    base_features = validate_ucs_windows(df, config=config)

    fold_metrics = []
    pooled_val_preds = []
    pooled_val_targets = []
    print(f"\n{'='*70}\nRunning Stacked & Calibrated Residual LSTM: {target_name.upper()} ({target_col})\n{'='*70}", flush=True)

    for fold in manifest_folds:
        fold_id = fold["fold_id"]
        held_out_episode = fold["held_out_episode_id"]
        train_indices = fold["train_indices"]
        test_indices = fold["test_indices"]
        attack_type = df.loc[df["episode_id"] == held_out_episode, "label_attack_type"].iloc[0]

        train_df = df.iloc[train_indices].copy().reset_index()
        test_df = df.iloc[test_indices].copy()
        train_episodes = train_df["episode_id"].unique()

        # Step 1: Leakage-safe Out-of-Fold LR predictions for training windows
        oof_lr_logits = np.zeros(len(train_df))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_ep_idx, val_ep_idx in kf.split(train_episodes):
            tr_eps = set(train_episodes[tr_ep_idx])
            val_eps = set(train_episodes[val_ep_idx])
            tr_mask = train_df["episode_id"].isin(tr_eps)
            val_mask = train_df["episode_id"].isin(val_eps)

            y_inner_tr = train_df.loc[tr_mask, target_col].to_numpy(dtype=int)
            if len(np.unique(y_inner_tr)) < 2:
                # If inner fold lacks both classes, skip or fallback
                continue

            scaler_inner = StandardScaler()
            x_inner_tr = scaler_inner.fit_transform(train_df.loc[tr_mask, lr_features].to_numpy(dtype=float))
            x_inner_val = scaler_inner.transform(train_df.loc[val_mask, lr_features].to_numpy(dtype=float))

            lr_inner = LogisticRegression(solver="liblinear", max_iter=2000, random_state=42)
            lr_inner.fit(x_inner_tr, y_inner_tr)
            oof_lr_logits[val_mask] = lr_inner.decision_function(x_inner_val)

        # Full LR on all training data -> predict on test fold
        scaler_full = StandardScaler()
        x_tr_full = scaler_full.fit_transform(train_df[lr_features].to_numpy(dtype=float))
        y_tr_full = train_df[target_col].to_numpy(dtype=int)

        lr_full = LogisticRegression(solver="liblinear", max_iter=2000, random_state=42)
        lr_full.fit(x_tr_full, y_tr_full)

        x_te_full = scaler_full.transform(test_df[lr_features].to_numpy(dtype=float))
        test_lr_logits = lr_full.decision_function(x_te_full)

        # Step 2: Chronological 80/20 train/validation split on train_indices
        fold_frame = df.copy()
        fold_frame["split"] = "none"
        train_df_sorted = fold_frame.loc[train_indices].sort_values("window_start_utc")
        n_train = len(train_df_sorted)
        val_start = int(n_train * 0.8)
        actual_train_indices = train_df_sorted.index[:val_start]
        actual_val_indices = train_df_sorted.index[val_start:]

        fold_frame.loc[actual_train_indices, "split"] = "train"
        fold_frame.loc[actual_val_indices, "split"] = "val"
        fold_frame.loc[test_indices, "split"] = "test"

        # Map LR base logits to fold_frame
        orig_idx_to_train_pos = {train_df["index"].iloc[i]: i for i in range(len(train_df))}
        fold_frame["base_logit"] = 0.0
        for idx, pos in orig_idx_to_train_pos.items():
            fold_frame.loc[idx, "base_logit"] = oof_lr_logits[pos]
        for idx, logit in zip(test_indices, test_lr_logits):
            fold_frame.loc[idx, "base_logit"] = logit

        # PCA fitted strictly on training data
        purged = fold_frame[fold_frame["split"].isin(["train", "val", "test"])].copy()
        transformed, pca = apply_training_only_pca(purged, base_features, 32, random_state=42)
        pca_features = [f"pca_{i}" for i in range(32)]
        full_transformed = pca.transform(fold_frame)
        full_np = full_transformed[pca_features].to_numpy(dtype=np.float32)

        idx_to_pos = {idx: pos for pos, idx in enumerate(fold_frame.index)}
        base_logits_arr = fold_frame["base_logit"].values
        targets_arr = fold_frame[target_col].values

        def extract_sequences(indices):
            X_seqs, base_vals, y_vals = [], [], []
            for idx in indices:
                t_pos = idx_to_pos[idx]
                start = t_pos - config.lookback_windows + 1
                if start >= 0:
                    hist = full_np[start : t_pos + 1]
                    if len(hist) == config.lookback_windows:
                        X_seqs.append(hist)
                        base_vals.append(base_logits_arr[t_pos])
                        y_vals.append(targets_arr[t_pos])
            return (
                np.asarray(X_seqs, dtype=np.float32),
                np.asarray(base_vals, dtype=np.float32),
                np.asarray(y_vals, dtype=np.float32),
            )

        X_train, base_train, y_train = extract_sequences(actual_train_indices)
        X_val, base_val, y_val = extract_sequences(actual_val_indices)
        X_test, base_test, y_test = extract_sequences(test_indices)

        if len(y_test) == 0:
            print(f"Skipping fold {fold_id}, no valid test sequences.")
            continue

        # Step 3: Train Residual LSTM
        model = ResidualLSTMClassifier(input_size=32, hidden_size=64, dropout=0.20).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(base_train),
            torch.from_numpy(y_train),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        for epoch in range(1, epochs + 1):
            model.train()
            for xb, bb, yb in train_loader:
                xb, bb, yb = xb.to(device), bb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model.forward_logits(xb, bb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_xb = torch.from_numpy(X_val).to(device)
                val_bb = torch.from_numpy(base_val).to(device)
                val_yb = torch.from_numpy(y_val).to(device)
                v_logits_t = model.forward_logits(val_xb, val_bb)
                val_loss = criterion(v_logits_t, val_yb).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        # Step 4: Temperature Calibration on Validation Split ONLY
        model.eval()
        with torch.no_grad():
            v_logits = model.forward_logits(torch.from_numpy(X_val).to(device), torch.from_numpy(base_val).to(device)).cpu().numpy()
            te_logits = model.forward_logits(torch.from_numpy(X_test).to(device), torch.from_numpy(base_test).to(device)).cpu().numpy()

        T = fit_temperature_on_val(v_logits, y_val)
        val_probs_raw = 1.0 / (1.0 + np.exp(-v_logits))
        val_probs_cal = 1.0 / (1.0 + np.exp(-v_logits / T))
        val_ece_raw = compute_ece(val_probs_raw, y_val)
        val_ece_cal = compute_ece(val_probs_cal, y_val)

        # Step 5: Test Evaluation
        te_probs_raw = 1.0 / (1.0 + np.exp(-te_logits))
        te_probs_cal = 1.0 / (1.0 + np.exp(-te_logits / T))
        test_ece_raw = compute_ece(te_probs_raw, y_test)
        test_ece_cal = compute_ece(te_probs_cal, y_test)

        pred_cal = (te_probs_cal >= 0.5).astype(int)
        y_test_int = y_test.astype(int)

        f1 = float(f1_score(y_test_int, pred_cal, zero_division=0))
        prec = float(precision_score(y_test_int, pred_cal, zero_division=0))
        rec = float(recall_score(y_test_int, pred_cal, zero_division=0))

        n_classes = len(np.unique(y_test_int))
        roc_auc = float(roc_auc_score(y_test_int, te_probs_cal)) if n_classes > 1 else None
        pr_auc = float(average_precision_score(y_test_int, te_probs_cal)) if n_classes > 1 else None

        cm = confusion_matrix(y_test_int, pred_cal, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

        # Step 6: Save Checkpoint & Metadata Sidecar for Live Deployment
        target_artifacts_dir = output_dir / "lstm_stacked" / target_name
        target_artifacts_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = target_artifacts_dir / f"model_fold_{fold_id}.pt"
        sidecar_path = target_artifacts_dir / f"sidecar_fold_{fold_id}.json"

        checkpoint_data = {
            "fold_id": fold_id,
            "held_out_episode_id": held_out_episode,
            "attack_type": attack_type,
            "target": target_col,
            "target_name": target_name,
            "model_state_dict": best_state,
            "temperature": float(T),
            "lr_coef": lr_full.coef_.tolist(),
            "lr_intercept": lr_full.intercept_.tolist(),
            "scaler_mean": scaler_full.mean_.tolist(),
            "scaler_scale": scaler_full.scale_.tolist(),
            "lr_features": lr_features,
            "pca_components": pca.pca.components_.tolist(),
            "pca_mean": pca.pca.mean_.tolist() if pca.pca.mean_ is not None else None,
            "pca_columns": pca.columns,
            "val_predictions": val_probs_cal.tolist(),
            "val_targets": y_val.tolist(),
            "val_logits": v_logits.tolist(),
        }
        torch.save(checkpoint_data, ckpt_path)

        sidecar_data = {
            "fold_id": fold_id,
            "held_out_episode_id": held_out_episode,
            "attack_type": attack_type,
            "target": target_col,
            "target_name": target_name,
            "checkpoint_file": f"model_fold_{fold_id}.pt",
            "temperature": float(T),
            "lr_coef": lr_full.coef_.tolist(),
            "lr_intercept": lr_full.intercept_.tolist(),
            "scaler_mean": scaler_full.mean_.tolist(),
            "scaler_scale": scaler_full.scale_.tolist(),
            "lr_features": lr_features,
            "pca_columns": pca.columns,
            "val_predictions": [float(x) for x in val_probs_cal],
            "val_targets": [int(x) for x in y_val],
            "val_ece_raw": float(val_ece_raw),
            "val_ece_cal": float(val_ece_cal),
            "test_f1": float(f1),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_fpr": float(fpr),
            "test_roc_auc": float(roc_auc) if roc_auc is not None else None,
            "test_pr_auc": float(pr_auc) if pr_auc is not None else None,
        }
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_data, f, indent=2)

        # Save canonical PCA artifact from first valid fold
        canonical_pca_path = output_dir / "lstm_stacked" / "pca_32_stacked.pkl"
        if not canonical_pca_path.exists():
            import pickle
            with open(canonical_pca_path, "wb") as pf:
                pickle.dump({"pca": pca, "features": base_features}, pf)

        pooled_val_preds.extend(val_probs_cal.tolist())
        pooled_val_targets.extend(y_val.tolist())

        fold_metrics.append({
            "fold_id": fold_id,
            "held_out_episode_id": held_out_episode,
            "attack_type": attack_type,
            "target": target_col,
            "train_sequences": len(X_train),
            "validation_sequences": len(X_val),
            "test_sequences": len(y_test),
            "f1": f1,
            "pr_auc": pr_auc,
            "precision": prec,
            "recall": rec,
            "fpr": fpr,
            "roc_auc": roc_auc,
            "val_ece_raw": val_ece_raw,
            "val_ece_cal": val_ece_cal,
            "test_ece_raw": test_ece_raw,
            "test_ece_cal": test_ece_cal,
            "temperature": T,
            "checkpoint_path": str(ckpt_path),
        })

        print(
            f"Fold {fold_id:02d} ({held_out_episode:30s} | {attack_type:15s}): "
            f"F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | FPR={fpr:.4f} | "
            f"T={T:.2f} | Val ECE: {val_ece_raw:.4f}->{val_ece_cal:.4f} [Saved: {ckpt_path.name}]",
            flush=True,
        )

    # Save pooled validation residuals for real conformal prediction calibration
    pooled_val_arr_p = np.array(pooled_val_preds, dtype=float)
    pooled_val_arr_y = np.array(pooled_val_targets, dtype=float)
    residuals = np.abs(pooled_val_arr_y - pooled_val_arr_p)
    pooled_conformal_meta = {
        "target": target_col,
        "target_name": target_name,
        "n_samples": int(len(residuals)),
        "mean_residual": float(np.mean(residuals)),
        "median_residual": float(np.median(residuals)),
        "q90_residual": float(np.quantile(residuals, 0.90)),
        "q95_residual": float(np.quantile(residuals, 0.95)),
        "val_predictions": pooled_val_preds,
        "val_targets": pooled_val_targets,
    }
    with open(target_artifacts_dir / "val_residuals_pooled.json", "w", encoding="utf-8") as f:
        json.dump(pooled_conformal_meta, f, indent=2)

    out_df = pd.DataFrame(fold_metrics)
    out_csv = output_dir / f"lstm_{target_name}_loeo_folds.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(out_df)} folds and checkpoints to {target_artifacts_dir}", flush=True)
    return out_df

def main():
    set_seed(42)
    data_path = repo_root / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
    if not data_path.exists():
        data_path = ml1_dir / "data" / "ucs" / "ucs_windows.parquet"
    manifest_path = ml1_dir / "artifacts" / "loeo" / "corrected_37fold_manifest.json"
    feature_path = ml1_dir / "artifacts" / "lr_set_a" / "set_a_features.json"
    output_dir = ml1_dir / "artifacts" / "lstm"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path).sort_values("window_start_utc").reset_index(drop=True)
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(feature_path) as f:
        features_meta = json.load(f)
    lr_features = features_meta["ordered_feature_names"]

    # 1. Run Detection
    run_stacked_target(
        df=df,
        manifest_folds=manifest["folds"],
        lr_features=lr_features,
        target_col="label_binary",
        output_dir=output_dir,
        target_name="detection",
        epochs=10,
    )

    # 2. Run Onset
    run_stacked_target(
        df=df,
        manifest_folds=manifest["folds"],
        lr_features=lr_features,
        target_col="future_attack_label",
        output_dir=output_dir,
        target_name="onset",
        epochs=10,
    )

    # 3. Create Master Conformal Calibration Artifact
    stacked_dir = output_dir / "lstm_stacked"
    det_res_file = stacked_dir / "detection" / "val_residuals_pooled.json"
    onset_res_file = stacked_dir / "onset" / "val_residuals_pooled.json"
    if det_res_file.exists() and onset_res_file.exists():
        with open(det_res_file) as f:
            det_conformal = json.load(f)
        with open(onset_res_file) as f:
            onset_conformal = json.load(f)
        master_conformal = {
            "description": "Validation-split empirical residuals pooled across 37 LOEO folds with zero test-set leakage.",
            "onset": onset_conformal,
            "detection": det_conformal,
        }
        master_path = stacked_dir / "conformal_calibration_residuals.json"
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(master_conformal, f, indent=2)
        print(f"Master conformal calibration artifact created at: {master_path}", flush=True)

    print("\nAll stacked LOEO evaluations completed successfully!")

if __name__ == "__main__":
    main()
