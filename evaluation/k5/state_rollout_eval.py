"""
Phase C-t1: State-Space Rollout K=1..5 under 37-fold LOEO
Evaluates Gaussian world model forward rollout K=1..5 against:
1. Persistence S(t+k) = S(t)
2. Training-fold mean S_bar_train
3. Linear AR(1) fitted on training fold

Computes pooled and per-fold RMSE and MAE per K, with paired episode-level bootstrap 95% CI
(1,000 resamples) for (model - persistence).
Determines H*_state = largest K where model beats persistence on BOTH RMSE and MAE with CI excluding 0.
Writes evaluation/k5/state_rollout_results.json and STATE_ROLLOUT.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "ml1") not in sys.path:
    sys.path.insert(0, str(REPO / "ml1"))

from lstm.model import LSTMGaussianWorldModel  # noqa: E402
from lstm.probabilistic import regression_metrics, train_gaussian  # noqa: E402
from lstm.ucs import UCSConfig, validate_ucs_windows  # noqa: E402

OUT_DIR = REPO / "evaluation" / "k5"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
PURGE_SECONDS = 2160.0  # 36 windows * 60s


def build_contiguous_lookbacks(
    df: pd.DataFrame,
    features: List[str],
    config: UCSConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized extraction of all contiguous lookback sequences in df.
    Returns:
      valid_target_indices: array of window indices t where [t-30:t] is contiguous
      X_all: (N_valid, 30, 406)
      y_all: (N_valid, 406)
    """
    times = pd.to_datetime(df["window_start_utc"], utc=True)
    time_diffs = times.diff().dt.total_seconds().values
    is_step_60 = (time_diffs == 60.0).astype(int)

    L = config.lookback_windows
    cumsum = np.cumsum(np.insert(is_step_60, 0, 0))

    values = df[features].to_numpy(dtype=np.float32)
    valid_targets = []
    for t in range(L, len(df)):
        # Contiguous if all steps between t-L and t are 60s
        if (cumsum[t] - cumsum[t - L + 1]) == (L - 1):
            valid_targets.append(t)

    valid_targets = np.array(valid_targets, dtype=int)
    X_list = [values[t - L : t] for t in valid_targets]
    y_list = [values[t] for t in valid_targets]

    X_all = np.stack(X_list, axis=0).astype(np.float32)
    y_all = np.stack(y_list, axis=0).astype(np.float32)
    return valid_targets, X_all, y_all


def fit_linear_ar1(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Fit 1-step linear model S(t+1) = W * S(t) + b on training transitions."""
    last_states = X_train[:, -1, :]
    lr = LinearRegression()
    lr.fit(last_states, y_train)
    return lr


def rollout_ar1(lr: LinearRegression, s0: np.ndarray, k_max: int = 5) -> np.ndarray:
    """Recursively roll out linear AR(1) for k=1..k_max steps."""
    preds = []
    curr = s0.reshape(1, -1)
    for _ in range(k_max):
        next_s = lr.predict(curr)
        preds.append(next_s.ravel())
        curr = next_s
    return np.stack(preds, axis=0)  # (k_max, state_dim)


def rollout_lstm(
    model: LSTMGaussianWorldModel,
    history_30x406: np.ndarray,
    device: torch.device,
    k_max: int = 5,
) -> np.ndarray:
    """Recursively roll forward the predicted mean of the Gaussian world model."""
    model.eval()
    preds = []
    curr_hist = history_30x406.copy()[np.newaxis, :, :]  # (1, 30, 406)

    with torch.no_grad():
        for _ in range(k_max):
            mean, _ = model(torch.as_tensor(curr_hist, dtype=torch.float32).to(device))
            mean_np = mean.cpu().numpy()  # (1, 406)
            preds.append(mean_np[0])
            # Slide history window
            curr_hist = np.concatenate([curr_hist[:, 1:, :], mean_np[:, np.newaxis, :]], axis=1)

    return np.stack(preds, axis=0)  # (k_max, state_dim)


def run_evaluation(
    data_path: str = "data-engineering/data/ucs/ucs_windows_models_v1.parquet",
    manifest_path: str = "ml1/artifacts/loeo/corrected_37fold_manifest.json",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    patience: int = 5,
    min_delta: float = 0.001,
    seed: int = 42,
    device_str: str = "cpu",
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)

    print(f"Loading dataset: {data_path}")
    df = pd.read_parquet(REPO / data_path).sort_values("window_start_utc").reset_index(drop=True)
    times = pd.to_datetime(df["window_start_utc"], utc=True)
    time_diffs = times.diff().dt.total_seconds().values
    is_step_60 = (time_diffs == 60.0).astype(int)
    cumsum = np.cumsum(np.insert(is_step_60, 0, 0))

    config = UCSConfig()
    features = validate_ucs_windows(df, config=config)
    n_feats = len(features)
    print(f"Validated features: {n_feats}")

    with open(REPO / manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    folds = manifest["folds"]
    print(f"Loaded {len(folds)} LOEO folds from manifest")

    # Vectorized lookback sequence preparation
    print("Building global lookback transitions...")
    valid_targets, X_all, y_all = build_contiguous_lookbacks(df, features, config)
    target_to_seq_idx = {t: idx for idx, t in enumerate(valid_targets)}
    print(f"Total contiguous transitions available: {len(valid_targets)}")

    raw_values = df[features].to_numpy(dtype=np.float32)

    # Collect per-window rollout evaluations across folds
    # Store: fold_id, episode_id, attack_type, window_idx, k, model_pred, persistence_pred, mean_pred, ar1_pred, actual
    all_window_records = []

    total_folds = len(folds)
    for f_idx, fold in enumerate(folds):
        fold_id = fold["fold_id"]
        ep_id = fold["held_out_episode_id"]
        train_indices = set(fold["train_indices"])
        test_indices = fold["test_indices"]

        # Determine attack type of held-out episode
        ep_rows = df[df["episode_id"] == ep_id]
        attack_type = ep_rows["label_attack_type"].iloc[0] if len(ep_rows) > 0 else "Unknown"

        print(f"\n--- Fold {fold_id} ({f_idx+1}/{total_folds}) | Episode {ep_id} ({attack_type}) ---")

        # Map training indices to precomputed sequence indices
        tr_seq_indices = [target_to_seq_idx[t] for t in sorted(train_indices) if t in target_to_seq_idx]
        cut = int(len(tr_seq_indices) * 0.8)
        tr_idx_sub = tr_seq_indices[:cut]
        val_idx_sub = tr_seq_indices[cut:]

        X_train = X_all[tr_idx_sub]
        y_train = y_all[tr_idx_sub]
        X_val = X_all[val_idx_sub]
        y_val = y_all[val_idx_sub]

        print(f"Train seqs: {len(X_train)}, Val seqs: {len(X_val)}")

        # Baselines
        # 1. Training fold mean
        train_mean_vec = np.mean(y_train, axis=0)

        # 2. Linear AR(1)
        lr_ar1 = fit_linear_ar1(X_train, y_train)

        # 3. Gaussian World Model
        model_ckpt = CHECKPOINT_DIR / f"gaussian_world_model_fold_{fold_id}.pt"
        model = LSTMGaussianWorldModel(
            input_size=n_feats,
            hidden_size=64,
            state_dim=n_feats,
            num_layers=1,
            dropout=0.2,
        )

        if model_ckpt.exists():
            print(f"Loading existing checkpoint: {model_ckpt.name}")
            state_dict = torch.load(model_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state_dict["model_state_dict"])
            model.to(device)
        else:
            t0 = time.time()
            train_gaussian(
                model,
                X_train,
                y_train,
                X_val,
                y_val,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                weight_decay=weight_decay,
                patience=patience,
                min_delta=min_delta,
                seed=seed,
                device=device_str,
                checkpoint_path=model_ckpt,
            )
            print(f"Trained fold {fold_id} in {time.time() - t0:.1f}s -> saved {model_ckpt.name}")

        model.eval()

        # Evaluate on test windows
        # A test window t can be rolled out for K=1..5 if t has 30 lookback windows and 5 contiguous future windows
        L = config.lookback_windows
        n_evaluated = 0
        for t in test_indices:
            if t < L or t + 5 >= len(df):
                continue
            # Check lookback contiguity [t-L : t]
            if (cumsum[t] - cumsum[t - L + 1]) != (L - 1):
                continue
            # Check future contiguity [t : t+5]
            if (cumsum[t + 5] - cumsum[t + 1]) != 4:
                continue

            hist_30 = raw_values[t - L : t]  # (30, 406)
            s_current = raw_values[t]        # (406,)
            actuals_5 = raw_values[t + 1 : t + 6]  # (5, 406)

            # Rollouts
            lstm_preds = rollout_lstm(model, hist_30, device, k_max=5)  # (5, 406)
            ar1_preds = rollout_ar1(lr_ar1, s_current, k_max=5)         # (5, 406)

            for step in range(5):
                k = step + 1
                actual_k = actuals_5[step]
                pred_lstm = lstm_preds[step]
                pred_persist = s_current
                pred_mean = train_mean_vec
                pred_ar1 = ar1_preds[step]

                all_window_records.append({
                    "fold_id": fold_id,
                    "episode_id": ep_id,
                    "attack_type": attack_type,
                    "window_index": int(t),
                    "k": k,
                    "target_window_index": int(t + k),
                    "actual_label_binary": int(df.loc[t + k, "label_binary"]),
                    "current_label_binary": int(df.loc[t, "label_binary"]),
                    # Errors
                    "err_lstm": float(np.mean((pred_lstm - actual_k) ** 2)),
                    "err_persist": float(np.mean((pred_persist - actual_k) ** 2)),
                    "err_mean": float(np.mean((pred_mean - actual_k) ** 2)),
                    "err_ar1": float(np.mean((pred_ar1 - actual_k) ** 2)),
                    "mae_lstm": float(np.mean(np.abs(pred_lstm - actual_k))),
                    "mae_persist": float(np.mean(np.abs(pred_persist - actual_k))),
                    "mae_mean": float(np.mean(np.abs(pred_mean - actual_k))),
                    "mae_ar1": float(np.mean(np.abs(pred_ar1 - actual_k))),
                    # Store rolled state for C-t2 reuse
                    "pred_lstm_state": pred_lstm,
                })
            n_evaluated += 1

        print(f"Evaluated {n_evaluated} valid test windows on fold {fold_id}")

    print(f"\nTotal test window records collected: {len(all_window_records)}")
    records_df = pd.DataFrame(all_window_records)

    # Save rollout predictions cache for C-t2
    cache_path = OUT_DIR / "rollout_predictions_cache.joblib"
    import joblib
    joblib.dump(records_df, cache_path)
    print(f"Saved rollout predictions cache to {cache_path}")

    # Compute Pooled and Per-Fold Metrics
    print("\nComputing metrics and paired episode-level bootstrap 95% CI...")
    pooled_metrics_per_k = {}
    bootstrap_results_per_k = {}

    unique_episodes = sorted(records_df["episode_id"].unique())
    n_episodes = len(unique_episodes)
    print(f"Unique test episodes: {n_episodes}")

    n_bootstraps = 1000
    rng = np.random.default_rng(seed)

    for k in range(1, 6):
        k_df = records_df[records_df["k"] == k]

        # Pooled metrics
        rmse_lstm = float(np.sqrt(np.mean(k_df["err_lstm"])))
        mae_lstm = float(np.mean(k_df["mae_lstm"]))

        rmse_persist = float(np.sqrt(np.mean(k_df["err_persist"])))
        mae_persist = float(np.mean(k_df["mae_persist"]))

        rmse_mean = float(np.sqrt(np.mean(k_df["err_mean"])))
        mae_mean = float(np.mean(k_df["mae_mean"]))

        rmse_ar1 = float(np.sqrt(np.mean(k_df["err_ar1"])))
        mae_ar1 = float(np.mean(k_df["mae_ar1"]))

        # Group data by episode for fast bootstrap sampling
        ep_groups = {ep: grp for ep, grp in k_df.groupby("episode_id")}

        boot_delta_rmse = []
        boot_delta_mae = []

        for _ in range(n_bootstraps):
            sampled_eps = rng.choice(unique_episodes, size=n_episodes, replace=True)
            sampled_err_lstm = []
            sampled_err_persist = []
            sampled_mae_lstm = []
            sampled_mae_persist = []

            for ep in sampled_eps:
                grp = ep_groups[ep]
                sampled_err_lstm.extend(grp["err_lstm"].values)
                sampled_err_persist.extend(grp["err_persist"].values)
                sampled_mae_lstm.extend(grp["mae_lstm"].values)
                sampled_mae_persist.extend(grp["mae_persist"].values)

            b_rmse_m = np.sqrt(np.mean(sampled_err_lstm))
            b_rmse_p = np.sqrt(np.mean(sampled_err_persist))
            b_mae_m = np.mean(sampled_mae_lstm)
            b_mae_p = np.mean(sampled_mae_persist)

            boot_delta_rmse.append(b_rmse_m - b_rmse_p)
            boot_delta_mae.append(b_mae_m - b_mae_p)

        point_delta_rmse = rmse_lstm - rmse_persist
        ci_rmse_low = float(np.percentile(boot_delta_rmse, 2.5))
        ci_rmse_high = float(np.percentile(boot_delta_rmse, 97.5))

        point_delta_mae = mae_lstm - mae_persist
        ci_mae_low = float(np.percentile(boot_delta_mae, 2.5))
        ci_mae_high = float(np.percentile(boot_delta_mae, 97.5))

        # Model beats persistence if CI upper bound < 0 (strictly negative on both)
        beats_rmse = bool(ci_rmse_high < 0)
        beats_mae = bool(ci_mae_high < 0)
        beats_both = bool(beats_rmse and beats_mae)

        pooled_metrics_per_k[str(k)] = {
            "n_windows": int(len(k_df)),
            "lstm_gaussian": {"rmse": round(rmse_lstm, 4), "mae": round(mae_lstm, 4)},
            "persistence": {"rmse": round(rmse_persist, 4), "mae": round(mae_persist, 4)},
            "training_mean": {"rmse": round(rmse_mean, 4), "mae": round(mae_mean, 4)},
            "linear_ar1": {"rmse": round(rmse_ar1, 4), "mae": round(mae_ar1, 4)},
            "delta_model_minus_persistence": {
                "rmse": {
                    "point_estimate": round(point_delta_rmse, 4),
                    "ci_95": [round(ci_rmse_low, 4), round(ci_rmse_high, 4)],
                    "beats_persistence": beats_rmse,
                },
                "mae": {
                    "point_estimate": round(point_delta_mae, 4),
                    "ci_95": [round(ci_mae_low, 4), round(ci_mae_high, 4)],
                    "beats_persistence": beats_mae,
                },
                "beats_persistence_both": beats_both,
            },
        }

    # Determine H*_state
    h_star_state = 0
    for k in range(1, 6):
        if pooled_metrics_per_k[str(k)]["delta_model_minus_persistence"]["beats_persistence_both"]:
            h_star_state = k
        else:
            break

    print(f"\nH*_state determined: {h_star_state}")

    # Per-fold breakdown table
    per_fold_summary = []
    for f in folds:
        f_id = f["fold_id"]
        ep_id = f["held_out_episode_id"]
        f_df = records_df[records_df["fold_id"] == f_id]
        if f_df.empty:
            continue
        at = f_df["attack_type"].iloc[0]
        n_w = len(f_df[f_df["k"] == 1])

        fold_row = {
            "fold_id": f_id,
            "episode_id": ep_id,
            "attack_type": at,
            "n_test_windows": n_w,
            "horizons": {},
        }
        for k in range(1, 6):
            kf = f_df[f_df["k"] == k]
            if kf.empty:
                continue
            fold_row["horizons"][str(k)] = {
                "rmse_lstm": round(float(np.sqrt(np.mean(kf["err_lstm"]))), 4),
                "mae_lstm": round(float(np.mean(kf["mae_lstm"])), 4),
                "rmse_persist": round(float(np.sqrt(np.mean(kf["err_persist"]))), 4),
                "mae_persist": round(float(np.mean(kf["mae_persist"])), 4),
            }
        per_fold_summary.append(fold_row)

    final_results = {
        "generated_by": "evaluation/k5/state_rollout_eval.py",
        "dataset": data_path,
        "manifest": manifest_path,
        "n_folds": len(folds),
        "n_episodes": n_episodes,
        "total_test_windows_evaluated": len(records_df[records_df["k"] == 1]),
        "h_star_state": h_star_state,
        "pooled_results_per_k": pooled_metrics_per_k,
        "per_fold_results": per_fold_summary,
    }

    results_json_path = OUT_DIR / "state_rollout_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"Wrote {results_json_path}")

    # Generate STATE_ROLLOUT.md
    md_lines = [
        "# State-Space Rollout Evaluation (K = 1..5) under 37-fold LOEO",
        "",
        f"Generated by `evaluation/k5/state_rollout_eval.py` on dataset `{data_path}`.",
        f"Manifest: `{manifest_path}` ({len(folds)} folds, {n_episodes} held-out attack episodes).",
        "",
        "## Summary of Findings",
        "",
        f"- **H\\*_state**: **{h_star_state}**",
        f"- At H\\*_state = {h_star_state}, the Gaussian world model {'beats' if h_star_state > 0 else 'does NOT beat'} "
        "persistence on both RMSE and MAE with a paired episode-level bootstrap 95% CI strictly excluding 0.",
        "",
        "## Pooled Results per Horizon K",
        "",
        "| Horizon K | Model RMSE | Persistence RMSE | AR(1) RMSE | Mean RMSE | Δ RMSE [95% CI] | Model MAE | Persistence MAE | AR(1) MAE | Mean MAE | Δ MAE [95% CI] | Beats Both? |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for k in range(1, 6):
        res = pooled_metrics_per_k[str(k)]
        m_rmse = res["lstm_gaussian"]["rmse"]
        p_rmse = res["persistence"]["rmse"]
        ar1_rmse = res["linear_ar1"]["rmse"]
        mean_rmse = res["training_mean"]["rmse"]
        d_rmse = res["delta_model_minus_persistence"]["rmse"]
        ci_r = f"{d_rmse['point_estimate']:+.4f} [{d_rmse['ci_95'][0]:+.4f}, {d_rmse['ci_95'][1]:+.4f}]"

        m_mae = res["lstm_gaussian"]["mae"]
        p_mae = res["persistence"]["mae"]
        ar1_mae = res["linear_ar1"]["mae"]
        mean_mae = res["training_mean"]["mae"]
        d_mae = res["delta_model_minus_persistence"]["mae"]
        ci_m = f"{d_mae['point_estimate']:+.4f} [{d_mae['ci_95'][0]:+.4f}, {d_mae['ci_95'][1]:+.4f}]"

        beats = "YES" if res["delta_model_minus_persistence"]["beats_persistence_both"] else "NO"
        md_lines.append(
            f"| K={k} (+{k}m) | {m_rmse:.4f} | {p_rmse:.4f} | {ar1_rmse:.4f} | {mean_rmse:.4f} | {ci_r} | "
            f"{m_mae:.4f} | {p_mae:.4f} | {ar1_mae:.4f} | {mean_mae:.4f} | {ci_m} | {beats} |"
        )

    md_lines.extend([
        "",
        "## Statistical Methodology",
        "",
        "1. **Baselines**:",
        "   - **Persistence**: $\\hat{S}(t+k) = S(t)$",
        "   - **Linear AR(1)**: $\\hat{S}(t+1) = W S(t) + b$ fitted on training fold transitions",
        "   - **Training Mean**: constant vector $\\bar{S}_{train}$ across all training fold windows",
        "2. **Confidence Intervals**:",
        "   - Paired episode-level bootstrap with 1,000 resamples over the 37 held-out episodes.",
        "   - CI bounds represent the empirical 2.5% and 97.5% quantiles of $\\Delta = \\text{Error}_{model} - \\text{Error}_{persistence}$.",
        "   - The criterion for beating persistence requires the upper bound of the 95% CI to be strictly negative on both RMSE and MAE.",
        "",
    ])

    md_path = OUT_DIR / "STATE_ROLLOUT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_evaluation(epochs=args.epochs, patience=args.patience, device_str=args.device)
