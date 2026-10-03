"""s2-fix5: measure the effect of using ONE PCA (pca_32_stacked.pkl) for all 37 stacked fold
models instead of each fold checkpoint's own PCA (pca_components / pca_mean).

For 20 windows of ucs_windows_models_v1.parquet (RandomState(42), windows with a full
30-window history), computes the 37-fold mean detection and onset probability two ways:
  single_pca : sequence projected with ml1/artifacts/lstm/lstm_stacked/pca_32_stacked.pkl
  per_fold   : sequence projected with each checkpoint's pca_mean / pca_components
               (as in evaluation/benchmark/reproduce_stacked_benchmark.py)
Both use each fold's own LR scaler and coefficients on the current window, and its temperature.

Writes docs/demo_check/stacked_pca_measurement.json.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ml1"))
sys.path.insert(0, str(REPO / "evaluation" / "benchmark"))

from reproduce_stacked_benchmark import LOOKBACK, ResidualLSTM  # noqa: E402

DATA = "data-engineering/data/ucs/ucs_windows_models_v1.parquet"
STACKED = REPO / "ml1/artifacts/lstm/lstm_stacked"
PCA_COLS = [f"pca_{i}" for i in range(32)]


def load_folds(task):
    folds = []
    for f in sorted((STACKED / task).glob("model_fold_*.pt"), key=lambda p: int(p.stem.split("_")[-1])):
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        m = ResidualLSTM()
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        folds.append((ckpt, m))
    return folds


def fold_prob(ckpt, model, seq32, x_now):
    z = ((x_now - np.array(ckpt["scaler_mean"])) / np.array(ckpt["scaler_scale"])) @ np.array(
        ckpt["lr_coef"]).reshape(-1) + np.array(ckpt["lr_intercept"]).reshape(-1)[0]
    with torch.no_grad():
        logit = model.forward_logits(torch.tensor(seq32[None], dtype=torch.float32),
                                     torch.tensor([z], dtype=torch.float32)).numpy()[0]
    return float(1.0 / (1.0 + np.exp(-logit / ckpt["temperature"])))


def main():
    df = pd.read_parquet(REPO / DATA).sort_values("window_start_utc").reset_index(drop=True)
    with open(STACKED / "pca_32_stacked.pkl", "rb") as f:
        shared = pickle.load(f)
    feats = shared["features"]
    shared_z = shared["pca"].transform(df[feats])[PCA_COLS].to_numpy(np.float32)

    rng = np.random.RandomState(42)
    idx = sorted(int(i) for i in rng.choice(np.arange(LOOKBACK - 1, len(df)), size=20, replace=False))

    rows = []
    summary = {}
    for task in ("detection", "onset"):
        folds = load_folds(task)
        single, per_fold = [], []
        for i in idx:
            ps, pf = [], []
            for ckpt, m in folds:
                x_now = df.loc[i, ckpt["lr_features"]].to_numpy(float)
                ps.append(fold_prob(ckpt, m, shared_z[i - LOOKBACK + 1: i + 1], x_now))
                hist = df.loc[i - LOOKBACK + 1: i, ckpt["pca_columns"]].to_numpy(float)
                own = ((hist - np.array(ckpt["pca_mean"])) @ np.array(ckpt["pca_components"]).T).astype(np.float32)
                pf.append(fold_prob(ckpt, m, own, x_now))
            single.append(float(np.mean(ps)))
            per_fold.append(float(np.mean(pf)))
        single, per_fold = np.array(single), np.array(per_fold)
        diff = np.abs(single - per_fold)
        summary[task] = {
            "mean_abs_diff": float(diff.mean()),
            "max_abs_diff": float(diff.max()),
            "pearson_r": float(np.corrcoef(single, per_fold)[0, 1]),
        }
        for j, i in enumerate(idx):
            if task == "detection":
                rows.append({"window_index": i, "window_start_utc": str(df.loc[i, "window_start_utc"]),
                             "label_attack_type": str(df.loc[i, "label_attack_type"])})
            rows[j][f"{task}_single_pca"] = float(single[j])
            rows[j][f"{task}_per_fold_pca"] = float(per_fold[j])

    out = REPO / "docs/demo_check/stacked_pca_measurement.json"
    out.write_text(json.dumps({"data": DATA, "n_windows": len(idx), "summary": summary, "windows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
