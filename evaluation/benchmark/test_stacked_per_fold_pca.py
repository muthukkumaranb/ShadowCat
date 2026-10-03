"""s2b-t4: the deployed onset probability equals the reference computation of
evaluation/benchmark/reproduce_stacked_benchmark.py (each fold's own checkpoint PCA and LR
scaler, mean over the 37 onset folds) within 1e-5, for 20 windows of the models' dataset.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "evaluation" / "benchmark"))

from reproduce_stacked_benchmark import LOOKBACK, ResidualLSTM  # noqa: E402

DATA = REPO / "data-engineering/data/ucs/ucs_windows_models_v1.parquet"
ONSET = REPO / "ml1/artifacts/lstm/lstm_stacked/onset"


def reference_onset(df, i, folds):
    probs = []
    for ckpt, model in folds:
        x = df.loc[i, ckpt["lr_features"]].to_numpy(float)
        lr_logit = ((x - np.array(ckpt["scaler_mean"])) / np.array(ckpt["scaler_scale"])) @ np.array(
            ckpt["lr_coef"]).reshape(-1) + np.array(ckpt["lr_intercept"]).reshape(-1)[0]
        hist = df.loc[i - LOOKBACK + 1: i, ckpt["pca_columns"]].to_numpy(float)
        seq = ((hist - np.array(ckpt["pca_mean"])) @ np.array(ckpt["pca_components"]).T).astype(np.float32)
        with torch.no_grad():
            logit = model.forward_logits(torch.tensor(seq[None]), torch.tensor([lr_logit], dtype=torch.float32)).numpy()[0]
        probs.append(1.0 / (1.0 + np.exp(-logit / ckpt["temperature"])))
    return float(np.mean(probs))


def test_deployed_onset_equals_per_fold_reference():
    if not DATA.exists():
        pytest.skip(f"missing {DATA}")
    from backend.predict import get_pipeline

    df = pd.read_parquet(DATA).sort_values("window_start_utc").reset_index(drop=True)
    folds = []
    for f in sorted(ONSET.glob("model_fold_*.pt")):
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        m = ResidualLSTM()
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        folds.append((ckpt, m))
    assert len(folds) == 37

    rng = np.random.RandomState(42)
    idx = sorted(int(i) for i in rng.choice(np.arange(LOOKBACK - 1, len(df)), size=20, replace=False))
    pipeline = get_pipeline()
    diffs = []
    for i in idx:
        res = pipeline.predict(df.iloc[i - LOOKBACK + 1: i + 1], source_type="windows")
        deployed = res["forecast_trajectory"]["onset_probability"]
        diffs.append(abs(deployed - reference_onset(df, i, folds)))
    assert max(diffs) < 1e-5, json.dumps({"max_diff": max(diffs), "windows": idx})
