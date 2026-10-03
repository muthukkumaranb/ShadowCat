"""
s2-fix3: Real equivalence test for hazard head integrity.
Replaces weak ratio variability tests with exact numerical equivalence verification.
For 20 windows from ucs_windows_models_v1.parquet, asserts that the dashboard hazard H=1/2/5
equals an independent computation (models/pca_32.pkl + scaler -> mean of the 37 fold models)
within 1e-5 tolerance.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ml1"))
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "data-engineering" / "src"))

_V1_PARQUET = REPO_ROOT / "data-engineering" / "data" / "ucs" / "ucs_windows_models_v1.parquet"
_PCA_FILE = REPO_ROOT / "models" / "pca_32.pkl"
_HAZARD_BASE = REPO_ROOT / "ml1" / "artifacts" / "lstm" / "hazard_head_v3"


def test_hazard_real_equivalence_20_windows():
    """Assert dashboard hazard H=1, 2, 5 equals independent computation within 1e-5."""
    if not _V1_PARQUET.exists():
        pytest.skip(f"Dataset not found: {_V1_PARQUET}")
    if not _PCA_FILE.exists():
        pytest.skip(f"PCA file not found: {_PCA_FILE}")

    from backend.models import LSTMClassifier
    from backend.predict import get_pipeline
    from ucs_extractor import UCSExtractor

    # 1. Independent setup: load PCA, scaler, features
    with open(_PCA_FILE, "rb") as f:
        pca_data = pickle.load(f)
    h_pca = pca_data["pca"]
    h_scaler = pca_data["scaler"]
    h_features = pca_data["features"]

    # Load 37 fold models independently for H=1, 2, 5
    h_models = {1: [], 2: [], 5: []}
    for h in (1, 2, 5):
        h_dir = _HAZARD_BASE / f"H{h}"
        assert h_dir.exists(), f"Hazard head dir missing: {h_dir}"
        ckpt_files = sorted(h_dir.glob("model_fold_*.pt"))
        assert len(ckpt_files) == 37, f"Expected 37 fold models for H={h}, got {len(ckpt_files)}"
        for f_path in ckpt_files:
            m = LSTMClassifier(input_size=32, hidden_size=64, num_layers=1, dropout=0.2)
            state = torch.load(f_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "lstm.weight_ih_l0" in state:
                m.load_state_dict(state)
            else:
                m.load_state_dict(state.get("model_state_dict", state))
            m.eval()
            h_models[h].append(m)

    # 2. Pipeline setup
    pipeline = get_pipeline()
    extractor = UCSExtractor()
    df = pd.read_parquet(_V1_PARQUET)

    n_windows = len(df)
    assert n_windows >= 100, f"Expected >=100 windows, got {n_windows}"

    rng = np.random.RandomState(42)
    max_start = max(0, n_windows - 40)
    starts = rng.choice(max_start + 1, size=20, replace=False)

    max_diffs = {1: 0.0, 2: 0.0, 5: 0.0}

    for start_idx in starts:
        s = int(start_idx)
        slice_df = df.iloc[s : s + 35].copy()

        # A. Pipeline inference
        pipe_res = pipeline.predict(slice_df, source_type="flows")
        pipe_hazards = pipe_res["forecast_trajectory"]["step_hazards"]
        p_h1 = pipe_hazards[0]
        p_h2 = pipe_hazards[1]
        p_h5 = pipe_hazards[4]

        assert p_h1 is not None, "H=1 hazard is None"
        assert p_h2 is not None, "H=2 hazard is None"
        assert p_h5 is not None, "H=5 hazard is None"

        # B. Independent computation
        model_tensor = extractor.extract_model_tensor(slice_df, source_type="flows")
        seq_406 = model_tensor[-30:].copy()  # (30, 406)

        seq_df = pd.DataFrame(seq_406, columns=h_features)
        scaled_df = pd.DataFrame(h_scaler.transform(seq_df[h_features]), columns=h_features)
        pca_df = h_pca.transform(scaled_df)
        pca_cols = [f"pca_{k}" for k in range(32)]
        seq_32 = pca_df[pca_cols].to_numpy(dtype=np.float32)
        seq_tensor = torch.as_tensor(seq_32).unsqueeze(0)

        indep_probs = {}
        with torch.no_grad():
            for h in (1, 2, 5):
                probs = [float(m(seq_tensor).cpu().numpy()[0]) for m in h_models[h]]
                indep_probs[h] = float(np.mean(probs))

        diff1 = abs(p_h1 - indep_probs[1])
        diff2 = abs(p_h2 - indep_probs[2])
        diff5 = abs(p_h5 - indep_probs[5])

        max_diffs[1] = max(max_diffs[1], diff1)
        max_diffs[2] = max(max_diffs[2], diff2)
        max_diffs[5] = max(max_diffs[5], diff5)

        assert diff1 < 1e-5, f"H=1 diff {diff1:.2e} >= 1e-5 (pipe={p_h1}, indep={indep_probs[1]})"
        assert diff2 < 1e-5, f"H=2 diff {diff2:.2e} >= 1e-5 (pipe={p_h2}, indep={indep_probs[2]})"
        assert diff5 < 1e-5, f"H=5 diff {diff5:.2e} >= 1e-5 (pipe={p_h5}, indep={indep_probs[5]})"

    print(
        f"\nEquivalence test PASSED across 20 windows! "
        f"Max diffs: H1={max_diffs[1]:.2e}, H2={max_diffs[2]:.2e}, H5={max_diffs[5]:.2e}"
    )


def test_dashboard_references_models_v1_parquet():
    """data_provider.py must reference ucs_windows_models_v1.parquet."""
    dp_path = REPO_ROOT / "frontend" / "data_provider.py"
    assert dp_path.exists(), f"data_provider.py not found at {dp_path}"
    content = dp_path.read_text(encoding="utf-8")
    assert "ucs_windows_models_v1.parquet" in content, (
        "data_provider.py does not reference ucs_windows_models_v1.parquet"
    )
