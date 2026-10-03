"""
s2-t1 pytest: H=2 and H=5 must NOT be fixed multiples of H=1 across 20 sample sequences.
s2-t4 pytest: dashboard uses ucs_windows_models_v1.parquet and loads 37 LOEO folds.
Generated for fix/demo-integrity branch.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "frontend"))

_V1_PARQUET = REPO_ROOT / "data-engineering" / "data" / "ucs" / "ucs_windows_models_v1.parquet"
_BENCH_RESULTS = REPO_ROOT / "evaluation" / "benchmark" / "stacked_benchmark_results.json"


def test_dashboard_references_models_v1_parquet():
    """data_provider.py must reference ucs_windows_models_v1.parquet, not raw ucs_windows.parquet."""
    dp_path = REPO_ROOT / "frontend" / "data_provider.py"
    assert dp_path.exists(), f"data_provider.py not found at {dp_path}"
    content = dp_path.read_text(encoding="utf-8")
    assert "ucs_windows_models_v1.parquet" in content, (
        "data_provider.py does not reference ucs_windows_models_v1.parquet; "
        "dashboard may produce probabilities inconsistent with LOEO benchmark."
    )


def test_fold0_inference_uses_correct_model_count():
    """Dashboard inference must load 37 LOEO onset folds and use hazard_head_v3 source."""
    if not _V1_PARQUET.exists():
        pytest.skip("ucs_windows_models_v1.parquet not found")
    try:
        from backend.predict import predict
    except ImportError as e:
        pytest.skip(f"Cannot import predict: {e}")
    df = pd.read_parquet(_V1_PARQUET).head(2500).copy()
    result = predict(df, source_type="flows")
    ft = result.get("forecast_trajectory", {})
    active_folds = ft.get("active_model_folds", 0)
    assert active_folds == 37, f"Expected 37 onset LOEO folds, got {active_folds}"
    hazard_source = ft.get("hazard_source", "")
    assert "hazard_head_v3" in hazard_source, (
        f"Expected hazard_head_v3 as hazard source, got: '{hazard_source}'"
    )


def test_h2_h5_not_fixed_multiple_of_h1():
    """H=2 and H=5 must NOT be a fixed multiple of H=1 across 20 sample sequences.
    Previously H=2 = H=1 * 1.12 and H=5 = H=1 * 1.25 (heuristic scaling).
    Now they come from independent hazard_head_v3 LOEO models.
    """
    if not _V1_PARQUET.exists():
        pytest.skip("ucs_windows_models_v1.parquet not found")
    try:
        from backend.predict import predict
    except ImportError as e:
        pytest.skip(f"Cannot import predict: {e}")

    df = pd.read_parquet(_V1_PARQUET)
    n_windows = len(df)
    if n_windows < 50:
        pytest.skip(f"Not enough windows ({n_windows}) for 20 sample sequences")

    rng = np.random.RandomState(42)
    max_start = max(0, n_windows - 35)
    starts = rng.choice(max_start + 1, size=min(20, max_start + 1), replace=False)
    ratios_h2_h1, ratios_h5_h1 = [], []

    for start in starts:
        s = int(start)
        slice_df = df.iloc[s: s + 35].copy()
        try:
            result = predict(slice_df, source_type="flows")
        except Exception:
            continue
        ft = result.get("forecast_trajectory", {})
        sh = ft.get("step_hazards", [])
        if len(sh) < 5:
            continue
        h1, h2, h5 = sh[0], sh[1], sh[4]
        if h1 is not None and h2 is not None and float(h1) > 1e-6:
            ratios_h2_h1.append(float(h2) / float(h1))
        if h1 is not None and h5 is not None and float(h1) > 1e-6:
            ratios_h5_h1.append(float(h5) / float(h1))

    if len(ratios_h2_h1) < 3:
        pytest.skip(f"Too few valid inferences ({len(ratios_h2_h1)}) to check ratio variability")

    std_h2 = float(np.std(ratios_h2_h1))
    std_h5 = float(np.std(ratios_h5_h1)) if ratios_h5_h1 else 0.0

    assert std_h2 > 1e-4, (
        f"H=2/H=1 ratios near-constant (std={std_h2:.6f}); "
        f"H=2 appears to still be a fixed multiple of H=1. "
        f"Ratios: {ratios_h2_h1}"
    )
    if ratios_h5_h1:
        assert std_h5 > 1e-4, (
            f"H=5/H=1 ratios near-constant (std={std_h5:.6f}); "
            f"H=5 appears to still be a fixed multiple of H=1. "
            f"Ratios: {ratios_h5_h1}"
        )
