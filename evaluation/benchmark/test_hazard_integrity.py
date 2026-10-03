"""
Hazard-head integrity (s2-fix1b).

hazard_head_v3 (H=1/2/5) is not shown by the dashboard: its reported LOEO ROC-AUC
(0.789/0.843/0.770) does not reproduce through models/pca_32.pkl + scaler, the input
path the dashboard would have to use (see evaluation/benchmark/reproduce_hazard_v3.py
and hazard_v3_results.json). These tests pin that decision:
  1. the committed reproduction result shows the pca_32.pkl path does NOT reproduce;
  2. predict() returns a single onset probability and no per-horizon hazard values.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

_V1_PARQUET = REPO_ROOT / "data-engineering" / "data" / "ucs" / "ucs_windows_models_v1.parquet"
_RESULTS = REPO_ROOT / "evaluation" / "benchmark" / "hazard_v3_results.json"


def test_hazard_v3_does_not_reproduce_via_pca32():
    runs = json.loads(_RESULTS.read_text())["runs"]
    pca32 = [r for r in runs.values() if r["pca_mode"] == "pca32"]
    assert pca32, "no pca32 reproduction run recorded"
    assert not any(r["all_horizons_reproduce"] for r in pca32)


def test_predict_has_no_per_horizon_hazards():
    if not _V1_PARQUET.exists():
        pytest.skip(f"Dataset not found: {_V1_PARQUET}")
    from backend.predict import get_pipeline

    df = pd.read_parquet(_V1_PARQUET).head(40)
    res = get_pipeline().predict(df, source_type="flows")
    fc = res["forecast_trajectory"]
    for key in ("step_hazards", "horizon_is_modelled", "hazard_source", "rollout_infiltration_prob"):
        assert key not in fc
    assert fc["risk"] == [fc["onset_probability"]]
    assert fc["onset_probability_label"] == "P(attack within the next 5 minutes)"
    assert 0.0 <= fc["onset_probability"] <= 1.0


def test_dashboard_references_models_v1_parquet():
    """data_provider.py must reference ucs_windows_models_v1.parquet."""
    content = (REPO_ROOT / "frontend" / "data_provider.py").read_text(encoding="utf-8")
    assert "ucs_windows_models_v1.parquet" in content
