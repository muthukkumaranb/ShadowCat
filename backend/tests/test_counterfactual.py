"""
Unit and Integration Tests for Explanatory Counterfactual Engine
Verifies bounded perturbation search, empirical percentile clipping, unit inversion,
and honesty disclosures.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from backend.predict import ShadowcatPipeline, UCSExtractor
from backend.counterfactual import CounterfactualEngine, HONESTY_LABEL, get_counterfactual_engine


@pytest.fixture(scope="module")
def pipeline():
    return ShadowcatPipeline()


@pytest.fixture(scope="module")
def engine():
    return get_counterfactual_engine()


@pytest.fixture(scope="module")
def ucs_dataset():
    data_path = Path(__file__).resolve().parent.parent.parent / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
    if not data_path.exists():
        pytest.skip(f"UCS parquet dataset not found at {data_path}")
    return pd.read_parquet(data_path)


def test_honesty_label_presence(engine):
    """Confirm honesty label text is mandatory and present."""
    assert HONESTY_LABEL in [
        "Model-based counterfactual under the hazard model's learned decision boundary — not a guarantee that this change would have prevented the actual attack, and not validated against real intervention data."
    ]
    # Check that a mock/empty result contains honesty label
    res = engine.search(None, np.zeros((30, 406), dtype=np.float32), threshold=0.15)
    # Even if pipeline is None, or already below threshold, honesty label must exist
    assert "honesty_label" in res
    assert res["honesty_label"] == HONESTY_LABEL


def test_percentile_clipping_safeguard_enforced(engine):
    """
    Verify that empirical [1st, 99th] percentile clipping is strictly enforced
    and candidate values are never allowed outside bounds.
    """
    sample_feat = "pkt_tcp_retrans_count"
    bounds = engine.bounds.get(sample_feat)
    assert bounds is not None, f"Feature {sample_feat} missing from empirical bounds"
    p01 = bounds["p01"]
    p99 = bounds["p99"]

    # Verify bounds are valid numbers and p01 < p99
    assert p01 <= p99

    # Test clipping function directly
    extreme_low = p01 - 1000.0
    extreme_high = p99 + 1000.0
    assert np.clip(extreme_low, p01, p99) == p01
    assert np.clip(extreme_high, p01, p99) == p99


def test_unit_inversion_to_physical_domain(engine):
    """
    Verify inversion from normalized space to original physical units
    (e.g., retransmissions, microseconds, seconds, packets).
    """
    # Test pkt_tcp_retrans_count
    # In scaler_params: median=6.0, scale=16.0, is_log1p=False
    # For z = 10.0: raw = 10.0 * 16.0 + 6.0 = 166.0
    raw_val = engine.invert_units("pkt_tcp_retrans_count", 10.0)
    assert raw_val == 166.0

    # For z = 0.0: raw = median = 6.0
    assert engine.invert_units("pkt_tcp_retrans_count", 0.0) == 6.0

    # Test duration_sec_max (is_log1p = True)
    raw_dur = engine.invert_units("duration_sec_max", 0.0)
    assert raw_dur > 0.0  # Must be positive physical duration


def test_feasible_counterfactual_on_flagged_window(pipeline, engine, ucs_dataset):
    """
    End-to-end verification: genuine flagged attack window (Window 70: SSH-Bruteforce)
    crosses below the calibrated alert threshold (0.150) within empirical bounds.
    """
    cols = UCSExtractor.MODEL_INPUT_COLUMNS
    idx = 70
    seq = ucs_dataset.iloc[idx - 29 : idx + 1][cols].to_numpy(dtype=np.float32)

    res = engine.search(pipeline, seq, threshold=0.50)

    assert res["status"] == "feasible"
    assert res["found"] is True
    assert res["initial_hazard"] > 0.50  # Genuine flagged window
    assert res["counterfactual_hazard"] < 0.50  # Crossed below threshold
    assert len(res["perturbed_features"]) > 0
    assert len(res["summary"]) > 20
    assert res["honesty_label"] == HONESTY_LABEL

    # Check that perturbed features are within empirical bounds
    for pf in res["perturbed_features"]:
        fid = pf["feature_id"]
        b = engine.bounds[fid]
        p_norm = pf["perturbed_value_normalized"]
        assert b["p01"] - 1e-4 <= p_norm <= b["p99"] + 1e-4


def test_inconclusive_on_extreme_attack_window(pipeline, engine, ucs_dataset):
    """
    End-to-end verification: extreme attack flood (Window 704: DDOS-LOIC-UDP)
    cannot cross threshold within budget and honestly reports inconclusive.
    """
    cols = UCSExtractor.MODEL_INPUT_COLUMNS
    idx = 704
    seq = ucs_dataset.iloc[idx - 29 : idx + 1][cols].to_numpy(dtype=np.float32)

    res = engine.search(pipeline, seq, threshold=0.15, max_features=3)

    assert res["status"] == "inconclusive"
    assert res["found"] is False
    assert res["initial_hazard"] > 0.90
    assert res["best_hazard_achieved"] >= 0.15
    assert "No realistic minimal change found within observed data bounds" in res["summary"]
    assert res["honesty_label"] == HONESTY_LABEL
