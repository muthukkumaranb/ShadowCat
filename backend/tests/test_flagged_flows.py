"""s2b-t8: flows are ranked by the onset LR stage, never padded with placeholder values."""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "frontend"))

DEMO = REPO / "frontend/demo_data"


def test_window_input_has_no_flows():
    from data_provider import run_core_ml_inference

    res = run_core_ml_inference(pd.read_parquet(DEMO / "benign_02-03-2018_windows.parquet"), source_type="windows")
    assert res["flagged_flows"] == []
    assert len(res["lr_contributions"]) == 10


def test_csv_flows_ranked_by_lr_score():
    path = DEMO / "botnet_02-03-2018.csv.gz"
    if not path.exists():
        pytest.skip(f"missing {path}")
    from data_provider import run_core_ml_inference

    raw = pd.read_csv(path)
    flows = run_core_ml_inference(raw, source_type="csv")["flagged_flows"]
    assert flows
    scores = [f["lr_score"] for f in flows]
    assert scores == sorted(scores, reverse=True)
    for f in flows:
        assert f["source"] is None and f["destination"] is None  # this CSV has no IP columns
        assert raw.loc[f["row"], "Timestamp"] == f["timestamp"]
