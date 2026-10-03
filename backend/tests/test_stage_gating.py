"""s2b-t5: stage output is gated and never uses invented ATT&CK IDs."""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backend"))

SLICES = ["benign_02-03-2018", "botnet_02-03-2018", "ssh_14-02-2018"]


@pytest.mark.parametrize("name", SLICES)
def test_stage_gating(name):
    path = REPO / "frontend/demo_data" / f"{name}_windows.parquet"
    if not path.exists():
        pytest.skip(f"missing {path}")
    from backend.predict import get_pipeline

    fc = get_pipeline().predict(pd.read_parquet(path), source_type="windows")["forecast_trajectory"]
    for stage, md in zip(fc["stage"], fc["mitre_details"]):
        assert md["tactic_id"] not in ("TA0000",) and md["technique_id"] not in ("T0000",)
        if stage == "No stage determined":
            assert md["tactic_id"] is None and md["technique_id"] is None
            assert md["likely_next_techniques"] == []
        else:
            assert md["stage_confidence"] > 0.5
            assert md["stage_head_top_class"] == stage != "Unknown/Other"
            assert md["technique_id"].startswith("T")


def test_resolve_unknown_has_no_ids():
    from backend.mitre_kb import get_mitre_kb

    info = get_mitre_kb().resolve_stage("Unknown/Other")
    assert info["tactic_id"] is None and info["technique_id"] is None
