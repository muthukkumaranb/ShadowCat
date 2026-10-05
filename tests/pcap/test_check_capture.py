"""data-engineering/scripts/check_pcap_capture.py: capture-clock mapping and the per-minute PCAP-vs-dataset run.

Uses a small synthetic TEST FIXTURE capture placed at the 14-02-2018 SSH onset on the capture clock; it exercises
the code path only (the traffic is synthetic, so no result is reported from it).
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
pytest.importorskip("cicflowmeter")
spec = importlib.util.spec_from_file_location("check_pcap_capture", REPO / "data-engineering" / "scripts" / "check_pcap_capture.py")
cpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpc)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_pcap_ingest as fx  # noqa: E402


def test_capture_clock_maps_to_dataset_clock():
    ds = pd.read_parquet(cpc.DATASET, columns=["window_start_utc"])
    minutes = set(pd.to_datetime(ds.window_start_utc, utc=True))
    # 14-02-2018: SSH-Bruteforce starts at 02:01 in the CSVs, 18:01 on the capture clock (+16 h)
    assert cpc.capture_to_dataset_minute(pd.Timestamp("2018-02-14 18:01", tz="UTC"), minutes) == pd.Timestamp("2018-02-14 02:01", tz="UTC")
    # 12:30 CSV time is 16:30 on the capture clock (+4 h)
    assert cpc.capture_to_dataset_minute(pd.Timestamp("2018-02-14 16:30", tz="UTC"), minutes) == pd.Timestamp("2018-02-14 12:30", tz="UTC")
    # outside the dataset
    assert cpc.capture_to_dataset_minute(pd.Timestamp("2018-02-14 02:00", tz="UTC"), minutes) is None


def test_run_lines_up_pcap_and_dataset_minutes(tmp_path, monkeypatch):
    monkeypatch.setenv("SHADOWCAT_CHECK_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("SHADOWCAT_RUNTIME_DIR", str(tmp_path))
    from scapy.all import wrpcap
    monkeypatch.setattr(fx, "T0", 1518631140.0)  # 2018-02-14 17:59:00 UTC, capture clock
    pcap = tmp_path / "fixture.pcap"
    wrpcap(str(pcap), fx._fixture_packets(minutes=3))
    res = cpc.run(pcap, tmp_path / "check.csv")
    assert len(res) == 3 and (tmp_path / "check.csv").exists()
    assert res.p_onset_from_pcap.between(0, 1).all()
    assert res.dataset_minute.notna().all() and res.p_onset_from_dataset.between(0, 1).all()
    assert list(res.label_attack_now) == [0, 0, 1]  # 01:59, 02:00 benign; 02:01 SSH-Bruteforce in the dataset


def test_feature_comparison_runs_and_ranks_features(tmp_path, monkeypatch):
    monkeypatch.setenv("SHADOWCAT_CHECK_RUNTIME_DIR", str(tmp_path))
    from scapy.all import wrpcap
    spec2 = importlib.util.spec_from_file_location(
        "compare_pcap_features", REPO / "data-engineering" / "scripts" / "compare_pcap_features.py")
    cmp_ = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(cmp_)
    monkeypatch.setattr(fx, "T0", 1518631140.0)
    pcap = tmp_path / "fixture.pcap"
    wrpcap(str(pcap), fx._fixture_packets(minutes=3))
    feat, fam = cmp_.report(*cmp_.build(pcap), top=5, out_dir=tmp_path)
    assert len(feat) == 406 and (tmp_path / "pcap_feature_diff.csv").exists()
    assert feat.abs_logit_contrib_diff.is_monotonic_decreasing
    assert "TCP flags" in fam.index
