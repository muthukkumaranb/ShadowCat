"""PCAP upload path: capture -> CIC-IDS2018-schema flows + packet features -> UCS windows -> predict().

The capture used here is a small synthetic TEST FIXTURE written with scapy (it exercises the code path only;
it is not data and no result is reported from it).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data-engineering"))

scapy_all = pytest.importorskip("scapy.all")
pytest.importorskip("cicflowmeter")

T0 = 1518573600.0  # 2018-02-14 02:00:00 UTC


def _fixture_packets(minutes=3):
    from scapy.all import Ether, IP, TCP, UDP, Raw
    pkts = []
    n = int(minutes * 60 / 0.45)
    for i in range(n):
        t = T0 + i * 0.45
        c, s, sp = f"10.0.0.{10 + i % 5}", "10.0.1.25", 40000 + i % 37
        p = Ether() / IP(src=c, dst=s, ttl=64) / TCP(sport=sp, dport=22, flags="PA", seq=1000 + i) / Raw(b"x" * (20 + i % 100))
        p.time = t
        r = Ether() / IP(src=s, dst=c, ttl=63) / TCP(sport=22, dport=sp, flags="A", seq=5000 + i) / Raw(b"y" * (20 + i % 60))
        r.time = t + 0.01
        pkts += [p, r]
        if i % 9 == 0:
            u = Ether() / IP(src=c, dst="10.0.2.53", ttl=64) / UDP(sport=53000 + i, dport=53) / Raw(b"q" * 40)
            u.time = t + 0.2
            pkts.append(u)
    return pkts


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    from scapy.all import wrpcap
    path = tmp_path_factory.mktemp("pcap") / "fixture.pcap"
    wrpcap(str(path), _fixture_packets())
    return path


def test_flows_use_the_cic_ids2018_schema(capture):
    from src.pcap_ingest import pcap_to_flows
    flows = pcap_to_flows(str(capture))
    for col in ("Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
                "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Flow IAT Mean", "Src IP", "Dst IP"):
        assert col in flows.columns, col
    ts = pd.to_datetime(flows["Timestamp"], format="%d/%m/%Y %H:%M:%S", utc=True)
    assert ts.min() >= pd.Timestamp("2018-02-14 02:00:00", tz="UTC")  # UTC, not the machine's local time
    # durations are converted to microseconds (CIC-IDS2018 unit): the longest flow spans ~3 minutes
    assert 1e8 < flows["Flow Duration"].max() < 2e8


def test_packet_features_cover_every_window(capture):
    from src.pcap_ingest import ingest_pcap
    flows = ingest_pcap(str(capture))
    pk = pd.DataFrame(flows.attrs["packet_features_by_window"])
    assert len(pk) == 3
    assert (pk["pkt_ttl_max"] == 64).all() and (pk["pkt_ttl_min"] == 63).all()
    assert (pk["pkt_payload_size_p50"] > 0).all()


def test_extractor_marks_packet_features_present(capture):
    from src.pcap_ingest import ingest_pcap
    from src.ucs_extractor import UCSExtractor
    ex = UCSExtractor()
    flows = ingest_pcap(str(capture))
    windows = ex.extract(flows, source_type="pcap")
    assert len(windows) == 3
    assert (windows["mask_has_packet_level_features"] == 1.0).all()
    assert windows[UCSExtractor.MODEL_INPUT_COLUMNS].notna().all().all()


def test_predict_runs_on_a_capture(capture, tmp_path, monkeypatch):
    monkeypatch.setenv("SHADOWCAT_RUNTIME_DIR", str(tmp_path))
    from src.pcap_ingest import ingest_pcap
    from backend.predict import predict
    out = predict(ingest_pcap(str(capture)), source_type="pcap")
    p = out["forecast_trajectory"]["risk"][0]
    assert 0.0 <= p <= 1.0
    assert len(out["flagged_flows"]) > 0


def test_pcapng_is_accepted(tmp_path):
    from scapy.utils import PcapNgWriter
    from src.pcap_ingest import ingest_pcap
    path = tmp_path / "fixture.pcapng"
    w = PcapNgWriter(str(path))
    for p in _fixture_packets(minutes=1):
        w.write(p)
    w.close()
    flows = ingest_pcap(str(path))
    assert len(flows) > 0


def test_non_capture_file_is_rejected(tmp_path):
    from src.pcap_ingest import ingest_pcap, PcapIngestError
    bad = tmp_path / "not_a_capture.pcap"
    bad.write_bytes(b"hello world, not a capture")
    with pytest.raises(PcapIngestError):
        ingest_pcap(str(bad))
