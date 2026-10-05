"""
PCAP -> UCS ingestion for uploaded captures (dashboard + CLI).

    raw PCAP / PCAPNG
      ├─► flow records in the CSE-CIC-IDS2018 CICFlowMeter schema (Python `cicflowmeter` port)
      └─► 12 packet-level features per 1-minute window (src/pcap_extractor.py, same capture clock)
    → predict(flows, source_type="pcap")

Design notes
- Flows: the `cicflowmeter` Python package (a port of the Java CICFlowMeter used to build the
  CIC-IDS2018 CSVs). Its column names map onto the CIC-IDS2018 CSV schema by name normalisation
  (78/80 columns; "CWE Flag Count" is aliased, "Label" does not exist for new captures). The port
  reports durations / inter-arrival / active / idle times in seconds; CIC-IDS2018 uses microseconds,
  so those columns are converted. Flow timestamps are written in UTC (the port uses local time).
- Packet features: computed with the project's own parser (`extract_features_from_pcap_stream`) on the
  SAME capture, for the same 1-minute windows, with no CIC-IDS2018 clock offset (flows and packets
  share the capture clock).
- Values come only from the capture. Nothing is filled with defaults here; anything the capture
  cannot provide is left to the extractor's documented mask / frozen-median handling.

CLI:
    python data-engineering/src/pcap_ingest.py capture.pcap [--out flows.csv]
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

PCAP_MAGICS = {b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"}
PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"

# cicflowmeter (Python) reports these in seconds; CIC-IDS2018 CSVs use microseconds.
_SECONDS_TO_MICROS = [
    "flow_duration",
    "flow_iat_mean", "flow_iat_max", "flow_iat_min", "flow_iat_std",
    "fwd_iat_tot", "fwd_iat_max", "fwd_iat_min", "fwd_iat_mean", "fwd_iat_std",
    "bwd_iat_tot", "bwd_iat_max", "bwd_iat_min", "bwd_iat_mean", "bwd_iat_std",
    "active_max", "active_min", "active_mean", "active_std",
    "idle_max", "idle_min", "idle_mean", "idle_std",
]
_ALIASES = {"cwr_flag_count": "CWE Flag Count", "src_ip": "Src IP", "dst_ip": "Dst IP", "src_port": "Src Port"}


class PcapIngestError(ValueError):
    pass


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "")


def _ensure_libpcap(path: str) -> str:
    """Return a classic-libpcap path for `path` (PCAPNG is rewritten to a temporary .pcap)."""
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic in PCAP_MAGICS:
        return path
    if magic != PCAPNG_MAGIC:
        raise PcapIngestError("File is not a PCAP or PCAPNG capture (unrecognised magic number).")
    from scapy.utils import PcapNgReader, PcapWriter

    out = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False).name
    with PcapNgReader(path) as r, PcapWriter(out, sync=False) as w:
        for pkt in r:
            w.write(pkt)
    return out


def pcap_to_flows(pcap_path: str) -> pd.DataFrame:
    """Build CIC-IDS2018-schema flow records from a capture with the `cicflowmeter` Python port."""
    try:
        from cicflowmeter.flow_session import FlowSession
        from cicflowmeter.features import packet_time
    except ImportError as e:  # pragma: no cover
        raise PcapIngestError("PCAP ingestion needs the `cicflowmeter` package (pip install -r requirements.txt).") from e
    from scapy.all import IP, TCP, UDP, PcapReader

    # The port formats flow start times in the machine's local timezone; write them in UTC instead.
    def _utc_timestamp(self):
        return datetime.fromtimestamp(float(self.flow.packets[0][0].time), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    original = packet_time.PacketTime.get_timestamp
    packet_time.PacketTime.get_timestamp = _utc_timestamp
    tmp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    try:
        session = FlowSession(output_mode="csv", output=tmp_csv)
        last_gc, n_ip = None, 0
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                if IP in pkt and (TCP in pkt or UDP in pkt):
                    session.process(pkt)
                    n_ip += 1
                    t = float(pkt.time)
                    if last_gc is None or t - last_gc > 60:
                        session.garbage_collect(t)
                        last_gc = t
        session.flush_flows()
        if n_ip == 0 or os.path.getsize(tmp_csv) == 0:
            raise PcapIngestError("The capture contains no IPv4 TCP/UDP packets.")
        raw = pd.read_csv(tmp_csv)
    finally:
        packet_time.PacketTime.get_timestamp = original
        try:
            os.unlink(tmp_csv)
        except OSError:
            pass

    for c in _SECONDS_TO_MICROS:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce") * 1e6

    import yaml
    mapping = yaml.safe_load((HERE.parent / "configs" / "canonical_mapping.yaml").read_text())["mapping"]
    by_norm = {_norm(k): k for k in mapping}
    rename = {}
    for c in raw.columns:
        if c in _ALIASES:
            rename[c] = _ALIASES[c]
        elif _norm(c) in by_norm:
            rename[c] = by_norm[_norm(c)]
    flows = raw.rename(columns=rename)
    ts = pd.to_datetime(flows["Timestamp"], format="%Y-%m-%d %H:%M:%S", utc=True)
    flows["Timestamp"] = ts.dt.strftime("%d/%m/%Y %H:%M:%S")  # CIC-IDS2018 CSV format, UTC
    flows = flows.sort_values("Timestamp", key=lambda s: pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S")).reset_index(drop=True)
    return flows


def pcap_packet_features(pcap_path: str, flows: pd.DataFrame) -> pd.DataFrame:
    """12 packet-level features per 1-minute window covering the flows, on the capture's own clock."""
    from src.pcap_extractor import extract_features_from_pcap_stream

    ts = pd.to_datetime(flows["Timestamp"], format="%d/%m/%Y %H:%M:%S", utc=True)
    starts = pd.date_range(ts.min().floor("min"), ts.max().floor("min"), freq="min")
    bounds = [(s, s + pd.Timedelta(minutes=1), s.isoformat()) for s in starts]
    pkt = extract_features_from_pcap_stream(pcap_path, bounds, apply_cic2018_clock_offset=False)
    pkt["window_start_utc"] = pkt["window_id"]  # ISO-8601 UTC window start
    return pkt.drop(columns=["window_id"])


def ingest_pcap(pcap_path: str) -> pd.DataFrame:
    """Flows in the CIC-IDS2018 schema, with per-window packet features attached in
    `df.attrs["packet_features_by_window"]` (a dict of lists, so pandas can copy/compare attrs)
    for UCSExtractor (source_type="pcap")."""
    try:
        import cicflowmeter  # noqa: F401  (pulls in scapy)
    except ImportError as e:
        raise PcapIngestError("PCAP ingestion needs the `cicflowmeter` package, which requires Python 3.12+ "
                              "(pip install -r requirements.txt on Python 3.12 or newer).") from e
    path = _ensure_libpcap(pcap_path)
    flows = pcap_to_flows(path)
    flows.attrs["packet_features_by_window"] = pcap_packet_features(path, flows).to_dict(orient="list")
    flows.attrs["source_capture"] = os.path.basename(pcap_path)
    return flows


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Convert a PCAP/PCAPNG capture to CIC-IDS2018-schema flows + packet features")
    ap.add_argument("pcap")
    ap.add_argument("--out", default=None, help="write the flow records to this CSV")
    a = ap.parse_args()
    df = ingest_pcap(a.pcap)
    pk = pd.DataFrame(df.attrs["packet_features_by_window"])
    print(f"{len(df)} flows, {len(pk)} one-minute windows with packet features "
          f"({pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S').min()} .. "
          f"{pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S').max()} UTC)")
    if a.out:
        df.to_csv(a.out, index=False)
        pk.to_csv(Path(a.out).with_suffix(".packet_features.csv"), index=False)
        print(f"wrote {a.out}")
