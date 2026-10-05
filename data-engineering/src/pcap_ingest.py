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
import time
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


class _CachedPacket:
    """Wraps a scapy packet for the cicflowmeter port, caching what it recomputes for every feature.

    The port calls len(packet), packet.sprintf("%TCP.flags%") and str(packet) many times per packet; on a scapy
    packet each call rebuilds the bytes or the text representation. The wrapper computes them once. Everything else
    (layers, fields, payloads, time) is passed through to the real packet, so the flow features are unchanged.
    """

    __slots__ = ("_pkt", "_len", "_flags", "time")

    def __init__(self, pkt):
        self._pkt = pkt
        self._len = len(pkt)
        self._flags = pkt.sprintf("%TCP.flags%") if "TCP" in pkt else ""
        self.time = pkt.time

    def __len__(self):
        return self._len

    def sprintf(self, fmt, *a, **k):
        return self._flags if fmt == "%TCP.flags%" else self._pkt.sprintf(fmt, *a, **k)

    def __str__(self):  # used only in the port's debug log line
        return "packet"

    def __contains__(self, layer):
        return layer in self._pkt

    def __getitem__(self, layer):
        return self._pkt[layer]

    def __getattr__(self, name):
        return getattr(self._pkt, name)


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


class _Fields:
    """Header fields of one layer, read straight from the packet bytes."""

    __slots__ = ("src", "dst", "ihl", "ttl", "proto", "sport", "dport", "window", "payload")


class _FastPacket:
    """A parsed Ethernet/IPv4/TCP-or-UDP packet exposing exactly what the cicflowmeter port reads.

    scapy dissects every packet into Python objects, which dominates run time on large captures. For the common case
    (Ethernet, optional 802.1Q tag, IPv4, first fragment, TCP or UDP) the fields are read with struct instead. The
    values match scapy's: length is the captured frame length, the TCP/UDP payload runs to the end of the frame
    (scapy keeps Ethernet padding after the payload), the time is built the same way as scapy's PcapReader, and the
    flag representations are scapy's own for the same flag bits. Anything else falls back to scapy.
    """

    __slots__ = ("time", "_len", "_ip", "_l4", "_l4name", "_tcpflags", "flags", "proto")

    @staticmethod
    def _name(layer):
        return layer if isinstance(layer, str) else getattr(layer, "__name__", "")

    def __contains__(self, layer):
        n = self._name(layer)
        return n == "IP" or n == self._l4name

    def __getitem__(self, layer):
        n = self._name(layer)
        if n == "IP":
            return self._ip
        if n == self._l4name:
            return self._l4
        raise IndexError(f"Layer [{n}] not found")

    def __len__(self):
        return self._len

    def sprintf(self, fmt, *a, **k):
        if fmt == "%TCP.flags%":
            return self._tcpflags
        raise NotImplementedError(fmt)

    def __str__(self):
        return "packet"


def _iter_packets(pcap_path: str):
    """Yield packets for the flow session: _FastPacket where possible, otherwise a scapy packet wrapped in
    _CachedPacket. Packets without IPv4 TCP/UDP are skipped (as before)."""
    import socket
    import struct
    from decimal import Decimal
    from scapy.all import IP, TCP, UDP, RawPcapReader, conf
    from scapy.utils import EDecimal

    tcp_flag_str, ip_flags = {}, {}
    with RawPcapReader(pcap_path) as reader:
        ll = reader.linktype
        LLcls = conf.l2types.num2layer.get(ll)
        power = Decimal(10) ** Decimal(-9 if getattr(reader, "nano", False) else -6)
        for data, info in reader:
            ts = EDecimal(info.sec + power * info.usec)
            off = None
            if ll == 1 and len(data) >= 14:  # Ethernet
                et = struct.unpack_from("!H", data, 12)[0]
                off = 14
                if et == 0x8100 and len(data) >= 18:  # one 802.1Q tag
                    et = struct.unpack_from("!H", data, 16)[0]
                    off = 18
                if et != 0x0800:
                    off = None
            fast = None
            if off is not None and len(data) >= off + 20:
                vihl, _, tot_len, _, frag, ttl, proto = struct.unpack_from("!BBHHHBB", data, off)
                ihl = vihl & 0x0F
                l4 = off + ihl * 4
                if vihl >> 4 == 4 and ihl >= 5 and (frag & 0x1FFF) == 0 and proto in (6, 17):
                    need = 20 if proto == 6 else 8
                    if len(data) >= l4 + need:
                        ipf = _Fields()
                        ipf.src = socket.inet_ntoa(data[off + 12:off + 16])
                        ipf.dst = socket.inet_ntoa(data[off + 16:off + 20])
                        ipf.ihl, ipf.ttl, ipf.proto = ihl, ttl, proto
                        l4f = _Fields()
                        if proto == 6:
                            l4f.sport, l4f.dport, _, _, doff_flags, l4f.window = struct.unpack_from("!HHIIHH", data, l4)
                            hdr = (doff_flags >> 12) * 4
                            fbits = doff_flags & 0x1FF
                            if hdr >= 20 and len(data) >= l4 + hdr:
                                l4f.payload = data[l4 + hdr:]
                                fast = _FastPacket()
                                fast._l4name = "TCP"
                                if fbits not in tcp_flag_str:
                                    tcp_flag_str[fbits] = TCP(flags=fbits).sprintf("%TCP.flags%")
                                fast._tcpflags = tcp_flag_str[fbits]
                        else:
                            l4f.sport, l4f.dport = struct.unpack_from("!HH", data, l4)
                            l4f.window = None
                            l4f.payload = data[l4 + 8:]
                            fast = _FastPacket()
                            fast._l4name = "UDP"
                            fast._tcpflags = "??"
                        if fast is not None:
                            ipfl = frag >> 13
                            if ipfl not in ip_flags:
                                ip_flags[ipfl] = IP(flags=ipfl).flags
                            fast.time, fast._len, fast._ip, fast._l4 = ts, len(data), ipf, l4f
                            fast.flags, fast.proto = ip_flags[ipfl], proto
            if fast is not None:
                yield fast
                continue
            # Anything else (other link types, ICMP errors carrying TCP/UDP headers, fragments, ...): scapy.
            if LLcls is None:
                continue
            try:
                pkt = LLcls(data)
            except Exception:
                continue
            pkt.time = ts
            if IP in pkt and (TCP in pkt or UDP in pkt):
                yield _CachedPacket(pkt)


def pcap_to_flows(pcap_path: str, progress: bool = False) -> pd.DataFrame:
    """Build CIC-IDS2018-schema flow records from a capture with the `cicflowmeter` Python port."""
    try:
        from cicflowmeter.flow_session import FlowSession
        from cicflowmeter.features import packet_time
    except ImportError as e:  # pragma: no cover
        raise PcapIngestError("PCAP ingestion needs the `cicflowmeter` package (pip install -r requirements.txt).") from e

    # The port formats flow start times in the machine's local timezone; write them in UTC instead.
    def _utc_timestamp(self):
        return datetime.fromtimestamp(float(self.flow.packets[0][0].time), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    original = packet_time.PacketTime.get_timestamp
    packet_time.PacketTime.get_timestamp = _utc_timestamp
    tmp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    try:
        session = FlowSession(output_mode="csv", output=tmp_csv)
        last_gc, n_ip = None, 0
        t_start = time.time()
        for pkt in _iter_packets(pcap_path):
            session.process(pkt)
            n_ip += 1
            t = float(pkt.time)
            if last_gc is None or t - last_gc > 60:
                session.garbage_collect(t)
                last_gc = t
            if progress and n_ip % 200_000 == 0:
                print(f"  {n_ip:,} packets, {time.time() - t_start:.0f}s", flush=True)
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


def ingest_pcap(pcap_path: str, progress: bool = False) -> pd.DataFrame:
    """Flows in the CIC-IDS2018 schema, with per-window packet features attached in
    `df.attrs["packet_features_by_window"]` (a dict of lists, so pandas can copy/compare attrs)
    for UCSExtractor (source_type="pcap")."""
    try:
        import cicflowmeter  # noqa: F401  (pulls in scapy)
    except ImportError as e:
        raise PcapIngestError("PCAP ingestion needs the `cicflowmeter` package, which requires Python 3.12+ "
                              "(pip install -r requirements.txt on Python 3.12 or newer).") from e
    path = _ensure_libpcap(pcap_path)
    flows = pcap_to_flows(path, progress=progress)
    if progress:
        print(f"  {len(flows):,} flows; computing packet-level features ...", flush=True)
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
