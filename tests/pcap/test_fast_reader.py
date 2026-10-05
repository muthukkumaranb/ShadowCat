"""The struct-based packet reader in src/pcap_ingest.py must give the same flow records as scapy's own dissection.

The capture is a synthetic TEST FIXTURE covering the cases that differ between parsers: Ethernet padding on short
frames, 802.1Q tags, IP and TCP options, UDP, fragments, ICMP errors that carry a TCP header, FIN/RST and
non-Ethernet frames. No result is reported from it.
"""
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data-engineering"))
pytest.importorskip("cicflowmeter")
from scapy.all import Ether, Dot1Q, IP, TCP, UDP, ICMP, Raw, IPOption_RR, wrpcap, PcapReader, fragment  # noqa: E402

import src.pcap_ingest as pi  # noqa: E402

T0 = 1518631080.0


def _capture(path):
    pkts, t = [], T0
    def add(p, dt=0.01):
        nonlocal t
        t += dt
        p.time = t
        pkts.append(p)
    c, s = "10.0.0.5", "172.31.69.25"
    for i in range(40):
        sp = 40000 + i % 7
        add(Ether() / IP(src=c, dst=s) / TCP(sport=sp, dport=22, flags="S", options=[("MSS", 1460), ("SAckOK", b"")]))
        add(Ether() / IP(src=s, dst=c) / TCP(sport=22, dport=sp, flags="SA"))
        add(Ether() / IP(src=c, dst=s) / TCP(sport=sp, dport=22, flags="A"))  # 54-byte frame, padded to 60 by Ether
        add(Ether() / IP(src=c, dst=s, options=[IPOption_RR()]) / TCP(sport=sp, dport=22, flags="PA") / Raw(b"x" * (10 + i)))
        add(Ether() / Dot1Q(vlan=7) / IP(src=s, dst=c) / TCP(sport=22, dport=sp, flags="PA") / Raw(b"y" * (30 + i)))
        add(Ether() / IP(src=c, dst=s) / TCP(sport=sp, dport=22, flags="FA" if i % 2 else "RA"))
        add(Ether() / IP(src=c, dst="10.0.2.53") / UDP(sport=53000 + i, dport=53) / Raw(b"q" * 30))
        add(Ether() / IP(src="10.0.2.53", dst=c) / UDP(sport=53, dport=53000 + i) / Raw(b"a" * 90))
        add(Ether() / IP(src=s, dst=c) / ICMP(type=3, code=3) / IP(src=c, dst=s) / TCP(sport=sp, dport=22))
        if i % 10 == 0:
            for fr in fragment(IP(src=c, dst=s) / UDP(sport=5000, dport=6000) / Raw(b"z" * 3000), fragsize=1000):
                add(Ether() / fr)
    # pad the short frames exactly as a NIC would (Ethernet minimum 60 bytes without FCS)
    padded = []
    for p in pkts:
        b = bytes(p)
        q = Ether(b + b"\x00" * max(0, 60 - len(b)))
        q.time = p.time
        padded.append(q)
    wrpcap(str(path), padded)


def _reference_flows(path):
    """The same pipeline with scapy's full dissection (the implementation before the fast reader)."""
    from cicflowmeter.flow_session import FlowSession
    from scapy.all import IP as IPc, TCP as TCPc, UDP as UDPc
    real_iter = pi._iter_packets

    def scapy_iter(p):
        with PcapReader(p) as r:
            for pkt in r:
                if IPc in pkt and (TCPc in pkt or UDPc in pkt):
                    yield pi._CachedPacket(pkt)

    pi._iter_packets = scapy_iter
    try:
        return pi.pcap_to_flows(str(path))
    finally:
        pi._iter_packets = real_iter


def test_fast_reader_matches_scapy(tmp_path):
    cap = tmp_path / "mixed.pcap"
    _capture(cap)
    fast = pi.pcap_to_flows(str(cap)).reset_index(drop=True)
    ref = _reference_flows(cap).reset_index(drop=True)
    assert len(fast) == len(ref) > 30
    key = ["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol", "Timestamp", "Flow Duration"]
    fast = fast.sort_values(key).reset_index(drop=True)
    ref = ref.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(fast, ref, check_exact=False, rtol=1e-9)
