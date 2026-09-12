"""
Fast Memory-Mapped/Buffered PCAP Reconciliation & Feature Extractor
SIH26153 - Cyber World Model Architecture (Data Engineer Track)

Loads PCAP into memory buffer and extracts all 12 genuine packet features
across all 543 windows for 14-02-2018 with full timestamp reconciliation.
"""

import os
import sys
import math
import struct
import socket
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

PCAP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_pcap", "UCAP172.31.69.25.pcap"))
UCS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "ucs", "ucs_windows.parquet"))
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "ucs", "packet_features.parquet"))


def calculate_port_scan_monotonicity(ports: List[int]) -> float:
    if len(ports) < 3:
        return 0.0
    ports_arr = np.array(ports, dtype=np.int64)
    diffs = np.diff(ports_arr)
    seq_steps = np.abs(diffs) == 1
    seq_ratio = float(np.mean(seq_steps))
    unique, counts = np.unique(ports_arr, return_counts=True)
    probs = counts / len(ports_arr)
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
    max_entropy = math.log2(len(unique)) if len(unique) > 1 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return float(np.clip(0.7 * seq_ratio + 0.3 * norm_entropy, 0.0, 1.0))


def run_reconciliation_and_extraction(
    pcap_path: str = PCAP_PATH,
    ucs_path: str = UCS_PATH,
    output_path: str = OUTPUT_PATH,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print(f"[*] Loading UCS windows: {ucs_path}", flush=True)
    ucs_df = pd.read_parquet(ucs_path)
    d14 = ucs_df[ucs_df["source_day"].str.contains("14-02-2018", na=False)].sort_values("window_start_utc").reset_index(drop=True)
    print(f"    Total 14-02-2018 windows in UCS: {len(d14)}", flush=True)

    # 1. Build Reconciled Window Target Bounds
    reconciled_windows = []
    window_data = {}
    for _, r in d14.iterrows():
        w_start = pd.to_datetime(r["window_start_utc"])
        w_end = pd.to_datetime(r["window_end_utc"])
        w_id = r["window_id"]
        
        # Timezone offset mapping:
        # Afternoon AST (01:00 - 09:59 CSV) -> 17:00 - 01:59 UTC (+16h) [SSH Attack is at 02:00-03:35 CSV -> 18:00-19:35 UTC]
        # Morning AST (10:00 - 12:59 CSV) -> 14:00 - 16:59 UTC (+4h) [FTP Attack is at 10:30-12:15 CSV -> 14:30-16:15 UTC]
        if w_start.hour < 10:
            offset = pd.Timedelta(hours=16)
        else:
            offset = pd.Timedelta(hours=4)
            
        pcap_s = (w_start + offset).timestamp()
        pcap_e = (w_end + offset).timestamp()
        reconciled_windows.append((pcap_s, pcap_e, w_id, r["label_attack_type"]))
        
        window_data[w_id] = {
            "window_id": w_id,
            "ttls": [],
            "frag_mf_count": 0,
            "frag_df_count": 0,
            "payload_sizes": [],
            "seen_tcp_seqs": set(),
            "retrans_count": 0,
            "dst_ports_by_src": {},
        }

    # Sort boundaries for binary search
    reconciled_windows = sorted(reconciled_windows, key=lambda x: x[0])
    w_starts = np.array([x[0] for x in reconciled_windows])
    w_ends = np.array([x[1] for x in reconciled_windows])
    w_ids = [x[2] for x in reconciled_windows]

    # 2. Fast In-Memory Parsing of Entire PCAP
    print(f"[*] Reading full PCAP into memory buffer ({os.path.getsize(pcap_path) / (1024*1024):.2f} MB)...", flush=True)
    t0 = time.time()
    with open(pcap_path, "rb") as f:
        pcap_bytes = f.read()
    print(f"    Loaded {len(pcap_bytes) / (1024*1024):.2f} MB into memory in {time.time() - t0:.2f}s", flush=True)

    magic = struct.unpack_from("<I", pcap_bytes, 0)[0]
    endian = "<" if magic == 0xa1b2c3d4 else (">" if magic == 0xd4c3b2a1 else "<")
    
    offset = 24  # Skip global header
    total_len = len(pcap_bytes)
    pkt_count = 0
    matched_pkts = 0
    t_parse_start = time.time()

    while offset + 16 <= total_len:
        ts_sec, ts_usec, incl_len, orig_len = struct.unpack_from(f"{endian}IIII", pcap_bytes, offset)
        offset += 16
        if offset + incl_len > total_len:
            break
        pkt_ts = ts_sec + (ts_usec / 1e6)
        pkt_start = offset
        offset += incl_len
        pkt_count += 1

        # Binary search interval match
        idx = np.searchsorted(w_starts, pkt_ts, side="right") - 1
        if idx < 0 or idx >= len(w_starts):
            continue
        if not (w_starts[idx] <= pkt_ts < w_ends[idx]):
            continue

        w_id = w_ids[idx]
        wd = window_data[w_id]
        matched_pkts += 1

        # Check Ethernet + IPv4
        if incl_len < 34:
            continue
        ethertype = struct.unpack_from(">H", pcap_bytes, pkt_start + 12)[0]
        if ethertype != 0x0800:
            continue

        ip_offset = pkt_start + 14
        ip_b0 = pcap_bytes[ip_offset]
        ihl = (ip_b0 & 0x0F) * 4
        tot_len = struct.unpack_from(">H", pcap_bytes, ip_offset + 2)[0]
        flags_frag = struct.unpack_from(">H", pcap_bytes, ip_offset + 6)[0]
        flags = (flags_frag >> 13) & 0x07
        ttl = pcap_bytes[ip_offset + 8]
        proto = pcap_bytes[ip_offset + 9]
        src_ip = socket.inet_ntoa(pcap_bytes[ip_offset + 12 : ip_offset + 16])
        dst_ip = socket.inet_ntoa(pcap_bytes[ip_offset + 16 : ip_offset + 20])

        # 1. TTL
        wd["ttls"].append(int(ttl))

        # 2. Fragmentation flags
        if flags & 0x02:
            wd["frag_df_count"] += 1
        if flags & 0x01:
            wd["frag_mf_count"] += 1

        # 3. Payload size
        payload_len = max(0, min(incl_len - 14, tot_len) - ihl)
        wd["payload_sizes"].append(payload_len)

        # 4. TCP Retransmissions & Destination Ports
        if proto == 6 and incl_len >= 14 + ihl + 20:
            tcp_offset = ip_offset + ihl
            sport, dport = struct.unpack_from(">HH", pcap_bytes, tcp_offset)
            seq_num = struct.unpack_from(">I", pcap_bytes, tcp_offset + 4)[0]
            
            flow_key = (src_ip, dst_ip, sport, dport)
            seq_sig = (flow_key, seq_num)
            if seq_sig in wd["seen_tcp_seqs"]:
                wd["retrans_count"] += 1
            else:
                wd["seen_tcp_seqs"].add(seq_sig)

            if src_ip not in wd["dst_ports_by_src"]:
                wd["dst_ports_by_src"][src_ip] = []
            wd["dst_ports_by_src"][src_ip].append(int(dport))

    print(f"[+] Parsed {pkt_count:,} packets ({matched_pkts:,} matched to windows) in {time.time() - t_parse_start:.2f}s", flush=True)

    # 3. Compile Aggregated DataFrame
    rows = []
    for _, r in d14.iterrows():
        w_id = r["window_id"]
        wd = window_data[w_id]
        ttls = wd["ttls"]
        payloads = wd["payload_sizes"]

        if ttls:
            ttl_min = float(np.min(ttls))
            ttl_max = float(np.max(ttls))
            ttl_std = float(np.std(ttls))
            vals, counts = np.unique(ttls, return_counts=True)
            ttl_mode = float(vals[np.argmax(counts)])
        else:
            ttl_min, ttl_max, ttl_std, ttl_mode = 0.0, 0.0, 0.0, 0.0

        if payloads:
            p25 = float(np.percentile(payloads, 25))
            p50 = float(np.percentile(payloads, 50))
            p75 = float(np.percentile(payloads, 75))
            p95 = float(np.percentile(payloads, 95))
        else:
            p25, p50, p75, p95 = 0.0, 0.0, 0.0, 0.0

        scan_scores = [
            calculate_port_scan_monotonicity(ports)
            for ports in wd["dst_ports_by_src"].values()
            if len(ports) >= 3
        ]
        port_scan_score = float(np.max(scan_scores)) if scan_scores else 0.0

        rows.append({
            "window_id": w_id,
            "pkt_ttl_min": ttl_min,
            "pkt_ttl_max": ttl_max,
            "pkt_ttl_std": ttl_std,
            "pkt_ttl_mode": ttl_mode,
            "pkt_frag_mf_count": float(wd["frag_mf_count"]),
            "pkt_frag_df_count": float(wd["frag_df_count"]),
            "pkt_payload_size_p25": p25,
            "pkt_payload_size_p50": p50,
            "pkt_payload_size_p75": p75,
            "pkt_payload_size_p95": p95,
            "pkt_tcp_retrans_count": float(wd["retrans_count"]),
            "pkt_port_scan_seq_score": port_scan_score,
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"[+] Successfully exported genuine packet features to: {output_path} ({len(out_df)} rows)", flush=True)

    active_windows = (out_df["pkt_ttl_max"] > 0).sum()
    print(f"[+] Telemetry Summary: {active_windows}/{len(out_df)} windows populated with genuine physical packets ({active_windows/len(out_df)*100:.1f}%)", flush=True)

    return out_df, {
        "total_packets": pkt_count,
        "matched_packets": matched_pkts,
        "active_windows": active_windows,
        "total_windows": len(out_df)
    }


if __name__ == "__main__":
    run_reconciliation_and_extraction()
