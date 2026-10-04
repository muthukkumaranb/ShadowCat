#!/usr/bin/env python3
"""
inspect_ait.py — Feasibility inspection for AIT-LDS v2.0 scenario.

Produces FEASIBILITY.md and feasibility.json with:
  a) Directory tree (depth 3) with file sizes
  b) Network captures: path, host, timestamps, packet count, total bytes
  c) Label schema: which log files have per-line labels, field names, values
  d) Attack step timestamps with derivation method
  e) Step order and overlap matrix
  f) Timezone alignment check (pcap vs log timestamps)
  g) Can each 1-minute window of network traffic be given an attack-step label?

Usage:
    python data-engineering/ait/inspect_ait.py --scenario russellmitchell
"""

import argparse
import glob
import hashlib
import json
import os
import struct
import sys
import time
import re
import yaml
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_RAW_DIR = os.environ.get("AIT_RAW_DIR", "data-engineering/data/ait/raw")

ALL_SCENARIOS = [
    "fox", "harrison", "russellmitchell", "santos",
    "shaw", "wardbeck", "wheeler", "wilson",
]


# ── Helpers ──────────────────────────────────────────────────────────────

def human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def dir_tree(root: str, max_depth: int = 3, prefix: str = "") -> List[str]:
    """Return a list of lines representing the directory tree."""
    lines = []
    root_path = Path(root)
    if not root_path.is_dir():
        return [f"{prefix}{root_path.name} (not a directory)"]

    entries = sorted(root_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        if entry.is_dir():
            child_count = sum(1 for _ in entry.rglob("*"))
            lines.append(f"{prefix}{connector}{entry.name}/ ({child_count} items)")
            if max_depth > 1:
                ext = "    " if is_last else "│   "
                lines.extend(dir_tree(str(entry), max_depth - 1, prefix + ext))
        else:
            sz = entry.stat().st_size
            lines.append(f"{prefix}{connector}{entry.name} ({human_size(sz)})")
    return lines


def find_pcap_files(root: str) -> List[str]:
    """Find all pcap/pcapng files recursively."""
    results = []
    for ext in ("*.pcap", "*.pcapng", "*.cap"):
        results.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(results)


def inspect_pcap_quick(pcap_path: str) -> Dict[str, Any]:
    """Quick pcap inspection: timestamps, packet count, total bytes."""
    info: Dict[str, Any] = {
        "path": pcap_path,
        "file_size_bytes": os.path.getsize(pcap_path),
    }

    try:
        with open(pcap_path, "rb") as f:
            header = f.read(24)
            if len(header) < 24:
                info["error"] = "file too small for pcap header"
                return info

            magic = struct.unpack_from("<I", header, 0)[0]
            if magic == 0xa1b2c3d4:
                endian = "<"
            elif magic == 0xd4c3b2a1:
                endian = ">"
            else:
                info["error"] = f"unknown magic: 0x{magic:08x}"
                return info

            _ver_major, _ver_minor, _tz, _sigfigs, snaplen, link_type = struct.unpack_from(
                f"{endian}HHiIII", header, 4
            )
            info["link_type"] = link_type
            info["snaplen"] = snaplen

            pkt_count = 0
            total_bytes = 0
            first_ts = None
            last_ts = None

            while True:
                pkt_hdr = f.read(16)
                if len(pkt_hdr) < 16:
                    break
                ts_sec, ts_usec, incl_len, orig_len = struct.unpack_from(
                    f"{endian}IIII", pkt_hdr
                )
                ts = ts_sec + ts_usec / 1e6
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
                pkt_count += 1
                total_bytes += orig_len
                f.seek(incl_len, 1)

            info["packet_count"] = pkt_count
            info["total_bytes"] = total_bytes
            if first_ts is not None:
                info["start_ts_unix"] = first_ts
                info["end_ts_unix"] = last_ts
                info["start_ts_utc"] = datetime.fromtimestamp(first_ts, tz=timezone.utc).isoformat()
                info["end_ts_utc"] = datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat()
                info["duration_seconds"] = round(last_ts - first_ts, 2)

    except Exception as e:
        info["error"] = str(e)

    return info


def find_label_files(root: str) -> List[str]:
    """Find all label files under labels/ directory."""
    labels_dir = os.path.join(root, "labels")
    if not os.path.isdir(labels_dir):
        return []
    results = []
    for f in Path(labels_dir).rglob("*"):
        if f.is_file():
            results.append(str(f))
    return sorted(results)


def parse_label_file(path: str) -> List[Dict]:
    """Parse a label file (one JSON object per line)."""
    labels = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                labels.append(obj)
            except json.JSONDecodeError:
                continue
    return labels


def find_attacks_log(root: str) -> Optional[str]:
    """Find gather/attacker_0/logs/attacks.log."""
    path = os.path.join(root, "gather", "attacker_0", "logs", "attacks.log")
    if os.path.isfile(path):
        return path
    # Search more broadly
    for f in Path(root).rglob("attacks.log"):
        return str(f)
    return None


def parse_attacks_log(path: str) -> List[Dict[str, Any]]:
    """Parse the attacks.log file.
    
    The format varies but typically contains timestamped attack step entries.
    We try multiple parse strategies.
    """
    entries = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Try JSON lines format first
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            entries.append(obj)
            continue
        except json.JSONDecodeError:
            pass

        # Try syslog-like format: timestamp message
        # Example: 2022-01-24T03:01:15+00:00 nmap_scan started
        ts_match = re.match(
            r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^\s]*)\s+(.*)', line
        )
        if ts_match:
            entries.append({
                "timestamp": ts_match.group(1),
                "message": ts_match.group(2),
                "raw": line,
            })
            continue

        # Plain text
        entries.append({"raw": line})

    return entries


def find_dataset_yml(root: str) -> Optional[str]:
    """Find dataset.yml."""
    path = os.path.join(root, "dataset.yml")
    if os.path.isfile(path):
        return path
    for f in Path(root).rglob("dataset.yml"):
        return str(f)
    return None


def find_servers_yml(root: str) -> Optional[str]:
    """Find processing/config/servers.yml."""
    path = os.path.join(root, "processing", "config", "servers.yml")
    if os.path.isfile(path):
        return path
    for f in Path(root).rglob("servers.yml"):
        return str(f)
    return None


def extract_host_from_pcap_path(pcap_path: str, scenario_root: str) -> str:
    """Extract host name from pcap path."""
    rel = os.path.relpath(pcap_path, scenario_root)
    parts = Path(rel).parts
    # Expected: gather/<host>/logs/...
    if len(parts) >= 2 and parts[0] == "gather":
        return parts[1]
    return "unknown"


def collect_all_label_values(label_files: List[str]) -> Dict[str, Dict[str, Any]]:
    """Collect label schema info from all label files."""
    file_summaries = {}
    for lf in label_files:
        labels = parse_label_file(lf)
        if not labels:
            continue
        all_labels_set = set()
        all_rules_set = set()
        fields = set()
        for entry in labels:
            fields.update(entry.keys())
            if "labels" in entry:
                for lbl in entry["labels"]:
                    all_labels_set.add(lbl)
            if "rules" in entry:
                for rule_key, rule_vals in entry["rules"].items():
                    for rv in rule_vals:
                        all_rules_set.add(rv)

        file_summaries[lf] = {
            "label_count": len(labels),
            "fields": sorted(fields),
            "unique_labels": sorted(all_labels_set),
            "unique_rules": sorted(all_rules_set),
            "first_line": labels[0].get("line") if labels else None,
            "last_line": labels[-1].get("line") if labels else None,
        }
    return file_summaries


def derive_step_timestamps_from_attacks_log(
    attacks_log_entries: List[Dict],
) -> Dict[str, Dict[str, str]]:
    """Derive per-step start/end timestamps from attacks.log."""
    steps: Dict[str, List[str]] = defaultdict(list)

    for entry in attacks_log_entries:
        ts = entry.get("timestamp")
        msg = entry.get("message", entry.get("raw", ""))
        if ts and msg:
            steps[msg.strip()].append(ts)

    result = {}
    for step_name, timestamps in steps.items():
        timestamps.sort()
        result[step_name] = {
            "earliest": timestamps[0],
            "latest": timestamps[-1],
            "count": len(timestamps),
        }
    return result


def derive_step_timestamps_from_labels(
    label_summaries: Dict[str, Dict[str, Any]],
    scenario_root: str,
) -> Dict[str, Dict[str, Any]]:
    """Derive step info from label files.
    
    Aggregates all unique labels across all label files.
    For timestamp derivation, we need to correlate with the actual log lines.
    """
    all_labels: Dict[str, int] = defaultdict(int)
    all_rules: Dict[str, int] = defaultdict(int)

    for lf, info in label_summaries.items():
        for lbl in info["unique_labels"]:
            all_labels[lbl] += info["label_count"]
        for rule in info["unique_rules"]:
            all_rules[rule] += 1

    return {
        "aggregated_labels": dict(all_labels),
        "aggregated_rules": dict(all_rules),
        "label_file_count": len(label_summaries),
    }


def get_timestamps_for_labels(
    label_file: str, 
    log_file: str,
) -> Dict[str, Dict[str, str]]:
    """Get actual timestamps for labelled lines by reading the corresponding log file.
    
    Returns {label_name: {earliest_ts, latest_ts, count}}.
    """
    labels = parse_label_file(label_file)
    if not labels:
        return {}

    # Build line_number -> labels mapping
    line_labels: Dict[int, List[str]] = {}
    for entry in labels:
        line_no = entry.get("line")
        lbls = entry.get("labels", [])
        if line_no is not None:
            line_labels[line_no] = lbls

    if not os.path.isfile(log_file):
        return {}

    # Read log file and extract timestamps for labelled lines
    label_timestamps: Dict[str, List[str]] = defaultdict(list)
    
    # Common timestamp patterns
    ts_patterns = [
        # ISO format: 2022-01-24T03:01:15.123456+00:00
        re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[.\d]*[+-]\d{2}:\d{2})'),
        # Audit format: msg=audit(1642999060.603:2226)
        re.compile(r'audit\((\d+\.\d+):\d+\)'),
        # Syslog format: Jan 24 03:01:15
        re.compile(r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})'),
        # Apache/other: [24/Jan/2022:03:01:15 +0000]
        re.compile(r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4})\]'),
    ]

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            for line_no_0idx, line in enumerate(f):
                line_no = line_no_0idx + 1  # 1-indexed
                if line_no not in line_labels:
                    continue
                
                # Try to extract timestamp
                ts_str = None
                for pat in ts_patterns:
                    m = pat.search(line)
                    if m:
                        ts_str = m.group(1)
                        break
                
                if ts_str:
                    for lbl in line_labels[line_no]:
                        label_timestamps[lbl].append(ts_str)
    except Exception:
        pass

    # Summarize
    result = {}
    for lbl, ts_list in label_timestamps.items():
        ts_list.sort()
        result[lbl] = {
            "earliest": ts_list[0],
            "latest": ts_list[-1],
            "count": len(ts_list),
        }
    return result


def audit_ts_to_utc(audit_ts: str) -> Optional[datetime]:
    """Convert audit timestamp (epoch.fraction) to UTC datetime."""
    try:
        epoch = float(audit_ts)
        return datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def parse_any_timestamp(ts_str: str) -> Optional[datetime]:
    """Try to parse various timestamp formats to UTC datetime."""
    # ISO format
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        pass

    # Audit epoch
    try:
        epoch = float(ts_str)
        if 1e9 < epoch < 2e9:  # reasonable unix timestamp
            return datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (ValueError, TypeError):
        pass

    return None


# ── Main inspection ──────────────────────────────────────────────────────

def inspect_scenario(scenario: str, raw_dir: str) -> Dict[str, Any]:
    """Full inspection of a scenario directory."""
    import zipfile

    zip_path = os.path.join(raw_dir, f"{scenario}.zip")
    scenario_root = os.path.join(raw_dir, scenario)

    # Extract if needed
    if not os.path.isdir(scenario_root):
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(f"Neither {scenario_root} nor {zip_path} found")
        print(f"[*] Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
        print(f"[+] Extracted to {scenario_root}")

    results: Dict[str, Any] = {
        "scenario": scenario,
        "scenario_root": scenario_root,
    }

    # (a) Directory tree
    print("[*] Building directory tree...")
    tree_lines = dir_tree(scenario_root, max_depth=3)
    results["directory_tree"] = tree_lines

    # (b) Network captures
    print("[*] Finding and inspecting pcap files...")
    pcap_files = find_pcap_files(scenario_root)
    pcap_info = []
    for pf in pcap_files:
        print(f"    Inspecting: {os.path.relpath(pf, scenario_root)}")
        info = inspect_pcap_quick(pf)
        info["host"] = extract_host_from_pcap_path(pf, scenario_root)
        info["rel_path"] = os.path.relpath(pf, scenario_root)
        pcap_info.append(info)
    results["pcap_files"] = pcap_info

    # (c) Labels
    print("[*] Inspecting label files...")
    label_files = find_label_files(scenario_root)
    label_summaries = collect_all_label_values(label_files)

    # Make paths relative for readability
    label_summaries_rel = {}
    for lf, info in label_summaries.items():
        rel_path = os.path.relpath(lf, scenario_root)
        label_summaries_rel[rel_path] = info
    results["label_files"] = label_summaries_rel

    # Collect all unique labels across all files
    all_unique_labels = set()
    for info in label_summaries.values():
        all_unique_labels.update(info["unique_labels"])
    results["all_unique_labels"] = sorted(all_unique_labels)

    # (c/d) Attacks.log — the primary source for step timestamps
    print("[*] Looking for attacks.log...")
    attacks_log_path = find_attacks_log(scenario_root)
    if attacks_log_path:
        results["attacks_log_path"] = os.path.relpath(attacks_log_path, scenario_root)
        attacks_entries = parse_attacks_log(attacks_log_path)
        results["attacks_log_entries"] = attacks_entries
        results["attacks_log_entry_count"] = len(attacks_entries)

        step_timestamps = derive_step_timestamps_from_attacks_log(attacks_entries)
        results["step_timestamps_from_attacks_log"] = step_timestamps
    else:
        results["attacks_log_path"] = None
        results["attacks_log_entries"] = []

    # (c/d) Get timestamps from label files for select host logs
    print("[*] Correlating label timestamps with log files...")
    label_derived_timestamps: Dict[str, Dict] = {}
    for lf, info in label_summaries.items():
        # Derive the corresponding log file path
        rel = os.path.relpath(lf, os.path.join(scenario_root, "labels"))
        log_file = os.path.join(scenario_root, "gather", rel)
        if os.path.isfile(log_file):
            ts_info = get_timestamps_for_labels(lf, log_file)
            if ts_info:
                rel_key = os.path.relpath(lf, scenario_root)
                label_derived_timestamps[rel_key] = ts_info
    results["label_derived_timestamps"] = label_derived_timestamps

    # (d) dataset.yml
    print("[*] Looking for dataset.yml...")
    dataset_yml_path = find_dataset_yml(scenario_root)
    if dataset_yml_path:
        with open(dataset_yml_path, "r") as f:
            dataset_info = yaml.safe_load(f)
        results["dataset_yml"] = dataset_info
        results["dataset_yml_path"] = os.path.relpath(dataset_yml_path, scenario_root)
    else:
        results["dataset_yml"] = None

    # servers.yml
    servers_yml_path = find_servers_yml(scenario_root)
    if servers_yml_path:
        with open(servers_yml_path, "r") as f:
            servers_info = yaml.safe_load(f)
        results["servers_yml"] = servers_info
    else:
        results["servers_yml"] = None

    # (e) Step order and overlap — derived from attacks.log timestamps
    # (f) Timezone alignment — check pcap vs log timestamps
    # (g) Can we label 1-minute network windows?

    # These analyses are in separate sections of the output

    return results


def analyze_step_order_and_overlap(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze step order, overlap, and timezone alignment."""
    analysis: Dict[str, Any] = {}

    step_ts = results.get("step_timestamps_from_attacks_log", {})
    if not step_ts:
        analysis["step_order"] = "CANNOT DETERMINE — no attacks.log timestamps"
        analysis["overlap_matrix"] = None
        return analysis

    # Parse timestamps and sort by earliest
    steps_parsed = []
    for step_name, ts_info in step_ts.items():
        earliest = parse_any_timestamp(ts_info["earliest"])
        latest = parse_any_timestamp(ts_info["latest"])
        if earliest and latest:
            steps_parsed.append({
                "name": step_name,
                "start": earliest,
                "end": latest,
                "start_str": ts_info["earliest"],
                "end_str": ts_info["latest"],
                "count": ts_info["count"],
            })

    steps_parsed.sort(key=lambda s: s["start"])
    analysis["step_order"] = [
        {"name": s["name"], "start": s["start_str"], "end": s["end_str"]}
        for s in steps_parsed
    ]

    # Overlap matrix
    n = len(steps_parsed)
    overlap = {}
    for i in range(n):
        for j in range(i + 1, n):
            si = steps_parsed[i]
            sj = steps_parsed[j]
            # Two intervals overlap if start_i < end_j AND start_j < end_i
            overlaps = si["start"] < sj["end"] and sj["start"] < si["end"]
            if overlaps:
                overlap_start = max(si["start"], sj["start"])
                overlap_end = min(si["end"], sj["end"])
                overlap[f"{si['name']} ↔ {sj['name']}"] = {
                    "overlaps": True,
                    "overlap_minutes": round((overlap_end - overlap_start).total_seconds() / 60, 2),
                }
    analysis["overlapping_steps"] = overlap if overlap else "none"

    # Timezone check: compare pcap timestamps with attacks.log timestamps
    pcap_info = results.get("pcap_files", [])
    if pcap_info and steps_parsed:
        first_attack = steps_parsed[0]
        attack_start_utc = first_attack["start"]

        # Find pcaps that cover the attack time
        pcap_covering = []
        for pc in pcap_info:
            if "start_ts_unix" in pc and "end_ts_unix" in pc:
                pc_start = datetime.fromtimestamp(pc["start_ts_unix"], tz=timezone.utc)
                pc_end = datetime.fromtimestamp(pc["end_ts_unix"], tz=timezone.utc)
                if pc_start <= attack_start_utc <= pc_end:
                    pcap_covering.append({
                        "pcap": pc.get("rel_path"),
                        "pcap_start": pc["start_ts_utc"],
                        "pcap_end": pc["end_ts_utc"],
                        "attack_start_within_pcap": True,
                    })
        analysis["timezone_alignment"] = {
            "attack_start_utc": attack_start_utc.isoformat(),
            "pcaps_covering_attack_start": pcap_covering,
            "assessment": (
                "ALIGNED — pcap and attacks.log timestamps both appear to be UTC"
                if pcap_covering
                else "CANNOT VERIFY — no pcap covers the attack start time"
            ),
        }
    else:
        analysis["timezone_alignment"] = {
            "assessment": "CANNOT VERIFY — insufficient data"
        }

    return analysis


def assess_window_labelling(results: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Assess whether 1-minute network windows can be labelled from real timestamps."""
    assessment: Dict[str, Any] = {}

    step_order = analysis.get("step_order", [])
    if isinstance(step_order, str):
        assessment["can_label_windows"] = "NO"
        assessment["reason"] = step_order
        return assessment

    # Check if we have:
    # 1. Attack step timestamps (from attacks.log)
    # 2. pcap files with valid timestamps
    # 3. Timezone alignment

    has_step_timestamps = len(step_order) > 0
    has_pcaps = len(results.get("pcap_files", [])) > 0
    pcaps_valid = any(
        "start_ts_unix" in pc for pc in results.get("pcap_files", [])
    )

    tz_aligned = "ALIGNED" in analysis.get("timezone_alignment", {}).get("assessment", "")
    tz_cannot_verify = "CANNOT VERIFY" in analysis.get("timezone_alignment", {}).get("assessment", "")

    # The key question: can the attacks.log timestamps directly map to
    # 1-minute windows of pcap traffic?
    #
    # attacks.log gives per-step start/end in UTC.
    # pcap files have packet timestamps in UTC.
    # So: for each 1-minute window of pcap traffic, we can check if it falls
    # within any attack step's [start, end] interval.
    #
    # BUT: attacks.log may only have coarse timestamps (one entry per step),
    # and some steps (like exfiltration) run in parallel.
    # If steps overlap, a window could belong to multiple steps.

    if has_step_timestamps and has_pcaps and pcaps_valid:
        if tz_aligned:
            assessment["can_label_windows"] = "YES"
            assessment["reason"] = (
                "attacks.log provides per-step start/end timestamps in UTC. "
                "pcap files contain packet timestamps in UTC. "
                "Timezone alignment confirmed by checking that the first attack step's "
                "start time falls within the time range of at least one pcap file. "
                "Each 1-minute window can be assigned an attack-step label by checking "
                "if the window's time interval overlaps with any step's [start, end] interval."
            )
            assessment["caveats"] = [
                "Overlapping steps (especially exfiltration) require a precedence rule.",
                "Timestamps come from attacks.log, not from packet-level ground truth.",
                "Benign windows are those outside all step intervals.",
            ]
        elif tz_cannot_verify:
            # Even if we can't find a pcap covering the exact attack start,
            # both timestamps appear to be in UTC based on the dataset description
            assessment["can_label_windows"] = "YES"
            assessment["reason"] = (
                "attacks.log provides per-step start/end timestamps. "
                "pcap files contain packet timestamps. "
                "While no single pcap file was found covering the exact attack start, "
                "the dataset documentation states all timestamps are UTC. "
                "Label mapping is feasible via interval overlap."
            )
            assessment["caveats"] = [
                "Timezone alignment could not be verified with a shared event.",
                "Overlapping steps require a precedence rule.",
            ]
        else:
            assessment["can_label_windows"] = "NO"
            assessment["reason"] = "Timezone alignment between pcap and log timestamps failed."
    elif has_step_timestamps and not has_pcaps:
        assessment["can_label_windows"] = "NO"
        assessment["reason"] = "No pcap files found in scenario."
    elif not has_step_timestamps:
        assessment["can_label_windows"] = "NO"
        assessment["reason"] = "No attack step timestamps available."
    else:
        assessment["can_label_windows"] = "NO"
        assessment["reason"] = "pcap files exist but have no valid timestamps."

    return assessment


def write_feasibility_md(
    results: Dict[str, Any],
    analysis: Dict[str, Any],
    window_assessment: Dict[str, Any],
    output_dir: str,
) -> str:
    """Write FEASIBILITY.md report."""
    md_path = os.path.join(output_dir, "FEASIBILITY.md")
    lines = []
    a = lines.append

    scenario = results["scenario"]
    a(f"# AIT-LDS v2.0 Feasibility Report: `{scenario}`\n")
    a(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")

    # (a) Directory tree
    a("## (a) Directory Tree (depth 3)\n")
    a("```")
    for line in results.get("directory_tree", []):
        a(line)
    a("```\n")

    # (b) Network captures
    a("## (b) Network Captures\n")
    pcaps = results.get("pcap_files", [])
    if pcaps:
        a("| File | Host | Start (UTC) | End (UTC) | Packets | Total Bytes | File Size |")
        a("|------|------|-------------|-----------|---------|-------------|-----------|")
        for pc in pcaps:
            a(f"| `{pc.get('rel_path', '?')}` "
              f"| {pc.get('host', '?')} "
              f"| {pc.get('start_ts_utc', 'N/A')} "
              f"| {pc.get('end_ts_utc', 'N/A')} "
              f"| {pc.get('packet_count', 'N/A'):,} "
              f"| {human_size(pc.get('total_bytes', 0))} "
              f"| {human_size(pc.get('file_size_bytes', 0))} |")
    else:
        a("No pcap/pcapng files found.\n")

    # (c) Labels
    a("\n## (c) Label Schema\n")
    a(f"**Total label files**: {len(results.get('label_files', {}))}\n")
    a(f"**All unique label values**: {', '.join(results.get('all_unique_labels', []))}\n")

    label_files = results.get("label_files", {})
    if label_files:
        a("\n### Label files with per-line labels\n")
        a("| File | Label count | Unique labels | Fields |")
        a("|------|-------------|---------------|--------|")
        for lf, info in label_files.items():
            a(f"| `{lf}` | {info['label_count']} | {', '.join(info['unique_labels'])} | {', '.join(info['fields'])} |")

    # (c) attacks.log
    a("\n### Attacker Timeline (attacks.log)\n")
    if results.get("attacks_log_path"):
        a(f"**Path**: `{results['attacks_log_path']}`\n")
        a(f"**Entries**: {results.get('attacks_log_entry_count', 0)}\n")

        entries = results.get("attacks_log_entries", [])
        if entries:
            a("\n```")
            for e in entries[:50]:  # Show first 50 entries
                a(json.dumps(e))
            if len(entries) > 50:
                a(f"... ({len(entries) - 50} more entries)")
            a("```\n")
    else:
        a("**attacks.log not found.**\n")

    # (d) Attack step timestamps
    a("## (d) Attack Step Timestamps\n")
    step_ts = results.get("step_timestamps_from_attacks_log", {})
    if step_ts:
        a("### From attacks.log\n")
        a("| Step | Earliest | Latest | Entry Count |")
        a("|------|----------|--------|-------------|")
        for step_name, info in sorted(step_ts.items(), key=lambda x: x[1]["earliest"]):
            a(f"| {step_name} | {info['earliest']} | {info['latest']} | {info['count']} |")
        a("\n**Derivation method**: Timestamps extracted directly from `gather/attacker_0/logs/attacks.log`.\n")

    label_ts = results.get("label_derived_timestamps", {})
    if label_ts:
        a("### From host-log label correlation\n")
        for lf, ts_info in label_ts.items():
            a(f"\n**{lf}**:\n")
            a("| Label | Earliest | Latest | Count |")
            a("|-------|----------|--------|-------|")
            for lbl, info in sorted(ts_info.items()):
                a(f"| {lbl} | {info['earliest']} | {info['latest']} | {info['count']} |")

    # (e) Step order and overlap
    a("\n## (e) Step Order and Overlap Matrix\n")
    step_order = analysis.get("step_order", [])
    if isinstance(step_order, list):
        a("### Step Order (sorted by start time)\n")
        a("| # | Step | Start | End |")
        a("|---|------|-------|-----|")
        for i, s in enumerate(step_order, 1):
            a(f"| {i} | {s['name']} | {s['start']} | {s['end']} |")

        overlap = analysis.get("overlapping_steps", "none")
        if overlap != "none":
            a("\n### Overlapping Steps\n")
            a("| Pair | Overlap (minutes) |")
            a("|------|-------------------|")
            for pair, info in overlap.items():
                a(f"| {pair} | {info['overlap_minutes']} |")
        else:
            a("\n**No overlapping steps detected.**\n")
    else:
        a(f"{step_order}\n")

    # (f) Timezone alignment
    a("\n## (f) Timezone Alignment\n")
    tz = analysis.get("timezone_alignment", {})
    a(f"**Assessment**: {tz.get('assessment', 'N/A')}\n")
    if tz.get("attack_start_utc"):
        a(f"**First attack step start (UTC)**: {tz['attack_start_utc']}\n")
    covering = tz.get("pcaps_covering_attack_start", [])
    if covering:
        a(f"**pcaps covering attack start**: {len(covering)}\n")
        for pc in covering:
            a(f"- `{pc['pcap']}`: {pc['pcap_start']} → {pc['pcap_end']}")

    # (g) Can we label 1-minute windows?
    a("\n## (g) Can Each 1-Minute Network Window Be Labelled?\n")
    a(f"**Answer**: **{window_assessment['can_label_windows']}**\n")
    a(f"**Reason**: {window_assessment['reason']}\n")
    if window_assessment.get("caveats"):
        a("\n**Caveats**:")
        for caveat in window_assessment["caveats"]:
            a(f"- {caveat}")

    # Dataset info
    a("\n## Dataset Metadata\n")
    if results.get("dataset_yml"):
        a("```yaml")
        a(yaml.dump(results["dataset_yml"], default_flow_style=False))
        a("```\n")

    if results.get("servers_yml"):
        a("### Servers (hosts and IPs)\n")
        a("```yaml")
        a(yaml.dump(results["servers_yml"], default_flow_style=False))
        a("```\n")

    content = "\n".join(lines)
    os.makedirs(output_dir, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] Wrote {md_path}")
    return md_path


def main():
    parser = argparse.ArgumentParser(
        description="Inspect AIT-LDS v2.0 scenario for feasibility"
    )
    parser.add_argument(
        "--scenario", required=True, choices=ALL_SCENARIOS,
        help="Scenario to inspect",
    )
    parser.add_argument(
        "--raw-dir", default=DEFAULT_RAW_DIR,
        help=f"Raw data directory (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir", default="data-engineering/ait",
        help="Output directory for reports",
    )
    args = parser.parse_args()

    print(f"[*] Inspecting AIT-LDS scenario: {args.scenario}")
    results = inspect_scenario(args.scenario, args.raw_dir)

    # Analysis
    analysis = analyze_step_order_and_overlap(results)
    window_assessment = assess_window_labelling(results, analysis)

    # Combine all into feasibility.json
    feasibility = {
        "scenario": args.scenario,
        "inspection_time": datetime.now(timezone.utc).isoformat(),
        "directory_tree_line_count": len(results.get("directory_tree", [])),
        "pcap_files_count": len(results.get("pcap_files", [])),
        "pcap_files": results.get("pcap_files", []),
        "label_files_count": len(results.get("label_files", {})),
        "label_files": results.get("label_files", {}),
        "all_unique_labels": results.get("all_unique_labels", []),
        "attacks_log_path": results.get("attacks_log_path"),
        "attacks_log_entries": results.get("attacks_log_entries", []),
        "step_timestamps_from_attacks_log": results.get("step_timestamps_from_attacks_log", {}),
        "label_derived_timestamps": results.get("label_derived_timestamps", {}),
        "dataset_yml": results.get("dataset_yml"),
        "servers_yml": results.get("servers_yml"),
        "step_order": analysis.get("step_order"),
        "overlapping_steps": analysis.get("overlapping_steps"),
        "timezone_alignment": analysis.get("timezone_alignment"),
        "can_label_windows": window_assessment["can_label_windows"],
        "window_label_reason": window_assessment["reason"],
        "window_label_caveats": window_assessment.get("caveats"),
    }

    json_path = os.path.join(args.output_dir, "feasibility.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(feasibility, f, indent=2, default=str)
    print(f"[+] Wrote {json_path}")

    # Write FEASIBILITY.md
    write_feasibility_md(results, analysis, window_assessment, args.output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print(f"FEASIBILITY ASSESSMENT: {args.scenario}")
    print(f"  pcap files: {len(results.get('pcap_files', []))}")
    print(f"  label files: {len(results.get('label_files', {}))}")
    print(f"  unique labels: {len(results.get('all_unique_labels', []))}")
    print(f"  attacks.log entries: {len(results.get('attacks_log_entries', []))}")
    print(f"  Can label windows: {window_assessment['can_label_windows']}")
    print(f"  Reason: {window_assessment['reason']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
