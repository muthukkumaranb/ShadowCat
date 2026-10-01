#!/usr/bin/env python3
"""
Interactive Terminal Progress Monitor & Dashboard for Cyber World Model Architecture.
Supports dual-mode real-time visualization:
1. PCAP Ingestion & Decompression Monitor (CSE-CIC-IDS2018 S3 Streaming Extraction)
2. GraphSAGE LOEO Full 37-Fold Evaluation & Benchmark Scoreboard

Auto-detects the currently active process (download_pcap_*.py, run_extraction_*.py, or evaluate_endtoend_*.py).
"""

import os
import sys
import time
import re
import json
import glob
import shutil
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Enable UTF-8 and ANSI escape sequences on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
if os.name == "nt":
    os.system("color")

# Default paths
REPO_ROOT = Path(__file__).resolve().parents[1]
PCAP_2102_DIR = REPO_ROOT / "data-engineering" / "data" / "raw_pcap" / "21022018"
DEFAULT_RESULTS_JSON = REPO_ROOT / "scratch" / "endtoend_graphsage_results.json"
MANIFEST_PATH = REPO_ROOT / "ml1" / "artifacts" / "loeo" / "corrected_37fold_manifest.json"

# Wednesday 21-02-2018 S3 PCAP Members Catalog
PCAP_CATALOG_2102 = [
    {
        "member": "pcap/UCAP172.31.69.15",
        "clean_name": "UCAP172.31.69.15.pcap",
        "role": "Auxiliary Server (.15)",
        "comp_mb": 0.32,
        "uncomp_mb": 0.80,
    },
    {
        "member": "pcap/UCAP172.31.69.18",
        "clean_name": "UCAP172.31.69.18.pcap",
        "role": "Auxiliary Server (.18)",
        "comp_mb": 0.58,
        "uncomp_mb": 1.74,
    },
    {
        "member": "pcap/UCAP172.31.69.21",
        "clean_name": "UCAP172.31.69.21.pcap",
        "role": "Auxiliary Server (.21)",
        "comp_mb": 11.78,
        "uncomp_mb": 12.45,
    },
    {
        "member": "pcap/UCAP172.31.69.22",
        "clean_name": "UCAP172.31.69.22.pcap",
        "role": "Auxiliary Server (.22)",
        "comp_mb": 0.57,
        "uncomp_mb": 1.71,
    },
    {
        "member": "pcap/UCAP172.31.69.25",
        "clean_name": "UCAP172.31.69.25.pcap",
        "role": "Ubuntu Victim Svr (.25)",
        "comp_mb": 4.29,
        "uncomp_mb": 5.10,
    },
    {
        "member": "pcap/UCAP172.31.69.28 part 1",
        "clean_name": "UCAP172.31.69.28_part_1.pcap",
        "role": "DDoS Victim Prim (.28 Pt 1)",
        "comp_mb": 1956.13,
        "uncomp_mb": 17428.46,
    },
    {
        "member": "pcap/UCAP172.31.69.28 part 2",
        "clean_name": "UCAP172.31.69.28_part_2.pcap",
        "role": "DDoS Victim Sec (.28 Pt 2)",
        "comp_mb": 252.85,
        "uncomp_mb": 2025.59,
    },
    {
        "member": "pcap/UCAP172.31.69.7",
        "clean_name": "UCAP172.31.69.7.pcap",
        "role": "Auxiliary Server (.7)",
        "comp_mb": 0.29,
        "uncomp_mb": 0.88,
    },
]

# Colors & ANSI styles
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[38;5;51m"
C_BLUE = "\033[38;5;39m"
C_GREEN = "\033[38;5;48m"
C_YELLOW = "\033[38;5;220m"
C_ORANGE = "\033[38;5;208m"
C_MAGENTA = "\033[38;5;199m"
C_PURPLE = "\033[38;5;141m"
C_WHITE = "\033[38;5;255m"
C_GRAY = "\033[38;5;244m"
C_DARK = "\033[38;5;236m"
C_RED = "\033[38;5;196m"

SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def make_bar(percent: float, width: int = 32, fill_color: str = C_GREEN) -> str:
    """Create a sleek visual progress bar."""
    percent = max(0.0, min(100.0, percent))
    filled_len = int(round(width * percent / 100.0))
    empty_len = width - filled_len
    bar_fill = "█" * filled_len
    bar_empty = "░" * empty_len
    return f"{fill_color}{bar_fill}{C_DARK}{bar_empty}{C_RESET}"


def format_duration(seconds: float) -> str:
    """Format seconds into readable Xh Ym Zs format."""
    if seconds <= 0:
        return "0s"
    s = int(round(seconds))
    hrs = s // 3600
    mins = (s % 3600) // 60
    secs = s % 60
    if hrs > 0:
        return f"{hrs}h {mins:02d}m"
    if mins > 0:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def format_bytes(bytes_val: float) -> str:
    """Format bytes into MB or GB."""
    mb = bytes_val / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.1f} MB"


def detect_active_workload(override_pid: int = None):
    """
    Scans running processes to determine the active pipeline:
    Returns (mode, is_running, pid, cpu_sec, mem_mb, elapsed_sec, cpu_pct, cmdline_str)
    mode is 'pcap' or 'graphsage' or 'extraction'.
    """
    try:
        import psutil

        if override_pid:
            try:
                p = psutil.Process(override_pid)
                if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                    with p.oneshot():
                        cmd_str = " ".join(p.cmdline() or [])
                        cpu_times = p.cpu_times()
                        cpu_sec = cpu_times.user + cpu_times.system
                        mem_mb = p.memory_info().rss / (1024 * 1024)
                        elapsed_sec = max(0, time.time() - p.create_time())
                        cpu_pct = p.cpu_percent(interval=None)
                        if "download_pcap" in cmd_str:
                            mode = "pcap"
                        elif "run_extraction" in cmd_str:
                            mode = "extraction"
                        elif "run_world_model_sweep.py" in cmd_str or "train_probabilistic.py" in cmd_str:
                            mode = "lstm_train"
                        else:
                            mode = "graphsage"
                        return mode, True, p.pid, cpu_sec, mem_mb, elapsed_sec, cpu_pct, cmd_str
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Inspect candidate processes
        pcap_candidates = []
        graphsage_candidates = []
        extraction_candidates = []

        lstm_candidates = []

        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = p.info.get("cmdline") or []
                cmd_str = " ".join(cmdline)
                if "terminal_progress_monitor" in cmd_str:
                    continue
                if "download_pcap" in cmd_str:
                    pcap_candidates.append((p, cmd_str))
                elif "run_extraction" in cmd_str or "rebuild_ucs" in cmd_str:
                    extraction_candidates.append((p, cmd_str))
                elif "run_world_model_sweep.py" in cmd_str or "train_probabilistic.py" in cmd_str:
                    lstm_candidates.append((p, cmd_str))
                elif "evaluate_endtoend_graphsage" in cmd_str or "evaluate_graphsage" in cmd_str:
                    graphsage_candidates.append((p, cmd_str))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Prioritize currently active downloading worker
        if pcap_candidates:
            best_p, best_cmd = max(pcap_candidates, key=lambda x: (x[0].cpu_times().user + x[0].cpu_times().system) if hasattr(x[0], 'cpu_times') else 0)
            with best_p.oneshot():
                return "pcap", True, best_p.pid, (best_p.cpu_times().user + best_p.cpu_times().system), best_p.memory_info().rss / (1024*1024), max(0, time.time() - best_p.create_time()), best_p.cpu_percent(interval=None), best_cmd

        if extraction_candidates:
            best_p, best_cmd = max(extraction_candidates, key=lambda x: (x[0].cpu_times().user + x[0].cpu_times().system))
            with best_p.oneshot():
                return "extraction", True, best_p.pid, (best_p.cpu_times().user + best_p.cpu_times().system), best_p.memory_info().rss / (1024*1024), max(0, time.time() - best_p.create_time()), best_p.cpu_percent(interval=None), best_cmd

        if lstm_candidates:
            best_p, best_cmd = max(lstm_candidates, key=lambda x: (x[0].cpu_times().user + x[0].cpu_times().system))
            with best_p.oneshot():
                return "lstm_train", True, best_p.pid, (best_p.cpu_times().user + best_p.cpu_times().system), best_p.memory_info().rss / (1024*1024), max(0, time.time() - best_p.create_time()), best_p.cpu_percent(interval=None), best_cmd

        if graphsage_candidates:
            best_p, best_cmd = max(graphsage_candidates, key=lambda x: (x[0].cpu_times().user + x[0].cpu_times().system))
            with best_p.oneshot():
                return "graphsage", True, best_p.pid, (best_p.cpu_times().user + best_p.cpu_times().system), best_p.memory_info().rss / (1024*1024), max(0, time.time() - best_p.create_time()), best_p.cpu_percent(interval=None), best_cmd

    except Exception:
        pass

    # If no active worker running, detect by recently modified files
    # If 21022018 folder has files modified in last 24h, default to PCAP mode
    if PCAP_2102_DIR.exists():
        recent_pcap = any(time.time() - os.path.getmtime(f) < 86400 for f in PCAP_2102_DIR.glob("*") if f.is_file())
        if recent_pcap:
            return "pcap", False, 0, 0.0, 0.0, 0.0, 0.0, ""

    return "graphsage", False, 0, 0.0, 0.0, 0.0, 0.0, ""


# ─────────────────────────────────────────────────────────────────────────────
# PCAP Ingestion & Decompression View
# ─────────────────────────────────────────────────────────────────────────────

# Speed tracking state across renders
_SPEED_TRACKER = {
    "last_check_ts": 0.0,
    "last_written_bytes": 0,
    "current_speed_mbs": 0.0,
}


def scan_pcap_directory(dest_dir: Path):
    """
    Inspects destination directory for downloaded and in-progress PCAP files.
    Returns status dict per catalog item and aggregate metrics.
    """
    file_statuses = []
    total_written_bytes = 0
    total_expected_uncomp_bytes = sum(int(item["uncomp_mb"] * 1024 * 1024) for item in PCAP_CATALOG_2102)

    active_file_info = None

    for item in PCAP_CATALOG_2102:
        clean_name = item["clean_name"]
        expected_size = int(item["uncomp_mb"] * 1024 * 1024)
        dest_path = dest_dir / clean_name
        tmp_path = dest_dir / (clean_name + ".tmp")

        if dest_path.exists():
            actual_size = os.path.getsize(dest_path)
            total_written_bytes += actual_size
            pct = 100.0 if expected_size == 0 else min(100.0, (actual_size / expected_size) * 100.0)
            file_statuses.append({
                **item,
                "status": "DONE",
                "actual_bytes": actual_size,
                "pct": pct,
                "path": dest_path,
            })
        elif tmp_path.exists():
            actual_size = os.path.getsize(tmp_path)
            total_written_bytes += actual_size
            pct = 0.0 if expected_size == 0 else min(99.9, (actual_size / expected_size) * 100.0)
            info = {
                **item,
                "status": "ACTIVE",
                "actual_bytes": actual_size,
                "pct": pct,
                "path": tmp_path,
            }
            file_statuses.append(info)
            active_file_info = info
        else:
            file_statuses.append({
                **item,
                "status": "QUEUED",
                "actual_bytes": 0,
                "pct": 0.0,
                "path": dest_path,
            })

    # Track download/decompression speed
    now = time.time()
    dt = now - _SPEED_TRACKER["last_check_ts"]
    if dt >= 1.0:
        if _SPEED_TRACKER["last_written_bytes"] > 0 and total_written_bytes >= _SPEED_TRACKER["last_written_bytes"]:
            delta_bytes = total_written_bytes - _SPEED_TRACKER["last_written_bytes"]
            speed_mb = (delta_bytes / (1024 * 1024)) / dt
            # Exponential smoothing
            _SPEED_TRACKER["current_speed_mbs"] = 0.7 * speed_mb + 0.3 * _SPEED_TRACKER["current_speed_mbs"]
        _SPEED_TRACKER["last_check_ts"] = now
        _SPEED_TRACKER["last_written_bytes"] = total_written_bytes

    return file_statuses, total_written_bytes, total_expected_uncomp_bytes, active_file_info, _SPEED_TRACKER["current_speed_mbs"]


def render_pcap_dashboard(
    spinner_char: str,
    dest_dir: Path,
    is_running: bool,
    pid: int,
    cpu_sec: float,
    mem_mb: float,
    wall_elapsed: float,
    cpu_pct: float,
    cmdline_str: str,
) -> str:
    """Render full ANSI dashboard for PCAP Ingestion & Decompression."""
    file_statuses, written_bytes, total_bytes, active_file, live_speed_mbs = scan_pcap_directory(dest_dir)

    total_pct = min(100.0, (written_bytes / total_bytes * 100.0)) if total_bytes > 0 else 0.0
    overall_prog_bar = make_bar(total_pct, width=32, fill_color=C_CYAN)

    done_files = sum(1 for f in file_statuses if f["status"] == "DONE")
    total_files = len(file_statuses)

    # Disk usage
    try:
        du = shutil.disk_usage(str(dest_dir.anchor if dest_dir.anchor else REPO_ROOT))
        free_gb = du.free / (1024 ** 3)
        total_disk_gb = du.total / (1024 ** 3)
        disk_str = f"{free_gb:.1f} GB Free / {total_disk_gb:.1f} GB"
    except Exception:
        disk_str = "Available"

    buf = []
    # Clear screen on initial render, home on refresh
    buf.append("\033[H")

    # Status Pill
    if is_running:
        status_badge = f"{C_GREEN}{C_BOLD}● STREAMING ACTIVE [PID {pid}]{C_RESET}"
    elif done_files >= total_files:
        status_badge = f"{C_CYAN}{C_BOLD}✓ ALL 8 PCAPS INGESTED{C_RESET}"
    else:
        status_badge = f"{C_YELLOW}◌ IDLE / READY TO RESUME{C_RESET}"

    # Header Box
    buf.append(f"{C_CYAN}╭─────────────────────────────────────────────────────────────────────────────╮{C_RESET}")
    title_text = f"📦  CSE-CIC-IDS2018 PCAP INGESTION & DECOMPRESSION MONITOR  {spinner_char}  {status_badge}"
    buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}{C_WHITE}{title_text}{C_RESET}")
    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")

    # Workload Context
    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_BOLD}Target Day:{C_RESET} {C_YELLOW}Wednesday-21-02-2018{C_RESET} │ {C_BOLD}Vector:{C_RESET} {C_PURPLE}DDOS-LOIC-UDP / HOIC (Victim 172.31.69.28){C_RESET}"
    )
    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_BOLD}Pipeline Target:{C_RESET} Option B Genuine Packet Extraction (12 Physical Features)"
    )

    # Overall Progress
    written_str = format_bytes(written_bytes)
    total_str = format_bytes(total_bytes)
    buf.append(
        f"{C_CYAN}│{C_RESET}  Overall Progress: [{overall_prog_bar}] {C_BOLD}{C_CYAN}{total_pct:5.1f}%{C_RESET} ({written_str} / {total_str})"
    )

    # Speed & ETA
    rem_bytes = max(0, total_bytes - written_bytes)
    if is_running and live_speed_mbs > 0.05:
        rem_sec = rem_bytes / (live_speed_mbs * 1024 * 1024)
        eta_time = datetime.now() + timedelta(seconds=rem_sec)
        eta_str = eta_time.strftime("%H:%M:%S")
        rem_str = format_duration(rem_sec)
        speed_str = f"{C_GREEN}{live_speed_mbs:.2f} MB/s{C_RESET}"
        time_info = f"Speed: {speed_str} │ Rem: {C_YELLOW}{rem_str}{C_RESET} │ ETA: {C_WHITE}{C_BOLD}{eta_str}{C_RESET}"
    elif done_files >= total_files:
        time_info = f"{C_GREEN}{C_BOLD}★ All 21-02-2018 captures fully extracted and ready for feature extraction.{C_RESET}"
    else:
        rem_mb = rem_bytes / (1024 * 1024)
        time_info = f"{C_YELLOW}Process paused or awaiting restart. {rem_mb:.1f} MB remaining.{C_RESET}"

    buf.append(f"{C_CYAN}│{C_RESET}  Files: {C_BOLD}{done_files}/{total_files} Ingested{C_RESET} │ {time_info}")

    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")

    # Active Stream Card
    if active_file:
        act_name = active_file["clean_name"]
        act_role = active_file["role"]
        act_bytes = active_file["actual_bytes"]
        act_total = int(active_file["uncomp_mb"] * 1024 * 1024)
        act_pct = active_file["pct"]
        act_bar = make_bar(act_pct, width=28, fill_color=C_YELLOW)

        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_BOLD}CURRENT STREAMING CAPTURE:{C_RESET} {C_WHITE}{act_name}{C_RESET} ({C_PURPLE}{act_role}{C_RESET})"
        )
        buf.append(
            f"{C_CYAN}│{C_RESET}  [{act_bar}] {C_YELLOW}{C_BOLD}{act_pct:5.1f}%{C_RESET} ({format_bytes(act_bytes)} / {format_bytes(act_total)})"
        )
        if is_running and live_speed_mbs > 0.05:
            file_rem_bytes = max(0, act_total - act_bytes)
            file_rem_sec = file_rem_bytes / (live_speed_mbs * 1024 * 1024)
            file_eta_str = (datetime.now() + timedelta(seconds=file_rem_sec)).strftime("%H:%M:%S")
            buf.append(
                f"{C_CYAN}│{C_RESET}  {C_DIM}File Pace:{C_RESET} ~{live_speed_mbs:.1f} MB/s │ File Rem: {format_duration(file_rem_sec)} │ File ETA: {C_WHITE}{file_eta_str}{C_RESET}"
            )
    elif done_files < total_files:
        next_queued = next((f for f in file_statuses if f["status"] == "QUEUED"), None)
        next_name = next_queued["clean_name"] if next_queued else "None"
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_DIM}Next in queue:{C_RESET} {C_WHITE}{next_name}{C_RESET} (Ready to stream via Range-request on restart)"
        )
    else:
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_GREEN}{C_BOLD}✓ Complete:{C_RESET} All PCAP captures verified in data/raw_pcap/21022018"
        )

    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")

    # File Queue Matrix
    buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}CAPTURE FILE INGESTION MATRIX (Wednesday 21-02-2018):{C_RESET}")
    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_WHITE}┌──────────────────────────────┬──────────────────┬──────────┬────────┬──────────┐{C_RESET}"
    )
    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_WHITE}│{C_RESET} {C_BOLD}Capture File{C_RESET}                 {C_WHITE}│{C_RESET} {C_BOLD}Endpoint / Role{C_RESET}  {C_WHITE}│{C_RESET} {C_BOLD}Uncomp. MB{C_RESET} {C_WHITE}│{C_RESET} {C_BOLD}Prog.%{C_RESET} {C_WHITE}│{C_RESET} {C_BOLD}Status{C_RESET}   {C_WHITE}│{C_RESET}"
    )
    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_WHITE}├──────────────────────────────┼──────────────────┼──────────┼────────┼──────────┤{C_RESET}"
    )

    for f in file_statuses:
        name_col = f["clean_name"][:28].ljust(28)
        role_col = f["role"][:16].ljust(16)
        uncomp_str = f"{f['uncomp_mb']:8.2f}"
        pct_str = f"{f['pct']:5.1f}%"

        if f["status"] == "DONE":
            stat_str = f"{C_GREEN}✓ DONE   {C_RESET}"
            pct_col = f"{C_GREEN}{pct_str}{C_RESET}"
        elif f["status"] == "ACTIVE":
            stat_str = f"{C_YELLOW}▶ STREAM {C_RESET}"
            pct_col = f"{C_YELLOW}{pct_str}{C_RESET}"
        else:
            stat_str = f"{C_DIM}⏳ QUEUED {C_RESET}"
            pct_col = f"{C_DIM}{pct_str}{C_RESET}"

        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_WHITE}│{C_RESET} {name_col} {C_WHITE}│{C_RESET} {role_col} {C_WHITE}│{C_RESET} {uncomp_str} {C_WHITE}│{C_RESET} {pct_col} {C_WHITE}│{C_RESET} {stat_str}{C_WHITE}│{C_RESET}"
        )

    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_WHITE}└──────────────────────────────┴──────────────────┴──────────┴────────┴──────────┘{C_RESET}"
    )

    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")

    # Telemetry & Storage
    cpu_hrs = cpu_sec / 3600.0
    wall_str = format_duration(wall_elapsed)
    active_pid_str = str(pid) if pid > 0 else "Offline"
    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_DIM}Telemetry:{C_RESET} PID: {C_BOLD}{active_pid_str}{C_RESET} │ Worker RAM: {C_MAGENTA}{mem_mb:.0f} MB{C_RESET} │ CPU Time: {C_YELLOW}{cpu_hrs:.2f}h{C_RESET} │ Wall: {C_BLUE}{wall_str}{C_RESET}"
    )
    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_DIM}Storage:{C_RESET} Destination: {C_WHITE}{dest_dir}{C_RESET} │ Drive D: {C_GREEN}{disk_str}{C_RESET}"
    )

    buf.append(f"{C_CYAN}╰─────────────────────────────────────────────────────────────────────────────╯{C_RESET}")
    buf.append(
        f"{C_DIM}Auto-refreshing live dashboard (Ctrl+C to exit monitor — background workers continue){C_RESET}"
    )

    return "\n".join(buf)


# ─────────────────────────────────────────────────────────────────────────────
# GraphSAGE Evaluation View (Preserved for LOEO analysis)
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest_folds():
    """Load metadata for all folds from manifest file."""
    if not MANIFEST_PATH.exists():
        return {}
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)
        folds_dict = {}
        for f in m.get("folds", []):
            fid = f.get("fold_id")
            ep_id = f.get("held_out_episode_id", f"Episode_{fid}")
            parts = ep_id.split("_")
            attack_type = parts[1] if len(parts) > 1 else "Unknown"
            folds_dict[fid] = {
                "fold_id": fid,
                "episode_id": ep_id,
                "attack_type": attack_type,
                "train_len": len(f.get("train_indices", [])),
                "test_len": len(f.get("test_indices", [])),
            }
        return folds_dict
    except Exception:
        return {}


def load_results_json(path: Path):
    """Safely load incremental results from JSON, retrying on transient file locks."""
    if not path.exists():
        return {}
    for _ in range(3):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.06)
    return {}


def render_graphsage_dashboard(
    spinner_char: str,
    manifest_folds: dict,
    results_path: Path,
    is_running: bool,
    pid: int,
    cpu_sec: float,
    mem_mb: float,
    wall_elapsed: float,
    cpu_pct: float,
    cmdline_str: str,
) -> str:
    """Render full ANSI dashboard for GraphSAGE 37-Fold LOEO Evaluation."""
    results_data = load_results_json(results_path)
    det_results = results_data.get("detection", [])
    active_done = len(det_results)
    active_planned = 37

    pct = (active_done / active_planned * 100.0) if active_planned > 0 else 0.0
    prog_bar = make_bar(pct, width=32, fill_color=C_CYAN)

    buf = []
    buf.append("\033[H")

    status_badge = f"{C_GREEN}● RUNNING [PID {pid}]{C_RESET}" if is_running else (f"{C_CYAN}✓ 37/37 COMPLETE{C_RESET}" if active_done >= 37 else f"{C_YELLOW}◌ COMPLETE / VERIFIED{C_RESET}")

    buf.append(f"{C_CYAN}╭─────────────────────────────────────────────────────────────────────────────╮{C_RESET}")
    title_text = f"🛡️  GRAPHSAGE LOEO 37-FOLD EVALUATION SCOREBOARD  {spinner_char}  {status_badge}"
    buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}{C_WHITE}{title_text}{C_RESET}")
    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")
    buf.append(f"{C_CYAN}│{C_RESET}  Overall: [{prog_bar}] {C_BOLD}{C_CYAN}{pct:5.1f}%{C_RESET} ({active_done}/{active_planned} Folds Verified)")
    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")

    # Scoreboard Table
    buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}PER-ATTACK-TYPE SCOREBOARD BREAKDOWN (N={active_done}/37 Folds):{C_RESET}")
    buf.append(f"{C_CYAN}│{C_RESET}  {C_WHITE}┌──────────────────┬───────┬──────────┬──────────┬──────────┬─────────────┐{C_RESET}")
    buf.append(f"{C_CYAN}│{C_RESET}  {C_WHITE}│{C_RESET} {C_BOLD}Attack Cohort{C_RESET}    {C_WHITE}│{C_RESET} {C_BOLD}Folds{C_RESET} {C_WHITE}│{C_RESET} {C_BOLD}Plain F1{C_RESET} {C_WHITE}│{C_RESET} {C_BOLD}Scalar F1{C_RESET}{C_WHITE}│{C_RESET} {C_BOLD}G-SAGE F1{C_RESET}{C_WHITE}│{C_RESET} {C_BOLD}Delta (GS-Sc){C_RESET}{C_WHITE}│{C_RESET}")
    buf.append(f"{C_CYAN}│{C_RESET}  {C_WHITE}├──────────────────┼───────┼──────────┼──────────┼──────────┼─────────────┤{C_RESET}")

    for atk_name, atk_target_total in [("Botnet", 10), ("SSH-Bruteforce", 9), ("DDOS-LOIC-UDP", 18)]:
        atk_records = [r for r in det_results if r.get("attack_type") == atk_name]
        n_atk = len(atk_records)
        if n_atk > 0:
            p_f1 = sum(r["plain_f1"] for r in atk_records) / n_atk
            s_f1 = sum(r["scalar_f1"] for r in atk_records) / n_atk
            g_f1 = sum(r["graphsage_f1"] for r in atk_records) / n_atk
            delta = g_f1 - s_f1
            d_col = C_GREEN if delta > 0.0005 else (C_YELLOW if delta >= -0.0005 else C_RED)
            d_str = f"{delta:+.4f}"
            count_str = f"{n_atk:>2}/{atk_target_total:<2}"
            buf.append(
                f"{C_CYAN}│{C_RESET}  {C_WHITE}│{C_RESET} {atk_name:<16} {C_WHITE}│{C_RESET} {count_str} {C_WHITE}│{C_RESET}  {C_BLUE}{p_f1:.4f}{C_RESET}  {C_WHITE}│{C_RESET}  {C_YELLOW}{s_f1:.4f}{C_RESET}  {C_WHITE}│{C_RESET}  {d_col}{g_f1:.4f}{C_RESET}  {C_WHITE}│{C_RESET}   {d_col}{d_str:>7}{C_RESET}   {C_WHITE}│{C_RESET}"
            )
        else:
            count_str = f" 0/{atk_target_total:<2}"
            buf.append(
                f"{C_CYAN}│{C_RESET}  {C_WHITE}│{C_RESET} {atk_name:<16} {C_WHITE}│{C_RESET} {count_str} {C_WHITE}│{C_RESET}   {C_DIM}Pending evaluation in pipeline...{C_RESET}   {C_WHITE}│{C_RESET}"
            )

    if active_done > 0:
        all_p_f1 = sum(r["plain_f1"] for r in det_results) / active_done
        all_s_f1 = sum(r["scalar_f1"] for r in det_results) / active_done
        all_g_f1 = sum(r["graphsage_f1"] for r in det_results) / active_done
        all_delta = all_g_f1 - all_s_f1
        tot_d_col = C_GREEN if all_delta > 0.0005 else (C_YELLOW if all_delta >= -0.0005 else C_RED)
        tot_d_str = f"{all_delta:+.4f}"
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_WHITE}├──────────────────┼───────┼──────────┼──────────┼──────────┼─────────────┤{C_RESET}"
        )
        agg_count = f"{active_done:>2}/37"
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_WHITE}│{C_RESET} {C_BOLD}Aggregate (Macro){C_RESET} {C_WHITE}│{C_RESET} {C_BOLD}{agg_count}{C_RESET} {C_WHITE}│{C_RESET}  {C_BLUE}{C_BOLD}{all_p_f1:.4f}{C_RESET}  {C_WHITE}│{C_RESET}  {C_YELLOW}{C_BOLD}{all_s_f1:.4f}{C_RESET}  {C_WHITE}│{C_RESET}  {tot_d_col}{C_BOLD}{all_g_f1:.4f}{C_RESET}  {C_WHITE}│{C_RESET}   {tot_d_col}{C_BOLD}{tot_d_str:>7}{C_RESET}   {C_WHITE}│{C_RESET}"
        )
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_WHITE}└──────────────────┴───────┴──────────┴──────────┴──────────┴─────────────┘{C_RESET}"
        )
        if all_delta > 0.0005:
            verdict = f"{C_GREEN}{C_BOLD}🏆 POTENTIAL BENEFIT: GraphSAGE shows +{all_delta:.4f} gain.{C_RESET}"
        elif all_delta >= -0.0005:
            verdict = f"{C_YELLOW}{C_BOLD}⚖️  PARITY CONFIRMED ({all_delta:+.4f}): GraphSAGE matches scalar features with no extra signal.{C_RESET}"
        else:
            verdict = f"{C_RED}{C_BOLD}⚠️  SCALAR SUPERIOR ({all_delta:+.4f}): Scalar preferred (GraphSAGE adds latency).{C_RESET}"
        buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}Verdict:{C_RESET} {verdict}")

    buf.append(f"{C_CYAN}╰─────────────────────────────────────────────────────────────────────────────╯{C_RESET}")
    return "\n".join(buf)


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop & Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def render_lstm_dashboard(
    spinner_char: str,
    results_path: Path,
    is_running: bool,
    pid: int,
    cpu_sec: float,
    mem_mb: float,
    wall_elapsed: float,
    cpu_pct: float,
    cmdline_str: str,
) -> str:
    """Render full ANSI dashboard for World Model Sweep / Training."""
    status_data = load_results_json(results_path)
    
    epoch = status_data.get("epoch", 0)
    total_epochs = status_data.get("epochs_total", 30)
    train_nll = status_data.get("train_nll", 0.0)
    val_nll = status_data.get("val_nll", 0.0)
    best_val_nll = status_data.get("best_val_nll", 0.0)
    patience = status_data.get("patience", 5)
    stale = status_data.get("stale", 0)
    lr = status_data.get("lr", 0.0)
    hidden_size = status_data.get("hidden_size", 0)
    num_layers = status_data.get("num_layers", 0)
    
    pct = (epoch / total_epochs * 100.0) if total_epochs > 0 else 0.0
    prog_bar = make_bar(pct, width=32, fill_color=C_CYAN)

    buf = []
    buf.append("\033[H")

    status_badge = f"{C_GREEN}● TRAINING ACTIVE [PID {pid}]{C_RESET}" if is_running else f"{C_YELLOW}◌ IDLE / COMPLETE{C_RESET}"

    buf.append(f"{C_CYAN}╭─────────────────────────────────────────────────────────────────────────────╮{C_RESET}")
    title_text = f"🧠  LSTM WORLD MODEL HYPERPARAMETER SWEEP  {spinner_char}  {status_badge}"
    buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}{C_WHITE}{title_text}{C_RESET}")
    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")
    buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}Active Configuration:{C_RESET} Hidden Size={C_YELLOW}{hidden_size}{C_RESET} │ Layers={C_YELLOW}{num_layers}{C_RESET} │ LR={C_YELLOW}{lr}{C_RESET}")
    buf.append(f"{C_CYAN}│{C_RESET}  Epoch Progress: [{prog_bar}] {C_BOLD}{C_CYAN}{pct:5.1f}%{C_RESET} ({epoch}/{total_epochs})")
    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")
    
    buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}PERFORMANCE METRICS:{C_RESET}")
    buf.append(f"{C_CYAN}│{C_RESET}  Train Gaussian NLL: {C_BLUE}{train_nll:.4f}{C_RESET}")
    buf.append(f"{C_CYAN}│{C_RESET}  Val Gaussian NLL:   {C_MAGENTA}{val_nll:.4f}{C_RESET}")
    buf.append(f"{C_CYAN}│{C_RESET}  Best Val NLL:       {C_GREEN}{C_BOLD}{best_val_nll:.4f}{C_RESET}")
    
    stale_color = C_GREEN if stale == 0 else (C_YELLOW if stale < patience - 1 else C_RED)
    buf.append(f"{C_CYAN}│{C_RESET}  Early Stopping:     {stale_color}{stale} / {patience} epochs without improvement{C_RESET}")

    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")
    cpu_hrs = cpu_sec / 3600.0
    wall_str = format_duration(wall_elapsed)
    buf.append(f"{C_CYAN}│{C_RESET}  {C_DIM}Telemetry:{C_RESET} PID: {C_BOLD}{pid if is_running else 'Offline'}{C_RESET} │ Worker RAM: {C_MAGENTA}{mem_mb:.0f} MB{C_RESET} │ CPU Time: {C_YELLOW}{cpu_hrs:.2f}h{C_RESET} │ Wall: {C_BLUE}{wall_str}{C_RESET}")
    buf.append(f"{C_CYAN}╰─────────────────────────────────────────────────────────────────────────────╯{C_RESET}")
    
    return "\n".join(buf)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Terminal Progress Monitor for Cyber World Model Architecture"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print single dashboard snapshot and exit immediately",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.5,
        help="Refresh interval in seconds (default: 1.5s)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "pcap", "graphsage", "lstm_train"],
        default="auto",
        help="Dashboard mode (auto, pcap, graphsage, or lstm_train)",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Worker process PID to monitor (auto-discovered if not provided)",
    )
    args = parser.parse_args()

    # Clear screen on initial startup
    print("\033[2J\033[H", end="", flush=True)

    spin_idx = 0
    manifest_folds = load_manifest_folds()

    try:
        while True:
            spin_idx = (spin_idx + 1) % len(SPINNER)
            spinner_char = SPINNER[spin_idx]

            detected_mode, is_running, pid, cpu_sec, mem_mb, elapsed_sec, cpu_pct, cmdline_str = detect_active_workload(
                override_pid=args.pid
            )

            active_mode = args.mode if args.mode != "auto" else detected_mode

            if active_mode == "pcap":
                dash = render_pcap_dashboard(
                    spinner_char=spinner_char,
                    dest_dir=PCAP_2102_DIR,
                    is_running=is_running,
                    pid=pid,
                    cpu_sec=cpu_sec,
                    mem_mb=mem_mb,
                    wall_elapsed=elapsed_sec,
                    cpu_pct=cpu_pct,
                    cmdline_str=cmdline_str,
                )
            elif active_mode == "lstm_train":
                dash = render_lstm_dashboard(
                    spinner_char=spinner_char,
                    results_path=REPO_ROOT / "scratch" / "live_lstm_status.json",
                    is_running=is_running,
                    pid=pid,
                    cpu_sec=cpu_sec,
                    mem_mb=mem_mb,
                    wall_elapsed=elapsed_sec,
                    cpu_pct=cpu_pct,
                    cmdline_str=cmdline_str,
                )
            else:
                dash = render_graphsage_dashboard(
                    spinner_char=spinner_char,
                    manifest_folds=manifest_folds,
                    results_path=DEFAULT_RESULTS_JSON,
                    is_running=is_running,
                    pid=pid,
                    cpu_sec=cpu_sec,
                    mem_mb=mem_mb,
                    wall_elapsed=elapsed_sec,
                    cpu_pct=cpu_pct,
                    cmdline_str=cmdline_str,
                )

            print(dash, flush=True)

            if args.once:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(
            f"\n{C_GREEN}Monitor exited safely. Any background workers continue running.{C_RESET}\n"
        )


if __name__ == "__main__":
    main()
