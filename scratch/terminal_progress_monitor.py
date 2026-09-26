#!/usr/bin/env python3
"""
Interactive Terminal Progress Monitor for End-to-End Trainable GraphSAGE LOEO Evaluation.
Displays real-time visual progress bars, attack-cohort progression (Botnet, SSH-Bruteforce, DDOS-LOIC-UDP),
model benchmark scoreboard with per-attack-type breakdown (Plain vs Scalar vs GraphSAGE),
process telemetry, fold-level loss curves, Go/No-Go assessment, and accurate ETA.
"""

import os
import sys
import time
import re
import json
import glob
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
REPO_ROOT = Path(r"d:\sih2026")
DEFAULT_RESULTS_JSON = REPO_ROOT / "scratch" / "endtoend_graphsage_results.json"
MANIFEST_PATH = REPO_ROOT / "ml1" / "artifacts" / "loeo" / "corrected_37fold_manifest.json"

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


def find_active_process(target_script: str = "evaluate_endtoend_graphsage_loeo", override_pid: int = None):
    """
    Dynamically discover the running evaluation worker process using psutil.
    Returns (is_running, pid, cpu_sec, mem_mb, elapsed_sec, cpu_pct, cmdline_str).
    """
    try:
        import psutil

        if override_pid:
            try:
                p = psutil.Process(override_pid)
                if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                    with p.oneshot():
                        cpu_times = p.cpu_times()
                        cpu_sec = cpu_times.user + cpu_times.system
                        mem_mb = p.memory_info().rss / (1024 * 1024)
                        elapsed_sec = max(0, time.time() - p.create_time())
                        cpu_pct = p.cpu_percent(interval=None)
                        cmdline_str = " ".join(p.cmdline() or [])
                        return True, p.pid, cpu_sec, mem_mb, elapsed_sec, cpu_pct, cmdline_str
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return False, 0, 0.0, 0.0, 0.0, 0.0, ""

        candidates = []
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = p.info.get("cmdline") or []
                cmd_str = " ".join(cmdline)
                # Ignore self monitor process
                if "terminal_progress_monitor" in cmd_str:
                    continue
                if target_script in cmd_str or "evaluate_graphsage" in cmd_str:
                    candidates.append((p, cmd_str))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not candidates:
            return False, 0, 0.0, 0.0, 0.0, 0.0, ""

        # Sort candidates by CPU times to select the primary compute worker
        best_p = None
        best_cmd = ""
        max_cpu = -1.0
        for p, cmd_str in candidates:
            try:
                t = p.cpu_times()
                cpu_total = t.user + t.system
                if cpu_total > max_cpu:
                    max_cpu = cpu_total
                    best_p = p
                    best_cmd = cmd_str
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if best_p:
            with best_p.oneshot():
                pid = best_p.pid
                cpu_times = best_p.cpu_times()
                cpu_sec = cpu_times.user + cpu_times.system
                mem_mb = best_p.memory_info().rss / (1024 * 1024)
                create_ts = best_p.create_time()
                elapsed_sec = max(0, time.time() - create_ts)
                try:
                    cpu_pct = best_p.cpu_percent(interval=None)
                except Exception:
                    cpu_pct = 0.0
                return True, pid, cpu_sec, mem_mb, elapsed_sec, cpu_pct, best_cmd
    except Exception:
        pass

    return False, 0, 0.0, 0.0, 0.0, 0.0, ""


def find_latest_task_log(active_pid: int = None) -> Path:
    """Scan brain task directories to locate the most recently modified evaluation log."""
    search_pattern = r"C:\Users\MUTHUKUMARAN\.gemini\antigravity-ide\brain\*\.system_generated\tasks\task-*.log"
    logs = glob.glob(search_pattern)
    if not logs:
        return Path(r"d:\sih2026\scratch\evaluate.log")

    # Sort newest modified first
    logs.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    # First check logs that have recent evaluation markers
    for l_path in logs[:30]:
        try:
            with open(l_path, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(2048)
                if (
                    "evaluate_endtoend_graphsage_loeo" in head
                    or "Building real per-window graph cache" in head
                    or "TARGET: DETECTION" in head
                    or "TARGET: ONSET" in head
                ):
                    return Path(l_path)
        except Exception:
            continue

    return Path(logs[0])


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
    for _ in range(4):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.06)
    return {}


def parse_log(log_path: Path):
    """Parse output stream and progress metadata from the active task log."""
    default_meta = {
        "current_target": "DETECTION",
        "has_onset_queued": False,
        "detection_planned": 37,
        "onset_planned": 37,
        "clean_lines": [],
        "early_signal_text": "",
        "assessment_text": "",
    }
    if not log_path or not log_path.exists():
        return default_meta

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return default_meta

    # Target detection
    has_onset_started = "TARGET: ONSET" in text
    has_onset_queued = "--target both" in text or "target='both'" in text

    if has_onset_started and text.rfind("TARGET: ONSET") > text.rfind("TARGET: DETECTION"):
        current_target = "ONSET"
    else:
        current_target = "DETECTION"

    # Number of folds per stage
    det_matches = re.findall(r"Evaluating\s+(\d+)\s+folds for detection", text)
    detection_planned = int(det_matches[-1]) if det_matches else 37

    onset_matches = re.findall(r"Evaluating\s+(\d+)\s+folds for onset", text)
    onset_planned = int(onset_matches[-1]) if onset_matches else detection_planned

    # Early signal snippet
    early_signal_text = ""
    if "-- EARLY SIGNAL" in text:
        parts = text.split("-- EARLY SIGNAL")[-1].split("--------------------------------------")
        if parts:
            early_signal_text = parts[0].strip()

    # Go/No-Go assessment snippet
    assessment_text = ""
    if "GO/NO-GO ASSESSMENT" in text:
        assessment_part = text.split("GO/NO-GO ASSESSMENT")[-1].strip()
        lines = [l.strip() for l in assessment_part.splitlines() if l.strip() and not l.startswith("=")]
        assessment_text = " ".join(lines[:2])

    # Clean recent log lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    clean_lines = [
        l
        for l in lines
        if not l.startswith("warnings.warn")
        and "FutureWarning" not in l
        and "RuntimeWarning" not in l
        and "torch.jit.script" not in l
        and "Full results saved to" not in l
        and "Building real per-window" not in l
        and not l.startswith("D:\\sih2026")
    ][-5:]

    return {
        "current_target": current_target,
        "has_onset_queued": has_onset_queued,
        "detection_planned": detection_planned,
        "onset_planned": onset_planned,
        "clean_lines": clean_lines,
        "early_signal_text": early_signal_text,
        "assessment_text": assessment_text,
    }


def make_bar(percent: float, width: int = 34, fill_color: str = C_GREEN) -> str:
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


def render_dashboard(
    spinner_char: str,
    manifest_folds: dict,
    results_path: Path,
    log_path: Path,
    is_running: bool,
    pid: int,
    cpu_sec: float,
    mem_mb: float,
    wall_elapsed: float,
    cpu_pct: float,
    cmdline_str: str,
) -> str:
    """Render full ANSI dashboard string."""
    results_data = load_results_json(results_path)
    log_data = parse_log(log_path)

    current_target = log_data["current_target"]
    det_results = results_data.get("detection", [])
    onset_results = results_data.get("onset", [])

    det_done = len(det_results)
    det_planned = max(log_data["detection_planned"], det_done, 37)

    onset_done = len(onset_results)
    onset_planned = max(log_data["onset_planned"], onset_done, 37)

    # Active target calculations
    if current_target == "ONSET":
        active_results = onset_results
        active_done = onset_done
        active_planned = onset_planned
        stage_num = "Stage 2/2"
    else:
        active_results = det_results
        active_done = det_done
        active_planned = det_planned
        stage_num = "Stage 1/1 (Detection Target)"

    # Pace & ETA estimation
    durations = [r.get("elapsed_sec", 0.0) for r in active_results if r.get("elapsed_sec", 0.0) > 0]
    # Use recent folds (last 5) for current pace if available
    recent_durations = durations[-6:] if len(durations) >= 6 else durations
    avg_fold_sec = (sum(recent_durations) / len(recent_durations)) if recent_durations else 120.0
    remaining_folds = max(0, active_planned - active_done)

    # Active fold info
    active_fold_id = active_done
    if active_fold_id < active_planned:
        f_meta = manifest_folds.get(active_fold_id, {})
        active_ep = f_meta.get("episode_id", f"Fold_{active_fold_id}")
        active_attack = f_meta.get("attack_type", "Pending")
    else:
        active_ep = "All 37 LOEO Folds Complete"
        active_attack = "Completed"

    # Current fold elapsed time: accurately computed from results file modification timestamp
    if is_running and active_done < active_planned:
        if results_path.exists():
            last_file_mod = os.path.getmtime(results_path)
            current_fold_elapsed = max(0.0, time.time() - last_file_mod)
        else:
            current_fold_elapsed = wall_elapsed % max(1.0, avg_fold_sec)
    else:
        current_fold_elapsed = 0.0

    rem_sec = max(0.0, (remaining_folds * avg_fold_sec) - current_fold_elapsed)
    eta_time = datetime.now() + timedelta(seconds=rem_sec)
    eta_clock_str = eta_time.strftime("%H:%M:%S")

    # Percentage
    pct = (active_done / active_planned * 100.0) if active_planned > 0 else 0.0
    prog_bar = make_bar(pct, width=34, fill_color=C_CYAN)

    # Cohort breakdown
    cohort_counts = {"Botnet": (0, 10), "SSH-Bruteforce": (0, 9), "DDOS-LOIC-UDP": (0, 18)}
    for r in active_results:
        atk = r.get("attack_type", "")
        if atk in cohort_counts:
            d, tot = cohort_counts[atk]
            cohort_counts[atk] = (d + 1, tot)

    buf = []
    # Cursor home to prevent terminal flickering
    buf.append("\033[H")

    # Status Pill
    if is_running:
        status_badge = f"{C_GREEN}{C_BOLD}● RUNNING [PID {pid}]{C_RESET}"
    elif active_done >= active_planned and active_done > 0:
        status_badge = f"{C_CYAN}{C_BOLD}✓ 37/37 FOLDS COMPLETE{C_RESET}"
    else:
        status_badge = f"{C_YELLOW}◌ IDLE / COMPLETE{C_RESET}"

    # Banner Header
    buf.append(f"{C_CYAN}╭─────────────────────────────────────────────────────────────────────────────╮{C_RESET}")
    title_text = f"🛡️  GRAPHSAGE LOEO FULL 37-FOLD EVALUATION MONITOR  {spinner_char}  {status_badge}"
    buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}{C_WHITE}{title_text}{C_RESET}")
    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")

    # Target Phase & Overall Progress Bar
    target_desc = "Binary Attack Detection Across All 37 Episodes"
    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_BOLD}Target Phase [{stage_num}]:{C_RESET} {C_YELLOW}{current_target}{C_RESET} ({target_desc})"
    )
    buf.append(
        f"{C_CYAN}│{C_RESET}  [{prog_bar}] {C_BOLD}{C_CYAN}{pct:5.1f}%{C_RESET} ({C_WHITE}{active_done}/{active_planned}{C_RESET} Folds Done)"
    )

    # Cohort progression pills
    b_done, b_tot = cohort_counts["Botnet"]
    s_done, s_tot = cohort_counts["SSH-Bruteforce"]
    d_done, d_tot = cohort_counts["DDOS-LOIC-UDP"]

    b_str = f"{C_GREEN}✓ Botnet: {b_done}/{b_tot}{C_RESET}" if b_done >= b_tot else f"{C_YELLOW}● Botnet: {b_done}/{b_tot}{C_RESET}"
    s_str = f"{C_GREEN}✓ SSH-Brute: {s_done}/{s_tot}{C_RESET}" if s_done >= s_tot else (f"{C_CYAN}▶ SSH-Brute: {s_done}/{s_tot}{C_RESET}" if b_done >= b_tot else f"{C_DIM}SSH-Brute: {s_done}/{s_tot}{C_RESET}")
    d_str = f"{C_GREEN}✓ DDOS-LOIC: {d_done}/{d_tot}{C_RESET}" if d_done >= d_tot else (f"{C_CYAN}▶ DDOS-LOIC: {d_done}/{d_tot}{C_RESET}" if (b_done >= b_tot and s_done >= s_tot) else f"{C_DIM}DDOS-LOIC: {d_done}/{d_tot}{C_RESET}")

    buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}Cohorts:{C_RESET} {b_str}  │  {s_str}  │  {d_str}")

    if active_done < active_planned and is_running:
        fold_el_str = format_duration(current_fold_elapsed)
        avg_fold_str = format_duration(avg_fold_sec)
        rem_str = format_duration(rem_sec)
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_DIM}Active:{C_RESET} {C_BOLD}Fold {active_fold_id:02d}{C_RESET} ({C_PURPLE}{active_attack}{C_RESET}) │ Episode: {active_ep[:23]} │ Time: {C_YELLOW}{fold_el_str}{C_RESET}"
        )
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_DIM}Pace:{C_RESET} ~{avg_fold_str}/fold │ {C_DIM}Remaining:{C_RESET} {C_YELLOW}{remaining_folds} folds (~{rem_str}){C_RESET} │ {C_DIM}ETA:{C_RESET} {C_WHITE}{C_BOLD}{eta_clock_str}{C_RESET}"
        )
    elif active_done >= active_planned:
        total_time_str = format_duration(sum(durations))
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_GREEN}{C_BOLD}★ All 37 Folds Fully Evaluated!{C_RESET} (Total run time: {total_time_str})"
        )

    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")

    # Process Telemetry
    cpu_hrs = cpu_sec / 3600.0
    wall_str = format_duration(wall_elapsed)
    active_pid_str = str(pid) if pid > 0 else "N/A"
    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_DIM}Telemetry:{C_RESET} PID: {C_BOLD}{active_pid_str}{C_RESET} │ CPU Compute: {C_YELLOW}{cpu_hrs:.2f} hrs{C_RESET} │ RAM: {C_MAGENTA}{mem_mb:.0f} MB{C_RESET} │ Session Wall: {C_BLUE}{wall_str}{C_RESET}"
    )
    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")

    # Per-Attack-Type Scoreboard Breakdown
    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_BOLD}PER-ATTACK-TYPE SCOREBOARD BREAKDOWN (N={active_done}/37 Folds):{C_RESET}"
    )

    if active_done > 0:
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_WHITE}┌──────────────────┬───────┬──────────┬──────────┬──────────┬─────────────┐{C_RESET}"
        )
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_WHITE}│{C_RESET} {C_BOLD}Attack Cohort{C_RESET}    {C_WHITE}│{C_RESET} {C_BOLD}Folds{C_RESET} {C_WHITE}│{C_RESET} {C_BOLD}Plain F1{C_RESET} {C_WHITE}│{C_RESET} {C_BOLD}Scalar F1{C_RESET}{C_WHITE}│{C_RESET} {C_BOLD}G-SAGE F1{C_RESET}{C_WHITE}│{C_RESET} {C_BOLD}Delta (GS-Sc){C_RESET}{C_WHITE}│{C_RESET}"
        )
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_WHITE}├──────────────────┼───────┼──────────┼──────────┼──────────┼─────────────┤{C_RESET}"
        )

        for atk_name, atk_target_total in [("Botnet", 10), ("SSH-Bruteforce", 9), ("DDOS-LOIC-UDP", 18)]:
            atk_records = [r for r in active_results if r.get("attack_type") == atk_name]
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

        # Aggregate Row
        all_p_f1 = sum(r["plain_f1"] for r in active_results) / active_done
        all_s_f1 = sum(r["scalar_f1"] for r in active_results) / active_done
        all_g_f1 = sum(r["graphsage_f1"] for r in active_results) / active_done
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

        # Verdict Summary
        if all_delta > 0.0005:
            verdict = f"{C_GREEN}{C_BOLD}🏆 POTENTIAL BENEFIT: GraphSAGE shows +{all_delta:.4f} gain. Proceed to latency benchmark.{C_RESET}"
        elif all_delta >= -0.0005:
            verdict = f"{C_YELLOW}{C_BOLD}⚖️  PARITY CONFIRMED ({all_delta:+.4f}): GraphSAGE matches scalar features with no extra signal.{C_RESET}"
        else:
            verdict = f"{C_RED}{C_BOLD}⚠️  SCALAR SUPERIOR ({all_delta:+.4f}): GNN topological fusion degrades performance.{C_RESET}"

        buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}Verdict:{C_RESET} {verdict}")
    else:
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_YELLOW}Training in progress for Fold 00... (Scoreboard populates after Fold 00){C_RESET}"
        )

    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")

    # Recent Fold Breakdown Table (Last 4 folds)
    buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}Recent Completed LOEO Folds:{C_RESET}")
    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_DIM}Fold   Attack Type       Plain F1   Scalar F1   GraphSAGE F1   Delta   Duration{C_RESET}"
    )

    if active_results:
        for r in active_results[-4:]:
            fid = r["fold_id"]
            atk = r.get("attack_type", "Unknown")[:14].ljust(14)
            p_f1 = f"{r['plain_f1']:.4f}"
            s_f1 = f"{r['scalar_f1']:.4f}"
            g_f1 = f"{r['graphsage_f1']:.4f}"
            delta_val = r["graphsage_f1"] - r["scalar_f1"]
            d_str = f"{delta_val:+.4f}"
            d_col = (
                C_GREEN
                if delta_val > 0.0005
                else (C_YELLOW if delta_val >= -0.0005 else C_RED)
            )
            t_str = f"{r.get('elapsed_sec', 0.0):.1f}s"
            buf.append(
                f"{C_CYAN}│{C_RESET}  Fold {fid:02d} {atk}    {C_BLUE}{p_f1}{C_RESET}     {C_YELLOW}{s_f1}{C_RESET}       {d_col}{g_f1}{C_RESET}     {d_col}{d_str:>7}{C_RESET}   {t_str:>7}"
            )

        # Loss curve snippet for the most recently completed fold
        latest_r = active_results[-1]
        losses = latest_r.get("graphsage_loss_curve", [])
        if losses and len(losses) >= 2:
            l_start = f"{losses[0]:.4f}"
            l_mid = f"{losses[len(losses)//2]:.4f}"
            l_end = f"{losses[-1]:.4f}"
            buf.append(
                f"{C_CYAN}│{C_RESET}  {C_DIM}Latest Fold {latest_r['fold_id']:02d} GraphSAGE Loss Curve:{C_RESET} E1: {l_start} → E{len(losses)//2}: {l_mid} → E{len(losses)}: {l_end}"
            )
    else:
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_DIM}(No completed folds recorded yet. Currently executing initial epoch...){C_RESET}"
        )

    # Formal Assessment Banner if present
    if log_data["assessment_text"]:
        buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")
        buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}Formal Assessment:{C_RESET} {C_WHITE}{log_data['assessment_text'][:70]}{C_RESET}")

    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")

    # Live Log Stream
    buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}Latest Task Log Stream:{C_RESET}")
    clean_lines = log_data.get("clean_lines", [])
    if clean_lines:
        for cl in clean_lines[-3:]:
            trunc_cl = (cl[:68] + "...") if len(cl) > 68 else cl.ljust(71)
            buf.append(f"{C_CYAN}│{C_RESET}  {C_DIM}> {trunc_cl}{C_RESET}")
    else:
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_DIM}> Initializing PyTorch Geometric graph cache and LOEO manifest...{C_RESET}"
        )

    buf.append(f"{C_CYAN}╰─────────────────────────────────────────────────────────────────────────────╯{C_RESET}")
    buf.append(
        f"{C_DIM}Live auto-refresh every 1.5s (Press Ctrl+C to exit monitor — background evaluation will continue){C_RESET}"
    )

    return "\n".join(buf)


def main():
    parser = argparse.ArgumentParser(
        description="Live Terminal Progress Monitor for End-to-End GraphSAGE LOEO Evaluation"
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
        "--results",
        type=str,
        default=str(DEFAULT_RESULTS_JSON),
        help="Path to endtoend_graphsage_results.json",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="",
        help="Path to task log file (auto-discovered dynamically if not provided)",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Worker process PID to monitor (auto-discovered if not provided)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    manifest_folds = load_manifest_folds()

    if args.once:
        is_running, pid, cpu_sec, mem_mb, elapsed_sec, cpu_pct, cmdline_str = find_active_process(
            override_pid=args.pid
        )
        log_path = Path(args.log) if args.log else find_latest_task_log(active_pid=pid)
        dash = render_dashboard(
            spinner_char="●",
            manifest_folds=manifest_folds,
            results_path=results_path,
            log_path=log_path,
            is_running=is_running,
            pid=pid,
            cpu_sec=cpu_sec,
            mem_mb=mem_mb,
            wall_elapsed=elapsed_sec,
            cpu_pct=cpu_pct,
            cmdline_str=cmdline_str,
        )
        print(dash)
        return

    # Clear screen on initial startup
    print("\033[2J\033[H", end="", flush=True)

    spin_idx = 0
    ticks = 0
    cached_log_path = Path(args.log) if args.log else None

    try:
        while True:
            spin_idx = (spin_idx + 1) % len(SPINNER)
            spinner_char = SPINNER[spin_idx]
            ticks += 1

            is_running, pid, cpu_sec, mem_mb, elapsed_sec, cpu_pct, cmdline_str = find_active_process(
                override_pid=args.pid
            )

            # Re-discover log file dynamically every 4 ticks (~6s) or if not yet set
            if not args.log and (cached_log_path is None or not cached_log_path.exists() or ticks % 4 == 0):
                cached_log_path = find_latest_task_log(active_pid=pid)

            dash = render_dashboard(
                spinner_char=spinner_char,
                manifest_folds=manifest_folds,
                results_path=results_path,
                log_path=cached_log_path,
                is_running=is_running,
                pid=pid,
                cpu_sec=cpu_sec,
                mem_mb=mem_mb,
                wall_elapsed=elapsed_sec,
                cpu_pct=cpu_pct,
                cmdline_str=cmdline_str,
            )
            print(dash, flush=True)
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(
            f"\n{C_GREEN}Monitor exited safely. Background process (PID {pid if 'pid' in locals() else 'active'}) continues running.{C_RESET}\n"
        )


if __name__ == "__main__":
    main()
