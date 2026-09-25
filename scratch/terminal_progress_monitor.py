#!/usr/bin/env python3
"""
Interactive Terminal Progress Monitor for End-to-End Trainable GraphSAGE LOEO Evaluation.
Displays real-time visual progress bars, model scoreboard (Plain vs Scalar vs GraphSAGE),
process telemetry, fold-level loss curves, Go/No-Go assessment, and ETA.
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
DEFAULT_RESULTS_JSON = Path(r"d:\sih2026\scratch\endtoend_graphsage_results.json")
MANIFEST_PATH = Path(r"d:\sih2026\ml1\artifacts\loeo\corrected_37fold_manifest.json")
DEFAULT_KNOWN_LOG = Path(
    r"C:\Users\MUTHUKUMARAN\.gemini\antigravity-ide\brain\9432d738-b122-4fb9-aeae-fcf713c93025\.system_generated\tasks\task-343.log"
)

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
    Returns (is_running, pid, cpu_sec, mem_mb, elapsed_sec, cpu_pct).
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
                        return True, p.pid, cpu_sec, mem_mb, elapsed_sec, cpu_pct
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return False, 0, 0.0, 0.0, 0.0, 0.0

        candidates = []
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = p.info.get("cmdline") or []
                cmd_str = " ".join(cmdline)
                if target_script in cmd_str or "evaluate_graphsage" in cmd_str:
                    candidates.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not candidates:
            return False, 0, 0.0, 0.0, 0.0, 0.0

        # Sort candidates by CPU times to select the primary compute worker
        best_p = None
        max_cpu = -1.0
        for p in candidates:
            try:
                t = p.cpu_times()
                cpu_total = t.user + t.system
                if cpu_total > max_cpu:
                    max_cpu = cpu_total
                    best_p = p
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
                return True, pid, cpu_sec, mem_mb, elapsed_sec, cpu_pct
    except Exception:
        pass

    return False, 0, 0.0, 0.0, 0.0, 0.0


def find_latest_task_log() -> Path:
    """Scan brain task directories to locate the most recently modified evaluation log."""
    search_pattern = r"C:\Users\MUTHUKUMARAN\.gemini\antigravity-ide\brain\*\.system_generated\tasks\task-*.log"
    logs = glob.glob(search_pattern)
    if not logs:
        return DEFAULT_KNOWN_LOG

    # Sort newest first
    logs.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    for l_path in logs[:12]:
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

    return DEFAULT_KNOWN_LOG if DEFAULT_KNOWN_LOG.exists() else Path(logs[0])


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
            time.sleep(0.08)
    return {}


def parse_log(log_path: Path):
    """Parse output stream and progress metadata from the active task log."""
    if not log_path.exists():
        return {
            "current_target": "DETECTION",
            "has_onset_queued": False,
            "detection_planned": 10,
            "onset_planned": 10,
            "clean_lines": [],
            "early_signal_text": "",
            "assessment_text": "",
        }

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return {
            "current_target": "DETECTION",
            "has_onset_queued": False,
            "detection_planned": 10,
            "onset_planned": 10,
            "clean_lines": [],
            "early_signal_text": "",
            "assessment_text": "",
        }

    # Target detection
    has_onset_started = "TARGET: ONSET" in text
    has_detection_started = "TARGET: DETECTION" in text
    has_onset_queued = "--target both" in text or "target='both'" in text

    if has_onset_started and text.rfind("TARGET: ONSET") > text.rfind("TARGET: DETECTION"):
        current_target = "ONSET"
    else:
        current_target = "DETECTION"

    # Number of folds per stage
    det_matches = re.findall(r"Evaluating\s+(\d+)\s+folds for detection", text)
    detection_planned = int(det_matches[-1]) if det_matches else 10

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
    ][-4:]

    return {
        "current_target": current_target,
        "has_onset_queued": has_onset_queued,
        "detection_planned": detection_planned,
        "onset_planned": onset_planned,
        "clean_lines": clean_lines,
        "early_signal_text": early_signal_text,
        "assessment_text": assessment_text,
    }


def make_bar(percent: float, width: int = 36, fill_color: str = C_GREEN) -> str:
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
) -> str:
    """Render full ANSI dashboard string."""
    results_data = load_results_json(results_path)
    log_data = parse_log(log_path)

    current_target = log_data["current_target"]
    det_results = results_data.get("detection", [])
    onset_results = results_data.get("onset", [])

    det_done = len(det_results)
    det_planned = max(log_data["detection_planned"], det_done)

    onset_done = len(onset_results)
    onset_planned = max(log_data["onset_planned"], onset_done)

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
        stage_num = "Stage 1/2" if log_data["has_onset_queued"] else "Stage 1/1"

    # Pace & ETA estimation
    total_fold_durations = sum(r.get("elapsed_sec", 0.0) for r in active_results)
    avg_fold_sec = (total_fold_durations / active_done) if active_done > 0 else 135.0
    remaining_folds = max(0, active_planned - active_done)

    # Active fold info
    active_fold_id = active_done
    if active_fold_id < active_planned:
        f_meta = manifest_folds.get(active_fold_id, {})
        active_ep = f_meta.get("episode_id", f"Fold_{active_fold_id}")
        active_attack = f_meta.get("attack_type", "Botnet")
    else:
        active_ep = "All Target Folds Complete"
        active_attack = "Completed"

    # Approximate time on current fold
    if is_running and active_done < active_planned:
        current_fold_elapsed = max(0.0, wall_elapsed - total_fold_durations)
        if current_fold_elapsed > (avg_fold_sec * 2.0):
            current_fold_elapsed = wall_elapsed % max(30.0, avg_fold_sec)
    else:
        current_fold_elapsed = 0.0

    rem_sec = max(0.0, (remaining_folds * avg_fold_sec) - current_fold_elapsed)
    eta_time = datetime.now() + timedelta(seconds=rem_sec)
    eta_clock_str = eta_time.strftime("%H:%M:%S")

    # Percentage
    pct = (active_done / active_planned * 100.0) if active_planned > 0 else 0.0
    prog_bar = make_bar(pct, width=36, fill_color=C_CYAN)

    buf = []
    buf.append("\033[H")

    # Status Pill
    if is_running:
        status_badge = f"{C_GREEN}{C_BOLD}● RUNNING{C_RESET}"
    elif active_done >= active_planned and active_done > 0:
        status_badge = f"{C_CYAN}{C_BOLD}✓ COMPLETED{C_RESET}"
    else:
        status_badge = f"{C_YELLOW}◌ IDLE / FINISHED{C_RESET}"

    # Banner Header
    buf.append(f"{C_CYAN}╭─────────────────────────────────────────────────────────────────────────────╮{C_RESET}")
    title_text = f"🛡️  END-TO-END GRAPHSAGE LOEO EVALUATION MONITOR  {spinner_char}  [{status_badge}]"
    buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}{C_WHITE}{title_text}{C_RESET}")
    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")

    # Target Phase & Progress Bar
    target_desc = (
        "Binary Attack Detection"
        if current_target == "DETECTION"
        else "Early Warning Prediction"
    )
    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_BOLD}Target Phase [{stage_num}]:{C_RESET} {C_YELLOW}{current_target}{C_RESET} ({target_desc})"
    )
    buf.append(
        f"{C_CYAN}│{C_RESET}  [{prog_bar}] {C_BOLD}{C_CYAN}{pct:5.1f}%{C_RESET} ({active_done}/{active_planned} Folds Done)"
    )

    if active_done < active_planned and is_running:
        fold_el_str = format_duration(current_fold_elapsed)
        avg_fold_str = format_duration(avg_fold_sec)
        rem_str = format_duration(rem_sec)
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_DIM}Active:{C_RESET} {C_BOLD}Fold {active_fold_id:02d}{C_RESET} ({C_PURPLE}{active_attack}{C_RESET}) │ {C_DIM}Episode:{C_RESET} {active_ep[:23]} │ Time: {C_YELLOW}{fold_el_str}{C_RESET}"
        )
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_DIM}Pace:{C_RESET} ~{avg_fold_str}/fold │ {C_DIM}Remaining:{C_RESET} {C_YELLOW}{remaining_folds} folds (~{rem_str}){C_RESET} │ {C_DIM}ETA:{C_RESET} {C_WHITE}{C_BOLD}{eta_clock_str}{C_RESET}"
        )
    elif active_done >= active_planned:
        total_time_str = format_duration(total_fold_durations)
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_GREEN}{C_BOLD}★ Target '{current_target}' Complete!{C_RESET} ({active_done} folds in {total_time_str})"
        )

    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")

    # Telemetry Bar
    cpu_hrs = cpu_sec / 3600.0
    wall_str = format_duration(wall_elapsed)
    active_pid = pid if pid > 0 else "N/A"
    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_DIM}Telemetry:{C_RESET} PID: {C_BOLD}{active_pid}{C_RESET} │ CPU Compute: {C_YELLOW}{cpu_hrs:.2f} hrs{C_RESET} │ RAM: {C_MAGENTA}{mem_mb:.0f} MB{C_RESET} │ Wall: {C_BLUE}{wall_str}{C_RESET}"
    )
    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")

    # Model Benchmark Scoreboard
    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_BOLD}MODEL BENCHMARK SCOREBOARD ({current_target} Macro-Average, N={active_done}):{C_RESET}"
    )

    if active_done > 0:
        plain_f1_avg = sum(r["plain_f1"] for r in active_results) / active_done
        plain_pr_avg = sum(r["plain_prec"] for r in active_results) / active_done
        plain_rc_avg = sum(r["plain_rec"] for r in active_results) / active_done

        scalar_f1_avg = sum(r["scalar_f1"] for r in active_results) / active_done
        scalar_pr_avg = sum(r["scalar_prec"] for r in active_results) / active_done
        scalar_rc_avg = sum(r["scalar_rec"] for r in active_results) / active_done

        graphsage_f1_avg = sum(r["graphsage_f1"] for r in active_results) / active_done
        graphsage_pr_avg = sum(r["graphsage_prec"] for r in active_results) / active_done
        graphsage_rc_avg = sum(r["graphsage_rec"] for r in active_results) / active_done

        delta_gs_sc = graphsage_f1_avg - scalar_f1_avg

        # Comparison Table
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_WHITE}┌──────────────────────────────────────┬─────────┬─────────┬─────────┐{C_RESET}"
        )
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_WHITE}│{C_RESET} {C_BOLD}Model Architecture{C_RESET}                   {C_WHITE}│{C_RESET} {C_BOLD}F1 Score{C_RESET}{C_WHITE}│{C_RESET} {C_BOLD}Precis.{C_RESET} {C_WHITE}│{C_RESET} {C_BOLD}Recall{C_RESET}  {C_WHITE}│{C_RESET}"
        )
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_WHITE}├──────────────────────────────────────┼─────────┼─────────┼─────────┤{C_RESET}"
        )

        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_WHITE}│{C_RESET} 1. Plain Stacked LSTM (32 PCA)        {C_WHITE}│{C_RESET}  {C_BLUE}{plain_f1_avg:.4f}{C_RESET} {C_WHITE}│{C_RESET}  {plain_pr_avg:.4f} {C_WHITE}│{C_RESET}  {plain_rc_avg:.4f} {C_WHITE}│{C_RESET}"
        )
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_WHITE}│{C_RESET} 2. Scalar Graph-Augmented (32+4)      {C_WHITE}│{C_RESET}  {C_YELLOW}{scalar_f1_avg:.4f}{C_RESET} {C_WHITE}│{C_RESET}  {scalar_pr_avg:.4f} {C_WHITE}│{C_RESET}  {scalar_rc_avg:.4f} {C_WHITE}│{C_RESET}"
        )

        gs_color = (
            C_GREEN
            if delta_gs_sc > 0.0005
            else (C_YELLOW if delta_gs_sc >= -0.0005 else C_RED)
        )
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_WHITE}│{C_RESET} 3. Trainable GraphSAGE Fusion (32+GNN){C_WHITE}│{C_RESET}  {gs_color}{graphsage_f1_avg:.4f}{C_RESET} {C_WHITE}│{C_RESET}  {graphsage_pr_avg:.4f} {C_WHITE}│{C_RESET}  {graphsage_rc_avg:.4f} {C_WHITE}│{C_RESET}"
        )
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_WHITE}└──────────────────────────────────────┴─────────┴─────────┴─────────┘{C_RESET}"
        )

        # Decision Indicator
        if delta_gs_sc > 0.0005:
            decision_badge = f"{C_GREEN}{C_BOLD}🏆 ADVANTAGE GRAPHSAGE (+{delta_gs_sc:.4f}) → PROCEED TO LATENCY BENCHMARK{C_RESET}"
        elif delta_gs_sc >= -0.0005:
            decision_badge = f"{C_YELLOW}{C_BOLD}⚖️  PARITY / TIED ({delta_gs_sc:+.4f}) → SCALAR PREFERRED (LOW LATENCY){C_RESET}"
        else:
            decision_badge = f"{C_RED}{C_BOLD}⚠️  SCALAR SUPERIOR ({delta_gs_sc:+.4f}) → TOPOLOGY ADDS NO SIGNAL{C_RESET}"

        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_BOLD}Delta (GraphSAGE - Scalar):{C_RESET} {gs_color}{delta_gs_sc:+.4f}{C_RESET} │ {decision_badge}"
        )
    else:
        buf.append(
            f"{C_CYAN}│{C_RESET}  {C_YELLOW}Training in progress for Fold 00... (Scoreboard populates after Fold 00){C_RESET}"
        )

    buf.append(f"{C_CYAN}├─────────────────────────────────────────────────────────────────────────────┤{C_RESET}")

    # Recent Fold Breakdown Table
    buf.append(f"{C_CYAN}│{C_RESET}  {C_BOLD}Recent Completed LOEO Folds:{C_RESET}")
    buf.append(
        f"{C_CYAN}│{C_RESET}  {C_DIM}Fold   Attack Type       Plain F1   Scalar F1   GraphSAGE F1   Delta   Duration{C_RESET}"
    )

    if active_results:
        for r in active_results[-5:]:
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
            l_end = f"{losses[-1]:.4f}"
            buf.append(
                f"{C_CYAN}│{C_RESET}  {C_DIM}Latest Fold {latest_r['fold_id']:02d} GraphSAGE Loss Curve:{C_RESET} Epoch 1: {l_start} → Epoch {len(losses)}: {l_end}"
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
        f"{C_DIM}Live auto-refresh (Ctrl+C to exit monitor — background evaluation will continue){C_RESET}"
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
        help="Path to task log file (auto-discovered if not provided)",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Worker process PID to monitor (auto-discovered if not provided)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    log_path = Path(args.log) if args.log else find_latest_task_log()
    manifest_folds = load_manifest_folds()

    if args.once:
        is_running, pid, cpu_sec, mem_mb, elapsed_sec, cpu_pct = find_active_process(
            override_pid=args.pid
        )
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
        )
        print(dash)
        return

    # Clear screen on initial startup
    print("\033[2J\033[H", end="", flush=True)

    spin_idx = 0
    try:
        while True:
            spin_idx = (spin_idx + 1) % len(SPINNER)
            spinner_char = SPINNER[spin_idx]

            # Re-check log path if not explicitly provided
            if not args.log and (not log_path or not log_path.exists()):
                log_path = find_latest_task_log()

            is_running, pid, cpu_sec, mem_mb, elapsed_sec, cpu_pct = find_active_process(
                override_pid=args.pid
            )

            dash = render_dashboard(
                spinner_char=spinner_char,
                manifest_folds=manifest_folds,
                results_path=results_path,
                log_path=log_path,
                is_running=is_running,
                pid=pid,
                cpu_sec=cpu_sec,
                mem_mb=mem_mb,
                wall_elapsed=elapsed_sec,
                cpu_pct=cpu_pct,
            )
            print(dash, flush=True)
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(
            f"\n{C_GREEN}Monitor exited safely. Background process continues running.{C_RESET}\n"
        )


if __name__ == "__main__":
    main()
