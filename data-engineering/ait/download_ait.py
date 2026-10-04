#!/usr/bin/env python3
"""
download_ait.py — Download AIT-LDS v2.0 scenarios from Zenodo.

Downloads a named scenario from Zenodo record 5789064 into AIT_RAW_DIR
(default: data-engineering/data/ait/raw), with HTTP resume support and
md5 checksum verification against Zenodo's listed checksums.

Usage:
    python data-engineering/ait/download_ait.py --scenario russellmitchell
    python data-engineering/ait/download_ait.py --scenario fox --raw-dir /mnt/data/ait

Environment:
    AIT_RAW_DIR  — override the default download directory
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("ERROR: 'requests' is required.  pip install requests")


# ── Zenodo record metadata ──────────────────────────────────────────────
ZENODO_RECORD_ID = "5789064"
ZENODO_FILES_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files"
ZENODO_DOI = "10.5281/zenodo.5789064"

# Pre-baked checksums from the Zenodo API (md5)
CHECKSUMS = {
    "fox":             "md5:b6cd133398fbed6f3e7434dafbc45756",
    "harrison":        "md5:4fbe3dd17a194776b2851116f8b51380",
    "russellmitchell": "md5:78e9b7d169b438f03816019b52ab3ff7",
    "santos":          "md5:fbd2663a41c83f345aa08a7fb11a6c6c",
    "shaw":            "md5:872bbec7ffe12f14e06364e328a96088",
    "wardbeck":        "md5:ff524634e64c7779def2eb804e8c52ac",
    "wheeler":         "md5:e3f5605d867fcc644f93ae6c736cdb30",
    "wilson":          "md5:952dd37cb262ef0054ad6d5371bdbe70",
}

FILE_SIZES = {
    "fox":             15_837_626_707,
    "harrison":        16_773_705_987,
    "russellmitchell":  7_132_670_599,
    "santos":          10_004_843_924,
    "shaw":            17_594_549_155,
    "wardbeck":        17_163_231_368,
    "wheeler":         19_593_129_576,
    "wilson":          26_544_770_737,
}

ALL_SCENARIOS = sorted(CHECKSUMS.keys())

DEFAULT_RAW_DIR = os.environ.get("AIT_RAW_DIR", "data-engineering/data/ait/raw")
DOWNLOAD_LOG = "data-engineering/ait/download_log.json"


def _md5_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute md5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download_url(scenario: str) -> str:
    return f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{scenario}.zip"


def download_scenario(scenario: str, raw_dir: str) -> dict:
    """Download a scenario zip with resume, return a log dict."""
    if scenario not in CHECKSUMS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Choose from: {ALL_SCENARIOS}"
        )

    os.makedirs(raw_dir, exist_ok=True)
    dest = os.path.join(raw_dir, f"{scenario}.zip")
    url = _download_url(scenario)
    expected_md5 = CHECKSUMS[scenario].split(":")[1]
    expected_size = FILE_SIZES[scenario]

    log_entry = {
        "scenario": scenario,
        "url": url,
        "dest": dest,
        "expected_md5": expected_md5,
        "expected_size_bytes": expected_size,
    }

    # ── Resume support ───────────────────────────────────────────────
    existing_size = 0
    if os.path.exists(dest):
        existing_size = os.path.getsize(dest)
        if existing_size == expected_size:
            # Verify checksum of completed file
            print(f"[*] File already exists ({existing_size} bytes), verifying md5...")
            actual_md5 = _md5_file(dest)
            if actual_md5 == expected_md5:
                print(f"[+] {scenario}.zip already downloaded and verified.")
                log_entry.update({
                    "status": "already_complete",
                    "actual_md5": actual_md5,
                    "verified": True,
                    "download_seconds": 0,
                })
                return log_entry
            else:
                print(f"[!] Checksum mismatch, re-downloading from scratch.")
                existing_size = 0

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"[*] Resuming download from byte {existing_size:,} / {expected_size:,}")
    else:
        print(f"[*] Starting download: {scenario}.zip ({expected_size / 1e9:.2f} GB)")

    t0 = time.time()
    resp = requests.get(url, headers=headers, stream=True, timeout=60)

    if existing_size > 0 and resp.status_code == 206:
        mode = "ab"
    elif resp.status_code == 200:
        mode = "wb"
        existing_size = 0
    else:
        log_entry["status"] = f"http_error_{resp.status_code}"
        log_entry["download_seconds"] = time.time() - t0
        print(f"[!] HTTP {resp.status_code} for {url}")
        return log_entry

    downloaded = 0
    with open(dest, mode) as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                total = existing_size + downloaded
                pct = 100.0 * total / expected_size if expected_size else 0
                print(
                    f"\r    {total:,} / {expected_size:,} bytes ({pct:.1f}%)",
                    end="", flush=True,
                )
    print()

    elapsed = time.time() - t0
    final_size = os.path.getsize(dest)
    log_entry["downloaded_bytes"] = downloaded
    log_entry["final_size_bytes"] = final_size
    log_entry["download_seconds"] = round(elapsed, 2)

    # ── Checksum verification ────────────────────────────────────────
    print(f"[*] Verifying md5 checksum...")
    actual_md5 = _md5_file(dest)
    log_entry["actual_md5"] = actual_md5
    log_entry["verified"] = actual_md5 == expected_md5

    if actual_md5 == expected_md5:
        print(f"[+] {scenario}.zip verified OK ({elapsed:.1f}s, {downloaded / 1e6:.1f} MB)")
        log_entry["status"] = "success"
    else:
        print(f"[!] CHECKSUM MISMATCH: expected {expected_md5}, got {actual_md5}")
        log_entry["status"] = "checksum_mismatch"

    return log_entry


def _update_log(entry: dict, log_path: str) -> None:
    """Append or update download log JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    log = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            try:
                log = json.load(f)
            except json.JSONDecodeError:
                log = []

    entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log.append(entry)

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[*] Download log updated: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download AIT-LDS v2.0 scenario from Zenodo"
    )
    parser.add_argument(
        "--scenario", required=True, choices=ALL_SCENARIOS,
        help="Scenario name to download",
    )
    parser.add_argument(
        "--raw-dir", default=DEFAULT_RAW_DIR,
        help=f"Download directory (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--log-file", default=DOWNLOAD_LOG,
        help=f"Download log JSON path (default: {DOWNLOAD_LOG})",
    )
    args = parser.parse_args()

    entry = download_scenario(args.scenario, args.raw_dir)
    _update_log(entry, args.log_file)

    if entry.get("status") not in ("success", "already_complete"):
        sys.exit(1)


if __name__ == "__main__":
    main()
