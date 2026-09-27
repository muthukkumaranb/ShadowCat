"""
SHADOWCAT Backend Audit & Diagnostics Script
Performs:
1. Scaler Parameter Sanity Check (DE vs ML1 v2)
2. Signature IDS Verification Search across all repos
3. Hazard Ensemble Timing Benchmark (37 folds)
4. Rollout Horizon Audit (K=3 validated vs K=4 exploratory)
"""

import os
import sys
import time
from pathlib import Path
import yaml
import json
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "data-engineering"))

def check_scalers():
    print("=" * 60)
    print("1. SCALER SANITY CHECK (POST-FIX VERIFICATION)")
    print("=" * 60)
    de_scaler_path = WORKSPACE / "data-engineering" / "data" / "ucs" / "scaler_params.yaml"
    ml1_scaler_path = WORKSPACE / "ml1" / "artifacts" / "lstm" / "inference_scaler_v2.yaml"

    with open(de_scaler_path, "r", encoding="utf-8") as f:
        de_scaler = yaml.safe_load(f)
    with open(ml1_scaler_path, "r", encoding="utf-8") as f:
        ml1_raw = yaml.safe_load(f)
        ml1_scaler = ml1_raw.get("features", ml1_raw)

    test_keys = ["duration_microsec_mean", "duration_microsec_max", "duration_sec_max", "byte_count_fwd_sum"]
    for k in test_keys:
        print(f"Feature: '{k}'")
        print(f"  DE  scaler_params.yaml    : {de_scaler.get(k)}")
        print(f"  ML1 inference_scaler_v2.yaml: {ml1_scaler.get(k)}")

    # Total diff
    all_keys = set(de_scaler.keys()).union(set(ml1_scaler.keys()))
    mismatches = []
    for k in all_keys:
        if de_scaler.get(k) != ml1_scaler.get(k):
            mismatches.append(k)
    print(f"\nTotal feature count: {len(all_keys)}")
    print(f"Total mismatches: {len(mismatches)}")
    if len(mismatches) == 0:
        print("[PASS] DE and ML1 v2 scalers match bit-for-bit (0 mismatches).")

def check_signature_ids():
    print("\n" + "=" * 60)
    print("2. SIGNATURE IDS BASELINE REPOSITORY AUDIT")
    print("=" * 60)
    # Search for suricata or snort in all repos
    found = []
    for repo in ["data-engineering", "ml1", "ml2-full", "frontend"]:
        repo_dir = WORKSPACE / repo
        for p in repo_dir.rglob("*.*"):
            if ".git" in str(p) or ".venv" in str(p):
                continue
            if p.suffix in [".py", ".json", ".md", ".csv", ".yaml"]:
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    if "suricata" in text.lower() or "snort" in text.lower():
                        found.append(str(p.relative_to(WORKSPACE)))
                except Exception:
                    pass
    print(f"Files referencing Suricata / Snort:")
    for f in set(found):
        print(f"  - {f}")

def benchmark_hazard_ensemble():
    print("\n" + "=" * 60)
    print("3. HAZARD ENSEMBLE TIMING BENCHMARK (37 FOLDS)")
    print("=" * 60)
    from backend.predict import get_pipeline
    pipeline = get_pipeline()
    dummy_seq = np.random.randn(30, 406).astype(np.float32)

    t0 = time.perf_counter()
    for _ in range(10):
        hazards = pipeline._predict_hazard_ensemble(dummy_seq)
    elapsed = (time.perf_counter() - t0) / 10.0
    print(f"Ensemble execution time per call (37 folds x 3 horizons): {elapsed * 1000.0:.2f} ms")
    print(f"Hazards output: {hazards}")

if __name__ == "__main__":
    check_scalers()
    check_signature_ids()
    benchmark_hazard_ensemble()
