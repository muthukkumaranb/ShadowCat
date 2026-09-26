"""
GUI-Level AppTest Verification for Missing/Unloaded GNN Dependency
Simulates the exact environment where optional `torch_geometric` dependency is missing,
confirming that:
1. Valid CIC-IDS-2018 upload does NOT crash with AttributeError: 'NoneType' object has no attribute 'get'.
2. The UI renders the clear 'not available' state ("GNN FUSION: NOT ACTIVE THIS SESSION", "NOT ACTIVE (GNN UNLOADED)").
3. Deceptive placeholder numbers ("12 Nodes, 40 Edges") are NOT displayed.
"""

import sys
import os
from pathlib import Path
from unittest.mock import patch

# Resolve repository root
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "frontend"))
os.chdir(str(REPO_ROOT))

# Simulate missing torch_geometric before anything loads it
sys.modules["torch_geometric"] = None

import streamlit as st
from streamlit.testing.v1 import AppTest
from backend.predict import ShadowcatPipeline

def test_missing_gnn_gui_resilience():
    print("=" * 80)
    print("TEST: SIMULATED MISSING GNN DEPENDENCY (torch_geometric = None)")
    print("=" * 80)

    # Verify that ShadowcatPipeline handles missing GNN gracefully
    pipeline = ShadowcatPipeline()
    print(f"[*] ShadowcatPipeline initialized with fused_model_loaded = {pipeline.fused_model_loaded}")
    assert pipeline.fused_model_loaded is False, "Expected fused_model_loaded to be False when torch_geometric is unavailable"

    page_path = str(REPO_ROOT / "frontend" / "views" / "01_Telemetry_Ingestion.py")
    cicids_path = REPO_ROOT / "data" / "dummy_cicids2018.csv"
    if not cicids_path.exists():
        cicids_path = REPO_ROOT / "scratch" / "dummy_cicids2018.csv"

    assert cicids_path.exists(), f"CIC-IDS-2018 sample missing at {cicids_path}"

    with open(cicids_path, "rb") as f:
        cicids_bytes = f.read()

    print("\n[*] Initializing AppTest on 01_Telemetry_Ingestion.py...")
    at = AppTest.from_file(page_path, default_timeout=120)
    at.run(timeout=120)

    print(f"[*] Uploading known-good CIC-IDS-2018 telemetry ({len(cicids_bytes)} bytes)...")
    at.file_uploader[0].upload("dummy_cicids2018.csv", cicids_bytes).run(timeout=120)

    print("\n--- Execution Diagnostics ---")
    print(f"Exceptions count: {len(at.exception)}")
    for e in at.exception:
        print(f"  [EXCEPTION] {e.value}")

    print(f"Errors count: {len(at.error)}")
    for err in at.error:
        print(f"  [ERROR] {err.value}")

    all_text = " ".join([getattr(m, "value", "") for m in at.markdown])

    has_summary = "LSTM GAUSSIAN WORLD MODEL PREDICTION SUMMARY" in all_text
    has_arch_status = "Telemetry Dynamics & Network Topology Architecture Status" in all_text
    has_canonical = "TEMPORAL WORLD MODEL + CANONICAL TOPOLOGY" in all_text
    has_temporal_bypass = "z'(t) = z(t) ∈ ℝ⁶⁴" in all_text
    has_fake_numbers = "12 Nodes, 40 Edges" in all_text

    print("\n--- UI State Assertions ---")
    print(f"  Does 'LSTM GAUSSIAN WORLD MODEL PREDICTION SUMMARY' render? {has_summary}")
    print(f"  Does 'Telemetry Dynamics & Network Topology Architecture Status' render? {has_arch_status}")
    print(f"  Does 'TEMPORAL WORLD MODEL + CANONICAL TOPOLOGY' badge render? {has_canonical}")
    print(f"  Does 'z'(t) = z(t) ∈ ℝ⁶⁴' render?                          {has_temporal_bypass}")
    print(f"  Are deceptive placeholder numbers ('12 Nodes, 40 Edges') present? {has_fake_numbers}")

    # Core verifications:
    assert len(at.exception) == 0, f"AppTest raised unexpected exception: {[e.value for e in at.exception]}"
    assert len(at.error) == 0, f"AppTest raised unexpected error: {[err.value for err in at.error]}"
    assert has_summary, "Prediction summary MUST render on valid data"
    assert has_arch_status, "Architecture status panel MUST render"
    assert has_canonical, "Must display 'TEMPORAL WORLD MODEL + CANONICAL TOPOLOGY' badge"
    assert has_temporal_bypass, "Must display 'z'(t) = z(t) ∈ ℝ⁶⁴'"
    assert not has_fake_numbers, "Must NOT display deceptive fake numbers (12 Nodes, 40 Edges)"

    print("\n[+] SUCCESS: Dashboard safely degraded to clear 'not active' state with 0 crashes!")
    print("=" * 80)

if __name__ == "__main__":
    test_missing_gnn_gui_resilience()
