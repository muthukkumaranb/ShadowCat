"""
GUI-Level Ingestion Verification via Streamlit AppTest
Verifies that uploading real CTU-13 telemetry triggers the blocking error and halts page execution via st.stop(),
while a known-good CIC-IDS-2018 telemetry file passes validation and renders the full predictive dashboard.
"""

import sys
import os
from pathlib import Path

# Resolve repository root and frontend paths
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "frontend"))
os.chdir(str(REPO_ROOT))

import streamlit as st
from streamlit.testing.v1 import AppTest

def verify_gui_ingestion():
    print("=" * 80)
    print("STREAMLIT GUI-LEVEL TELEMETRY INGESTION VERIFICATION")
    print(f"Streamlit Version: {st.__version__}")
    print("Testing Framework: streamlit.testing.v1.AppTest")
    print("=" * 80)

    page_path = str(REPO_ROOT / "frontend" / "views" / "01_Telemetry_Ingestion.py")
    ctu13_path = REPO_ROOT / "scratch" / "CTU13_Attack_Traffic.csv"
    if not ctu13_path.exists():
        ctu13_path = REPO_ROOT / "data" / "ctu13_sample.csv"
    cicids_path = REPO_ROOT / "data" / "dummy_cicids2018.csv"
    if not cicids_path.exists():
        cicids_path = REPO_ROOT / "scratch" / "dummy_cicids2018.csv"

    assert ctu13_path.exists(), f"CTU-13 dataset missing at {ctu13_path}"
    assert cicids_path.exists(), f"CIC-IDS-2018 dataset missing at {cicids_path}"

    with open(ctu13_path, "rb") as f:
        ctu13_bytes = f.read()

    with open(cicids_path, "rb") as f:
        cicids_bytes = f.read()

    # -------------------------------------------------------------------------
    # PART 1: Real CTU-13 Telemetry Upload Verification
    # -------------------------------------------------------------------------
    print("\n[PART 1] Running AppTest on 01_Telemetry_Ingestion.py with Real CTU-13 CSV...")
    at_ctu13 = AppTest.from_file(page_path, default_timeout=120)
    at_ctu13.run(timeout=120)

    print(f"[*] Uploading {ctu13_path.name} ({len(ctu13_bytes):,} bytes) via file_uploader widget...")
    at_ctu13.file_uploader[0].upload("CTU13_Attack_Traffic.csv", ctu13_bytes).run(timeout=120)

    print("\n--- CTU-13 Execution Results ---")
    print(f"Exceptions count: {len(at_ctu13.exception)}")
    for e in at_ctu13.exception:
        print(f"  [EXCEPTION] {e.value}")

    print(f"Errors count: {len(at_ctu13.error)}")
    for err in at_ctu13.error:
        print(f"  [ERROR] {err.value}")

    print(f"Warnings count: {len(at_ctu13.warning)}")
    for w in at_ctu13.warning:
        print(f"  [WARNING] {w.value}")

    print(f"Successes count: {len(at_ctu13.success)}")
    for s in at_ctu13.success:
        print(f"  [SUCCESS] {s.value}")

    # Check halting behavior (st.stop())
    ctu13_all_text = " ".join([getattr(m, "value", "") for m in at_ctu13.markdown])
    has_summary_ctu13 = "LSTM GAUSSIAN WORLD MODEL PREDICTION SUMMARY" in ctu13_all_text
    has_fusion_ctu13 = "Telemetry Dynamics & Network Topology Architecture Status" in ctu13_all_text
    has_terminal_ctu13 = "Ingestion Event & System Log Stream" in ctu13_all_text

    print("\n--- CTU-13 Halting & Non-Rendering Verification ---")
    print(f"  Does 'LSTM GAUSSIAN WORLD MODEL PREDICTION SUMMARY' render? {has_summary_ctu13}")
    print(f"  Does 'Telemetry Dynamics & Network Topology Architecture Status' render?   {has_fusion_ctu13}")
    print(f"  Does 'Ingestion Event & System Log Stream' render?         {has_terminal_ctu13}")

    assert len(at_ctu13.error) == 1, f"Expected exactly 1 error, got {len(at_ctu13.error)}"
    assert "INGESTION BLOCKED: SCHEMA MISMATCH" in at_ctu13.error[0].value, "Error text missing block message"
    assert not has_summary_ctu13, "World model prediction summary should NOT render on CTU-13"
    assert not has_fusion_ctu13, "Architecture status should NOT render on CTU-13"
    assert not has_terminal_ctu13, "Terminal log stream should NOT render on CTU-13"
    print("[+] CTU-13 Rejection & GUI Halting Confirmed!")

    # -------------------------------------------------------------------------
    # PART 2: Known-Good CIC-IDS-2018 Telemetry Upload Verification
    # -------------------------------------------------------------------------
    print("\n[PART 2] Running AppTest on 01_Telemetry_Ingestion.py with Known-Good CIC-IDS-2018 CSV...")
    at_cicids = AppTest.from_file(page_path, default_timeout=120)
    at_cicids.run(timeout=120)

    print(f"[*] Uploading {cicids_path.name} ({len(cicids_bytes):,} bytes) via file_uploader widget...")
    at_cicids.file_uploader[0].upload("dummy_cicids2018.csv", cicids_bytes).run(timeout=120)

    print("\n--- CIC-IDS-2018 Execution Results ---")
    print(f"Exceptions count: {len(at_cicids.exception)}")
    for e in at_cicids.exception:
        print(f"  [EXCEPTION] {e.value}")

    print(f"Errors count: {len(at_cicids.error)}")
    for err in at_cicids.error:
        print(f"  [ERROR] {err.value}")

    print(f"Warnings count: {len(at_cicids.warning)}")
    for w in at_cicids.warning:
        print(f"  [WARNING] {w.value}")

    print(f"Successes count: {len(at_cicids.success)}")
    for s in at_cicids.success:
        print(f"  [SUCCESS] {s.value}")

    # Check dashboard rendering
    cicids_all_text = " ".join([getattr(m, "value", "") for m in at_cicids.markdown])
    has_summary_cicids = "LSTM GAUSSIAN WORLD MODEL PREDICTION SUMMARY" in cicids_all_text
    has_fusion_cicids = "Telemetry Dynamics & Network Topology Architecture Status" in cicids_all_text
    has_terminal_cicids = "Ingestion Event & System Log Stream" in cicids_all_text

    print("\n--- CIC-IDS-2018 Dashboard Rendering Verification ---")
    print(f"  Does 'LSTM GAUSSIAN WORLD MODEL PREDICTION SUMMARY' render? {has_summary_cicids}")
    print(f"  Does 'Telemetry Dynamics & Network Topology Architecture Status' render?   {has_fusion_cicids}")
    print(f"  Does 'Ingestion Event & System Log Stream' render?         {has_terminal_cicids}")

    assert len(at_cicids.error) == 0, f"Expected 0 errors on valid dataset, got {len(at_cicids.error)}"
    assert has_summary_cicids, "World model prediction summary MUST render on valid CIC-IDS-2018"
    assert has_fusion_cicids, "Architecture status MUST render on valid CIC-IDS-2018"
    assert has_terminal_cicids, "Terminal log stream MUST render on valid CIC-IDS-2018"
    print("[+] CIC-IDS-2018 Normal Dashboard Rendering Confirmed (No Regression)!")

    print("\n" + "=" * 80)
    print("ALL GUI-LEVEL INGESTION VERIFICATIONS PASSED")
    print("=" * 80)

if __name__ == "__main__":
    verify_gui_ingestion()
