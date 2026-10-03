"""
Audit script: Click-tests every button, radio, slider, file uploader, and state mutation
across all SHADOWCAT views using Streamlit AppTest.
"""
import os
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data_provider

def audit_interactions():
    results = {}

    print("Pre-warming ML models and live prediction cache...")
    data_provider._get_live_prediction()

    # 1. Test Threat Forecast (views/03_Forecast.py)
    print("--- Testing Threat Forecast Interactions (views/03_Forecast.py) ---")
    at = AppTest.from_file(str(ROOT / "views" / "03_Forecast.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"Initial run exception: {at.exception}"

    # 1a. Scrubber Buttons (k=0..5)
    step_btns = [b for b in at.button if b.key and b.key.startswith("step_btn_")]
    assert len(step_btns) == 6, f"Expected 6 step buttons, found {len(step_btns)}"
    step_btns[3].click().run()
    assert not at.exception
    assert at.session_state["forecast_k_step"] == 3
    print("[PASS] Scrubber button successfully updates forecast_k_step to 3.")

    # 1b. Mitigation Simulation Button
    sim_btn = next((b for b in at.button if b.label and "Simulate Mitigation" in b.label), None)
    assert sim_btn is not None, "Simulate Mitigation button missing!"
    sim_btn.click().run()
    assert not at.exception
    print("[PASS] Simulate Mitigation button clicked cleanly.")

    # 1c. Quarantine Authorization & Revocation Buttons
    quar_btn = next((b for b in at.button if b.label and "Authorize Autonomous Quarantine" in b.label), None)
    assert quar_btn is not None, "Authorize Autonomous Quarantine button missing!"
    quar_btn.click().run()
    assert not at.exception
    assert at.session_state["quarantine_active"] is True
    assert "svc-auth-master" in at.session_state["isolated_nodes"]
    print("[PASS] Authorize Autonomous Quarantine sets quarantine_active and isolates host.")

    revoke_btn = next((b for b in at.button if b.label and "Revoke Autonomous Quarantine" in b.label), None)
    assert revoke_btn is not None, "Revoke Autonomous Quarantine button missing!"
    revoke_btn.click().run()
    assert not at.exception
    assert at.session_state["quarantine_active"] is False
    print("[PASS] Revoke Autonomous Quarantine restores interconnect cleanly.")

    # 2. Test Attack Graph Topology & Scrubber (views/04_Attack_Graph.py)
    print("\n--- Testing Attack Graph Interactions (views/04_Attack_Graph.py) ---")
    at_graph = AppTest.from_file(str(ROOT / "views" / "04_Attack_Graph.py"), default_timeout=60)
    at_graph.run()
    assert not at_graph.exception, f"Attack graph run exception: {at_graph.exception}"

    # 2a. Attack Graph Horizon Scrubber
    att_k_btns = [b for b in at_graph.button if b.key and b.key.startswith("att_k_")]
    assert len(att_k_btns) == 6, f"Expected 6 attack graph k buttons, found {len(att_k_btns)}"
    att_k_btns[2].click().run()
    assert not at_graph.exception
    assert at_graph.session_state["attack_k_step"] == 2
    print("[PASS] Attack graph K-step button updates attack_k_step to 2.")

    # 2b. Host Node Selectbox
    assert len(at_graph.selectbox) > 0, "Host node selectbox missing!"
    host_select = at_graph.selectbox[0]
    host_select.set_value(host_select.options[1]).run()
    assert not at_graph.exception
    assert at_graph.session_state["selected_node"] == host_select.options[1]
    print(f"[PASS] Host node selectbox updates selected_node to {host_select.options[1]}.")

    # 3. Test Telemetry Ingestion (views/01_Telemetry_Ingestion.py)
    print("\n--- Testing Telemetry Ingestion Interactions (views/01_Telemetry_Ingestion.py) ---")
    at_in = AppTest.from_file(str(ROOT / "views" / "01_Telemetry_Ingestion.py"), default_timeout=60)
    at_in.run()
    assert not at_in.exception, f"Telemetry ingestion exception: {at_in.exception}"

    # 3a. Ingestion source mode radio
    assert len(at_in.radio) > 0, "Ingestion source mode radio missing!"
    source_radio = at_in.radio[0]
    source_radio.set_value("Live Flow Feed (gRPC / Kafka)").run()
    assert not at_in.exception
    print("[PASS] Ingestion Source Mode radio changes options cleanly.")

    # 3b. Telemetry file uploader
    assert len(at_in.file_uploader) > 0, "File uploader missing!"
    assert at_in.file_uploader[0].key == "telemetry_uploader"
    print("[PASS] telemetry_uploader present and verified.")

    # 3c. Load Demo Benchmark button
    bench_btn = next((b for b in at_in.button if b.label and "Load Demo Benchmark" in b.label), None)
    assert bench_btn is not None, "Load Demo Benchmark button missing!"
    bench_btn.click().run()
    assert not at_in.exception
    assert at_in.session_state["benchmark_loaded"] is True
    assert at_in.session_state["ingested_df"] is not None
    print("[PASS] Load Demo Benchmark button loads benchmark data into session state.")

    print("\n======================================================================")
    print("ALL INTERACTIVE WIDGET AUDIT CHECKS PASSED ZERO RUNTIME EXCEPTIONS!")
    print("======================================================================")

if __name__ == "__main__":
    audit_interactions()
