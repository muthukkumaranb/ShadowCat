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

def audit_interactions():
    results = {}

    # 1. Test Threat Forecast (views/01_Forecast.py)
    print("--- Testing Threat Forecast Interactions ---")
    at = AppTest.from_file(str(ROOT / "views" / "01_Forecast.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Initial run exception: {at.exception}"

    # 1a. Horizon Radio Selector
    radio = at.radio(key="horizon_radio_selector")
    assert radio is not None, "horizon_radio_selector missing!"
    radio.set_value(radio.options[0]).run()
    assert not at.exception
    assert at.session_state["selected_horizon_idx"] == 0, "Radio selection did not update selected_horizon_idx!"
    radio.set_value(radio.options[3]).run()
    assert not at.exception
    assert at.session_state["selected_horizon_idx"] == 3
    print("[PASS] horizon_radio_selector successfully updates selected_horizon_idx across options.")

    # 1b. Test page_link to Lateral Movement Graph
    rendered_fc = " ".join([m.value for m in at.markdown])
    assert "Dynamic Enterprise Attack Graph & Lateral Rollout" in rendered_fc
    print("[PASS] Threat Forecast preview card and navigation bridge rendered cleanly.")

    # 2. Test Lateral Movement Graph (views/01b_AttackGraph.py)
    print("\n--- Testing Lateral Movement Graph Interactions ---")
    at_ag = AppTest.from_file(str(ROOT / "views" / "01b_AttackGraph.py"), default_timeout=30)
    at_ag.run()
    assert not at_ag.exception, f"Initial run exception: {at_ag.exception}"

    # 2a. Attack Graph Horizon Slider
    slider = at_ag.slider(key="attack_graph_k_slider")
    assert slider is not None, "attack_graph_k_slider missing!"
    slider.set_value(0).run()
    assert not at_ag.exception
    assert at_ag.session_state["attack_graph_k"] == 0
    slider.set_value(4).run()
    assert not at_ag.exception
    assert at_ag.session_state["attack_graph_k"] == 4
    print("[PASS] attack_graph_k_slider successfully updates attack_graph_k.")

    # 2b. Inspect and Focus Buttons
    inspect_btns = [b for b in at_ag.button if "btn_inspect" in b.key]
    focus_btns = [b for b in at_ag.button if "btn_focus" in b.key]
    assert len(inspect_btns) == 5, f"Expected 5 inspect buttons, found {len(inspect_btns)}"
    assert len(focus_btns) == 5, f"Expected 5 focus buttons, found {len(focus_btns)}"

    # Test clicking inspect on 10.0.5.1 (Domain Controller)
    btn_dc = next(b for b in inspect_btns if "10.0.5.1" in b.key)
    btn_dc.click().run()
    assert not at_ag.exception
    assert at_ag.session_state["selected_graph_host"] == "10.0.5.1"
    assert "HOST TELEMETRY INSPECTOR: 10.0.5.1" in " ".join([m.value for m in at_ag.markdown])
    print("[PASS] btn_inspect correctly focuses host 10.0.5.1 and updates Telemetry Inspector.")

    # Test clicking focus on 10.0.4.10 (SSH Jump Host)
    btn_foc = next(b for b in focus_btns if "10.0.4.10" in b.key)
    btn_foc.click().run()
    assert not at_ag.exception
    assert at_ag.session_state["focused_graph_host"] == "10.0.4.10"
    print("[PASS] btn_focus correctly toggles blast radius isolation.")

    # Test unfocusing
    btn_foc_again = next(b for b in at_ag.button if "btn_focus_10.0.4.10" in b.key)
    btn_foc_again.click().run()
    assert not at_ag.exception
    assert at_ag.session_state["focused_graph_host"] is None
    print("[PASS] btn_focus correctly unfocuses when clicked a second time.")

    # 2. Test Telemetry Ingestion (views/01a_Input.py)
    print("\n--- Testing Telemetry Ingestion Interactions ---")
    at_in = AppTest.from_file(str(ROOT / "views" / "01a_Input.py"), default_timeout=30)
    at_in.run()
    assert not at_in.exception

    # 2a. Benchmark selector radio
    bench_radio = at_in.radio(key="benchmark_selector")
    assert bench_radio is not None
    bench_radio.set_value(bench_radio.options[1]).run()
    assert not at_in.exception
    print("[PASS] benchmark_selector radio changes options cleanly.")

    # 2b. Custom pipeline mode radio
    mode_radio = at_in.radio(key="custom_pipeline_mode")
    assert mode_radio is not None
    # Toggle to CSV mode
    mode_radio.set_value("Flow Records CSV (.csv, NetFlow / IPFIX)").run()
    assert not at_in.exception
    assert len(at_in.file_uploader) > 0
    assert at_in.file_uploader[0].key == "csv_uploader"
    print("[PASS] custom_pipeline_mode radio switches uploaders cleanly (PCAP <-> CSV).")

    # Toggle back to PCAP mode
    mode_radio.set_value("Raw Packet Capture (.pcap, .pcapng)").run()
    assert not at_in.exception
    assert at_in.file_uploader[0].key == "pcap_uploader"
    print("[PASS] pcap_uploader present in PCAP mode.")

    print("\n======================================================================")
    print("ALL INTERACTIVE WIDGET AUDIT CHECKS PASSED ZERO RUNTIME EXCEPTIONS!")
    print("======================================================================")

if __name__ == "__main__":
    audit_interactions()
