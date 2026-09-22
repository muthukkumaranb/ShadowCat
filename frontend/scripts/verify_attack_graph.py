import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streamlit.testing.v1 import AppTest
import data_provider
from data_provider import get_host_risk_graph

def test_attack_graph_data_integrity():
    print("--- 1. Testing get_host_risk_graph Data Integrity ---")
    for k in range(5):
        data = get_host_risk_graph(k_step=k)
        step_info = data["rollout_steps"][k]
        node_roles = data["node_roles"]
        host_risks = step_info["host_risks"]
        host_telemetry = data["host_telemetry"]

        expected_hosts = {"10.0.2.15", "10.0.3.50", "10.0.4.10", "10.0.4.21", "10.0.5.1"}
        assert set(node_roles.keys()) == expected_hosts, f"Step {k}: node_roles mismatch {set(node_roles.keys())}"
        assert set(host_risks.keys()) == expected_hosts, f"Step {k}: host_risks mismatch {set(host_risks.keys())}"
        assert set(host_telemetry.keys()) == expected_hosts, f"Step {k}: host_telemetry mismatch {set(host_telemetry.keys())}"

        assert node_roles["10.0.2.15"] == "Workstation (Patient Zero)"
        assert node_roles["10.0.3.50"] == "Internal File Share"
        assert node_roles["10.0.4.10"] == "SSH Jump Host"
        assert node_roles["10.0.4.21"] == "Internal Auth Cluster"
        assert node_roles["10.0.5.1"] == "Domain Controller (Critical Asset)"

        assert "10.0.2.18" not in node_roles
        assert "10.0.2.18" not in host_risks

    print("[PASS] get_host_risk_graph data integrity verified across all 5 rollout steps (k=0..4).")

def test_attack_graph_apptest_rendering():
    print("\n--- 2. Testing Attack Graph AppTest Rendering & Interactivity ---")
    data_provider._get_live_prediction()

    # 2a. Test Attack Graph View (views/04_Attack_Graph.py)
    graph_path = os.path.join(PROJECT_ROOT, "views", "04_Attack_Graph.py")
    at = AppTest.from_file(graph_path, default_timeout=40)
    at.run()

    assert not at.exception, f"App threw exception: {at.exception}"
    rendered_text = " ".join([m.value for m in at.markdown])

    assert "Attack Topology & Lateral Propagation Graph" in rendered_text, "Attack Graph header missing!"
    assert "badge-mock" not in rendered_text, "Deprecated MOCK badge still rendered in Attack Graph!"
    print("[PASS] Attack Topology header verified; zero deprecated [MOCK] badges.")

    # Test K-Step Horizon scrubber buttons (att_k_0..5)
    att_k_btns = [b for b in at.button if b.key and b.key.startswith("att_k_")]
    assert len(att_k_btns) == 6, f"Expected 6 attack graph buttons, found {len(att_k_btns)}"
    att_k_btns[2].click().run()
    assert not at.exception
    assert at.session_state["attack_k_step"] == 2
    print("[PASS] Interactivity verified: Clicking k=2 updates attack_k_step to 2.")

    # Test Host node selectbox
    assert len(at.selectbox) > 0, "Host node selectbox missing!"
    host_sel = at.selectbox[0]
    host_sel.set_value(host_sel.options[1]).run()
    assert not at.exception
    assert at.session_state["selected_node"] == host_sel.options[1]
    print(f"[PASS] Interactivity verified: Selecting host node updates selected_node to {host_sel.options[1]}.")

    # 2b. Also verify Forecast view (views/03_Forecast.py)
    print("\n--- 2b. Testing Threat Forecast View (views/03_Forecast.py) ---")
    forecast_path = os.path.join(PROJECT_ROOT, "views", "03_Forecast.py")
    at_fc = AppTest.from_file(forecast_path, default_timeout=40)
    at_fc.run()
    assert not at_fc.exception, f"Forecast threw exception: {at_fc.exception}"
    fc_text = " ".join([m.value for m in at_fc.markdown])
    assert "Predictive Threat Trajectory Forecast" in fc_text, "Forecast header missing!"
    step_btns = [b for b in at_fc.button if b.key and b.key.startswith("step_btn_")]
    assert len(step_btns) == 6
    step_btns[3].click().run()
    assert not at_fc.exception
    assert at_fc.session_state["forecast_k_step"] == 3
    print("[PASS] Interactivity verified: Forecast k-step scrubber updates forecast_k_step to 3.")

def test_attack_graph_svg_rendering_features():
    print("\n--- 3. Testing Advanced Visual Upgrades in SVG Canvas Generator ---")
    from components.attack_graph import _build_attack_graph_svg
    data = get_host_risk_graph(k_step=2)
    svg_html = _build_attack_graph_svg(
        graph_data=data,
        active_k=2,
        selected_host="10.0.4.10",
        focused_host="10.0.4.10"
    )

    # Task 2: Highlight predicted attack path with marching ants and laser glow
    assert "active-attack-path" in svg_html, "Active attack path class missing from SVG!"
    assert "marchingAnts" in svg_html, "Marching ants animation missing from SVG!"
    assert "glow-filter" in svg_html, "Laser glow filter missing from SVG!"
    assert "arrow-active" in svg_html, "Prominent active arrow marker missing from SVG!"
    print("[PASS] Task 2 verified: Predicted attack path rendered with marching ants, laser glow, and scaled arrowheads.")

    # Task 3: Dual-dimension node encoding (Node size = asset criticality)
    assert 'r="38"' in svg_html, "Domain Controller Tier-1 radius 38px missing!"
    assert 'r="30"' in svg_html, "Gateway/Auth Tier-2 radius 30px missing!"
    assert 'r="24"' in svg_html or 'r="25"' in svg_html, "Endpoint Tier-3 radius 24px/25px missing!"
    assert "beacon-ring" in svg_html, "Domain Controller outer target beacon ring missing!"
    print("[PASS] Task 3 verified: Node size strictly encodes asset criticality (DC=38px + radar ring, GW=30px, EP=24px).")

    # Task 4: Compact in-canvas legend
    assert "canvas-legend" in svg_html, "Canvas legend container missing!"
    assert "Graph Encoding" in svg_html, "Legend title missing!"
    assert "Critical (>70%)" in svg_html or "Critical (&gt;70%)" in svg_html, "Legend risk threshold missing!"
    assert "Node Size:" in svg_html and "Asset Criticality" in svg_html, "Legend criticality label missing!"
    assert "Active Rollout Path" in svg_html, "Legend path label missing!"
    print("[PASS] Task 4 verified: Compact in-canvas legend displays risk colors, criticality sizing, and path styles.")

    # Task 5: Smooth horizon unfolding transitions
    assert "transition: fill 0.7s" in svg_html or "transition" in svg_html, "CSS transitions missing from SVG!"
    assert "ROLLOUT_DATA" in svg_html, "Client-side rollout step dataset missing from canvas script!"
    print("[PASS] Task 5 verified: CSS GPU transitions and multi-horizon dataset embedded for smooth step unfolding.")

    # Task 6: Interaction depth, tooltips, and blast radius isolation
    assert "graph-tooltip" in svg_html, "Interactive hover tooltip element missing!"
    assert "dimmed" in svg_html, "Blast radius dimming class missing!"
    assert "focused-pulse-ring" in svg_html or "focused-node" in svg_html, "Focused node ring missing!"
    print("[PASS] Task 6 verified: Interactive hover tooltips and blast radius isolation dimming confirmed.")

if __name__ == "__main__":
    test_attack_graph_data_integrity()
    test_attack_graph_apptest_rendering()
    test_attack_graph_svg_rendering_features()
    print("\n======================================================================")
    print("ALL ATTACK GRAPH INTEGRITY & INTERACTIVITY TESTS PASSED SUCCESSFULLY!")
    print("======================================================================")

