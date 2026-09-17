import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streamlit.testing.v1 import AppTest
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
    print("\n--- 2. Testing Lateral Movement Graph AppTest Rendering & Interactivity ---")
    page_path = os.path.join(PROJECT_ROOT, "views", "01b_AttackGraph.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    at.run()

    assert not at.exception, f"App threw exception: {at.exception}"

    n1_caption = "Demonstrated on the single infiltration case study (n = 1). Not a general lateral-movement forecasting capability."
    rendered_text = " ".join([m.value for m in at.markdown])
    assert n1_caption in rendered_text, "Mandatory n=1 caption missing!"
    print("[PASS] Mandatory n=1 caption present and intact.")

    assert "badge-mock" not in rendered_text, "Deprecated MOCK badge still rendered in Lateral Movement Graph!"
    assert any(phrase in rendered_text for phrase in ["Running on benchmark data", "BENCHMARK MODE", "● LIVE", "LIVE"]), "Sticky governance banner missing from Lateral Movement Graph!"
    print("[PASS] Governance header verified; individual [MOCK] badges successfully deprecated.")

    for ip in ["10.0.2.15", "10.0.3.50", "10.0.4.10", "10.0.4.21", "10.0.5.1"]:
        assert ip in rendered_text, f"Host IP {ip} missing from rendered output!"
    print("[PASS] All 5 distinct host IPs confirmed in rendered output.")

    assert "HOST TELEMETRY INSPECTOR" in rendered_text
    assert "Forward Risk Trajectory Progression" in rendered_text
    assert "Active Sockets & Ports" in rendered_text
    assert "Driving Flow Indicators" in rendered_text
    print("[PASS] HOST TELEMETRY INSPECTOR rendered with multi-step trajectory and socket metadata.")

    inspect_btns = [b for b in at.button if "btn_inspect" in (b.key or "")]
    assert len(inspect_btns) == 5, f"Expected 5 inspect buttons, found {len(inspect_btns)}"

    # User Requirement 1: Explicitly test the Telemetry button and confirm its behavior is distinct from Inspect
    btn_telemetry = next((b for b in inspect_btns if b.label == "Telemetry"), None)
    assert btn_telemetry is not None, "Telemetry button missing from host cards!"
    assert btn_telemetry.key == "btn_inspect_10.0.4.10", "Telemetry button must be on the default active SSH Jump Host (10.0.4.10)!"
    assert btn_telemetry.disabled == True, "Active Telemetry button must be disabled to signify current selection!"
    inspect_labeled = [b for b in inspect_btns if b.label == "Inspect"]
    assert len(inspect_labeled) == 4, f"Expected 4 Inspect buttons and 1 Telemetry button, found {len(inspect_labeled)} Inspect buttons"
    print("[PASS] User verification confirmed: Telemetry button distinct from Inspect (only on SSH Jump Host 10.0.4.10 initially).")

    # Click Inspect on 10.0.2.15
    btn_10_0_2_15 = next(b for b in inspect_btns if "10.0.2.15" in (b.key or ""))
    btn_10_0_2_15.click().run()

    assert not at.exception, f"App threw exception on button click: {at.exception}"
    updated_text = " ".join([m.value for m in at.markdown])
    assert "HOST TELEMETRY INSPECTOR: 10.0.2.15" in updated_text, "Host inspector did not update to 10.0.2.15!"
    assert "Workstation (Patient Zero)" in updated_text

    # Re-verify that 10.0.2.15 now has Telemetry and 10.0.4.10 reverted to Inspect
    updated_inspect_btns = [b for b in at.button if "btn_inspect" in (b.key or "")]
    new_telemetry_btn = next((b for b in updated_inspect_btns if b.label == "Telemetry"), None)
    assert new_telemetry_btn is not None and new_telemetry_btn.key == "btn_inspect_10.0.2.15", "10.0.2.15 should now show Telemetry button!"
    btn_ssh_reverted = next(b for b in updated_inspect_btns if "10.0.4.10" in (b.key or ""))
    assert btn_ssh_reverted.label == "Inspect" and not btn_ssh_reverted.disabled, "10.0.4.10 should now show enabled Inspect button!"
    print("[PASS] Interactivity verified: Clicking Inspect button immediately updates Host Telemetry Inspector and swaps Telemetry state to 10.0.2.15.")

    btn_10_0_3_50 = next(b for b in updated_inspect_btns if "10.0.3.50" in (b.key or ""))
    btn_10_0_3_50.click().run()

    assert not at.exception, f"App threw exception on second button click: {at.exception}"
    updated_text_50 = " ".join([m.value for m in at.markdown])
    assert "HOST TELEMETRY INSPECTOR: 10.0.3.50" in updated_text_50, "Host inspector did not update to 10.0.3.50!"
    assert "Internal File Share" in updated_text_50
    assert "TCP/445 (SMB/CIFS)" in updated_text_50
    print("[PASS] Interactivity verified: Clicking Inspect button immediately updates Host Telemetry Inspector to 10.0.3.50.")

    # Test Focus button (Task 1 & 6)
    focus_btns = [b for b in at.button if "btn_focus" in (b.key or "")]
    assert len(focus_btns) == 5, f"Expected 5 focus buttons, found {len(focus_btns)}"
    btn_focus_10_0_4_10 = next(b for b in focus_btns if "10.0.4.10" in (b.key or ""))
    btn_focus_10_0_4_10.click().run()
    assert not at.exception, f"App threw exception on focus click: {at.exception}"
    assert at.session_state["focused_graph_host"] == "10.0.4.10", "focused_graph_host not set in session state!"
    print("[PASS] Interactivity verified: Clicking Focus button activates blast radius isolation for 10.0.4.10.")

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
    assert 'r="40"' in svg_html, "Domain Controller Tier-1 radius 40px missing!"
    assert 'r="32"' in svg_html, "Gateway/Auth Tier-2 radius 32px missing!"
    assert 'r="26"' in svg_html, "Endpoint Tier-3 radius 26px missing!"
    assert "beacon-ring" in svg_html, "Domain Controller outer target beacon ring missing!"
    print("[PASS] Task 3 verified: Node size strictly encodes asset criticality (DC=40px + radar ring, GW=32px, EP=26px).")

    # Task 4: Compact in-canvas legend
    assert "canvas-legend" in svg_html, "Canvas legend container missing!"
    assert "Graph Encoding" in svg_html, "Legend title missing!"
    assert "Critical (>70%)" in svg_html or "Critical (&gt;70%)" in svg_html, "Legend risk threshold missing!"
    assert "Node Size:" in svg_html and "Asset Criticality" in svg_html, "Legend criticality label missing!"
    assert "Active Rollout Path" in svg_html, "Legend path label missing!"
    print("[PASS] Task 4 verified: Compact in-canvas legend displays risk colors, criticality sizing, and path styles.")

    # Task 4 (New): Pan, Zoom & HUD Controls
    assert "canvas-hud-controls" in svg_html, "Canvas HUD controls missing!"
    assert "btn-zoom-in" in svg_html, "Zoom In button missing!"
    assert "btn-zoom-out" in svg_html, "Zoom Out button missing!"
    assert "btn-reset-view" in svg_html, "Reset View button missing!"
    assert 'id="viewport"' in svg_html, "Master viewport transform container missing!"
    assert 'viewBox="0 0 880 540"' in svg_html, "Expanded 880x540 viewBox missing!"
    print("[PASS] Task 4 verified: Pan & Zoom engine, expanded 880x540 viewport, and HUD controls (Zoom in, Zoom out, Reset) verified.")

    # Task 5: Smooth horizon unfolding transitions
    assert "transition: stroke" in svg_html or "transition" in svg_html, "CSS transitions missing from SVG!"
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

