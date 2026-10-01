"""
Unit & Integration Test Suite for feature/dashboard-risk-ui
Validates:
1. Entity-Level Risk-Based Alerting (RBA) accumulator logic & threshold crossing
2. MITRE ATT&CK taxonomy reuse from mitre_mapper.json
3. Layered Explanation component data contracts (Layers 1-4)
4. Conformal Credibility / Out-of-Distribution flag schema contract (Task 3)
5. Streamlit AppTest verification across modified dashboard views (05_Alerts, 03_Forecast, 04_Attack_Graph, 02_Overview)
"""

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "frontend"))
os.chdir(str(REPO_ROOT))

import json
from frontend.data_provider import (
    get_conformal_credibility,
    get_conformal_forecast,
    get_entity_risk_summary,
    get_mitre_data,
    get_forecast_trajectory,
    get_attributions,
    get_counterfactual,
)
from frontend.components.risk_accumulator import calculate_entity_risk


def test_rba_entity_risk_logic():
    print("[TEST 1] Testing Entity-Level Risk-Based Alerting (RBA) Calculation...")

    # Test Host-level aggregation
    rba_host = calculate_entity_risk(threshold=100.0, lookback_hours=24, group_by="host")
    assert rba_host["threshold"] == 100.0
    assert rba_host["group_by"] == "host"
    assert rba_host["total_tracked_entities"] >= 5
    assert rba_host["total_raw_events"] > 0
    assert rba_host["noise_reduction_pct"] > 50.0  # RBA drastically reduces fatigue!

    print(f"  Tracked entities: {rba_host['total_tracked_entities']}")
    print(f"  Alerting entities (>= 100 pts): {rba_host['alerting_count']}")
    print(f"  Suppressed sub-threshold entities: {rba_host['sub_threshold_count']}")
    print(f"  Raw events analyzed: {rba_host['total_raw_events']}")
    print(f"  Noise reduction: {rba_host['noise_reduction_pct']}%")

    # Verify stage labels match MITRE taxonomy
    mitre_data = get_mitre_data()
    valid_stages = {m["stage"] for m in mitre_data}
    valid_stages.update({"Reconnaissance", "Initial Access", "Credential Access", "Lateral Movement", "Impact", "Discovery", "Command and Control"})

    for ent in rba_host["all_entities"]:
        for stg in ent["stage_breakdown"].keys():
            assert stg in valid_stages, f"Unknown stage {stg} not in MITRE taxonomy!"

    # Test Subnet-level aggregation
    rba_subnet = calculate_entity_risk(threshold=120.0, lookback_hours=12, group_by="subnet")
    assert rba_subnet["group_by"] == "subnet"
    assert len(rba_subnet["all_entities"]) > 0
    print(f"  Subnet aggregation passed: {len(rba_subnet['all_entities'])} subnets tracked.")
    print("  [PASS] RBA Calculation & MITRE taxonomy validation successful.")


def test_conformal_credibility_contract():
    print("\n[TEST 2] Testing Conformal Credibility / OOD Flag Contract (Task 3)...")
    cred = get_conformal_credibility()

    # Validate schema contract required for feature/conformal-mimo integration
    assert "status" in cred, "Missing 'status' key in conformal credibility contract"
    assert "is_in_distribution" in cred, "Missing 'is_in_distribution' boolean in contract"
    assert "badge_label" in cred, "Missing 'badge_label' in contract"
    assert "is_stub" in cred, "Missing 'is_stub' marker in contract"
    assert isinstance(cred["is_in_distribution"], bool)

    print(f"  Status: {cred['status']}")
    print(f"  In-Distribution: {cred['is_in_distribution']}")
    print(f"  Badge Label: {cred['badge_label']}")
    print(f"  Is Stub (Awaiting feature/conformal-mimo): {cred['is_stub']}")
    print("  [PASS] Conformal credibility contract conforms to specification.")


def test_conformal_forecast_intervals():
    print("\n[TEST 3] Testing Split Conformal Forecast Intervals (Task 2)...")
    cf = get_conformal_forecast()
    assert "intervals" in cf
    assert "coverage" in cf
    intervals = cf["intervals"]
    assert len(intervals) >= 5, f"Expected at least 5 horizon intervals, got {len(intervals)}"
    for idx, (lb, ub) in enumerate(intervals):
        assert 0.0 <= lb <= ub <= 1.0, f"Invalid interval at step {idx}: [{lb}, {ub}]"
        print(f"  Horizon t+{idx+1}: [{lb:.3f}, {ub:.3f}] (Coverage: {int(cf['coverage']*100)}%)")
    print("  [PASS] Conformal forecast intervals verified.")


def test_layered_explanation_contracts():
    print("\n[TEST 4] Testing Layered Explanation Data Contracts (Task 2)...")
    fc = get_forecast_trajectory()
    attributions = get_attributions()
    cf = get_counterfactual()

    # Layer 1
    assert "risk" in fc and len(fc["risk"]) > 0
    assert "stage" in fc and len(fc["stage"]) > 0
    print(f"  Layer 1 - Target Stage: {fc['stage'][0]}, Lead time: {fc.get('lead_time', ['1m 00s'])[0]}")

    # Layer 2
    conf = get_conformal_forecast()
    assert len(conf.get("intervals", [])) > 0
    print(f"  Layer 2 - Calibrated Coverage: {int(conf.get('coverage', 0.90)*100)}%")

    # Layer 3
    assert len(attributions) > 0
    print(f"  Layer 3 - Top Driver: {attributions[0].get('feature')} (+{attributions[0].get('contribution', 0)*100:.0f}%)")

    # Layer 4
    assert "summary" in cf
    assert "honesty_label" in cf
    print(f"  Layer 4 - Counterfactual: {cf['summary'][:60]}...")
    print("  [PASS] Layered explanation contracts verified.")


def test_apptest_views():
    print("\n[TEST 5] Testing Streamlit AppTest Execution Across Views...")
    import ast as _ast
    from streamlit.testing.v1 import AppTest

    views_to_test = [
        ("05_Alerts.py", "frontend/views/05_Alerts.py"),
        ("03_Forecast.py", "frontend/views/03_Forecast.py"),
        ("04_Attack_Graph.py", "frontend/views/04_Attack_Graph.py"),
        ("02_Overview.py", "frontend/views/02_Overview.py"),
    ]

    # --- Syntax pre-validation pass ---
    # AppTest.exception only captures runtime exceptions; it silently passes when a
    # SyntaxError causes compilation to fail before script execution even begins.
    # We guard against that class of blind spot with an explicit ast.parse() check.
    print("  [SYNTAX CHECK] Validating parse-ability of all view files before AppTest...")
    for v_name, v_path in views_to_test:
        full_p = str(REPO_ROOT / v_path)
        try:
            with open(full_p, "r", encoding="utf-8") as _fh:
                _src = _fh.read()
            _ast.parse(_src)
            print(f"    [SYNTAX OK] {v_name}")
        except SyntaxError as _se:
            raise AssertionError(
                f"[SYNTAX ERROR] {v_name} failed ast.parse(): {_se}\n"
                f"  File: {full_p}\n"
                f"  Line {_se.lineno}: {_se.text}"
            ) from _se
    print("  [SYNTAX CHECK] All files passed syntax validation.")

    # --- AppTest execution pass ---
    for v_name, v_path in views_to_test:
        full_p = str(REPO_ROOT / v_path)
        print(f"  Running AppTest on {v_name}...")
        at = AppTest.from_file(full_p, default_timeout=60)
        at.run(timeout=60)
        if len(at.exception) > 0:
            for ex in at.exception:
                print(f"    [EXCEPTION in {v_name}] {ex.value}")
            raise AssertionError(f"AppTest failed on {v_name} with {len(at.exception)} exceptions")
        print(f"    [PASS] {v_name} passed with 0 exceptions.")

    print("  [PASS] All views rendered successfully with 0 exceptions!")


if __name__ == "__main__":
    print("=" * 80)
    print("STARTING TEST SUITE: feature/dashboard-risk-ui")
    print("=" * 80)

    test_rba_entity_risk_logic()
    test_conformal_credibility_contract()
    test_conformal_forecast_intervals()
    test_layered_explanation_contracts()
    test_apptest_views()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED SUCCESSFULLY! (100% PASS RATE)")
    print("=" * 80)
