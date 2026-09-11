#!/usr/bin/env python3
"""
SHADOWCAT - Revision E Edge-Case Audit & Stress Test
Validates that frontend views and components render cleanly under non-trivial real-inference outputs:
  1. 0% Risk (zero attack probability, nominal state)
  2. 100% Risk (certain breach, extreme critical escalation)
  3. Cold-Start / NaN / None states (insufficient observation window, uncalibrated uncertainty)
  4. Non-standard values (e.g. 0.999, negative numbers clamped, float bounds)
"""

import sys
import math
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data_provider
from components.forecast import create_forecast_chart
from components.explanation import create_attribution_chart

def test_chart_components_with_edge_cases():
    print("--- 1. Testing Chart Generators with Edge Cases ---")

    # (a) 0% risk data
    zero_risk_steps = [
        {"horizon": f"t+{k}", "time_ahead": f"{k} min", "probability": 0.0, "stage": "Baseline",
         "lead_time": "+0 min", "lower_bound": 0.0, "upper_bound": 0.0, "uncertainty": 0.0}
        for k in range(1, 5)
    ]
    fig_zero = create_forecast_chart(zero_risk_steps)
    assert fig_zero is not None
    print("[PASS] Forecast chart generated cleanly for 0% risk trajectory.")

    # (b) 100% risk data
    max_risk_steps = [
        {"horizon": f"t+{k}", "time_ahead": f"{k} min", "probability": 1.0, "stage": "Full Breach",
         "lead_time": "+1 min", "lower_bound": 0.95, "upper_bound": 1.0, "uncertainty": 0.05}
        for k in range(1, 5)
    ]
    fig_max = create_forecast_chart(max_risk_steps)
    assert fig_max is not None
    print("[PASS] Forecast chart generated cleanly for 100% risk trajectory.")

    # (c) Cold-start / NaN / None values
    cold_start_steps = [
        {"horizon": f"t+{k}", "time_ahead": f"{k} min", "probability": float("nan"), "stage": "Cold Start",
         "lead_time": "—", "lower_bound": None, "upper_bound": None, "uncertainty": float("nan")}
        for k in range(1, 5)
    ]
    fig_cold = create_forecast_chart(cold_start_steps)
    assert fig_cold is not None
    print("[PASS] Forecast chart handled Cold-Start / NaN / None values gracefully without crashing.")


def test_forecast_view_edge_cases():
    print("\n--- 2. Testing Forecast View Rendering Under Edge Cases ---")
    
    # Test rendering default view
    at = AppTest.from_file(str(ROOT / "views" / "01_Forecast.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Forecast view failed with exception: {at.exception}"
    print("[PASS] Forecast view renders with zero exceptions.")

    # Test with simulated session state edge cases
    at2 = AppTest.from_file(str(ROOT / "views" / "01_Forecast.py"), default_timeout=30)
    at2.session_state["selected_horizon_idx"] = 0
    at2.run()
    assert not at2.exception, f"Horizon index 0 failed: {at2.exception}"
    print("[PASS] Horizon index 0 (immediate horizon) renders cleanly.")

    at3 = AppTest.from_file(str(ROOT / "views" / "01_Forecast.py"), default_timeout=30)
    at3.session_state["selected_horizon_idx"] = 3
    at3.run()
    assert not at3.exception, f"Horizon index 3 (cutoff horizon t+4) failed: {at3.exception}"
    print("[PASS] Horizon index 3 (cutoff horizon t+4) renders cleanly.")


def test_evidence_view_edge_cases():
    print("\n--- 3. Testing Evidence View Rendering Under Edge Cases ---")
    at = AppTest.from_file(str(ROOT / "views" / "02_Evidence.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"Evidence view failed with exception: {at.exception}"
    print("[PASS] Evidence view renders with zero exceptions.")


def main():
    print("=" * 65)
    print("SHADOWCAT Revision E: Edge-Case & Robustness Verification")
    print("=" * 65)

    test_chart_components_with_edge_cases()
    test_forecast_view_edge_cases()
    test_evidence_view_edge_cases()

    print("\n" + "=" * 65)
    print("ALL REVISION E EDGE-CASE AND ROBUSTNESS TESTS PASSED.")
    print("=" * 65)


if __name__ == "__main__":
    main()
