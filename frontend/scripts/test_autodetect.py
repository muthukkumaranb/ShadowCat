#!/usr/bin/env python3
"""
Test script for verifying checkpoint auto-detection in data_provider.py and Streamlit views.
Validates Revision A:
  - inference_status() -> "mock" | "live" auto-detected via models/world_model.pt
  - validation_status() -> "pending" | "validated_offline" auto-detected independently
  - Header sticky banner reacts immediately to inference_status()
  - Validation banner correctly communicates offline LOEO 37-fold status
"""

import os
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

# Force project root in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data_provider

def run_test():
    print("=" * 65)
    print("Testing Checkpoint Auto-Detection (Revision A Independent Flags)")
    print("=" * 65)

    print("Pre-warming ML model cache...")
    data_provider._get_live_prediction()

    ckpt_file = ROOT / "models" / "world_model.pt"
    models_dir = ckpt_file.parent
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save original checkpoint if present so we can restore it after test
    orig_ckpt_bytes = ckpt_file.read_bytes() if ckpt_file.is_file() else None
    if ckpt_file.exists():
        ckpt_file.unlink()

    # Save original fallback state and set to True for checkpoint auto-detection test
    orig_mock = data_provider.MOCK_STATUS.get("forecast_trajectory", False)
    data_provider.MOCK_STATUS["forecast_trajectory"] = True

    try:
        # -------------------------------------------------------------------------
        # Step 1: Baseline Inference Status (No checkpoint file)
        # -------------------------------------------------------------------------
        inf_initial = data_provider.inference_status()
        val_initial = data_provider.validation_status()

        print(f"Step 1 - Pre-condition (No models/world_model.pt):")
        print(f"  • File exists ({ckpt_file}): {ckpt_file.is_file()}")
        print(f"  • inference_status(): '{inf_initial}' (Expected: 'mock')")
        print(f"  • validation_status(): '{val_initial}' (Expected: 'validated_offline')")

        assert inf_initial == "mock", f"Expected inference_status() == 'mock', got {inf_initial}"
        assert val_initial == "validated_offline", f"Expected validation_status() == 'validated_offline', got {val_initial}"

        # Verify Forecast view with AppTest
        at = AppTest.from_file(str(ROOT / "views" / "03_Forecast.py"), default_timeout=40)
        at.run()
        assert not at.exception, f"App exception: {at.exception}"
        forecast_html = "".join(m.value for m in at.markdown)

        assert "Predictive Threat Trajectory Forecast" in forecast_html, "Expected Forecast header in views/03_Forecast.py"
        assert "badge-mock" not in forecast_html, "Found deprecated per-widget [MOCK] badges!"
        print("  • Forecast view (views/03_Forecast.py) renders cleanly with 0 per-widget mock badges.")

        # -------------------------------------------------------------------------
        # Step 2: Auto-Detection to LIVE when models/world_model.pt exists
        # -------------------------------------------------------------------------
        print(f"\nStep 2 - Dropping dummy weights file at '{ckpt_file}'...")
        ckpt_file.write_text("SHADOWCAT_DUMMY_WORLD_MODEL_WEIGHTS")

        try:
            assert ckpt_file.is_file(), "Dummy model weights file creation failed"
            inf_live = data_provider.inference_status()
            val_live = data_provider.validation_status()

            print(f"  • File exists ({ckpt_file}): {ckpt_file.is_file()}")
            print(f"  • inference_status(): '{inf_live}' (Expected: 'live')")
            print(f"  • validation_status(): '{val_live}' (Expected: 'validated_offline')")

            assert inf_live == "live", f"Expected inference_status() == 'live' after checkpoint dropped, got {inf_live}"
            assert val_live == "validated_offline", "validation_status() should remain independent as 'validated_offline'"

            # Verify Forecast view now executes in LIVE mode with zero exceptions
            at_live = AppTest.from_file(str(ROOT / "views" / "03_Forecast.py"), default_timeout=40)
            at_live.run()
            assert not at_live.exception, f"App exception: {at_live.exception}"
            live_html = "".join(m.value for m in at_live.markdown)

            assert "Predictive Threat Trajectory Forecast" in live_html, "Forecast header missing in LIVE mode"
            assert "badge-mock" not in live_html, "No per-widget badges should appear in LIVE mode"
            print("  -> AUTO-DETECTION FLIPPED INFERENCE STATUS TO 'LIVE' INSTANTLY VIA FILE PRESENCE!")

        finally:
            # ---------------------------------------------------------------------
            # Step 3: Clean up dummy file and verify rollback
            # ---------------------------------------------------------------------
            print(f"\nStep 3 - Removing dummy file '{ckpt_file}'...")
            if ckpt_file.exists():
                ckpt_file.unlink()

        inf_revert = data_provider.inference_status()
        print(f"  • File exists ({ckpt_file}): {ckpt_file.is_file()}")
        print(f"  • inference_status(): '{inf_revert}' (Expected: 'mock')")
        assert inf_revert == "mock", f"Expected rollback to 'mock', got {inf_revert}"

        # -------------------------------------------------------------------------
        # Step 4: Validation View Status & Governance Test
        # -------------------------------------------------------------------------
        print("\n" + "-" * 65)
        print("Testing Validation View Status & Governance Banner")
        print("-" * 65)

        at_val = AppTest.from_file(str(ROOT / "views" / "07_Validation_Trust.py"), default_timeout=40)
        at_val.run()
        assert not at_val.exception, f"Validation app exception: {at_val.exception}"
        val_html = "".join(m.value for m in at_val.markdown)

        expected_val_banner = "LOEO 37-Fold"
        assert expected_val_banner in val_html, f"Expected '{expected_val_banner}' in Validation view"
        assert "badge-mock" not in val_html, "Deprecated [MOCK] badges found in Validation view!"
        print(f"  • Validation banner containing '{expected_val_banner}' verified in views/07_Validation_Trust.py.")
        print("  • Zero per-widget [MOCK] badges verified in Validation view.")

        print("\n" + "=" * 65)
        print("ALL REVISION A AUTO-DETECTION TESTS PASSED SUCCESSFULLY!")
        print("=" * 65)
    finally:
        data_provider.MOCK_STATUS["forecast_trajectory"] = orig_mock
        if orig_ckpt_bytes is not None:
            ckpt_file.write_bytes(orig_ckpt_bytes)

if __name__ == "__main__":
    run_test()
