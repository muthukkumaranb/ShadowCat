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

    ckpt_file = ROOT / "models" / "world_model.pt"
    backup_file = ROOT / "models" / "world_model.pt.autodetect_backup"
    models_dir = ckpt_file.parent
    models_dir.mkdir(parents=True, exist_ok=True)

    # Safely backup any existing real checkpoint before testing
    if ckpt_file.exists():
        ckpt_file.rename(backup_file)

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

        # Verify Forecast view with AppTest initially shows BENCHMARK MODE sticky banner
        at = AppTest.from_file(str(ROOT / "views" / "01_Forecast.py"), default_timeout=30)
        at.run()
        assert not at.exception, f"App exception: {at.exception}"
        forecast_html = "".join(m.value for m in at.markdown)

        assert "BENCHMARK MODE" in forecast_html, "Expected 'BENCHMARK MODE' banner in initial Forecast view"
        assert "Running on benchmark data (CIC-IDS2018)" in forecast_html, "Expected benchmark explanation banner"
        assert "badge-mock" not in forecast_html, "Found deprecated per-widget [MOCK] badges!"
        print("  • Forecast view renders sticky 'BENCHMARK MODE' banner and 0 per-widget mock badges.")

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

            # Verify Forecast view now reflects LIVE in sticky header
            at_live = AppTest.from_file(str(ROOT / "views" / "01_Forecast.py"), default_timeout=30)
            at_live.run()
            assert not at_live.exception, f"App exception: {at_live.exception}"
            live_html = "".join(m.value for m in at_live.markdown)

            assert "● LIVE" in live_html, "Expected '● LIVE' pill in header when checkpoint present"
            assert "Live inference pipeline connected" in live_html, "Expected live connection text in header"
            assert "BENCHMARK MODE" not in live_html, "'BENCHMARK MODE' banner should be replaced by LIVE"
            assert "badge-mock" not in live_html, "No per-widget badges should appear in LIVE mode"
            print("  -> AUTO-DETECTION FLIPPED HEADER BANNER TO 'LIVE' INSTANTLY VIA FILE PRESENCE!")

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

        at_val = AppTest.from_file(str(ROOT / "views" / "03_Validation.py"), default_timeout=30)
        at_val.run()
        assert not at_val.exception, f"Validation app exception: {at_val.exception}"
        val_html = "".join(m.value for m in at_val.markdown)

        expected_val_banner = "Model validated offline (LOEO 37-fold)"
        assert expected_val_banner in val_html, f"Expected '{expected_val_banner}' in Validation view"
        assert "badge-mock" not in val_html, "Deprecated [MOCK] badges found in Validation view!"
        print(f"  • Validation banner '{expected_val_banner}' verified.")
        print("  • Zero per-widget [MOCK] badges verified in Validation view.")

    finally:
        # Restore real model weights if previously backed up
        if ckpt_file.exists():
            ckpt_file.unlink()
        if backup_file.exists():
            backup_file.rename(ckpt_file)
            print(f"[CLEANUP] Restored real model weights to '{ckpt_file}'.")

    print("\n" + "=" * 65)
    print("ALL REVISION A AUTO-DETECTION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_test()
