#!/usr/bin/env python3
"""
Test script for verifying checkpoint auto-detection in data_provider.py and Streamlit views.
Ensures that placing a checkpoint file removes the badge via auto-detection alone,
with MOCK_STATUS left untouched at True.
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
    print("Testing Checkpoint Auto-Detection (File-Presence Path)")
    print("=" * 65)

    key = "forecast_trajectory"
    ckpt_rel = data_provider.CHECKPOINT_PATHS[key]
    ckpt_file = ROOT / ckpt_rel
    models_dir = ckpt_file.parent
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Baseline: Ensure file does NOT exist, MOCK_STATUS is True
    if ckpt_file.exists():
        ckpt_file.unlink()
    data_provider.MOCK_STATUS[key] = True

    is_mock_initial = data_provider.is_using_mock_data(key)
    badge_initial = data_provider.get_mock_badge_html(key)
    print(f"Step 1 - Pre-condition (No checkpoint file):")
    print(f"  • MOCK_STATUS['{key}'] = {data_provider.MOCK_STATUS[key]}")
    print(f"  • File exists ({ckpt_rel}): {ckpt_file.is_file()}")
    print(f"  • is_using_mock_data('{key}'): {is_mock_initial} (Expected: True)")
    print(f"  • get_mock_badge_html('{key}'): {repr(badge_initial)} (Expected: non-empty)")
    assert is_mock_initial is True, "Expected is_using_mock_data to be True initially"
    assert "MOCK" in badge_initial, "Expected MOCK badge HTML initially"

    # Verify Forecast view with AppTest initially shows the badge
    at = AppTest.from_file(str(ROOT / "views" / "01_Forecast.py"), default_timeout=30)
    at.run()
    initial_html_blocks = [m.value for m in at.markdown]
    has_mock_init = any("badge-mock" in block for block in initial_html_blocks)
    print(f"  • Forecast view AppTest badge presence: {has_mock_init} (Expected: True)")
    assert has_mock_init is True, "Expected mock badges in Forecast view initially"

    # 2. Place dummy checkpoint file WITHOUT touching MOCK_STATUS
    print(f"\nStep 2 - Dropping dummy file at '{ckpt_rel}' (MOCK_STATUS untouched at True)...")
    ckpt_file.write_text("DUMMY_MODEL_WEIGHTS_FOR_TEST")
    try:
        assert ckpt_file.is_file(), "Dummy file creation failed"

        is_mock_after = data_provider.is_using_mock_data(key)
        badge_after = data_provider.get_mock_badge_html(key)
        print(f"  • MOCK_STATUS['{key}'] = {data_provider.MOCK_STATUS[key]} (UNTOUCHED)")
        print(f"  • File exists ({ckpt_rel}): {ckpt_file.is_file()}")
        print(f"  • is_using_mock_data('{key}'): {is_mock_after} (Expected: False via auto-detect)")
        print(f"  • get_mock_badge_html('{key}'): {repr(badge_after)} (Expected: '')")
        assert is_mock_after is False, "Expected is_using_mock_data to be False when checkpoint exists"
        assert badge_after == "", f"Expected empty badge HTML, got {badge_after}"

        # 3. Verify in Forecast view via AppTest
        at2 = AppTest.from_file(str(ROOT / "views" / "01_Forecast.py"), default_timeout=30)
        at2.run()
        blocks_after = [m.value for m in at2.markdown]
        # Specifically check "Projected Threat Risk" line
        risk_blocks = [b for b in blocks_after if "Projected Threat Risk" in b]
        assert len(risk_blocks) > 0, "Could not find Projected Threat Risk block"
        risk_has_mock = "badge-mock" in risk_blocks[0]
        print(f"  • 'Projected Threat Risk' HTML block has 'badge-mock': {risk_has_mock} (Expected: False)")
        assert not risk_has_mock, "Projected Threat Risk still has badge-mock even though checkpoint is present!"

        print("  -> AUTO-DETECTION FLIPPED BADGE OFF SUCCESSFULLY IN PYTHON & IN STREAMLIT VIEW!")

    finally:
        # 4. Clean up dummy file and verify rollback
        print(f"\nStep 3 - Removing dummy file '{ckpt_rel}'...")
        if ckpt_file.exists():
            ckpt_file.unlink()

    is_mock_revert = data_provider.is_using_mock_data(key)
    badge_revert = data_provider.get_mock_badge_html(key)
    print(f"  • File exists ({ckpt_rel}): {ckpt_file.is_file()}")
    print(f"  • is_using_mock_data('{key}'): {is_mock_revert} (Expected: True)")
    print(f"  • get_mock_badge_html('{key}'): {repr(badge_revert)} (Expected: non-empty)")
    assert is_mock_revert is True, "Expected is_using_mock_data to revert to True"
    assert "MOCK" in badge_revert, "Expected badge to return after file removal"

    # =========================================================================
    # Step 4 - Validation View Auto-Detection & Top Warning Banner Test
    # =========================================================================
    print("\n" + "-" * 65)
    print("Testing Validation View Checkpoint Auto-Detection (comparison_table & validation_data)")
    print("-" * 65)

    comp_file = ROOT / data_provider.CHECKPOINT_PATHS["comparison_table"]
    val_file = ROOT / data_provider.CHECKPOINT_PATHS["validation_data"]

    # Pre-condition: both absent
    if comp_file.exists(): comp_file.unlink()
    if val_file.exists(): val_file.unlink()

    at_val_init = AppTest.from_file(str(ROOT / "views" / "03_Validation.py"), default_timeout=30)
    at_val_init.run()
    val_init_html = "".join([m.value for m in at_val_init.markdown])
    has_banner_init = "Illustrative Benchmark Data — Pending Live Model Integration" in val_init_html
    print(f"  • Initial Validation Banner Present: {has_banner_init} (Expected: True)")
    assert has_banner_init, "Expected validation warning banner initially"

    # Place both checkpoint files
    comp_file.write_text("[]")
    val_file.write_text("{}")
    try:
        at_val_after = AppTest.from_file(str(ROOT / "views" / "03_Validation.py"), default_timeout=30)
        at_val_after.run()
        val_after_html = "".join([m.value for m in at_val_after.markdown])
        has_banner_after = "Illustrative Benchmark Data — Pending Live Model Integration" in val_after_html
        print(f"  • Validation Banner Present after Checkpoints: {has_banner_after} (Expected: False)")
        assert not has_banner_after, "Warning banner still present after checkpoints dropped!"

        has_comp_badge = "badge-mock" in val_after_html
        print(f"  • Badges present on Validation view: {has_comp_badge} (Expected: False)")
        assert not has_comp_badge, "Badges still present on Validation view!"
        print("  -> AUTO-DETECTION FLIPPED VALIDATION PAGE & REMOVED BANNER AUTOMATICALLY!")
    finally:
        if comp_file.exists(): comp_file.unlink()
        if val_file.exists(): val_file.unlink()

    print("\n" + "=" * 65)
    print("ALL AUTO-DETECTION TESTS (FORECAST & VALIDATION) PASSED.")
    print("=" * 65)

if __name__ == "__main__":
    run_test()
