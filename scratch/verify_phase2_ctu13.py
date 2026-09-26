"""
Phase 2 Verification Script:
1. Verify raw (untranslated) CTU-13 is REJECTED by ucs_extractor.py with SchemaValidationError.
2. Verify translated CTU-13 PASSES CICFlowMeterValidator with EXACT_MATCH.
3. Verify translated CTU-13 PASSES ucs_extractor.extract().
4. Run translated CTU-13 through backend predict() end-to-end.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data-engineering"))

from src.ucs_extractor import UCSExtractor, SchemaValidationError
from src.csv_validator import CICFlowMeterValidator, ValidationStatus
from src.ctu13_translator import CTU13Translator, translate_ctu13
from backend.predict import predict

def verify_phase2():
    print("=" * 80)
    print("PHASE 2: HONEST CTU-13 TRANSLATOR VERIFICATION")
    print("=" * 80)

    # 1. Load raw CTU-13
    raw_ctu_path = REPO_ROOT / "data" / "ctu13_sample.csv"
    assert raw_ctu_path.exists(), f"Missing {raw_ctu_path}"
    raw_ctu_df = pd.read_csv(raw_ctu_path)
    print(f"[*] Loaded raw CTU-13 sample: {len(raw_ctu_df)} rows, {len(raw_ctu_df.columns)} columns")
    print(f"    Raw columns: {list(raw_ctu_df.columns)}")

    # 2. Verify raw CTU-13 is REJECTED by ucs_extractor
    print("\n[STEP 1] Testing rejection of raw (untranslated) CTU-13 by UCSExtractor...")
    extractor = UCSExtractor()
    rejected = False
    try:
        extractor.extract(raw_ctu_df, source_type="csv")
    except SchemaValidationError as e:
        rejected = True
        print(f"    [PASS] Expected SchemaValidationError caught: {e}")
    except Exception as e:
        print(f"    [FAIL] Unexpected exception: {type(e)}: {e}")

    assert rejected, "Raw CTU-13 was NOT rejected by UCSExtractor!"
    print("    [CONFIRMED] Defensive validation contract preserved without modification.")

    # 3. Translate CTU-13 using CTU13Translator
    print("\n[STEP 2] Translating CTU-13 via CTU13Translator...")
    translator = CTU13Translator()
    translated_df, fidelity_report = translator.translate(raw_ctu_df)

    print(f"    Translated DataFrame: {len(translated_df)} rows, {len(translated_df.columns)} columns")
    print(f"    Fidelity Breakdown: {fidelity_report.direct_count} Direct, {fidelity_report.derived_count} Derived, {fidelity_report.unavailable_count} Unavailable")
    print(f"    Fidelity Score: {fidelity_report.fidelity_score_pct}%")

    # 4. Validate translated DataFrame against CICFlowMeterValidator
    print("\n[STEP 3] Validating translated DataFrame with CICFlowMeterValidator...")
    validator = CICFlowMeterValidator()
    val_res = validator.validate(translated_df)
    print(f"    Validation status: {val_res.status}")
    print(f"    Is valid: {val_res.is_valid}")
    print(f"    Missing columns: {val_res.missing_required_columns}")
    assert val_res.is_valid, f"Validation failed: {val_res.errors}"
    assert val_res.status == ValidationStatus.EXACT_MATCH, f"Expected EXACT_MATCH, got {val_res.status}"
    print("    [PASS] Translated DataFrame strictly conforms to 80-column schema contract.")

    # 5. Extract 410-column UCS tensor via UCSExtractor
    print("\n[STEP 4] Extracting 410-column UCS window tensor via UCSExtractor...")
    extracted = extractor.extract(translated_df, source_type="csv")
    print(f"    Extracted UCS windows shape: {extracted.shape}")
    assert extracted.shape[1] == 410, f"Expected 410 columns, got {extracted.shape[1]}"
    print("    [PASS] UCSExtractor successfully output 410 standardized UCS columns.")

    # 6. Run end-to-end inference through predict()
    print("\n[STEP 5] Running full live inference pipeline on translated CTU-13...")
    pred_res = predict(translated_df, source_type="csv")
    print(f"    Window ID: {pred_res.get('window_id')}")
    print(f"    Forecast Risk: {pred_res.get('risk_scores')}")
    print(f"    Stage Predictions: {pred_res.get('stage_predictions')}")
    print(f"    Notarized Via: {pred_res.get('notarized_via')}")
    lineage = pred_res.get("lineage", {})
    print(f"    Lineage ID: {lineage.get('lineage_id')}")
    print(f"    Raw Data Hash: {lineage.get('raw_data_hash')[:16]}...")
    print(f"    Feature Hash: {lineage.get('feature_hash')[:16]}...")
    print(f"    Prediction Hash: {lineage.get('prediction_hash')[:16]}...")

    print("\n" + "=" * 80)
    print("PHASE 2 ALL VERIFICATIONS SUCCEEDED CLEANLY")
    print("=" * 80)

if __name__ == "__main__":
    verify_phase2()
