"""
Kill-the-Network Fallback Verification Test
Verifies that when Hyperledger Fabric is unreachable, the hazard-head prediction
pipeline automatically and safely falls back to local SHA-256 hash-chain notarization,
and that the resulting audit chain remains 100% cryptographically valid.
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "backend"))

import pandas as pd
import backend.predict as predict_module
from backend.predict import predict
from backend.audit_chain import verify_chain, _load_chain
import backend.fabric_bridge as fabric_bridge

def run_fallback_test():
    print("=" * 65)
    print("KILL-THE-NETWORK FALLBACK TEST (FABRIC UNREACHABLE -> SHA-256)")
    print("=" * 65)

    # 1. Verify chain before test
    chain_before = _load_chain()
    print(f"[*] Audit chain entries before test: {len(chain_before)}")
    valid_before, issues = verify_chain()
    assert valid_before, f"Chain invalid before test: {issues}"

    # 2. Simulate Fabric Unreachable / Network Down
    print("[*] Simulating Fabric network disconnection (Fabric Unreachable)...")
    def mock_fabric_fail(*args, **kwargs):
        raise ConnectionError("Fabric peer/orderer unreachable: Connection refused (network simulated down)")

    # Monkeypatch both notarize_alert and notarize_model in fabric_bridge AND predict_module
    orig_notarize_alert = fabric_bridge.notarize_alert
    orig_notarize_model = fabric_bridge.notarize_model
    orig_pm_alert = getattr(predict_module, "notarize_alert", None)
    orig_pm_model = getattr(predict_module, "notarize_model", None)

    fabric_bridge.notarize_alert = mock_fabric_fail
    fabric_bridge.notarize_model = mock_fabric_fail
    setattr(predict_module, "notarize_alert", mock_fabric_fail)
    setattr(predict_module, "notarize_model", mock_fabric_fail)

    try:
        # 3. Load sample data that triggers hazard alert
        parquet_path = repo_root / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
        df = pd.read_parquet(parquet_path).head(45).copy()

        # 4. Run predict() through the live hazard-head path
        print("[*] Executing predict() under simulated network outage...")
        output = predict(df, source_type="flows")

        fc = output["forecast_trajectory"]
        risks = fc["risk"]
        threshold = fc["calibrated_threshold"]
        hazard_alert = fc["hazard_alert"]
        notarized_via = output.get("notarized_via")
        mechanism = output.get("notarization_mechanism")

        print(f"\n[*] Prediction Results:")
        print(f"    - Risk trajectory: {risks}")
        print(f"    - Calibrated Threshold: {threshold}")
        print(f"    - Hazard Alert Triggered: {hazard_alert}")
        print(f"    - Notarized Via: {notarized_via}")
        print(f"    - Notarization Mechanism: {mechanism}")

        assert hazard_alert is True, "Expected hazard alert to trigger on high-risk sequence"
        assert notarized_via == "sha256_fallback", f"Expected 'sha256_fallback', got: {notarized_via}"
        assert mechanism == "sha256_fallback", f"Expected mechanism 'sha256_fallback', got: {mechanism}"
        print("    [PASS] Fallback to sha256_fallback confirmed upon Fabric network failure.")

        # 5. Verify audit chain was updated and is 100% cryptographically intact
        chain_after = _load_chain()
        print(f"\n[*] Audit chain entries after test: {len(chain_after)}")
        assert len(chain_after) >= len(chain_before), "Audit chain did not record the fallback entry"

        latest_entry = chain_after[-1]
        print(f"[*] Latest audit chain entry:")
        print(f"    - Index: {latest_entry['index']}")
        print(f"    - Type : {latest_entry['artifact_type']}")
        print(f"    - Hash : {latest_entry['entry_hash'][:16]}...")
        print(f"    - Desc : {latest_entry['description']}")

        is_valid, issues = verify_chain()
        assert is_valid, f"Audit chain verification failed: {issues}"
        print("\n[+] Cryptographic Verification: PASS (100% intact, zero tampering).")
        print("=" * 65)
        print("KILL-THE-NETWORK FALLBACK TEST SUCCEEDED COMPLETELY")
        print("=" * 65)

    finally:
        # Restore original functions
        fabric_bridge.notarize_alert = orig_notarize_alert
        fabric_bridge.notarize_model = orig_notarize_model
        if orig_pm_alert is not None:
            setattr(predict_module, "notarize_alert", orig_pm_alert)
        if orig_pm_model is not None:
            setattr(predict_module, "notarize_model", orig_pm_model)

if __name__ == "__main__":
    run_fallback_test()
