"""
SHADOWCAT - Tamper-Evidence & Forensic Integrity Demo
Demonstrates blockchain-inspired cryptographic hash-chain verification
and deliberate adversarial tampering detection.

NOTE: All tampering tests operate strictly on temporary copies.
Real project artifacts and checkpoints are never modified.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Safe UTF-8 console output for Windows CLI environments
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from audit_chain import verify_chain, _load_chain, _sha256_of_file, CHAIN_PATH


def run_demo():
    print("=" * 75)
    print("   SHADOWCAT FORENSIC PROVENANCE — TAMPER DETECTION DEMO")
    print("   (Blockchain-Inspired Cryptographic Hash Chain Audit Trail)")
    print("=" * 75)

    # -------------------------------------------------------------
    # 1. Baseline Integrity Verification
    # -------------------------------------------------------------
    print("\n[PHASE 1] VERIFYING UNTOUCHED PRODUCTION ARTIFACTS...")
    chain = _load_chain()
    print(f"Loaded audit chain: {CHAIN_PATH} ({len(chain)} linked entries)")

    is_valid, issues = verify_chain()
    if is_valid:
        print("✅ [PASS] Official audit chain is 100% VALID.")
        for entry in chain:
            print(f"   • [{entry['index']}] {entry['artifact_type']:<22} | SHA256: {entry['artifact_hash'][:16]}... | {entry['description']}")
    else:
        print("❌ [FAIL] Official audit chain has issues:")
        for iss in issues:
            print(f"   - {iss}")
        return

    # -------------------------------------------------------------
    # 2. Adversarial Tamper Simulation (File Modification on a COPY)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("[PHASE 2] ADVERSARIAL SIMULATION: UNNOTICED ARTIFACT MUTATION")
    print("Scenario: An unauthorized actor modifies 1 byte in a model report.")
    print("Safety  : Operating strictly on a temporary copy — real files are untouched.")
    print("-" * 75)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Pick the hazard head evaluation report
        target_entry_idx = 2
        original_entry = chain[target_entry_idx]
        real_rel_path = original_entry["artifact_path"]
        real_full_path = os.path.normpath(os.path.join(BACKEND_DIR, real_rel_path))

        # Copy original file to temp directory
        tampered_artifact_copy = os.path.join(tmp_dir, "hazard_head_report_tampered.md")
        with open(real_full_path, "rb") as f_src:
            content = f_src.read()

        # Deliberately modify 1 byte in copy
        tampered_content = content + b"\n<!-- MALICIOUS INJECTION: spoofed evaluation score PR-AUC=0.999 -->\n"
        with open(tampered_artifact_copy, "wb") as f_dst:
            f_dst.write(tampered_content)

        print(f"Original Artifact : {real_rel_path}")
        print(f"Original SHA-256  : {_sha256_of_file(real_full_path)}")
        print(f"Tampered Copy SHA : {_sha256_of_file(tampered_artifact_copy)}")

        # Create a temporary chain copy pointing entry to tampered copy
        temp_chain = [dict(e) for e in chain]
        temp_chain[target_entry_idx]["artifact_path"] = tampered_artifact_copy
        temp_chain_path = os.path.join(tmp_dir, "temp_audit_chain.json")
        with open(temp_chain_path, "w", encoding="utf-8") as f:
            json.dump(temp_chain, f, indent=2)

        # Run verification against tampered scenario
        is_valid_tampered, tamper_issues = verify_chain(temp_chain_path)

        if not is_valid_tampered:
            print("\n🚨 [DETECTED] Hash mismatch caught immediately by verify_chain():")
            for issue in tamper_issues:
                print(f"   ⚠️  {issue}")
            print("\n🎯 Result: Cryptographic seal broken — post-hoc tampering is provably detectable!")
        else:
            print("❌ Unexpected: Tamper was not detected.")

    # -------------------------------------------------------------
    # 3. Adversarial Simulation (Chain Block / Link Tampering)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("[PHASE 3] ADVERSARIAL SIMULATION: AUDIT LEDGER FORGERY")
    print("Scenario: An attacker attempts to forge the audit_chain.json itself.")
    print("-" * 75)

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_chain = [dict(e) for e in chain]
        # Adversary attempts to overwrite the recorded hash in entry 1
        temp_chain[1]["artifact_hash"] = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        temp_chain_path = os.path.join(tmp_dir, "temp_forged_chain.json")
        with open(temp_chain_path, "w", encoding="utf-8") as f:
            json.dump(temp_chain, f, indent=2)

        is_valid_forged, forgery_issues = verify_chain(temp_chain_path)
        if not is_valid_forged:
            print("🚨 [DETECTED] Ledger tampering caught immediately:")
            for issue in forgery_issues:
                print(f"   ⚠️  {issue}")
            print("\n🎯 Result: Block hash verification prevented historical rewriting.")

    # -------------------------------------------------------------
    # 4. Final Safety & Status Check
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("✅ DEMO COMPLETE — No real files were touched, modified, or restored.")
    print("   Production audit chain remains intact and pristine.")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
