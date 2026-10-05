import sys
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

from audit_chain import verify_chain, _load_chain, chain_merkle_root
from fabric_bridge import _is_docker_available

if __name__ == "__main__":
    chain = _load_chain()
    print("=" * 70)
    print("TAMPER-EVIDENT AUDIT CHAIN VERIFICATION")
    print(f"Total entries in chain: {len(chain)}")
    print("=" * 70)

    is_valid, issues = verify_chain()
    if is_valid:
        print("✅ AUDIT CHAIN VALID — all entries intact, no tampering detected.")
        print(f"Merkle root ({len(chain)} entries): {chain_merkle_root()['merkle_root']}")
        if not _is_docker_available():
            print("[i] Docker: not found — Fabric notarization will use SHA-256 fallback (this is fine)\n")
        else:
            print("[i] Docker: available — Fabric notarization is supported\n")
        
        for entry in chain:
            print(f"  [{entry['index']}] {entry['artifact_type']:<22} | {entry['entry_hash'][:16]}... | {entry['description']}")
        sys.exit(0)
    else:
        print("❌ AUDIT CHAIN INVALID:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
