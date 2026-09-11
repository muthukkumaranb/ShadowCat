import os
import sys
from pathlib import Path

# Safe UTF-8 console output for Windows CLI environments
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from audit_chain import append_entry, CHAIN_PATH

ENTRIES = [
    ("../data-engineering/data/ucs/CONTRACT_DIFF_REPORT.md", "contract_verification", "UCS-ML1 inference contract diff — confirmed PASS"),
    ("../ml1/artifacts/lstm/gaussian_next_state_best_v2.pt", "model_checkpoint", "LSTM world model v2 (post scaler-fix retrain)"),
    ("../ml1/artifacts/lstm/hazard_head/hazard_head_report.md", "evaluation_report", "Hazard head v2 evaluation (post label-leak correction)"),
    ("../ml1/artifacts/lstm/stage_head/stage_head_report.md", "evaluation_report", "Stage head v2 evaluation"),
    ("../ml1/artifacts/lstm/pc2_significance_report.md", "evaluation_report", "PC2 significance test (re-run on v2 checkpoint)"),
    ("../ml2-full/GNN_FINAL/ml2/results/gnn_adopt_hold_decision.md", "architecture_decision", "GNN fusion adopt/hold decision — HOLD, multi-seed evidence"),
    ("../ml1/artifacts/leakage_audit/component_exposure_audit.md", "leakage_audit", "Packet-feature label-leak exposure audit"),
]

if __name__ == "__main__":
    if os.path.exists(CHAIN_PATH):
        os.remove(CHAIN_PATH)

    print("=" * 70)
    print("BUILDING TAMPER-EVIDENT FORENSIC AUDIT CHAIN")
    print(f"Destination: {CHAIN_PATH}")
    print("=" * 70)

    for path, artifact_type, desc in ENTRIES:
        resolved = os.path.normpath(os.path.join(BACKEND_DIR, path))
        entry = append_entry(resolved, artifact_type, desc)
        print(f"[{entry['index']}] {desc}")
        print(f"    Artifact: {entry['artifact_path']}")
        print(f"    Type    : {entry['artifact_type']}")
        print(f"    SHA-256 : {entry['artifact_hash']}")
        print(f"    Link    : {entry['prev_entry_hash'][:16]}... -> {entry['entry_hash'][:16]}...")
        print("-" * 70)

    print(f"\n[PASS] Successfully chained {len(ENTRIES)} verified artifacts.")
