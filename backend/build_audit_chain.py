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
    ("../data-engineering/data/ucs/PACKET_EXTRACTION_VERIFICATION.md", "extraction_verification", "Real Scapy PCAP extraction verification (Option B real packet telemetry)"),
    ("../data-engineering/data/ucs/CONTRACT_DIFF_REPORT.md", "contract_verification", "UCS-ML1 inference contract diff v3 — confirmed PASS"),
    ("../ml1/artifacts/lstm/gaussian_next_state_best_v3.pt", "model_checkpoint", "LSTM world model v3 (trained on genuine PCAP packet telemetry)"),
    ("../ml1/artifacts/lstm/hazard_head_v3/hazard_head_report_v3.md", "evaluation_report", "Hazard head v3 evaluation report (restored discriminative power under real packet telemetry)"),
    ("../ml1/artifacts/lstm/stage_head_v3/stage_head_report.md", "evaluation_report", "Stage head v3 evaluation report"),
    ("../ml1/artifacts/lstm/probabilistic_world_model_v3/pc2_attack_significance_report.md", "evaluation_report", "PC2 significance test report (evaluated on v3 checkpoint)"),
    ("../ml2-full/GNN_FINAL/ml2/results/gnn_adopt_hold_decision.md", "architecture_decision", "GNN fusion adopt/hold decision — HOLD, v3 sync addendum"),
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
