"""
SHADOWCAT Backend Smoke Test Script
Executes full predict() on canonical data and outputs formatted diagnostic report.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data-engineering"))

import json
import pandas as pd
import numpy as np
from backend.predict import predict

def run_smoke_test():
    print("=" * 60)
    print("SHADOWCAT BACKEND END-TO-END SMOKE TEST")
    print("=" * 60)

    # 1. Canonical Dataset Sample
    parquet_path = "data-engineering/data/ucs/ucs_windows.parquet"
    print(f"\n[1] Loading real canonical sample from: {parquet_path}")
    df = pd.read_parquet(parquet_path).head(45).copy()
    print(f"    Loaded {len(df)} windows.")

    # 2. Run predict()
    print("\n[2] Executing predict(df, source_type='flows')...")
    output = predict(df, source_type="flows")

    # 3. Validate Non-Degeneracy
    fc = output["forecast_trajectory"]
    risks = fc["risk"]
    stages = fc["stage"]
    novelty = output["novelty_score"]

    print("\n[3] Verification Checks:")
    print(f"    - Window ID: {output['window_id']}")
    print(f"    - Is Warmup: {output['is_warmup']}")
    print(f"    - Risk Trajectory [t+1..t+4]: {risks}")
    print(f"    - Stage Trajectory: {stages}")
    print(f"    - Novelty Score: {novelty['novelty_score']} ({novelty['novelty_status']})")
    print(f"    - Dominant Behavior: {novelty['dominant_behavior']}")

    # Assertions
    assert len(risks) == 4, f"Expected 4 risk horizons, got {len(risks)}"
    assert all(0.0 <= r <= 1.0 for r in risks), f"Risk values outside [0, 1]: {risks}"
    assert not all(r == 0.0 for r in risks), "Degenerate failure: all risks are exact 0.0"
    assert not any(np.isnan(r) for r in risks), "NaN values detected in risk scores"
    print("\n    [PASS] Probability boundedness & non-degeneracy verified.")

    print("\n[4] Top Feature Attributions:")
    for a in output["attributions"]:
        print(f"    - {a['feature']:<30}: {a['contribution']:.2f} ({a['category']}) | {a['delta']}")

    print("\n[5] Flagged Flows Sample:")
    for f in output["flagged_flows"][:3]:
        print(f"    - {f['id']}: {f['source']}:{f['sport']} -> {f['destination']}:{f['dport']} ({f['protocol']}) | {f['reason']}")

    print("\n[6] MITRE ATT&CK Knowledge Base Verification:")
    from backend.mitre_kb import get_mitre_kb
    kb = get_mitre_kb()
    meta = kb.get_corpus_metadata()
    print(f"    - Corpus Name: {meta.get('name')}")
    print(f"    - Version: {meta.get('version')} (Modified: {meta.get('modified')})")
    print(f"    - Total STIX Tactics: {len(kb.tactics_by_id)}")
    print(f"    - Total STIX Techniques: {len(kb.techniques_by_id)} (Active: {meta.get('active_techniques')})")

    # Assertions on real counts
    assert len(kb.tactics_by_id) == 15, f"Expected 15 tactics, got {len(kb.tactics_by_id)}"
    assert len(kb.techniques_by_id) >= 600, f"Expected >=600 techniques, got {len(kb.techniques_by_id)}"
    assert meta.get("active_techniques", 0) >= 500, f"Expected >=500 active techniques, got {meta.get('active_techniques')}"

    # Verify 6 attack_tactics_mapping.yaml technique IDs
    target_yaml_ids = ["T1110.001", "T1498.001", "T1071.001", "T1190", "T1189", "T1046"]
    for tid in target_yaml_ids:
        t_info = kb.get_technique(tid)
        assert t_info is not None, f"Technique {tid} not found in real STIX corpus!"
        assert t_info["url"].startswith("https://attack.mitre.org/"), f"Technique {tid} has invalid URL: {t_info['url']}"
        print(f"    - Verified {tid}: {t_info['full_name']} -> {t_info['url']}")

    # Verify predict() output contains real MITRE metadata
    assert "mitre_details" in fc, "Forecast trajectory missing 'mitre_details'!"
    assert len(fc["mitre_details"]) == 4, f"Expected 4 mitre_details entries, got {len(fc['mitre_details'])}"
    for idx, md in enumerate(fc["mitre_details"]):
        assert md["tactic_id"].startswith("TA"), f"Step {idx} invalid tactic_id: {md['tactic_id']}"
        assert md["technique_id"].startswith("T"), f"Step {idx} invalid technique_id: {md['technique_id']}"
        assert len(md["technique_description"]) > 20, f"Step {idx} empty technique description!"
        assert "is_heuristic_progression" in md, f"Step {idx} missing heuristic flag!"
        assert "attribution_source" in md, f"Step {idx} missing attribution source!"
    print(f"    - Trajectory MITRE Tactics: {fc.get('tactic_id')}")
    print(f"    - Trajectory Techniques: {fc.get('technique_id')}")
    print(f"    - Heuristic Progression Flags: {fc.get('is_heuristic_progression')}")
    print(f"    - Stage Attribution Sources: {fc.get('stage_attribution_source')}")
    print("    [PASS] MITRE ATT&CK Knowledge Base offline query layer verified.")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETED SUCCESSFULLY WITH ZERO DEFECTS")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()
