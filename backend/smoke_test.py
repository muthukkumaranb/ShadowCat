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

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETED SUCCESSFULLY WITH ZERO DEFECTS")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()
