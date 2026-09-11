"""
Validation Tests for Chronological ML1 Deviation Export
"""
import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))

EXPORT_DIR = workspace_dir / 'artifacts' / 'lstm' / 'ml2_export_chronological'
JSON_DIR = EXPORT_DIR / 'per_window_json'

def main():
    print("======================================================================")
    print("Chronological Deviation Export Tests")
    print("======================================================================")

    passed = 0
    total = 0

    def assert_test(condition, message):
        nonlocal passed, total
        total += 1
        if condition:
            print(f"  [PASS] {message}")
            passed += 1
        else:
            print(f"  [FAIL] {message}")

    # Test 1: Artifact Existence
    print("\nTest 1: Artifact Existence")
    assert_test(JSON_DIR.exists(), "per_window_json directory exists")
    json_files = list(JSON_DIR.glob("*.json"))
    assert_test(len(json_files) > 0, f"JSON files present ({len(json_files)} found)")
    assert_test((EXPORT_DIR / 'metadata.json').exists(), "metadata.json exists")
    assert_test((EXPORT_DIR / 'deviation_audit.csv').exists(), "deviation_audit.csv exists")

    if len(json_files) == 0:
        print("Missing JSON files. Cannot proceed.")
        return

    # Load audit
    audit = pd.read_csv(EXPORT_DIR / 'deviation_audit.csv')

    # Test 2: Chronological Ordering
    print("\nTest 2: Chronological Ordering")
    times = pd.to_datetime(audit['window_start_utc'])
    assert_test(times.is_monotonic_increasing, "Deviation audit windows are strictly chronologically ordered")

    # Test 3: Coverage Check (ML2 Val/Test are covered)
    print("\nTest 3: Coverage Check")
    assert_test(len(json_files) > 2000, f"Provides > 2000 windows (got {len(json_files)})")
    # Exact check against 2275
    assert_test(len(json_files) == 2275, f"Provides exactly 2275 windows (expected 2787 - 512, got {len(json_files)})")

    # Test 4: Adapter Schema
    print("\nTest 4: Adapter Schema")
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    assert_test(isinstance(data, dict), "JSON is a dictionary")
    if data:
        k, v = next(iter(data.items()))
        assert_test(isinstance(k, str), "JSON keys are strings (host IDs)")
        assert_test(isinstance(v, (int, float)), "JSON values are numeric (deviation scores)")

    # Test 5: No Prohibited Features Leakage
    print("\nTest 5: No Future/Prohibited Feature Leakage")
    assert_test('window_start_utc' not in data, "No window_start_utc in keys")
    assert_test('source_day' not in data, "No source_day in keys")
    assert_test('episode_id' not in data, "No episode_id in keys")

    # Test 6: Normalization Sanity
    print("\nTest 6: Normalization Sanity")
    scores = audit['normalized_deviation_score'].values
    assert_test(not np.isnan(scores).any(), "No NaN values in normalized scores")
    assert_test(not np.isinf(scores).any(), "No Inf values in normalized scores")
    # Normalized scores should generally be around 0 with unit variance, 
    # but since it's robust scaling on log1p, it won't be perfectly N(0,1)
    median = np.median(scores)
    assert_test(-1.0 < median < 1.0, f"Normalized median is roughly centered (median={median:.3f})")

    # Test 7: Scaler Fit Boundary Audit
    print("\nTest 7: Scaler Fit Boundary Audit")
    with open(EXPORT_DIR / 'metadata.json', 'r') as f:
        meta = json.load(f)
    
    assert_test('chunks' in meta, "Metadata contains chunk information")
    for i, chunk in enumerate(meta.get('chunks', [])):
        train_indices = chunk.get('train_indices', [])
        pred_indices = chunk.get('predict_indices', [])
        
        if not train_indices or not pred_indices:
            continue
            
        max_train = max(train_indices)
        min_pred = min(pred_indices)
        
        assert_test(max_train < min_pred, f"Chunk {i+1}: max(train_index) [{max_train}] < min(pred_index) [{min_pred}]")

    print("\n======================================================================")
    print(f"RESULTS: {passed} passed, {total - passed} failed, {total} total")
    print("======================================================================")
    
    if passed == total:
        print("\n*** ALL TESTS PASSED ***")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
