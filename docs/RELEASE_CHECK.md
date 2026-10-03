# Release Verification Checks: `release/judge-ready`

Branch: `release/judge-ready`
Base: `origin/main` (`04eb82e9334c48b26b8af4927990aa26907c757a`)
Date: 2026-10-03

All improvement branches merged in specified order:
1. `origin/feature/judge-readiness-v2` (`40c04fb`)
2. `origin/fix/demo-integrity` (`2cb16c7`)
3. `origin/docs/final-findings` (`4c35dd8`)
4. `origin/feature/deviation-evidence` (`34c1aa5`)
5. `origin/feature/rollout-delta-mc` (`9363b5f`)

Forecast notarisation: fixed (walked_edge_keys set -> list); smoke test shows no append warning; forecast files unique with exclusive creation ("x").

---

### Check 1: Backend Tests
- **Command:** `python -m pytest backend/tests`
- **Result:** PASS
- **Key Output Line:**
```text
======================= 37 passed, 3 warnings =======================
```

---

### Check 2: Evaluation Benchmark Tests
- **Command:** `python -m pytest evaluation/benchmark`
- **Result:** PASS
- **Key Output Line:**
```text
============================= 4 passed in 38.30s ==============================
```

---

### Check 3: Backend End-to-End Smoke Test
- **Command:** `python backend/smoke_test.py`
- **Result:** PASS (exit code 0)
- **Key Output Line:**
```text
============================================================
SMOKE TEST COMPLETED SUCCESSFULLY WITH ZERO DEFECTS
============================================================
```

---

### Check 4: Stacked Benchmark Reproduction
- **Command:** `python evaluation/benchmark/reproduce_stacked_benchmark.py --data data-engineering/data/ucs/ucs_windows_models_v1.parquet`
- **Result:** PASS (exit code 0, all 37 folds reproduce without error)
- **Key Output Line:**
```text
All 37 folds reproduced sidecar F1 for both detection and onset:
detection.folds_reproducing_sidecar_f1 = 37
onset.folds_reproducing_sidecar_f1 = 37
```

---

### Check 5: Tamper-Evident Audit Chain Verification
- **Sequence:**
  ```powershell
  # Verified on fresh SHADOWCAT_RUNTIME_DIR
  $env:SHADOWCAT_RUNTIME_DIR = "$freshDir"
  python backend/smoke_test.py
  python backend/verify_audit_chain.py
  ```
- **Result:** PASS (exit code 0, verifier reports zero issues)
- **Key Output Line:**
```text
✅ AUDIT CHAIN VALID — all entries intact, no tampering detected.
Total entries in chain: 3 (model_checkpoint, prediction_lineage, forecast_payload)
```

---

### Check 6: PEM Key Security Verification
- **Command:** `git ls-files | Select-String -Pattern "\.pem"`
- **Result:** PASS
- **Key Output Line:**
```text
backend/admin_public_key.pem
```

---

### Check 7: SOC Dashboard Demo Slices Inference
- **Command:** `python -m streamlit run frontend/app.py --server.port 8502 --server.headless true` (with `load_demo_slice` & `run_core_ml_inference`)
- **Result:** PASS
- **Key Output Line:**
```text
Benign demo slice onset probability: 0.314 (0.31363)
SSH-Bruteforce demo slice onset probability: 0.887 (0.88727)
```
Probabilities differ significantly as expected (~0.31 vs ~0.89). Dashboard stopped.
