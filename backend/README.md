# ShadowCat: Backend Integration Layer & Inference Pipeline

The backend track serves as the unified runtime integration layer for the SHADOWCAT platform. It binds the Data Engineering preprocessing contracts, ML1 causal LSTM world models, hazard/stage classification heads, and tamper-evident audit chains into an authoritative, decoupled `predict()` inference interface.

---

## Architecture & Modules

```
backend/
├── __init__.py              # Package initialization
├── audit_chain.py           # Blockchain-inspired cryptographic hash-chain engine
├── audit_chain.json         # Canonical tamper-evident audit ledger (27 verified blocks)
├── benchmark_real_latency.py# Profiling script for end-to-end processing latency
├── build_audit_chain.py     # Script to generate/rebuild the canonical audit chain
├── conformal.py             # Conformal prediction utilities for calibrated uncertainty
├── counterfactual.py        # Counterfactual explanation generation (Integrated Gradients)
├── demo_fabric_tamper.py    # Safe live demo of adversarial tampering detection for Fabric
├── demo_tamper_detection.py # Safe live demo of adversarial tampering detection for SHA-256
├── fabric_bridge.py         # Hyperledger Fabric chaincode integration & RPC bridge
├── graph_traversal.py       # Graph diffusion simulation and topology rules
├── mitre_kb.py              # MITRE ATT&CK Knowledge Base mapping
├── models.py                # PyTorch architectures (Gaussian LSTM, StageHead, Stacked Residual LSTM)
├── predict.py               # Production inference pipeline (predict(df, source_type=...))
├── proof_runner.py          # Helper for generating verifiable prediction proofs
├── reports_db.py            # Persistent storage layer for generated threat reports
├── smoke_test.py            # End-to-end integration smoke test
├── temporal_attribution.py  # Temporal feature attribution over lookback windows
├── test_fabric_bridge.py    # Unit tests for the Fabric bridge module
├── verify_audit.py          # Diagnostics & scaler parity verification script
├── verify_audit_chain.py    # Fast integrity verification CLI tool
└── requirements.txt         # Standalone backend dependencies
```

---

## Primary Inference Contract (predict.py)

The primary entry point is `predict(input_df, source_type='flows', config=None) -> dict`.

### Supported Input Modes:
1. **Raw / Processed Flows (`source_type='flows'`)**:
   - Accepts network flow DataFrames (e.g. CICFlowMeter schema).
   - Dynamically windowed and normalized using calibrated `data-engineering` scalers.
2. **Pre-computed UCS Windows (`source_type='windows'`)**:
   - Accepts 1-minute Unified Cyber State (UCS) feature windows ($F=406$ dimensions).
   - Requires sequence length $L=30$ lookback windows.

### Inference Pipeline Flow:
```
Input DataFrame (Flows / Windows)
               │
               ▼
   Dynamic Ingestion & Alignment
(Zero-infinities, robust feature extraction, 406-D UCS mapping)
               │
               ▼
   Feature Normalization (v3.0 Scalers)
(Strict post-purge training fit: Log1p + RobustScaler)
               │
               ▼
   LSTM World Model (z_t Latent Transition)
(Extracts 64-D causal temporal context from 30-window sequence)
               │
      ┌────────┴────────┐
      ▼                 ▼
Stacked Residual LSTM Ensemble (37-fold)     Stage Head v3
(Multi-horizon     (ATT&CK stage
 risk trajectory   classification:
 t+1 .. t+4)       Credential Access)
      │                 │
      └────────┬────────┘
               │
               ▼
Dual-Signal Anomaly & Attribution
(NLL-based novelty scoring + Deletion-tested feature attribution)
               │
               ▼
Authoritative Prediction Dictionary Output
```

---

## Tamper-Evident Hash Chain (audit_chain.py)

To answer the SIH theme (*Blockchain & Cybersecurity*) with genuine technical rigor, SHADOWCAT implements a lightweight, blockchain-inspired SHA-256 recursive hash chain over all verified project artifacts (Gate 0 verdicts, model checkpoint weights, contract diffs, and evaluation reports).

### Two-Tier Notarization Architecture
SHADOWCAT implements a real two-tier notarization design, with a live blockchain as the primary path and a cryptographic hash chain as a resilient fallback.
- **Primary Tier (Hyperledger Fabric):** The system connects to a real Go chaincode on a local Fabric test network (`shadowcat-notary-channel`). It records an atomic 4-stage prediction lineage (raw data, feature vector, model ID, prediction). The chaincode features autonomous incident response natively on-chain for HIGH/CRITICAL alerts.
- **Fallback Tier (SHA-256 Hash Chain):** If the Fabric network is unreachable, the system automatically falls back to a cryptographic SHA-256 recursive hash chain to ensure uninterrupted, tamper-evident lineage without pipeline failure. Every record is tagged `notarized_via: "fabric"` or `"sha256_fallback"`.

---

## Standalone Execution & Verification

### 1. Run Pipeline Smoke Test
Executes end-to-end inference against canonical UCS test data:
```bash
python backend/smoke_test.py
```

### 2. Verify Audit Chain Integrity
Walks all 27 chained blocks and validates byte-level SHA-256 matches:
```bash
python backend/verify_audit_chain.py
```

### 3. Run Live Tamper Detection Demo
Simulates an adversary modifying an evaluation score on a safe temporary copy and demonstrates immediate detection:
```bash
python backend/demo_tamper_detection.py
```

### 4. Run Diagnostics & Scaler Parity Audit
```bash
python backend/verify_audit.py
```


**Note**: The final deployed hazard threshold is globally calibrated to 0.15.
