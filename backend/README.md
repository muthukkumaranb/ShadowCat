# SHADOWCAT — Backend Integration Layer & Inference Pipeline

The backend track serves as the unified runtime integration layer for the SHADOWCAT platform. It binds the Data Engineering preprocessing contracts, ML1 causal LSTM world models, hazard/stage classification heads, and tamper-evident audit chains into an authoritative, decoupled `predict()` inference interface.

---

## 🏗️ Architecture & Modules

```
backend/
├── predict.py               # Production inference pipeline (predict(df, source_type=...))
├── models.py                # PyTorch architectures (Gaussian LSTM, HazardHead, StageHead)
├── smoke_test.py            # End-to-end integration smoke test
├── audit_chain.py           # Blockchain-inspired cryptographic hash-chain engine
├── audit_chain.json         # Canonical tamper-evident audit ledger (7 verified blocks)
├── build_audit_chain.py     # Script to generate/rebuild the canonical audit chain
├── verify_audit_chain.py    # Fast integrity verification CLI tool
├── demo_tamper_detection.py # Safe live demo of adversarial tampering detection
├── verify_audit.py          # Diagnostics & scaler parity verification script
└── requirements.txt         # Standalone backend dependencies
```

---

## ⚡ Primary Inference Contract (`predict.py`)

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
   Feature Normalization (v2 Scalers)
(Strict post-purge training fit: Log1p + RobustScaler)
               │
               ▼
   LSTM World Model (z_t Latent Transition)
(Extracts 128-D causal temporal context from 30-window sequence)
               │
      ┌────────┴────────┐
      ▼                 ▼
Hazard Head v2     Stage Head v2
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

## 🔗 Tamper-Evident Hash Chain (`audit_chain.py`)

To answer the SIH theme (*Blockchain & Cybersecurity*) with genuine technical rigor, SHADOWCAT implements a lightweight, blockchain-inspired SHA-256 recursive hash chain over all verified project artifacts (Gate 0 verdicts, model checkpoint weights, contract diffs, and evaluation reports).

### Why Hash Chains vs Full Distributed Ledgers?
A full distributed consensus network (e.g. proof-of-work/stake) introduces disproportionate computational and deployment overhead for a single enterprise/perimeter security gateway. A recursive cryptographic hash chain provides **provable, post-hoc tamper detection** (tampering with any past report or model weight breaks the downstream hash tree), which is the exact property required for forensic evidence integrity.

---

## 🚀 Standalone Execution & Verification

### 1. Run Pipeline Smoke Test
Executes end-to-end inference against canonical UCS test data:
```bash
python backend/smoke_test.py
```

### 2. Verify Audit Chain Integrity
Walks all 7 chained blocks and validates byte-level SHA-256 matches:
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
