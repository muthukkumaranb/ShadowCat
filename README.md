# SHADOWCAT

> **An AI world model that forecasts network attacks before they happen, learning network state dynamics from traffic and mapping predictions to MITRE ATT&CK stages, fully offline.**

---

## Problem Statement Reference

- **ID:** `SIH26153`
- **Organization:** National Technical Research Organisation (NTRO)
- **Theme:** Blockchain & Cybersecurity
- **Core Mission:** Shift perimeter intrusion defense from reactive signature matching (0 lead time, post-compromise alerts) to **predictive pre-emptive forecasting** with quantified lead time, empirical leakage protection, and tamper-evident cryptographic provenance.

---

## Pipeline Architecture

```
                                  SHADOWCAT PIPELINE
                                  
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ 1. DATA TRACK: Unified Cyber State (UCS)                                    │
   │ Raw NetFlows (CSV) ──► Ingestion & Sanitization ──► 1-Min Window Aggregator │
   │                                                     (406-D UCS Feature S_t) │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ 2. ML TRACK: Causal Temporal World Model                                    │
   │ 30-Window Lookback ──► Log1p + RobustScaler ──► Causal LSTM (64-D Latent z_t) │
   │                                                 p(S_{t+1} | S_t) Dynamics   │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ 3. MULTI-HEAD FORECASTING & EXPLAINABILITY                                  │
   │         ┌────────────────────────────┴────────────────────────────┐         │
   │         ▼                                                         ▼         │
   │   Hazard Head v2                                            Stage Head v2   │
   │   Multi-step Risk Trajectory                                MITRE ATT&CK    │
   │   (t+1 .. t+4 Horizons)                                     (Credential)    │
   │         │                                                         │         │
   │         └────────────────────────────┬────────────────────────────┘         │
   │                                      ▼                                      │
   │                    Integrated Gradients & NLL Novelty                       │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ 4. INTEGRATION & AUDIT LAYER (backend/predict.py)                           │
   │ Unified predict() Contract  ◄──►  Tamper-Evident SHA-256 Hash Chain Ledger  │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ 5. OPERATIONAL DASHBOARD (frontend/app.py)                                  │
   │ Streamlit Glassmorphic UI: Risk Trajectories, Attribution & Benchmark Audits│
   └─────────────────────────────────────────────────────────────────────────────┘

   * Note on Graph Branch (ML2): GraphSAGE encoder & multimodal fusion were built and
     rigorously evaluated across 3 seeds. Per empirical evidence (validation loss favored
     temporal baseline 1.666 vs 1.932), the GNN was HELD BACK to avoid test-leakage.
     See ml2-full/GNN_FINAL/ml2/results/gnn_adopt_hold_decision.md.
```

---

## Repository Structure & Track Ownership

| Directory | Track / Ownership | Description |
| :--- | :--- | :--- |
| [`/data-engineering`](data-engineering/README.md) | **Data Engineering Track** | Gate 0 protocols, UCS extraction pipeline, 406-D feature normalization, chronological splits, and purge+embargo zones. |
| [`/ml1`](ml1/README.md) | **ML1 Track** | Causal LSTM world model ($z_t$), probabilistic next-state Gaussian transitions, Hazard Head v2, and Stage Head v2. |
| [`/ml2-full`](ml2-full/README.md) | **ML2 Track** | Graph representation learning and multimodal fusion ablation (`GNN_FINAL/ml2` is canonical; `gnn/` is exploratory/superseded). |
| [`/backend`](backend/README.md) | **Backend Track** | Authoritative `predict()` inference boundary, smoke test suite, and blockchain-inspired SHA-256 hash-chain audit ledger. |
| [`/frontend`](frontend/README.md) | **Frontend Track** | Interactive dark-mode Streamlit dashboard, dual-signal telemetry viewer, and scientific benchmark comparison panels. |

---

## Installation & Usage Guide

Follow these steps to set up the environment, verify the models, and launch the ShadowCat platform.

### 1. Clone the Repository
```bash
git clone https://github.com/muthukkumaranb/ShadowCat.git
cd ShadowCat
```

### 2. Set Up a Virtual Environment (Recommended)
It's highly recommended to use an isolated Python environment (Python 3.9+ is supported). 

**Using `venv`:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

**Using Conda:**
```bash
conda create -n shadowcat python=3.10 -y
conda activate shadowcat
```

### 3. Install Dependencies
Install the required packages for the data pipeline, PyTorch models, and Streamlit frontend:
```bash
pip install -r requirements.txt
```

### 4. Verify System Integrity
Before launching the application, ensure all backend components and models are functioning correctly.

Run the integration smoke test to validate the inference pipeline:
```bash
python backend/smoke_test.py
```

Then, verify the cryptographic hash chain to confirm no evaluation metrics or models have been tampered with:
```bash
python backend/verify_audit_chain.py
```

### 5. Launch the Dashboard
Start the Streamlit operational interface:
```bash
streamlit run frontend/app.py
```
*The dashboard will automatically open in your default web browser at `http://localhost:8501`. Here you can view the live risk trajectories and forensic benchmark panels.*

---

## Summary of Key Scientific Results

| Evaluation Metric / Milestone | Result / Finding | Status & Reference |
| :--- | :--- | :--- |
<<<<<<< Updated upstream
| **Real PCAP Telemetry (Option B)** | Genuine Scapy-extracted packet stats for 14-02-2018; 0 label leakage | [Extraction Report](data-engineering/data/ucs/PACKET_EXTRACTION_VERIFICATION.md) |
| **Hazard Forecasting ROC-AUC (v3)** | **0.789** (H=1), **0.843** (H=2), **0.770** (H=5) across LOEO 37 Folds | [Hazard Report v3](ml1/artifacts/lstm/hazard_head_v3/hazard_head_report_v3.md) |
| **Lagged Logistic Regression Baseline** | F1: 0.738 — Caveat: heavily degraded on Fold 16 (temporal shift) | [Gate 0 Report](data-engineering/gate0_leakage_report.md) |
| **GNN Multimodal Fusion Ablation** | Validation Loss: Temporal-Only (1.666) beats Fused (1.932) | **HOLD** — [GNN Decision](ml2-full/GNN_FINAL/ml2/results/gnn_adopt_hold_decision.md) |
| **Stage-Head Scope (v3)** | Validated on *Credential Access / Brute Force* vs background (v3) | [Stage Report v3](ml1/artifacts/lstm/stage_head_v3/stage_head_report.md) |
| **PC2 Significance Test (v3)** | Persistence baseline significantly outperforms autoregressive rollouts on PC2 | [PC2 Report v3](ml1/artifacts/lstm/probabilistic_world_model_v3/pc2_attack_significance_report.md) |
| **Inference Contract Verification (v3)**| DE vs ML1 Scaler parameters match bit-for-bit (0 mismatches, 406-D) | [Contract Diff v3](data-engineering/data/ucs/CONTRACT_DIFF_REPORT.md) |
| **Tamper-Evident Hash Chain** | 10 chained blocks covering all verified extraction, v4 models, and FPR-calibrated reports | [Audit Chain](backend/audit_chain.json) |

---

## Blockchain-Inspired Tamper-Evident Audit Trail

To meet the SIH **Blockchain & Cybersecurity** theme without introducing the excessive infrastructure overhead of a full distributed ledger for a single-deployment gateway, SHADOWCAT implements a **cryptographic SHA-256 recursive hash chain** ([`backend/audit_chain.py`](backend/audit_chain.py)).

- **Provable Tamper-Evidence:** Every Gate 0 contract diff, PyTorch model checkpoint, and evaluation report is hashed by its exact raw bytes and cryptographically chained ($Hash_i = SHA256(Block_i \parallel Hash_{i-1})$).
- **Post-Hoc Verification:** Any unauthorized alteration to historical evaluation metrics or model weights breaks downstream link hashes and is instantly caught by `verify_chain()`.
- **Live Demo Script:** Run `python backend/demo_tamper_detection.py` to observe simulated adversarial tampering detection on safe temporary copies.

---

## Known Limitations & Scope Disclosures

In accordance with scientific integrity and engineering transparency:
1. **Dataset Nature:** Evaluated on the CSE-CIC-IDS2018 testbed dataset; real-world enterprise zero-day generalization requires continuous fine-tuning.
2. **Network Topology:** The evaluation environment reflects a single simulated enterprise topology.
3. **Onset Forecasting Limit ($H=5$):** Multi-horizon forecasting at horizon $H=5$ minutes represents an epistemic benchmark limit under Leave-One-Episode-Out cross-validation.
4. **ATT&CK Stage Granularity:** Stage head currently evaluates high-fidelity discrimination for initial access / credential brute-force stages; full 14-tactic multi-stage ATT&CK classification is exploratory.
5. **Rollout Horizon Boundaries:** Autoregressive state rollout is empirically validated for depths $K=1 \dots 3$; depth $K=4$ is classified as informational/exploratory due to compounding drift.
6. **Telemetry Extraction (Option B):** Packet-level telemetry is now genuinely extracted for the 14-02-2018 and 02-03-2018 PCAPs without label leakage.
7. **Hazard FPR Constraints:** Hazard-head calibration is enforced using an explicit False Positive Rate ceiling (global τ=0.45, ≤5% false-alarm rate) to prevent alert fatigue.
8. **Botnet Detection Shortfall:** Botnet onset detection remains a disclosed, unresolved limitation. Even with real packet telemetry, the signal-to-noise ratio is too weak (ROC-AUC ~0.63), which is insufficient for reliable, low-FPR alerting. This has not been artificially 'solved' via F1-only threshold manipulation.

---

## Track Leads & Contributors

- **Data Engineering:** Unified Cyber State schema, ingestion pipeline, leakage purging & embargo protocols.
- **ML1 Modeling:** Causal LSTM state dynamics, probabilistic Gaussian heads, hazard/stage classification.
- **ML2 Modeling:** Dynamic graph generation, GraphSAGE architecture, multi-seed fusion ablation.
- **Backend Integration:** Production `predict()` interface, pipeline contract verification, cryptographic audit chain.
- **Frontend / UX:** Streamlit dashboard, Plotly telemetry charts, analyst guidance interface.
