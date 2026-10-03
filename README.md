# SHADOWCAT

> **A network-traffic world model with a validated attack detector and onset estimator: it learns 1-minute network state dynamics from CICFlowMeter flows, scores P(attack within the next 5 minutes) with a stacked LOEO ensemble, and keeps a tamper-evident audit trail, fully offline. Its multi-step rollout does not yet beat a persistence baseline (H* = 0).**

---

## Deliverables

- **Demo video:** https://youtu.be/c_4-A5syubs
- **Architecture diagram:** [docs/assets/Shadowcat_Architecture.png](docs/assets/Shadowcat_Architecture.png) ([PDF](docs/assets/ShadowCat_Architecture.pdf))
- **Slides (content):** [docs/Shadowcat_PPT_Final_Slide_Content.md](docs/Shadowcat_PPT_Final_Slide_Content.md)
- **Architecture document:** [docs/SIH26153_Architecture_Final.md](docs/SIH26153_Architecture_Final.md)
- **Final findings:** [docs/FINAL_FINDINGS.md](docs/FINAL_FINDINGS.md)

---

## Problem Statement Reference

- **ID:** `SIH26153`
- **Organization:** National Technical Research Organisation (NTRO)
- **Theme:** Blockchain & Cybersecurity
- **Core Mission:** Shift perimeter intrusion defense from reactive signature matching (0 lead time, post-compromise alerts) to learned, leakage-controlled estimation of attack onset from traffic, with tamper-evident cryptographic provenance. Onset F1 0.9127 is measured mostly on windows already inside an attack; on true precursors (no attack yet, attack within 5 minutes) the stacked model reaches ROC-AUC 0.625 vs LR 0.520 (*pending merge*, `feature/deviation-evidence`).

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
   │   Stacked Residual LSTM Ensemble                            Stage Head v3   │
   │   One onset probability:                                    MITRE ATT&CK    │
   │   P(attack within t+1..t+5)                                 (Credential)    │
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
   │ Streamlit UI: Onset Probability, Attribution & Benchmark Audits             │
   └─────────────────────────────────────────────────────────────────────────────┘

   * Note on Graph Branch (ML2): GraphSAGE fusion (real per-window graph construction) is now
     wired in as an explicitly labeled experimental/held-back output, separate from the
     primary verified forecast. The underlying ablation evidence (validation loss 1.666 vs 1.932)
     remains the reason it's not in the primary path.
     See ml2-full/GNN_FINAL/ml2/results/gnn_adopt_hold_decision.md.
```

---

## Repository Structure & Track Ownership

| Directory | Track / Ownership | Description |
| :--- | :--- | :--- |
| [`/data-engineering`](data-engineering/README.md) | **Data Engineering Track** | Gate 0 protocols, UCS extraction pipeline, 406-D feature normalization, chronological splits, and purge+embargo zones. |
| [`/ml1`](ml1/README.md) | **ML1 Track** | Causal LSTM world model ($z_t$), probabilistic next-state Gaussian transitions, Stacked Residual LSTM Ensemble, and Stage Head v3. |
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

### Optional: Hyperledger Fabric Notarization
The blockchain notarization layer requires Docker Desktop (docker.com) to be installed and running.
This is OPTIONAL — if Docker isn't available, the pipeline automatically falls back to a SHA-256
hash-chain notarization (`notarized_via: "sha256_fallback"`), and all forecasting/scoring/graph
functionality works identically either way. Only the on-chain audit trail is affected.

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
*The dashboard will automatically open in your default web browser at `http://localhost:8501`. Load one of the three real demo slices on the Ingestion page; the onset probability, explanation and validation results update across the pages.*

### Optional: Reproducing the Hyperledger Fabric Layer

**Note:** This step is purely optional and for advanced verification. The full SHADOWCAT deliverable (world model, inference, explainability, offline demo, and benchmarks) works completely out-of-the-box via the automatic SHA-256 fallback mechanism.

By default, the backend scripts map the Windows host path `D:\sih2026` to the Fabric Docker containers. You can configure this path by setting the `SHADOWCAT_FABRIC_HOST_ROOT` environment variable. If unset, it gracefully defaults to `D:\sih2026` (preserving byte-for-byte identical behavior).

To stand up the real Fabric test-network layer yourself:
1. Ensure **Docker Desktop** is installed and running.
2. Install dependencies and normalize line endings (required once per environment): `apt-get update && apt-get install -y docker.io docker-compose-v2 jq curl dos2unix`, then run `dos2unix` on the shell scripts under `fabric-samples/test-network/` as `run_network.sh` does.
3. From within a bash-compatible terminal, navigate to the test-network directory and bring up the network with CA nodes and the required channel:
   ```bash
   cd fabric-experiment/fabric-samples/test-network
   ./network.sh up createChannel -c shadowcat-notary-channel -ca
   ```
4. Deploy the `shadowcat_notary` Go chaincode to the channel:
   ```bash
   ./network.sh deployCC -ccn shadowcat_notary -ccp ../../chaincode/shadowcat_notary -ccl go -c shadowcat-notary-channel
   ```

---

## Benchmark: Production Model vs. Logistic Regression Baseline

| Task | Model | F1 | Precision | Recall | FPR |
|---|---|---|---|---|---|
| Detection | Production (Stacked Residual LSTM Ensemble) | **0.9962** | 0.9929 | 1.0000 | 0.0080 |
| Detection | Logistic Regression baseline (same features) | 0.9730 | 0.9730 | 0.9730 | 0.0000 |
| Onset (forecasting) | Production (Stacked Residual LSTM Ensemble) | **0.9127** | 0.9684 | 0.9054 | 0.0095 |
| Onset (forecasting) | Logistic Regression baseline (same features) | 0.8880 | 0.9459 | 0.8784 | 0.0000 |

**Note**: The LR detection F1 of 0.973 is a mean across 37 folds where 36 score a perfect 1.0 and 1 fold (Fold 16, SSH-Bruteforce, 66 test windows) scores 0.0 — 0% recall, every attack window misclassified as benign.

LR scores 1.0 on 36/37 folds and 0.0 on Fold 16 (SSH-Bruteforce).

**Matched false-positive rate (~5%)** on the same 412 pooled LOEO test windows, threshold chosen by the same procedure for both models — *pending merge* (`fix/demo-integrity`, `evaluation/benchmark/stacked_benchmark_results.json`):

| Task | Model | F1 | Precision | Recall | FPR |
|---|---|---|---|---|---|
| Detection | Stacked Residual LSTM Ensemble | **0.958** | 0.920 | 1.000 | 0.049 |
| Detection | Logistic Regression baseline | 0.871 | 0.906 | 0.839 | 0.049 |
| Onset | Stacked Residual LSTM Ensemble | **0.936** | 0.928 | 0.945 | 0.048 |
| Onset | Logistic Regression baseline | 0.829 | 0.912 | 0.761 | 0.048 |

---

## Summary of Key Scientific Results

| Evaluation Metric / Milestone | Result / Finding | Status & Reference |
| :--- | :--- | :--- |
| **Real PCAP Telemetry (Option B)** | Packet stats extracted with the project's custom PCAP parser for SSH-Bruteforce (14-02-2018), DDOS-LOIC-UDP (21-02-2018), and Botnet (02-03-2018); 0 label leakage | [Extraction Report](data-engineering/data/ucs/PACKET_EXTRACTION_VERIFICATION.md) |
| **Hazard Forecasting ROC-AUC (v3)** | **0.789** (H=1), **0.843** (H=2), **0.770** (H=5) across LOEO 37 Folds; reproduces exactly only with the per-fold PCA refit it was trained with, not through the deployed input path, so the dashboard does not show it (*pending merge*, `fix/demo-integrity`, `hazard_v3_results.json`) | [Hazard Report v3](ml1/artifacts/lstm/hazard_head_v3/hazard_head_report_v3.md) |
| **Lagged Logistic Regression Baseline** | F1: 0.9730 — Caveat: 0% recall, every attack window misclassified as benign on Fold 16 | [Gate 0 Report](data-engineering/gate0_leakage_report.md) |
| **GNN Multimodal Fusion Ablation** | Validation Loss: Temporal-Only (1.666) beats Fused (1.932) | **HOLD** — [GNN Decision](ml2-full/GNN_FINAL/ml2/results/gnn_adopt_hold_decision.md) |
| **Stage-Head Scope (v3)** | Validated on *Credential Access / Brute Force* vs background (v3), 0.8627 accuracy | [Stage Report v3](ml1/artifacts/lstm/stage_head_v3/reeval_packetcov/stage_head_report.md) |
| **PC2 Significance Test (v3)** | Persistence baseline significantly outperforms autoregressive rollouts on PC2 | [PC2 Report v3](ml1/artifacts/lstm/probabilistic_world_model_v3/pc2_attack_significance_report.md) |
| **Inference Contract Verification (v3)**| DE vs ML1 Scaler parameters match bit-for-bit (0 mismatches, 406-D) | [Contract Diff v3](data-engineering/data/ucs/CONTRACT_DIFF_REPORT.md) |
| **Tamper-Evident Hash Chain** | 27 chained blocks covering all verified extraction, v4 models, and FPR-calibrated reports | [Audit Chain](backend/audit_chain.json) |

---

## Forecasting Advancement: Predictability Horizon Investigation

- A dedicated investigation was run to determine the model's genuine multi-step forecasting skill, separate from the single-step detection/onset benchmarks reported above.
- Rolling-origin evaluation (5 chronological splits) found the world model's forecast F1 does not exceed a trivial persistence baseline at any tested horizon (K=1 through K=5). Predictability Horizon H* = 0. This was independently reproduced multiple times, including by direct re-execution of the evaluation script against freshly trained checkpoints.
- Standard remediation attempts were tried and exhausted: rare-class data augmentation (latent-space jitter), narrower horizon scoping, task reframing (regression and lagged-classification instead of multi-step rollout), and a hyperparameter sweep of the base world model (hidden size, depth, learning rate, training length). None of these moved the result; the model consistently collapses to predicting the majority "Unknown/Other" class regardless of configuration.
- Root cause identified: the world model is trained with a Gaussian negative log-likelihood loss, which is mean-seeking. It is mathematically rewarded for predicting a smoothed average next-state rather than preserving rare attack-state signal, since rare states resemble noise relative to that average. This holds independent of model capacity.
- An architectural change (Mixture Density Network head, 5 Gaussian mixture components, replacing the single-Gaussian output) was implemented and evaluated to test whether a multimodal output distribution could preserve the signal a single Gaussian collapses. Results: F1 at K=1 through K=5 (0.5530, 0.5527, 0.5523, 0.5520, 0.5517) still does not beat the persistence baseline (0.9517 to 0.9059 across the same horizons); H* remains 0.
- A follow-up diagnostic found that the mixture components do differentiate internally between benign and attack windows (for example, one component's average weight rises from 5.03% on benign windows to 17.53% on attack windows), but this distinction is lost once the mixture is collapsed to a single expected value for downstream classification, which is required by the current stage-head architecture.
- A third remediation path was evaluated: data volume was expanded from 3 to 10 labeled days (4,545 total windows at commit a0ae7bc, full coverage of Credential Access, Impact, Discovery, and Command and Control classes). Re-evaluating the MDN world model on this expanded data still gives H* = 0. Note: `main` currently holds the 6-day dataset (2,787 windows); a rebuild of the 10-day dataset with main's pipeline is *pending merge* on `fix/restore-10day-dataset` and did not reproduce a0ae7bc's feature values (`REBUILD_CHECK.json`).
- Conclusion: all three independent remediation paths (capacity/hyperparameters, architecture, data volume) now agree, which is strong evidence the ceiling is a structural property of the single-step world-model objective itself, not any of these three factors. Passing the full predicted distribution (or a sample from it) into the stage head, instead of collapsing it to a mean, is identified as the next concrete direction, not yet implemented.
- This investigation did not change the validated Detection (F1 0.9962) and Onset (F1 0.9127) benchmarks reported above, which measure a different task (single-step classification, not multi-step autoregressive forecasting).
- Closing summary (details in [docs/FINAL_FINDINGS.md](docs/FINAL_FINDINGS.md)):
  - H* = 0 at K = 1..5, confirmed by the capacity sweep, the MDN head, the 10-day data and the CIC-IDS2017 exploration (combined data: model 0.2136 vs persistence 0.6051, *pending merge*); a delta-parameterised and 32-sample Monte Carlo rollout also gives H* = 0 in all four configurations (*pending merge*, `feature/rollout-delta-mc`).
  - Root cause: the Gaussian NLL objective is mean-seeking, and the scripted CSE-CIC-IDS2018 attacks have almost no observable precursors.
  - Onset F1 0.9127 is measured mostly on windows already inside an attack: of 984 onset-positive windows only 71 are true precursors; on precursors the stacked model reaches ROC-AUC 0.625 vs LR 0.520 (*pending merge*, `feature/deviation-evidence`).
  - The LSTM residual adds no measurable gain over the stacked model's own calibrated LR stage (history shuffled or removed gives the same F1, *pending merge*); the improvement over LR is not temporal learning.
  - The earlier interim claim that "more data eliminated H* = 0" is retracted: it compared against a day-of-week predictor, not the world model.

---

## Hyperledger Fabric Two-Tier Notarization Architecture

To meet the SIH **Blockchain & Cybersecurity** theme, SHADOWCAT implements a real two-tier notarization design, with a live blockchain as the primary path and a cryptographic hash chain as a resilient fallback.

- **Primary Tier (Hyperledger Fabric):** The system integrates a real Go chaincode deployed on a local Fabric test network (channel: `shadowcat-notary-channel`). It exposes functions like `RecordPredictionLineageWithTarget`, `NotarizeAlert`, and `RecordModelProvenance`. The chaincode features autonomous incident response: when severity is HIGH or CRITICAL, the Go chaincode natively creates an `IncidentResponseRecord` (e.g., `ISOLATE_HOST:<node_id>`) in the same transaction without Python-layer intervention. Real measured Fabric commit latency is a median of 2402.55ms (3/3 on-chain confirmed).
- **Fallback Tier (SHA-256 Hash Chain):** If the Fabric RPC path is severed, the system seamlessly falls back to a **cryptographic SHA-256 recursive hash chain** ([`backend/audit_chain.py`](backend/audit_chain.py)). Every Gate 0 contract diff, PyTorch model checkpoint, and evaluation report is hashed and chained ($Hash_i = SHA256(Block_i \parallel Hash_{i-1})$). A documented fallback test (`scratch/test_kill_network_fallback.py`) proves failover is clean and uninterrupted.
- **Live Demo Script:** Run `python backend/demo_fabric_tamper.py` to observe simulated adversarial tampering detection specifically for the Fabric path.

---

## Known Limitations & Scope Disclosures

In accordance with scientific integrity and engineering transparency:
1. **Dataset Nature:** Evaluated only on the CSE-CIC-IDS2018 testbed dataset. No claim is made that the models work on unseen attack types or other networks; the leave-one-family-out world-model deviation result (*pending merge*, `feature/deviation-evidence`) is the only unseen-family evidence.
2. **Network Topology:** The evaluation environment reflects a single simulated enterprise topology.
3. **Onset window ($H=5$):** The onset model gives one probability for "an attack window within the next 5 minutes"; it does not produce per-minute horizons.
4. **ATT&CK Stage Granularity:** Stage head currently evaluates high-fidelity discrimination for credential brute-force stages vs background; full 14-tactic multi-stage ATT&CK classification is exploratory: Discovery and Command & Control have zero test support in LOEO splits, while Impact-stage attacks (19 test windows) are not detected by the current Stage Head (0% recall in both v2 and v3 evaluations; the model defaults these to Unknown/Other).
5. **Rollout Horizon Boundaries:** Autoregressive state rollout does not beat a persistence baseline at any depth K = 1..5 (H* = 0); no rollout depth is validated.
6. **Telemetry Extraction (Option B):** Packet-level telemetry is now genuinely extracted for the SSH-Bruteforce (14-02-2018), DDOS-LOIC-UDP (21-02-2018), and Botnet (02-03-2018) PCAPs without label leakage.
7. **Alert threshold:** The dashboard alerts when the onset probability is >= 0.5 (the threshold of the reported per-fold F1). In a sweep over every dataset window, 3.8% of benign windows alert (these windows were in training for 36 of 37 folds, so this is not a held-out false-alarm rate; *pending merge*, `fix/demo-integrity`, `evaluation/demo_sweep_results.json`). No ≤5% held-out false-alarm guarantee is claimed.
8. **Botnet Detection Shortfall:** Botnet onset detection remains a disclosed, unresolved limitation. Even with real packet telemetry, the signal-to-noise ratio is too weak (ROC-AUC ~0.63), which is insufficient for reliable, low-FPR alerting. This has not been artificially 'solved' via F1-only threshold manipulation.
9. **GraphSAGE Fusion:** Real per-window graph construction is now wired in as an explicitly labeled experimental/held-back output, separate from the primary verified forecast; the underlying ablation evidence (validation loss 1.666 vs 1.932) remains the reason it's not in the primary path.
10. **Hyperledger Fabric Local Dependency:** The primary Hyperledger Fabric notarization path is not fully plug-and-play upon cloning the repository. It requires a local Fabric test-network running via Docker with the `shadowcat-notary-channel` and chaincode deployed, as the Windows path (`D:\sih2026\fabric-experiment`) is hardcoded in `backend/fabric_bridge.py`. Without this local network running, every notarization call gracefully and automatically falls back to the SHA-256 hash chain (`notarized_via="sha256_fallback"`).
11. **Predictability Horizon:** Multi-step autoregressive forecasting skill beyond the validated single-step benchmarks remains unresolved (H* = 0 across all tested configurations and one architectural remediation), and is disclosed as an open problem rather than masked.
12. **Expanded Dataset Limitations:** 5 of the 10 labeled days used for this investigation do not have genuine packet-level extraction (CSV-only), unlike the 3 original "Option B" verified days.

---

## Track Leads & Contributors

- **Data Engineering:** Unified Cyber State schema, ingestion pipeline, leakage purging & embargo protocols.
- **ML1 Modeling:** Causal LSTM state dynamics, probabilistic Gaussian heads, hazard/stage classification.
- **ML2 Modeling:** Dynamic graph generation, GraphSAGE architecture, multi-seed fusion ablation.
- **Backend Integration:** Production `predict()` interface, pipeline contract verification, cryptographic audit chain.
- **Frontend / UX:** Streamlit dashboard, Plotly telemetry charts, analyst guidance interface.
