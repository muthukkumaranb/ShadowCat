# SHADOWCAT - Final Presentation Slide Content

## Slide 1: Problem & Vision
* **Challenge:** Traditional intrusion defense relies on reactive signature matching (0 lead time, post-compromise alerts).
* **Solution:** **SHADOWCAT** - An AI world model that forecasts network attacks before they happen.
* **Mechanism:** Learns network state dynamics from traffic and maps predictions to MITRE ATT&CK stages, fully offline.

## Slide 2: Pipeline Architecture
* **Unified Cyber State (UCS):** Ingests raw NetFlows and Scapy-extracted packets into a 406-D feature vector ($S_t$) using 1-minute window aggregation.
* **Temporal World Model:** A Causal LSTM (30-window lookback) predicts network state dynamics $p(S_{t+1} | S_t)$.
* **Multi-Head Analysis:**
  * **Hazard Head:** Multi-step risk trajectories (t+1 .. t+4).
  * **Stage Head:** MITRE ATT&CK classification mapping.

## Slide 3: Real Packet Coverage & Explainability
* **Comprehensive Attack Coverage:** Fully upgraded pipeline now integrates real Scapy-extracted packet-level telemetry for **all 3 headline attacks** (SSH-Bruteforce, Botnet, and DDOS-LOIC-UDP) with zero label leakage.
* **Explainable AI:** Uses **Integrated Gradients** and **NLL Novelty** scoring to provide counterfactual explainability. Analysts can see exactly which feature perturbations (e.g., connection rates, rhythmic anomalies) triggered the forecast.

## Slide 4: Benchmark: Production Model vs. Logistic Regression Baseline
*Our production Stacked Residual LSTM Ensemble consistently outperforms the identical-feature linear baseline.*

| Task | Model | F1 | Precision | Recall | FPR |
|---|---|---|---|---|---|
| Detection | Production (Stacked Residual LSTM Ensemble) | **0.9962** | 0.9929 | 1.0000 | 0.0080 |
| Detection | Logistic Regression baseline (same features) | 0.9730 | 0.9730 | 0.9730 | 0.0000 |
| Onset (forecasting) | Production (Stacked Residual LSTM Ensemble) | **0.9127** | 0.9684 | 0.9054 | 0.0095 |
| Onset (forecasting) | Logistic Regression baseline (same features) | 0.8880 | 0.9459 | 0.8784 | 0.0045 |

**Caveat Disclosures:** 
1. The LR detection F1 of 0.973 is a mean across 37 folds where 36 score a perfect 1.0 and 1 fold (Fold 16, SSH-Bruteforce, 66 test windows) scores 0.0 — 0% recall, every attack window misclassified as benign.
2. The Stage Head, retrained on the full-packet-coverage dataset across all 3 attacks, achieves a verified **0.8627 accuracy**.

## Slide 5: Tamper-Evident Cryptographic Ledger
* **Provable Integrity:** To meet the SIH Blockchain & Cybersecurity theme without excess overhead, SHADOWCAT implements a SHA-256 recursive hash chain.
* **Immutable Provenance:** Every Gate 0 contract diff, PyTorch model checkpoint, and evaluation report is cryptographically chained.
* **Real-time Auditing:** Any unauthorized tampering with evaluation metrics or model weights instantly invalidates the chain, ensuring trust in the deployed gateway.
