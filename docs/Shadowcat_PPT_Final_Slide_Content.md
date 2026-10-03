# SHADOWCAT - Final Presentation Slide Content

Numbers match README.md and docs/FINAL_FINDINGS.md. Items marked *pending merge* come from committed
results files on branches not yet merged into main.

## Slide 1: Problem & Approach
* **Challenge:** Signature IDS alerts after the fact. SIH26153 asks for a world model that learns network state dynamics and estimates where an attack is heading.
* **SHADOWCAT:** A world model over 1-minute network states (406 features from CICFlowMeter flows and packet statistics), a stacked LOEO ensemble that scores **P(attack within the next 5 minutes)**, a stage head mapped to MITRE ATT&CK, and a tamper-evident audit trail, fully offline.
* **Honest status:** detection and onset are validated; multi-step forecasting is not (H* = 0).

## Slide 2: Pipeline Architecture
* **Unified Cyber State (UCS):** CICFlowMeter flows plus packet statistics from the project's custom PCAP parser, aggregated into 1-minute windows (406-D state $S_t$).
* **World model:** Causal LSTM (30-window lookback) learning $p(S_{t+1} | S_t)$.
* **Heads:**
  * **Stacked & calibrated residual LSTM ensemble (37 LOEO folds):** one onset probability for t+1..t+5, and one detection probability for the current window.
  * **Stage head:** MITRE ATT&CK stage, shown only when its top probability is > 0.5 for a real tactic.

## Slide 3: Explainability
* **Integrated Gradients** on the stacked onset logit: which inputs, in which of the 30 lookback windows, push the onset probability up or down.
* **Bounded counterfactual:** the smallest change, within observed data ranges, that would bring the onset probability below the alert threshold.

## Slide 4: Benchmark vs Logistic Regression (same features, 37 LOEO folds)

| Task | Model | F1 | Precision | Recall | FPR |
|---|---|---|---|---|---|
| Detection | Stacked ensemble | **0.9962** | 0.9929 | 1.0000 | 0.0080 |
| Detection | Logistic Regression | 0.9730 | 0.9730 | 0.9730 | 0.0000 |
| Onset | Stacked ensemble | **0.9127** | 0.9684 | 0.9054 | 0.0095 |
| Onset | Logistic Regression | 0.8880 | 0.9459 | 0.8784 | 0.0000 |

At matched FPR (~5%, 412 pooled test windows): detection 0.958 vs 0.871, onset 0.936 vs 0.829 (*pending merge*).

**Caveats:**
1. LR scores 1.0 on 36/37 folds and 0.0 on Fold 16 (SSH-Bruteforce).
2. Onset F1 0.9127 is measured mostly on windows already inside an attack: of 984 onset-positive windows only 71 are true precursors. On precursors the stacked model reaches ROC-AUC 0.625 vs LR 0.520 (*pending merge*).
3. The LSTM residual adds no measurable gain over the stacked model's calibrated LR stage; the improvement over LR is not temporal learning (*pending merge*).
4. The world model's rollout does not beat a persistence baseline at any K = 1..5 (H* = 0), confirmed by a capacity sweep, an MDN head, 10-day data, CIC-IDS2017 data and a delta / Monte Carlo rollout.

## Slide 5: Tamper-Evident Ledger
* **Hyperledger Fabric (optional, local Docker network):** Go chaincode records prediction lineage and model provenance; for HIGH/CRITICAL records it writes an `IncidentResponseRecord` on-chain.
* **SHA-256 hash chain (default when Fabric is not running):** every model load and alert is chained; any alteration breaks the chain.
