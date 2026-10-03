# SIH26153: SHADOWCAT Pipeline Architecture

**Theme:** Blockchain & Cybersecurity
**Organization:** National Technical Research Organisation (NTRO)

<img src="assets/Shadowcat_Architecture.png" alt="ShadowCat Architecture" style="width: 100%; max-height: 400px; object-fit: contain;">

Numbers match README.md and docs/FINAL_FINDINGS.md; *pending merge* marks results committed on branches not yet merged into main.

## 1. Unified Cyber State (UCS) & Data Ingestion
* **Feature extraction:** CICFlowMeter flows are aggregated into 1-minute windows, yielding a 406-dimensional UCS state $S_t$ (388 flow features, 6 presence masks, 12 packet features).
* **Packet-level coverage:** packet statistics from the project's custom PCAP parser are available for the SSH-Bruteforce, Botnet and DDOS-LOIC-UDP days; elsewhere `mask_has_packet_level_features` = 0.
* **Normalization:** Log1p + RobustScaler fitted on training windows only; purge + embargo at split boundaries.

## 2. World Model
* **Architecture:** a causal LSTM over a 30-window lookback learns $p(S_{t+1} | S_t)$ with a Gaussian head.
* **Forecasting status:** the autoregressive rollout does not beat a persistence baseline at any K = 1..5 (H* = 0), across a capacity sweep, an MDN head, 10-day data, CIC-IDS2017 data and a delta / Monte Carlo rollout (the last two *pending merge*). Root cause: the Gaussian NLL is mean-seeking, and the dataset's scripted attacks have few observable precursors.
* **Graph branch (held back):** a GraphSAGE fusion branch was explored; ablation (validation loss 1.666 vs 1.932) kept it out of the primary path. No prediction uses a GNN.

## 3. Heads and Explainability
* **Stacked & calibrated residual LSTM ensemble (37 LOEO folds):** one probability that an attack window occurs within the next 5 minutes (onset), and one for the current window (detection). LOEO mean F1: detection 0.9962 vs LR 0.9730; onset 0.9127 vs LR 0.8880. Onset is measured mostly on windows already inside an attack; on true precursors ROC-AUC is 0.625 vs LR 0.520 (*pending merge*). The gain over LR comes from the calibrated LR stage, not from the LSTM history (*pending merge*).
* **Stage head:** maps the world-model state to a MITRE ATT&CK tactic (0.8627 accuracy on its own test set, credential brute-force vs background). The dashboard shows a stage only when its top probability is > 0.5 for a real tactic; in a sweep over the dataset this never happened (*pending merge*).
* **Explainability:** Integrated Gradients on the stacked onset logit, plus a bounded counterfactual search on the onset probability.

## 4. Tamper-Evident Audit Ledger
* **Hyperledger Fabric (optional):** Go chaincode on a local test network (`shadowcat-notary-channel`) records a 4-stage prediction lineage and model provenance, and writes an `IncidentResponseRecord` on-chain for HIGH/CRITICAL records; measured median commit latency 2402.55 ms (3/3 confirmed).
* **SHA-256 hash chain (default when Fabric is unavailable):** every model load, alert and lineage record is chained; altering an earlier record breaks verification.

## 5. Operational Interface
* **Dashboard:** a Streamlit app that runs the pipeline on uploaded files or three real demo slices from CSE-CIC-IDS2018 and shows the onset and detection probabilities, Integrated Gradients, flow ranking, validation results and the audit chain. It processes files, not a live network feed.
