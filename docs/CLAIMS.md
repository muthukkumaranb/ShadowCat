# ShadowCat Claims Sheet

This document explicitly summarizes the rigorously verified capabilities and limitations of the ShadowCat system, grounded solely in the committed experimental artifacts from LOEO cross-validation.

## 1. What We Can Claim (Verified)

### A. Multi-Stage Stacked Onset Detection (Phase A)
- **Claim:** The stacked, temperature-calibrated residual LSTM ensemble reliably discriminates attack windows from benign traffic within the current minute.
- **Evidence:** 37-fold LOEO validation across all multi-episode families confirms an onset ROC-AUC of >0.99.

### B. Family & ATT&CK Stage Classification (Phase B)
- **Claim:** The system can classify the family and MITRE ATT&CK tactic of an ongoing attack across known multi-episode families.
- **Evidence:** The logistic regression classifier achieves a LOEO Macro-F1 of 0.4608 for family and 0.6402 for ATT&CK tactic on `Botnet`, `SSH-Bruteforce`, and `DDOS-LOIC-UDP`. (Note: Single-episode families cannot be evaluated under LOEO).

## 2. What We Cannot Claim (Refuted by Phase C)

### A. Anticipatory Forecasting (K > 0)
- **Refutation:** The system possesses **ZERO** anticipatory forecasting skill for cyber attacks. It cannot predict attacks before they happen.
- **Evidence (Direct Heads):** Direct multi-horizon logistic regression heads score at chance (~0.42-0.49 ROC-AUC) when evaluated strictly on precursor windows. Their apparent high performance across horizons is merely an artifact of predicting that an ongoing attack will continue (persistence).
- **Evidence (State-Space Rollout):** The LSTM-Gaussian world model fails to beat simple persistence in full-state RMSE/MAE on transition windows. The previously reported 0.79 ROC-AUC for rollout on change windows was a misleading artifact of inversely predicting the current state, and the model entirely fails to separate true onset transitions from benign stationary traffic.
