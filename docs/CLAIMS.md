# ShadowCat Claims Sheet

This document serves as the authoritative, verified claims sheet for ShadowCat (SIH26153). Every value is backed by a committed JSON artifact produced by a reproducible script and verified in git history.

---

## 1. Verified & Evaluated Claims Table

| Claim | Value | Source File | Commit | Status |
|---|---|---|---|---|
| **Stacked Detection Performance (Matched 5% FPR)** | F1 = 0.9582, Precision = 0.9198, Recall = 1.0000, FPR = 0.0494, ROC-AUC = 1.0000 (Threshold = 0.4014). Note: Pooled ROC-AUC is 1.0; F1 0.958 is at the 5%-FPR operating point; F1 at 0.5 is 0.987. CIC-IDS2018 detection is a known-easy in-distribution task. | `evaluation/benchmark/stacked_benchmark_results.json` | `a0687cf` | **validated** |
| **Linear Baseline Detection Performance (Matched 5% FPR)** | F1 = 0.8711, Precision = 0.9058, Recall = 0.8389, FPR = 0.0494, ROC-AUC = 0.9743 (Threshold = 0.0523) | `evaluation/benchmark/stacked_benchmark_results.json` | `a0687cf` | **updated**: computed on an earlier feature version; re-run on the final dataset gives F1 0.9582 / ROC-AUC 1.0000 (see Final-Dataset LR Baseline row) |
| **Stacked Onset Performance (Matched 5% FPR)** | F1 = 0.9362, Precision = 0.9277, Recall = 0.9448, FPR = 0.0482, ROC-AUC = 0.9621 (Threshold = 0.3749) | `evaluation/benchmark/stacked_benchmark_results.json` | `a0687cf` | **validated** |
| **Linear Baseline Onset Performance (Matched 5% FPR)** | F1 = 0.8294, Precision = 0.9118, Recall = 0.7607, FPR = 0.0482, ROC-AUC = 0.9148 (Threshold = 0.0951) | `evaluation/benchmark/stacked_benchmark_results.json` | `a0687cf` | **updated**: computed on an earlier feature version; re-run on the final dataset gives F1 0.9198 / ROC-AUC 0.9518 (see Final-Dataset LR Baseline row) |
| **LSTM Temporal Ablation** | Permuting history or repeating the last state leaves ROC-AUC and F1 essentially unchanged: on this dataset the calibrated LR stage of the stacked model carries the signal, and the LSTM residual adds little on top. | `evaluation/ablation/ABLATION.md` | `212ed88` | **validated** |
| **Final-Dataset LR Baseline (37-fold LOEO)** | Detection: F1 0.9582 at 5% FPR, ROC-AUC 1.0000 (stacked: same). Onset: F1 0.9198, ROC-AUC 0.9518 (stacked: 0.9362 / 0.9621, ahead). | `evaluation/leakage_checks/leakage_results.json` | `5bc481d` | **validated** |
| **Threshold Chosen Without Test Labels (37-fold LOEO)** | Stacked, per-fold validation threshold: detection F1 0.9898 (test FPR 0.0%), onset F1 0.9067 (test FPR 0.4%). | `evaluation/leakage_checks/leakage_results.json` | `5bc481d` | **validated** |
| **Unseen-Attack-Family Stress Test (leave-one-day-out, retrained)** | Supervised pooled ROC-AUC: detection 0.666 (LR 0.631), onset 0.633 (LR 0.618). World-model deviation on held-out families: 0.707. Ambiguous 'Benign'-type positives affect several days. | `evaluation/leakage_checks/leakage_results.json` | `5bc481d` | **research direction** |
| **World-Model Next-State Deviation (Pooled LOEO)** | ROC-AUC = 0.8201, PR-AUC = 0.7721 (40 LOEO folds, 312 positives / 768 test windows) | `evaluation/deviation/deviation_results.json` | `20a8b7d` | **validated** |
| **World-Model Next-State Deviation (Pooled Family-Out)** | ROC-AUC = 0.7070, PR-AUC = 0.4855 (Held-out attack days, 312 positives / 1,730 test windows) | `evaluation/deviation/deviation_results.json` | `20a8b7d` | **promising** |
| **Precursor Lead Time & Recall (Matched 5% FPR)** | Recall = 40.0% (6/15 precursors detected), Mean Lead Time = 3.00 min (range 1–5 min), FPR = 0.0444 | `evaluation/precursor/precursor_results.json` | `5513678` | **early signal (15 precursors)** |
| **Linear Baseline Precursor Detection** | Recall = 20.0% (3/15 precursors detected), Mean Lead Time = 3.33 min (range 2–5 min), FPR = 0.0444 | `evaluation/precursor/precursor_results.json` | `5513678` | **reference baseline** |
| **Stage Prediction Horizon from World-Model Rollout ($H^*$)** | $H^* = 0$ (rollout stage F1 $\approx 0.45\text{--}0.55$; persistence reference F1 $\approx 0.91\text{--}0.95$ across $K=1..5$) | `evaluation/rollout_v2/results.json` | `10b3578` | **measured horizon (data-limited)** |
| **State-Space Rollout Horizon ($H^*_{state}$)** | $H^*_{state} = 0$ (on RMSE and MAE the rollout stays within the persistence reference; bootstrap 95% CIs include or exceed 0) | `evaluation/k5/state_rollout_results.json` | `b05aeab` | **measured horizon (data-limited)** |
| **Direct Multi-Horizon Onset Heads ($K=1..5$)** | Overall ROC-AUC 0.95–0.99, mostly from attacks already in progress (>98% of positives). On the few true precursor windows, ROC-AUC is 0.41–0.48. | `evaluation/k5/direct_heads_results.json` | `b05aeab` | **measured horizon (data-limited)** |
| **Transition / Change-Window Prediction** | Change windows are rare (3–24 per K). Rollout change-window AUC 0.79 at K=5, where the inverse-persistence reference reaches 1.00; onset vs benign change-window CIs include 0.5 at every K. | `evaluation/k5/change_skill_results.json` | `b05aeab` | **measured horizon (data-limited)** |
| **Current-Window ATT&CK Stage Classification (LOEO)** | Tactic Macro-F1 = 0.6402 (Credential Access: 0.7101, C2: 0.6721, Impact: 0.5385); Family Macro-F1 = 0.4608. Note: LOEO covers 3/6 families; DDOS-LOIC-UDP family F1 = 0. | `evaluation/family/family_results.json` | `f4d6a9c` | **validated (3 of 6 families evaluable)** |
| **Dataset Provenance & Window Count** | 6 days, 2,787 windows (`ucs_windows_models_v1.parquet`); 10-day rebuild failed check 4 (226/406 features differed) | `data-engineering/data/ucs/REBUILD_CHECK.json` | `285bb8d` | **validated** |
| **Cryptographic Audit Chain Integrity** | Tamper-evident Ed25519 digital signature notarization over prediction lineages and model hashes | `backend/audit_chain.json`, `backend/tests/test_audit_chain.py` | `6bc1335` | **validated** |
| **Intra-Attack Forecasting (duration / end / intensity)** | Not evaluable on CIC-IDS2018: 40 attack runs, only 12 last ≥ 6 minutes (stop rule: ≥ 20 required); no model was trained | `evaluation/intra_attack/runs.json` | `fb799dc` | **not evaluable** |
| **Probabilistic K-step Forecast ($H^*_{prob}$)** | $H^*_{prob} = 0$. On CRPS the world model's predictive distribution stays within the noised-persistence / AR(1) references (e.g. F5 − best = +17.4 [+9.4, +29.5] at K=1); it stays close to climatology; 90% intervals cover 0.68–0.70. Same with no early stopping. | `evaluation/prob_forecast/prob_results.json` | `6ba39ee` | **measured horizon (data-limited)** |
| **World-Model Rollout as Early Warning** | On windows benign at t, rolled-out states give P(attack at t+K) with ROC-AUC 0.18 (K=1) to 0.27 (K=5); the Markov reference has the lower Brier score at every K (CI excludes 0) | `evaluation/prob_forecast/binary_results.json` | `6ba39ee` | **measured horizon (data-limited)** |

---

## 2. Reporting Guidance

How to describe each result accurately, so every statement matches a committed artifact:

1. **K-step rollout:** describe it as implemented and evaluated, with a measured predictability horizon of $H^* = 0$
   on CIC-IDS2018 (stage, state-space and probabilistic protocols). The horizon is set by the data: scripted attacks
   with very few precursor windows, and a mean-seeking Gaussian objective.
2. **Onset from rolled-out states:** report that onset is estimated from the observed window, not from rolled-out
   states (on rolled-out states the onset classifier reaches ROC-AUC $\le 0.4561$).
3. **Direct multi-horizon heads:** report their high overall ROC-AUC together with the precursor-only result
   (0.41–0.48), since most positives are attacks already in progress.
4. **Onset F1:** use the committed LOEO values, **0.9362** at 5% FPR (stacked ensemble) and **0.9067** with the
   threshold chosen without test labels. An early "0.835" figure came from an interim report and is not a
   committed result.
5. **Hazard heads (`hazard_head_v3`):** treated as an earlier experiment. They reproduce only with the PCA procedure
   they were trained with, so the dashboard uses the calibrated stacked ensemble instead.
6. **Dataset size:** models are trained and validated on the verified 6-day dataset (`ucs_windows_models_v1.parquet`,
   2,787 windows). A 10-day rebuild did not reproduce the canonical features (226 of 406 differed; `REBUILD_CHECK.json`),
   so it is described as an experiment.
7. **Future-stage prediction:** stage is reported for the current window (tactic macro-F1 0.640). Rolled-out stage
   F1 ($\approx 0.55$) is part of the forecasting research, alongside the persistence reference ($\approx 0.91\text{--}0.95$).
8. **LR baseline:** use the final-dataset baseline (detection 0.9582, onset 0.9198); the stacked model matches it on
   detection and is ahead on onset. The earlier 0.8711 / 0.8294 used an earlier feature version.
9. **Unseen attack families:** pair the supervised leave-one-day-out result (0.666 / 0.633) with the world-model
   deviation score (0.707 family-out), which is the component designed for unseen attacks.
