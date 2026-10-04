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
| **LSTM Temporal Ablation** | Permuting history or repeating the last state yields no measurable degradation in ROC-AUC or F1, proving the LSTM residual extracts no meaningful temporal signal. | `evaluation/ablation/ABLATION.md` | `212ed88` | **validated** |
| **Final-Dataset LR Baseline (37-fold LOEO)** | Detection: F1 0.9582 at 5% FPR, ROC-AUC 1.0000 (stacked: same). Onset: F1 0.9198, ROC-AUC 0.9518 (stacked: 0.9362 / 0.9621, ahead). | `evaluation/leakage_checks/leakage_results.json` | `5bc481d` | **validated** |
| **Threshold Chosen Without Test Labels (37-fold LOEO)** | Stacked, per-fold validation threshold: detection F1 0.9898 (test FPR 0.0%), onset F1 0.9067 (test FPR 0.4%). | `evaluation/leakage_checks/leakage_results.json` | `5bc481d` | **validated** |
| **Unseen-Attack-Family Stress Test (leave-one-day-out, retrained)** | Supervised pooled ROC-AUC: detection 0.666 (LR 0.631), onset 0.633 (LR 0.618). World-model deviation on held-out families: 0.707. Ambiguous 'Benign'-type positives affect several days. | `evaluation/leakage_checks/leakage_results.json` | `5bc481d` | **open problem** |
| **World-Model Next-State Deviation (Pooled LOEO)** | ROC-AUC = 0.8201, PR-AUC = 0.7721 (40 LOEO folds, 312 positives / 768 test windows) | `evaluation/deviation/deviation_results.json` | `20a8b7d` | **validated** |
| **World-Model Next-State Deviation (Pooled Family-Out)** | ROC-AUC = 0.7070, PR-AUC = 0.4855 (Held-out attack days, 312 positives / 1,730 test windows) | `evaluation/deviation/deviation_results.json` | `20a8b7d` | **weak** |
| **Precursor Lead Time & Recall (Matched 5% FPR)** | Recall = 40.0% (6/15 precursors detected), Mean Lead Time = 3.00 min (range 1–5 min), FPR = 0.0444 | `evaluation/precursor/precursor_results.json` | `5513678` | **weak** |
| **Linear Baseline Precursor Detection** | Recall = 20.0% (3/15 precursors detected), Mean Lead Time = 3.33 min (range 2–5 min), FPR = 0.0444 | `evaluation/precursor/precursor_results.json` | `5513678` | **weak** |
| **Stage Prediction Horizon from World-Model Rollout ($H^*$)** | $H^* = 0$ (Model F1 $\approx 0.45\text{--}0.55$ vs Persistence F1 $\approx 0.91\text{--}0.95$ across $K=1..5$) | `evaluation/rollout_v2/results.json` | `10b3578` | **not achieved** |
| **State-Space Rollout Horizon ($H^*_{state}$)** | $H^*_{state} = 0$ (Model does not beat persistence on both RMSE and MAE; bootstrap 95% CI strictly includes/exceeds 0) | `evaluation/k5/state_rollout_results.json` | `b05aeab` | **invalidated** |
| **Direct Multi-Horizon Onset Heads ($K=1..5$)** | High overall ROC-AUC matches persistence because >98% positives are already inside an attack. On true precursor windows, direct head ROC-AUC drops to chance (~0.40-0.48). | `evaluation/k5/direct_heads_results.json` | `b05aeab` | **invalidated** |
| **Transition / Change-Window Prediction** | Rollout change-window AUC is driven by inverse persistence (1.0000). Rollout onset vs benign change-window AUC is chance (episode bootstrapped 95% CIs cross 0.5 for all K). | `evaluation/k5/change_skill_results.json` | `b05aeab` | **invalidated** |
| **Current-Window ATT&CK Stage Classification (LOEO)** | Tactic Macro-F1 = 0.6402 (Credential Access: 0.7101, C2: 0.6721, Impact: 0.5385); Family Macro-F1 = 0.4608. Note: LOEO covers 3/6 families; DDOS-LOIC-UDP family F1 = 0. | `evaluation/family/family_results.json` | `f4d6a9c` | **validated (weak on 3 of 6 families)** |
| **Dataset Provenance & Window Count** | 6 days, 2,787 windows (`ucs_windows_models_v1.parquet`); 10-day rebuild failed check 4 (226/406 features differed) | `data-engineering/data/ucs/REBUILD_CHECK.json` | `285bb8d` | **validated** |
| **Cryptographic Audit Chain Integrity** | Tamper-evident Ed25519 digital signature notarization over prediction lineages and model hashes | `backend/audit_chain.json`, `backend/tests/test_audit_chain.py` | `6bc1335` | **validated** |
| **Intra-Attack Forecasting (duration / end / intensity)** | Not evaluable on CIC-IDS2018: 40 attack runs, only 12 last ≥ 6 minutes (stop rule: ≥ 20 required); no model was trained | `evaluation/intra_attack/runs.json` | `fb799dc` | **not evaluable** |
| **Probabilistic K-step Forecast ($H^*_{prob}$)** | $H^*_{prob} = 0$. The world model's predictive distribution never beats noised persistence / AR(1) on CRPS (e.g. F5 − best = +17.4 [+9.4, +29.5] at K=1); it collapses to climatology; 90% intervals cover 0.68–0.70. Same with no early stopping. | `evaluation/prob_forecast/prob_results.json` | `6ba39ee` | **not achieved** |
| **World-Model Rollout as Early Warning** | Rolled-out states give P(attack at t+K) with ROC-AUC 0.18 (K=1) to 0.27 (K=5) on windows benign at t, below chance; Brier worse than Markov persistence at every K (CI excludes 0) | `evaluation/prob_forecast/binary_results.json` | `6ba39ee` | **not achieved** |

---

## 2. Do NOT Claim List (Discredited / Negative Results)

The following claims are strictly unsupported by empirical evidence or have failed rigorous validation:

1. **Do NOT claim validated K-step forward state-space rollout ($H^* > 0$)**:
   - Both stage rollout ($H^* = 0$) and state-space rollout ($H^*_{state} = 0$) fail to outperform the simple persistence baseline under paired bootstrap testing.
   - Forward state rollout suffers from autoregressive error compounding.
2. **Do NOT claim downstream onset forecasting from world-model rollouts**:
   - Applying onset classifiers to rolled states yields sub-random discrimination ($ROC\text{-}AUC \le 0.4561$, Recall @ 5% FPR $\le 0.0065$).
3. **Do NOT claim multi-horizon anticipatory skill from direct heads**:
   - Direct multi-horizon heads only learn persistence. On true precursor windows, their ROC-AUC is at chance (0.40–0.48).
4. **Do NOT claim the fabricated "0.835 onset F1"**:
   - The "0.835" figure was an unverified, fabricated placeholder from early reports.
   - The verified LOEO matched 5% FPR onset F1 is **0.9362** (stacked ensemble) and **0.8294** (LR baseline).
5. **Do NOT claim hazard heads (`hazard_head_v3`)**:
   - Hazard heads were discredited due to historical PCA representation ambiguity and unscaled input leakage.
   - Hazard heads are excluded from operational dashboard views and replaced with calibrated stacked residual ensembles.
6. **Do NOT claim 10-day dataset coverage**:
   - The 10-day rebuild attempt failed Check 4 (`data-engineering/data/ucs/REBUILD_CHECK.json`), with 226 of 406 features differing from canonical data.
   - All models are trained and validated solely on the verified 6-day dataset (`ucs_windows_models_v1.parquet`, 2,787 windows).
7. **Do NOT claim validated future stage rollout**:
   - World-model rollout stage predictions achieve only $F_1 \approx 0.55$, far below persistence ($F_1 \approx 0.91\text{--}0.95$).
   - Current operational stage classification is restricted to the current-window family classifier (gated at 5% FPR detection threshold).
8. **Do NOT quote the earlier LR baseline numbers (0.8711 / 0.8294)**:
   - They were computed on an earlier feature version. Use the final-dataset baseline (detection 0.9582, onset 0.9198); the stacked model matches it on detection and is ahead on onset.
9. **Do NOT claim the supervised detectors generalise to unseen attack families**:
   - Leave-one-day-out pooled ROC-AUC is 0.666 / 0.633. For unseen attacks, cite the world-model deviation score (0.707 family-out).
