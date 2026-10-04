# ShadowCat Claims Sheet

This document serves as the authoritative, verified claims sheet for ShadowCat (SIH26153). Every value is backed by a committed JSON artifact produced by a reproducible script and verified in git history.

---

## 1. Verified & Evaluated Claims Table

| Claim | Value | Source File | Commit | Status |
|---|---|---|---|---|
| **Stacked Detection Performance (Matched 5% FPR)** | F1 = 0.9582, Precision = 0.9198, Recall = 1.0000, FPR = 0.0494, ROC-AUC = 1.0000 (Threshold = 0.4014) | `evaluation/benchmark/stacked_benchmark_results.json` | `a0687cf` | **validated** |
| **Linear Baseline Detection Performance (Matched 5% FPR)** | F1 = 0.8711, Precision = 0.9058, Recall = 0.8389, FPR = 0.0494, ROC-AUC = 0.9743 (Threshold = 0.0523) | `evaluation/benchmark/stacked_benchmark_results.json` | `a0687cf` | **validated** |
| **Stacked Onset Performance (Matched 5% FPR)** | F1 = 0.9362, Precision = 0.9277, Recall = 0.9448, FPR = 0.0482, ROC-AUC = 0.9621 (Threshold = 0.3749) | `evaluation/benchmark/stacked_benchmark_results.json` | `a0687cf` | **validated** |
| **Linear Baseline Onset Performance (Matched 5% FPR)** | F1 = 0.8294, Precision = 0.9118, Recall = 0.7607, FPR = 0.0482, ROC-AUC = 0.9148 (Threshold = 0.0951) | `evaluation/benchmark/stacked_benchmark_results.json` | `a0687cf` | **validated** |
| **World-Model Next-State Deviation (Pooled LOEO)** | ROC-AUC = 0.8201, PR-AUC = 0.7721 (40 LOEO folds, 312 positives / 768 test windows) | `evaluation/deviation/deviation_results.json` | `20a8b7d` | **validated** |
| **World-Model Next-State Deviation (Pooled Family-Out)** | ROC-AUC = 0.7070, PR-AUC = 0.4855 (Held-out attack days, 312 positives / 1,730 test windows) | `evaluation/deviation/deviation_results.json` | `20a8b7d` | **weak** |
| **Precursor Lead Time & Recall (Matched 5% FPR)** | Recall = 40.0% (6/15 precursors detected), Mean Lead Time = 3.00 min (range 1–5 min), FPR = 0.0444 | `evaluation/precursor/precursor_results.json` | `5513678` | **validated** |
| **Linear Baseline Precursor Detection** | Recall = 20.0% (3/15 precursors detected), Mean Lead Time = 3.33 min (range 2–5 min), FPR = 0.0444 | `evaluation/precursor/precursor_results.json` | `5513678` | **weak** |
| **Stage Prediction Horizon from World-Model Rollout ($H^*$)** | $H^* = 0$ (Model F1 $\approx 0.45\text{--}0.55$ vs Persistence F1 $\approx 0.91\text{--}0.95$ across $K=1..5$) | `evaluation/rollout_v2/results.json` | `10b3578` | **not achieved** |
| **State-Space Rollout Horizon ($H^*_{state}$)** | $H^*_{state} = 0$ (Model does not beat persistence on both RMSE and MAE; bootstrap 95% CI strictly includes/exceeds 0) | `evaluation/k5/state_rollout_results.json` | `d2192cb` | **pending merge** |
| **Direct Multi-Horizon Onset Heads ($K=1..5$)** | ROC-AUC: 0.9882 ($K=1$) to 0.9519 ($K=5$); Recall @ 5% FPR: 98.0% ($K=1$) to 91.4% ($K=5$) | `evaluation/k5/direct_heads_results.json` | `8fb74cc` | **pending merge** |
| **Rollout-Based Multi-Horizon Onset** | ROC-AUC: 0.4142 ($K=1$) to 0.3967 ($K=5$); Recall @ 5% FPR: 0.0% across all $K$ (error compounding) | `evaluation/k5/direct_heads_results.json` | `8fb74cc` | **pending merge** |
| **Current-Window ATT&CK Stage Classification (LOEO)** | Tactic Macro-F1 = 0.6402 (Credential Access: 0.7101, C2: 0.6721, Impact: 0.5385); Family Macro-F1 = 0.4608 | `evaluation/family/family_results.json` | `f4d6a9c` | **pending merge** |
| **Dataset Provenance & Window Count** | 6 days, 2,787 windows (`ucs_windows_models_v1.parquet`); 10-day rebuild failed check 4 (226/406 features differed) | `data-engineering/data/ucs/REBUILD_CHECK.json` | `285bb8d` | **validated** |
| **Cryptographic Audit Chain Integrity** | Tamper-evident Ed25519 digital signature notarization over prediction lineages and model hashes | `backend/audit_chain.json`, `backend/tests/test_audit_chain.py` | `6bc1335` | **validated** |

---

## 2. Do NOT Claim List (Discredited / Negative Results)

The following claims are strictly unsupported by empirical evidence or have failed rigorous validation:

1. **Do NOT claim validated K-step forward state-space rollout ($H^* > 0$)**:
   - Both stage rollout ($H^* = 0$) and state-space rollout ($H^*_{state} = 0$) fail to outperform the simple persistence baseline under paired bootstrap testing.
   - Forward state rollout suffers from autoregressive error compounding.
2. **Do NOT claim downstream onset forecasting from world-model rollouts**:
   - Applying onset classifiers to rolled states yields sub-random discrimination ($ROC\text{-}AUC \le 0.4561$, Recall @ 5% FPR $\le 0.0065$).
   - Multi-horizon forecasting is only validated when using direct multi-horizon heads.
3. **Do NOT claim the fabricated "0.835 onset F1"**:
   - The "0.835" figure was an unverified, fabricated placeholder from early reports.
   - The verified LOEO matched 5% FPR onset F1 is **0.9362** (stacked ensemble) and **0.8294** (LR baseline).
4. **Do NOT claim hazard heads (`hazard_head_v3`)**:
   - Hazard heads were discredited due to historical PCA representation ambiguity and unscaled input leakage.
   - Hazard heads are excluded from operational dashboard views and replaced with calibrated stacked residual ensembles.
5. **Do NOT claim 10-day dataset coverage**:
   - The 10-day rebuild attempt failed Check 4 (`data-engineering/data/ucs/REBUILD_CHECK.json`), with 226 of 406 features differing from canonical data.
   - All models are trained and validated solely on the verified 6-day dataset (`ucs_windows_models_v1.parquet`, 2,787 windows).
6. **Do NOT claim validated future stage rollout**:
   - World-model rollout stage predictions achieve only $F_1 \approx 0.55$, far below persistence ($F_1 \approx 0.91\text{--}0.95$).
   - Current operational stage classification is restricted to the current-window family classifier (gated at 5% FPR detection threshold).
