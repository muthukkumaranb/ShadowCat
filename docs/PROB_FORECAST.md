# Probabilistic K-step forecast (Part P): findings

**Question.** Point forecasts of the next state do not beat persistence (H* = 0, `docs/K5_FORECAST.md`).
Does the world model's *predictive distribution* over S(t+K) beat a fairly-noised persistence forecast under
proper scoring rules, at K = 1..5?

**Answer: no. H\*_prob = 0.** The same holds in a sensitivity run with no early stopping.

## Protocol

- **Script:** `evaluation/prob_forecast/run_prob_forecast.py` (scoring rules in `scoring.py`, forecasters in
  `ml1/prob_forecast/forecasters.py`, tests in `tests/prob_forecast/`).
- **Data:** `data-engineering/data/ucs/ucs_windows_models_v1.parquet`, the 406 schema features.
- **Cross-validation:** the 37-fold LOEO manifest. Test origins are t ≥ 29 with a contiguous 30-window lookback;
  409–411 origins per K.
- **Space:** per-fold z-score (dropping 53 zero-variance features) + PCA-32, fitted on the fold's training indices
  only.
- **Forecasters:**
  - F0: climatology.
  - F1: noised persistence N(S(t), Σ_K), with Σ_K the training variance of S(t+K) − S(t).
  - F2: F1 with Σ_K split by the stacked-detection decision at t (5%-FPR threshold, never the true label).
  - F3: diagonal AR(1) with closed-form K-step mean and variance.
  - F4: the existing `LSTMGaussianWorldModel`, trained per fold with the default `train_gaussian` settings and
    rolled out autoregressively with **64 sampled trajectories**.
  - F5: F4 with one variance-inflation factor per K, fitted on the inner validation split (factors 1.09–1.68).
- **World-model training split:** the first 80% of each fold's training windows train the model; the last 20% are
  used for early stopping and F5 calibration.
- **Scores:** CRPS, energy score, Gaussian log score, 50% and 90% interval coverage, 90% width, and PIT histograms.
  Each is reported on all windows, on windows benign at t, and on change windows.
- **Confidence intervals:** paired bootstrap over the 37 held-out episodes, 2,000 resamples.
- **Definition of H\*_prob:** the largest consecutive K at which F4 or F5 beats the best of F0–F3 on mean CRPS
  (all windows) with a 95% CI excluding 0, **and** has 90% coverage in [0.85, 0.95].

## Results (`evaluation/prob_forecast/prob_results.json`, `PROB_FORECAST.md`)

Mean CRPS in PCA-32 space, all windows (lower is better):

| K | F0 climatology | F1 noised persistence | F3 AR(1) | F4 world model | F5 recalibrated | F4 90% coverage |
|---|---|---|---|---|---|---|
| 1 | 38.23 | 29.99 | **20.71** | 38.17 | 38.15 | 0.70 |
| 3 | 41.53 | **35.69** | 37.89 | 41.46 | 41.42 | 0.69 |
| 5 | 45.78 | **40.38** | 44.38 | 45.73 | 45.69 | 0.68 |

- **The world model's distribution is practically identical to climatology** (CRPS within 0.2% of F0 at every K).
  This is the mean-seeking collapse already identified as the root cause of H* = 0: the rolled-out states revert
  to the training mean.
- **It never beats the best baseline.** For example, F5 − best baseline at K=1 is +17.4 [+9.4, +29.5]; at K=5 it is
  +5.3 [−2.7, +16.7].
- **It is not calibrated:** 90% intervals cover only 68–70% (F4) and 74–77% (F5).
- **Windows benign at t** (the robust view; the all-window means are dominated by a few very large attack windows):
  noised persistence F1 is best at every K. F5 is worse than F1 by +0.64 [+0.33, +1.06] at K=1 and by
  +0.28 [+0.09, +0.53] at K=5.
- **Change windows** (only 3–23 per K): at K=5, F5 is lower than climatology by −0.22 [−0.38, −0.10] on a CRPS of
  about 283. That is about 0.08%, from 23 windows, and practically negligible.

**Attack probability P(label_binary(t+K)=1)** (`binary_results.json`, Brier score, lower is better):

| K | Base rate | Markov, realistic current state | LR on features | World-model rollout |
|---|---|---|---|---|
| 1 | 0.235 | **0.031** | 0.014* | 0.221 |
| 5 | 0.238 | **0.058** | 0.064 | 0.223 |

\*LR on features is lower than the Markov baseline at K=1, but both mostly encode the current state.

- The world-model rollout is worse than the reference at every K, by +0.165 to +0.190 Brier, with CIs excluding 0.
- **On windows benign at t (early warning), its ROC-AUC is 0.18 at K=1 and 0.27 at K=5**, which is below chance.
  Its rolled-out states carry no forward-looking attack signal.

**Sensitivity** (`evaluation/prob_forecast/sensitivity_no_early_stop/`): 14 of 37 folds early-stopped at epoch 3.
Retraining every fold for the full 30 epochs changes nothing material: F4 CRPS is 38.10 vs 38.17 at K=1,
H\*_prob is still 0, and 90% coverage is 0.67–0.70.

## What we can claim

1. "Scored as a probability forecast with proper scoring rules, the world model's K=1..5 predictive distribution
   does not beat noised persistence or an AR(1) baseline at any K (H\*_prob = 0; `prob_results.json:h_star_prob`).
   Its distribution collapses to climatology, and its 90% intervals cover only 68–70%."
2. "Its rolled-out states give no early warning: ROC-AUC 0.18–0.27 on windows benign at t
   (`binary_results.json:horizons.*.benign_t`)."
3. Together with `docs/K5_FORECAST.md` and `docs/INTRA_ATTACK.md`, this closes the forecasting question on
   CIC-IDS2018: no K-step skill, measured as a point forecast, as a distribution, as a change-window classifier, and
   as an early-warning probability.

**Limitations:**
- 37 episodes over 6 days of scripted attacks.
- PCA-32 representation only. A z-space score for the world model would require inverting the PCA, so it is not
  reported.
- One world-model architecture, with its existing defaults (no tuning).
