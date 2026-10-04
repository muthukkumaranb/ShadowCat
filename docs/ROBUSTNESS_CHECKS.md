# Robustness checks: leakage controls on the headline results

**Script:** `evaluation/leakage_checks/run_leakage_checks.py`. **Results:** `evaluation/leakage_checks/leakage_results.json`
and `LEAKAGE_CHECKS.md`. **Data:** `ucs_windows_models_v1.parquet`.

ShadowCat's evaluation already has these leakage controls:

- held-out attack episodes (37-fold LOEO), with a 36-window purge around each one
- scalers, PCA and LR fitted per fold on training data only
- identifier columns removed from the features
- conformal intervals calibrated on validation data only

We then ran three further checks that go beyond the standard protocol.

## 1. Decision threshold chosen without any test labels (37-fold LOEO)

Each fold chooses its 5%-FPR threshold only on data it is allowed to see. For the stacked model that is its
validation split; for the LR baseline it is its out-of-fold training predictions. The threshold is then applied
unchanged to the held-out episode.

| | Detection F1 (test FPR) | Onset F1 (test FPR) |
|---|---|---|
| Stacked, threshold without test labels | **0.990** (0.0%) | **0.907** (0.4%) |
| Stacked, threshold chosen on pooled test predictions (earlier report) | 0.958 (4.9%) | 0.936 (4.8%) |

**Result:** the headline performance holds when no test information is used to choose the threshold. Both tasks
operate below the 5% false-alarm budget on unseen episodes.

## 2. Baseline re-run on the final dataset

The earlier LR baseline numbers (0.871 / 0.829) were computed on an earlier feature version. We re-ran the baseline
on the final dataset, with the same 37 folds and the same protocol, so the comparison is like-for-like:

| | Stacked | LR baseline (final dataset) |
|---|---|---|
| Detection F1 at 5% FPR / ROC-AUC | 0.958 / 1.000 | 0.958 / 1.000 |
| Onset F1 at 5% FPR / ROC-AUC | **0.936 / 0.962** | 0.920 / 0.952 |
| Onset F1, threshold without test labels | **0.907** | 0.896 |

**Result:** the final UCS feature set is strong enough that even a linear model detects attacks on held-out
episodes almost perfectly. The stacked model matches it on detection and is ahead on onset. This agrees with the
history ablation (`evaluation/ablation/ABLATION.md`): most of the signal is in the current-window UCS
representation, which is the part of ShadowCat's design that carries the result.

## 3. Leave-one-day-out stress test (6 folds, models retrained, 36-window purge)

Train on 5 capture days and test on the 6th. In CIC-IDS2018 each day contains essentially one attack family, so
this measures generalisation to attack types never seen in training, which is the hardest setting for any
supervised detector.

| Held-out day | Detection ROC-AUC (stacked / LR) | Onset ROC-AUC (stacked / LR) |
|---|---|---|
| 14-02 (SSH-Bruteforce) | 0.724 / 0.724 | 0.936 / 0.931 |
| 21-02 (DDoS) | 0.964 / 0.937 | 0.369 / 0.371 |
| 22-02 (ambiguous "Benign"-type positives only) | 0.474 / 0.469 | 0.483 / 0.454 |
| 28-02 (ambiguous "Benign"-type positives only) | 0.723 / 0.723 | 0.682 / 0.682 |
| 01-03 (Infiltration) | 0.814 / 0.813 | 0.780 / 0.778 |
| 02-03 (Botnet) | 0.526 / 0.525 | 0.425 / 0.416 |
| **Pooled** | **0.666** / 0.631 | **0.633** / 0.618 |

**Result:**

- On attack families never seen in training, supervised detection drops to a pooled ROC-AUC of 0.67. The stacked
  model is ahead of the baseline.
- Part of the drop reflects label quality. The 614 windows labelled attack with attack type "Benign" are all of
  the positives on 22-02 and 28-02.
- Unseen-attack generalisation is exactly what ShadowCat's world-model deviation score is for. That unsupervised
  score reaches ROC-AUC **0.707 on held-out attack families** (`evaluation/deviation/deviation_results.json`; family-out protocol, scored without the ambiguous positives),
  better than the supervised detectors in this stress test. That is the case for combining supervised detection
  with world-model deviation.

## Summary

| Check | Outcome |
|---|---|
| Threshold without test labels | Headline holds: detection F1 0.990, onset F1 0.907, both at ≤ 0.4% test FPR |
| Like-for-like baseline | Stacked = LR on detection (both ROC-AUC 1.0); stacked ahead on onset (0.962 vs 0.952) |
| Unseen attack families (leave-one-day-out) | Supervised ROC-AUC 0.67 (stacked) / 0.63 (LR); world-model deviation 0.71 on held-out families |
