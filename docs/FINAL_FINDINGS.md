# ShadowCat (SIH26153): Final Findings

Every number below is either on `main` (file named) or in a committed results file on an unmerged
branch, marked **pending merge** with its branch and file. Nothing here is new analysis.

## 1. Multi-step forecasting: H* = 0

The world model's autoregressive rollout does not beat a persistence baseline at any horizon K = 1..5
(predictability horizon H* = 0). This has been confirmed independently by:

1. **Capacity / hyperparameter sweep** of the Gaussian world model (hidden size, depth, learning rate,
   training length): `ml1/artifacts/lstm/sweep_v1/*/horizon_eval.txt`, README "Forecasting Advancement".
2. **Mixture Density Network head** (5 components): F1 0.5530 / 0.5527 / 0.5523 / 0.5520 / 0.5517 at
   K = 1..5 vs persistence 0.9517 .. 0.9059 (README, `data-engineering/data/ucs/EXPANDED_DATASET_REPORT.md`).
3. **10-day data** (4,545 windows at commit a0ae7bc): H* = 0
   (`data-engineering/data/ucs/EXPANDED_DATASET_REPORT.md`). Note: the 10-day dataset is not on `main`
   today; a rebuild with main's pipeline did not reproduce a0ae7bc's feature values (see §6).
4. **CIC-IDS2017 exploration** (combined data): model 0.2136 vs persistence 0.6051 —
   **pending merge** (`feature/cic-ids2017-exploratory`, closed, not merged).
5. **Delta parameterisation and 32-sample Monte Carlo rollout** (4 configs: gaussian/delta x mean/MC-32):
   H* = 0 in every config (6-day dataset) — **pending merge** (`feature/rollout-delta-mc`,
   `evaluation/rollout_v2/results.json`, `ROLLOUT_V2.md`).

**Root cause.** The world model is trained with a Gaussian negative log-likelihood, which is
mean-seeking: it is rewarded for predicting a smoothed average next state, so rare attack states are
averaged away. In addition, the scripted CSE-CIC-IDS2018 attacks have almost no observable precursors
(§3): there is little to forecast before an attack starts.

## 2. What is validated

Single-step classification, mean over 37 leave-one-episode-out (LOEO) folds, reproduced exactly from the
committed models on the dataset they were trained on (commit e200fdb):

| Task | Stacked calibrated LSTM F1 | LR baseline F1 |
|---|---|---|
| Detection (label_binary) | 0.9962 | 0.9730 |
| Onset (attack within t+1..t+5) | 0.9127 | 0.8880 |

Source: README benchmark table (`main`). The LR detection mean hides a split result: LR scores 1.0 on 36/37
folds and 0.0 on Fold 16 (SSH-Bruteforce) (README, `ml1/artifacts/lr_set_a_corrected/README.md`).

At a matched false-positive rate (~5%) on the same 412 pooled test windows — **pending merge**
(`fix/demo-integrity`, `evaluation/benchmark/stacked_benchmark_results.json`):

| Task | Stacked F1 | LR F1 |
|---|---|---|
| Detection | 0.958 | 0.871 |
| Onset | 0.936 | 0.829 |

**Hazard heads.** `hazard_head_v3` reports LOEO ROC-AUC 0.789 / 0.843 / 0.770 at H = 1 / 2 / 5
(`ml1/artifacts/lstm/hazard_head_v3/hazard_head_report_v3.md`). These reproduce exactly only with a
per-fold PCA refit on commit 371b855's data, the procedure they were trained with; through
`models/pca_32.pkl` (the dashboard path) they do not reproduce, so the dashboard no longer shows them —
**pending merge** (`fix/demo-integrity`, `evaluation/benchmark/hazard_v3_results.json`).

## 3. Onset is mostly detection of attacks already in progress

Of 984 onset-positive windows in the e200fdb dataset, 913 are already inside an attack and only 71 are
true precursors (no attack yet, attack within 5 minutes) — counts in `dataset_counts` of the precursor results
file below (pending merge). The onset F1 of 0.9127 is therefore measured
mostly on in-progress windows. On the 15 precursors in the LOEO test set — **pending merge**
(`feature/deviation-evidence`, `evaluation/precursor/precursor_results.json`, 6-day dataset):

| Model | ROC-AUC (precursor vs benign non-precursor) | Precursor recall at matched FPR |
|---|---|---|
| Stacked ensemble | 0.625 | 6/15 |
| LR baseline | 0.520 | 3/15 |

## 4. Ablation: the gain is not temporal learning

Shuffling the 30-window history or replacing it with the last window repeated leaves F1 unchanged, and
the stacked model's calibrated LR stage alone matches the full model — **pending merge**
(`fix/demo-integrity`, `stacked_benchmark_results.json`, `history_ablation_at_0.5`). The LSTM residual adds
no measurable gain; the improvement over the LR baseline comes from the stacked model's calibrated LR
stage. It should not be described as temporal learning.

## 5. World-model deviation as evidence

Next-state deviation separates attack from benign windows: pooled LOEO ROC-AUC 0.820, pooled
leave-one-family-out ROC-AUC 0.707 (6-day dataset) — **pending merge** (`feature/deviation-evidence`,
`evaluation/deviation/deviation_results.json`, `DEVIATION.md`). Only three families have two or more
episodes; the others are not evaluable.

## 6. Retraction and open items

- **Retracted:** the interim claim that "more data eliminated H* = 0". It compared Set A traffic features
  against a day-of-week predictor (`run_loeo_corrected.py`), not the world model
  (`data-engineering/data/ucs/EXPANDED_DATASET_REPORT.md`).
- **10-day dataset:** `main` holds the 6-day / 2,787-window file. Re-running main's pipeline over 10 days gives
  4,545 windows with identical order and labels, but 226 of 406 feature columns differ from a0ae7bc — **pending**
  (`fix/restore-10day-dataset`, `data-engineering/data/ucs/REBUILD_CHECK.json`; dataset not committed).
- **CIC-IDS2017:** inconclusive; work is on an unmerged, closed branch.
- **Future work:** multi-stage APT datasets with real precursor phases (e.g. DAPT2020); feeding the full
  predicted distribution (or samples) to the stage head instead of its mean.
