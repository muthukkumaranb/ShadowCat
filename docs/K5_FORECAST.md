# K=1..5 Forward Simulation and Multi-Horizon Evaluation

Evaluated under 37-fold Leave-One-Episode-Out (LOEO) cross-validation on `data-engineering/data/ucs/ucs_windows_models_v1.parquet` using the 36-window purge manifest.

---

## 1. Task C-t1: State-Space Rollout Error Curve

**Source**: `evaluation/k5/state_rollout_results.json` (`pooled_results_per_k_change` -> `delta_model_minus_persistence` -> `rmse`)  
**Metric**: RMSE and MAE on *change windows* (where state changes).

**Empirical Result**: On change windows (rare: 3 to 24 windows per K), the world model's RMSE is lower than the persistence reference at K=2, 4 and 5, with CIs that exclude 0 (e.g. K=5: Δ −240.53, CI [−2900.39, −0.02]). The margin is driven largely by the reference's error on transitions, and with so few windows it does not yet establish a practical forecasting horizon.

---

## 2. Task C-t2: Skill on Change Windows

**Source**: `evaluation/k5/change_skill_results.json` (`change_windows`, `onset_vs_benign` inside `horizons` -> `5`)  

**Empirical Result**: The rollout reaches a change-window ROC-AUC of 0.7926 at K=5. An inverse-persistence reference ($1 - y(t)$) reaches 1.0000 on the same windows, because every change window is by definition the opposite of the current state, so this metric is best read against that reference. Separating true onset changes ($0 \to 1$) from benign no-change windows ($0 \to 0$), the rollout ROC-AUC is 0.3806.

---

## 3. Task C-t3: Direct Multi-Horizon Onset Heads vs Rollout

**Source**: `evaluation/k5/direct_heads_results.json` (`horizons` -> `precursor_only` -> `direct_lr_head` -> `roc_auc`)  

**Empirical Result**: The direct heads reach an overall ROC-AUC of 0.95–0.99, and most of that comes from attacks already in progress: the large majority of positives at $t+K$ are inside an attack at time $t$. On a **precursor-only** evaluation (windows where $y(t) = 0$) the ROC-AUC is 0.41–0.48 across K=1..5 (0.4077 at K=1, 0.4801 at K=5), in line with the persistence reference. With only a handful of precursor windows in CIC-IDS2018, this is the measured limit of the data.

---

## 4. Summary

Based on the committed experimental JSON artifacts:

1. **State rollout:** the Gaussian world model's RMSE is lower than the persistence reference on the rare change
   windows (K=2, 4, 5); over all windows it stays within that reference, giving a measured horizon of H\*_state = 0.
2. **Direct heads:** their high overall ROC-AUC reflects attacks that span several windows. On true precursor
   windows the ROC-AUC is 0.41–0.48, which is what the few available precursors support.
3. **Change windows:** the 0.7926 rollout AUC is best read against the inverse-persistence reference (1.0000). For
   onset vs benign, episode-bootstrapped 95% CIs include 0.5 at every K (e.g. K=1 [0.3111, 0.7377] from 3 true
   positives; K=2 [0.4044, 0.9239] from 6), so the data are too sparse to resolve a difference.

**Conclusion**: On CIC-IDS2018 the measured predictability horizon is H\* = 0. The limiting factor is the data:
scripted attacks with very few observable precursors. The K-step pipeline is implemented and evaluated end to end,
ready for whole-network multi-stage data.
