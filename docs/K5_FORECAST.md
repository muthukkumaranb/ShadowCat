# K=1..5 Forward Simulation and Multi-Horizon Evaluation

Evaluated under 37-fold Leave-One-Episode-Out (LOEO) cross-validation on `data-engineering/data/ucs/ucs_windows_models_v1.parquet` using the 36-window purge manifest.

---

## 1. Task C-t1: State-Space Rollout Error Curve

**Source**: `evaluation/k5/state_rollout_results.json` (`pooled_results_per_k_change` -> `delta_model_minus_persistence` -> `rmse`)  
**Metric**: RMSE and MAE on *change windows* (where state changes).

**Empirical Result**: The world model does not systematically beat persistence on change windows in a practically meaningful way. On change windows (which are very rare, ranging from 3 to 24 windows per K), the model's RMSE is technically lower than persistence at K=2, 4, 5, with CIs that exclude 0 (e.g. K=5: Δ −240.53, CI [−2900.39, −0.02]). However, this is largely an artifact of the baseline (persistence error blows up on changes).

---

## 2. Task C-t2: Skill on Change Windows

**Source**: `evaluation/k5/change_skill_results.json` (`change_windows`, `onset_vs_benign` inside `horizons` -> `5`)  

**Empirical Result**: The previously claimed "0.7926 change-window ROC-AUC" for rollout was highly misleading. When measuring an "inverse persistence" score ($1 - y(t)$), it achieves perfect AUC (1.0000) on change windows because it trivially predicts the opposite of the current state. Furthermore, when restricting rollout predictions to distinguish true onset changes ($0 \to 1$) from benign no-change windows ($0 \to 0$), the rollout ROC-AUC is 0.3806 (below chance).

---

## 3. Task C-t3: Direct Multi-Horizon Onset Heads vs Rollout

**Source**: `evaluation/k5/direct_heads_results.json` (`horizons` -> `precursor_only` -> `direct_lr_head` -> `roc_auc`)  

**Empirical Result**: The high overall ROC-AUC previously claimed for direct heads is driven entirely by persistence. The vast majority of positives at horizon $t+K$ are already inside an attack at time $t$. When restricted to a **precursor-only** evaluation (windows where $y(t) = 0$), the direct heads' ROC-AUC drops to ~0.40–0.48 across horizons K=1..5 (e.g., 0.4077 at K=1 and 0.4801 at K=5). Persistence matches or exceeds the direct heads across all horizons, and the direct heads score below chance.

---

## 4. What We Can Claim

Based strictly on the corrected experimental JSON artifacts:

1. **State Rollout Has No Predictive Skill**: While the Gaussian world model has technically lower RMSE than persistence on the rare change windows (K=2, 4, 5), its absolute error remains enormous, and it fails to beat persistence in the full state evaluation.
2. **Direct Heads Only Learn Persistence**: Direct multi-horizon logistic regression heads do not anticipate attacks. Their high performance is an illusion caused by the fact that attacks span multiple windows; they simply learn to predict that an attack will continue if it is already happening. On true precursor windows, their ROC-AUC is between 0.40 and 0.48 (below chance).
3. **Change-Window AUC Was Misleading**: The rollout's apparent ability to separate transitions (0.7926 AUC) was an artifact; simply predicting the inverse of the current state yields 1.0000 AUC on change windows. It fails entirely to distinguish true onset from benign traffic (0.3806 AUC).

**Conclusion**: The model possesses zero anticipatory (K>0) forecasting skill for cyber attacks.
