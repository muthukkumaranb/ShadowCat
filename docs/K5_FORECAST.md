# K=1..5 Forward Simulation and Multi-Horizon Evaluation

Evaluated under 37-fold Leave-One-Episode-Out (LOEO) cross-validation on `data-engineering/data/ucs/ucs_windows_models_v1.parquet` using the 36-window purge manifest.

---

## 1. Task C-t1: State-Space Rollout Error Curve

**Source**: `evaluation/k5/state_rollout_results.json`  
**Metric**: RMSE and MAE on *change windows* (where state changes).

**Empirical Result**: The world model does not beat persistence on change windows. Across horizons, state-space rollout produces larger errors than simply predicting $S(t+k) = S(t)$, even precisely at the moments the state transitions.

---

## 2. Task C-t2: Skill on Change Windows

**Source**: `evaluation/k5/change_skill_results.json`  

**Empirical Result**: The previously claimed "0.7926 change-window ROC-AUC" for rollout was highly misleading. When measuring an "inverse persistence" score ($1 - y(t)$), it achieves perfect or near-perfect AUC on change windows because it trivially predicts the opposite of the current state. Furthermore, when restricting rollout predictions to distinguish true onset changes ($0 \to 1$) from benign no-change windows ($0 \to 0$), the rollout ROC-AUC is at chance.

---

## 3. Task C-t3: Direct Multi-Horizon Onset Heads vs Rollout

**Source**: `evaluation/k5/direct_heads_results.json`  

**Empirical Result**: The high ROC-AUC (e.g., 0.9882) previously claimed for direct heads is driven entirely by persistence. The vast majority of positives at horizon $t+K$ are already inside an attack at time $t$. When restricted to a **precursor-only** evaluation (windows where $y(t) = 0$), the direct heads' ROC-AUC drops to ~0.42–0.49 (chance level or worse). Persistence matches or exceeds the direct heads across all horizons.

---

## 4. What We Can Claim

Based strictly on the corrected experimental JSON artifacts:

1. **State Rollout Has No Predictive Skill**: The Gaussian world model fails to beat persistence in full-state RMSE/MAE, even on windows where the true state actually changes.
2. **Direct Heads Only Learn Persistence**: Direct multi-horizon logistic regression heads do not anticipate attacks. Their high performance is an illusion caused by the fact that attacks span multiple windows; they simply learn to predict that an attack will continue if it is already happening. On true precursor windows, they score at chance.
3. **Change-Window AUC Was Misleading**: The rollout's apparent ability to separate transitions was an artifact of predicting the opposite of the current state, failing entirely to distinguish onset from benign traffic.

**Conclusion**: The model possesses zero anticipatory (K>0) forecasting skill for cyber attacks.
