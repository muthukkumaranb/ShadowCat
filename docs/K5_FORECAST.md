# K=1..5 Forward Simulation and Multi-Horizon Evaluation

Evaluated under 37-fold Leave-One-Episode-Out (LOEO) cross-validation on `data-engineering/data/ucs/ucs_windows_models_v1.parquet` (2,787 windows, 6 days) using the authoritative 36-window purge manifest `ml1/artifacts/loeo/corrected_37fold_manifest.json` (Seed 42).

---

## 1. Task C-t1: State-Space Rollout Error Curve (K = 1..5)

**Source**: `evaluation/k5/state_rollout_results.json`  
**Model**: `LSTMGaussianWorldModel` (input_dim=406, hidden_dim=64, state_dim=406, num_layers=1, dropout=0.2)  
**Baselines**: Persistence $\hat{S}(t+k) = S(t)$, Linear AR(1) $\hat{S}(t+1) = W S(t) + b$, Training Fold Mean $\bar{S}_{train}$.  
**Statistical Criterion**: $H^*_{state}$ is the largest $K$ at which the world model beats persistence on **BOTH** RMSE and MAE with a paired episode-level bootstrap 95% CI (1,000 resamples) strictly excluding 0.

| Horizon K | Model RMSE | Persistence RMSE | AR(1) RMSE | Mean RMSE | Δ RMSE (Model - Persistence) [95% CI] | Model MAE | Persistence MAE | AR(1) MAE | Mean MAE | Δ MAE (Model - Persistence) [95% CI] | Beats Persistence Both? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **K=1 (+1m)** | 1136.5066 | 986.1863 | 4747.7077 | 1140.9507 | +150.3203 [-11.6023, +394.8618] | 22.9626 | 14.6483 | 258.7436 | 31.0661 | +8.3143 [+3.8756, +15.1609] | **NO** |
| **K=2 (+2m)** | 1625.2552 | 756.7327 | 71490.5736 | 1628.3590 | +868.5225 [+483.2188, +1599.1089] | 33.8526 | 7.8293 | 3347.0837 | 41.7164 | +26.0233 [+14.1331, +43.7713] | **NO** |
| **K=3 (+3m)** | 1233.4641 | 1107.7304 | 137552.3045 | 1237.5560 | +125.7336 [-42.4727, +454.4348] | 26.6346 | 18.1846 | 5065.2561 | 34.3452 | +8.4500 [+3.2336, +17.1582] | **NO** |
| **K=4 (+4m)** | 1640.5731 | 929.4387 | 1461290.0414 | 1643.6514 | +711.1344 [+382.3719, +1416.1142] | 35.5764 | 12.8608 | 53366.1388 | 43.1834 | +22.7156 [+11.6359, +39.1323] | **NO** |
| **K=5 (+5m)** | 1314.8689 | 1238.5680 | 2377968.5830 | 1318.7112 | +76.3009 [-113.3311, +382.5498] | 28.8971 | 22.3887 | 91050.3995 | 36.4457 | +6.5084 [+0.8022, +13.8772] | **NO** |

**Empirical Result**: **$H^*_{state} = 0$**. The world model does not beat persistence at any horizon. Linear AR(1) explodes under recursive compounding.

---

## 2. Task C-t2: Skill on Change Windows

**Source**: `evaluation/k5/change_skill_results.json`  
**Definition**: A change window has $y(t+K) \neq y(t)$ (state changes between observation time $t$ and horizon $t+K$). If $N_{change} < 20$, it is marked "too few to evaluate".

| Horizon K | Total Test Windows | Change Windows | Evaluation Status | Rollout ROC-AUC | Rollout Recall @ 5% FPR | Persistence ROC-AUC | No-Change F1 (Rollout / Persist) | All Windows F1 (Rollout / Persist) |
|---|---|---|---|---|---|---|---|---|
| **K=1 (+1m)** | 410 | 3 | too few to evaluate (3 < 20) | N/A | N/A | N/A | 0.0255 / 1.0000 | 0.0252 / 0.9899 |
| **K=2 (+2m)** | 410 | 8 | too few to evaluate (8 < 20) | N/A | N/A | N/A | 0.1371 / 1.0000 | 0.1339 / 0.9732 |
| **K=3 (+3m)** | 410 | 13 | too few to evaluate (13 < 20) | N/A | N/A | N/A | 0.1311 / 1.0000 | 0.1265 / 0.9565 |
| **K=4 (+4m)** | 410 | 18 | too few to evaluate (18 < 20) | N/A | N/A | N/A | 0.1167 / 1.0000 | 0.1111 / 0.9400 |
| **K=5 (+5m)** | 410 | 24 | evaluated | **0.7926** | **0.5333** | 0.0000 | 0.1181 / 1.0000 | 0.1111 / 0.9200 |

**Empirical Result**: At $K=1..4$, fewer than 20 test windows exhibit a state transition. At $K=5$, 24 change windows exist; on these change windows, the rollout model achieves 0.7926 ROC-AUC and 53.3% recall at 5% FPR, whereas persistence achieves 0.0000 ROC-AUC. However, across all windows combined, rollout onset F1 is 0.1111 due to false alarms on stationary windows.

---

## 3. Task C-t3: Direct Multi-Horizon Onset Heads vs Rollout-Based Probability

**Source**: `evaluation/k5/direct_heads_results.json`  
**Target**: An attack starts within $(t, t+K]$ (i.e. $\max_{j=1..K} y(t+j) = 1$).  
**Architecture**: Direct multinomial logistic regression per horizon on 406 `set_a_features` vs stacked onset head applied to rolled world-model states.

| Horizon K | Positives / Total | Direct Head ROC-AUC | Rollout ROC-AUC | Direct Head PR-AUC | Rollout PR-AUC | Direct Head Recall @ 5% FPR | Rollout Recall @ 5% FPR |
|---|---|---|---|---|---|---|---|
| **K=1 (+1m)** | 150 / 410 | **0.9882** | 0.4142 | **0.9905** | 0.3003 | **0.9800** | 0.0000 |
| **K=2 (+2m)** | 153 / 410 | **0.9784** | 0.4561 | **0.9822** | 0.3291 | **0.9608** | 0.0065 |
| **K=3 (+3m)** | 156 / 410 | **0.9700** | 0.4307 | **0.9772** | 0.3277 | **0.9423** | 0.0064 |
| **K=4 (+4m)** | 159 / 410 | **0.9592** | 0.4143 | **0.9694** | 0.3222 | **0.9308** | 0.0000 |
| **K=5 (+5m)** | 162 / 410 | **0.9519** | 0.3967 | **0.9646** | 0.3202 | **0.9136** | 0.0000 |

**Empirical Result**: Direct multi-horizon heads retain strong discrimination out to K=5 (ROC-AUC 0.9882 $\to$ 0.9519, recall at 5% FPR 0.9800 $\to$ 0.9136). In contrast, applying downstream onset heads to autoregressively rolled world-model states collapses discrimination to sub-random levels (ROC-AUC $\le$ 0.4561, recall at 5% FPR $\le$ 0.0065) due to compounding state-space simulation error.

---

## 4. What We Can Claim

Based strictly on the committed experimental JSON artifacts:

1. **State Rollout Horizon**: We can claim $H^*_{state} = 0$. The Gaussian world model does not outperform simple persistence in predicting full multi-dimensional network telemetry states across horizons $K=1..5$ under rigorous LOEO cross-validation.
2. **Direct Multi-Horizon Onset Heads**: Direct logistic regression heads trained explicitly for multi-horizon onset achieve validated discrimination across all evaluated horizons ($K=1..5$), with ROC-AUC ranging from 0.9882 (K=1) to 0.9519 (K=5), and recall at 5% matched FPR ranging from 98.0% (K=1) to 91.4% (K=5).
3. **Rollout Onset Limitation**: Downstream classification applied to world-model rollouts fails ($ROC\text{-}AUC \le 0.4561$ across all $K$). Forward state rollout cannot be used as an intermediate feature generator for multi-horizon attack classification without severe degradation.
4. **Transition Skill**: On change windows at $K=5$, the forward rollout model separates transition points better than persistence (0.7926 vs 0.0000 ROC-AUC), but suffers high false positive rates during stationary benign traffic (overall combined F1 of 0.1111).
