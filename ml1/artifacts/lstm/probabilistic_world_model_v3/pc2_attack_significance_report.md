# PC2 Attack Window Significance Test Report (v3 Retrain Cascade)

## Summary & Key Findings

- **Test Goal**: Evaluate whether the retrained LSTM Gaussian World Model (`v3`) outperforms the Persistence baseline on Principal Component 2 (PC2) during attack windows in the test set.
- **Model Checkpoint**: `gaussian_next_state_best_v3.pt`
- **Dataset**: `ucs_windows.parquet` (Test split, contiguous attack windows).
- **Statistical Test**: Wilcoxon signed-rank test (two-sided paired non-parametric test) + Cohen's d effect size.
- **Overall Verdict**: **PERSISTENCE BASELINE IS STATISTICALLY SUPERIOR ON PC2**.
  - At **K=1** ($N=165$ attack windows): Mean MAE difference $\Delta = +2.5121$ points (Persistence advantage), $p = 7.91e-29$, Cohen's $d = 1.6166$. Persistence wins 100.0\% (165/165) of attack windows.
  - At **K=2** ($N=166$ attack windows): Mean MAE difference $\Delta = +3.8187$ points (Persistence advantage), $p = 5.42e-29$, Cohen's $d = 1.7354$. Persistence wins 100.0\% (166/166) of attack windows.
  - At **K=3** ($N=167$ attack windows): Mean MAE difference $\Delta = +5.4872$ points (Persistence advantage), $p = 3.72e-29$, Cohen's $d = 1.8465$. Persistence wins 100.0\% (167/167) of attack windows.

---

## Detailed Results by Rollout Horizon $K$

### Rollout Horizon $K=1$ ($N = 165$ Attack Windows)
- **LSTM Mean MAE**: $2.54$ | **Persistence Mean MAE**: $0.02$ | **Mean Delta (LSTM - Pers)**: **$+2.5121$**
- **Median Delta (LSTM - Pers)**: $+1.9228$
- **Wilcoxon Signed-Rank Statistic**: $0.0$ ($p = 7.9145e-29$)
- **Cohen's d**: $1.6166$
- **Win Breakdown**: LSTM wins on 0 / 165 windows (0.0\%), Persistence wins on 165 / 165 windows (100.0\%).

#### PC2 MAE Percentile Distribution ($K=1$)
| Model | p5 | p25 | p50 (Median) | p75 | p95 | Max |
|---|---|---|---|---|---|---|
| **LSTM** | 0.91 | 1.42 | 1.96 | 3.17 | 5.49 | 10.17 |
| **Persistence** | 0.00 | 0.01 | 0.01 | 0.03 | 0.06 | 0.32 |

---

### Rollout Horizon $K=2$ ($N = 166$ Attack Windows)
- **LSTM Mean MAE**: $3.84$ | **Persistence Mean MAE**: $0.02$ | **Mean Delta (LSTM - Pers)**: **$+3.8187$**
- **Median Delta (LSTM - Pers)**: $+3.0023$
- **Wilcoxon Signed-Rank Statistic**: $0.0$ ($p = 5.4235e-29$)
- **Cohen's d**: $1.7354$
- **Win Breakdown**: LSTM wins on 0 / 166 windows (0.0\%), Persistence wins on 166 / 166 windows (100.0\%).

#### PC2 MAE Percentile Distribution ($K=2$)
| Model | p5 | p25 | p50 (Median) | p75 | p95 | Max |
|---|---|---|---|---|---|---|
| **LSTM** | 0.89 | 2.17 | 3.02 | 5.39 | 7.97 | 9.60 |
| **Persistence** | 0.00 | 0.01 | 0.01 | 0.02 | 0.05 | 0.34 |

---

### Rollout Horizon $K=3$ ($N = 167$ Attack Windows)
- **LSTM Mean MAE**: $5.51$ | **Persistence Mean MAE**: $0.02$ | **Mean Delta (LSTM - Pers)**: **$+5.4872$**
- **Median Delta (LSTM - Pers)**: $+5.6596$
- **Wilcoxon Signed-Rank Statistic**: $0.0$ ($p = 3.7165e-29$)
- **Cohen's d**: $1.8465$
- **Win Breakdown**: LSTM wins on 0 / 167 windows (0.0\%), Persistence wins on 167 / 167 windows (100.0\%).

#### PC2 MAE Percentile Distribution ($K=3$)
| Model | p5 | p25 | p50 (Median) | p75 | p95 | Max |
|---|---|---|---|---|---|---|
| **LSTM** | 1.14 | 2.78 | 5.67 | 8.03 | 10.30 | 10.96 |
| **Persistence** | 0.00 | 0.01 | 0.01 | 0.03 | 0.07 | 0.32 |

---
