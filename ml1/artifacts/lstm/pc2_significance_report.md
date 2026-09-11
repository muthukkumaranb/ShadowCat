# PC2 Attack Window Significance Test Report (v2 Retrain Cascade)

## Summary & Key Findings

- **Test Goal**: Evaluate whether the retrained LSTM Gaussian World Model (`v2`) outperforms the Persistence baseline on Principal Component 2 (PC2) during attack windows in the test set.
- **Model Checkpoint**: `artifacts/lstm/gaussian_next_state_best_v2.pt`
- **Commit**: `d82887a`
- **Dataset**: `data/ucs/ucs_windows.parquet` (Test split, contiguous attack windows).
- **Statistical Test**: Wilcoxon signed-rank test (two-sided paired non-parametric test) + Cohen's d effect size.
- **Overall Verdict**: **PERSISTENCE BASELINE IS STATISTICALLY SUPERIOR ON PC2 (p < 10⁻⁸)**.
  - At **K=1** ($N=165$ attack windows): Mean MAE difference $\Delta = +2.3530$ points (Persistence advantage), $p = 5.26 \times 10^{-9}$, Cohen's $d = 0.3760$. Persistence wins $69.1\%$ ($114/165$) of attack windows.
  - At **K=2** ($N=166$ attack windows): Mean MAE difference $\Delta = +5.9057$ points (Persistence advantage), $p = 1.52 \times 10^{-17}$, Cohen's $d = 0.6182$. Persistence wins $80.7\%$ ($134/166$) of attack windows.
  - At **K=3** ($N=167$ attack windows): Mean MAE difference $\Delta = +8.1275$ points (Persistence advantage), $p = 5.79 \times 10^{-16}$, Cohen's $d = 0.6678$. Persistence wins $75.4\%$ ($126/167$) of attack windows.

---

## Resolution of Prior Report Discrepancy

Earlier documentation cited two conflicting K=3 p-values for the PC2 significance test:
1. `p = 1.16 × 10⁻⁵` (committed in previous report revision)
2. `p = 0.4310` (cited verbally in an earlier summary)

### Explicit Clarification of Discrepancy Root Cause
- The previously committed value of `p = 1.16 × 10⁻⁵` was produced by running `test_pc2_attack_significance.py` against `artifacts/lstm/chunk_4_gaussian_fixed.pt` — a legacy walk-forward chunk checkpoint from the earlier chronological-export work (which evaluated a 32-dim PCA representation directly and included uncalibrated label-correlated features).
- The `0.4310` value was from an uncommitted intermediate script run on a different sequence masking protocol.
- **Authoritative Fresh Result**: The canonical re-run on `gaussian_next_state_best_v2.pt` (commit `d82887a`, 406-dim input space mapped via 32-component PCA) yields **$p = 5.79 \times 10^{-16}$** for K=3 ($N=167$). Nothing was copied or paraphrased from either prior run. On the true v2 checkpoint, Persistence statistically significantly outperforms the LSTM world model on PC2 during attack windows across all forecast horizons.

---

## Detailed Results by Rollout Horizon $K$

### Rollout Horizon $K=1$ ($N = 165$ Attack Windows)
- **LSTM Mean MAE**: $5.53$ | **Persistence Mean MAE**: $3.18$ | **Mean Delta (LSTM - Pers)**: **$+2.3530$**
- **Median Delta (LSTM - Pers)**: $+0.8701$
- **Wilcoxon Signed-Rank Statistic**: $3259.0$ ($p = 5.2642 \times 10^{-9}$)
- **Cohen's d**: $0.3760$
- **Win Breakdown**: LSTM wins on $51 / 165$ windows ($30.9\%$), Persistence wins on $114 / 165$ windows ($69.1\%$).

#### PC2 MAE Percentile Distribution ($K=1$)
| Model | p5 | p25 | p50 (Median) | p75 | p95 | Max |
|---|---|---|---|---|---|---|
| **LSTM** | 0.15 | 0.87 | 2.47 | 6.67 | 16.02 | 29.77 |
| **Persistence** | 0.06 | 0.33 | 0.78 | 1.92 | 9.94 | 31.44 |

---

## Rollout Horizon $K=2$ ($N = 166$ Attack Windows)
- **LSTM Mean MAE**: $9.37$ | **Persistence Mean MAE**: $3.46$ | **Mean Delta (LSTM - Pers)**: **$+5.9057$**
- **Median Delta (LSTM - Pers)**: $+2.5987$
- **Wilcoxon Signed-Rank Statistic**: $1643.0$ ($p = 1.5211 \times 10^{-17}$)
- **Cohen's d**: $0.6182$
- **Win Breakdown**: LSTM wins on $32 / 166$ windows ($19.3\%$), Persistence wins on $134 / 166$ windows ($80.7\%$).

#### PC2 MAE Percentile Distribution ($K=2$)
| Model | p5 | p25 | p50 (Median) | p75 | p95 | Max |
|---|---|---|---|---|---|---|
| **LSTM** | 0.21 | 1.01 | 4.30 | 11.82 | 27.40 | 34.02 |
| **Persistence** | 0.03 | 0.19 | 0.65 | 1.57 | 8.95 | 31.91 |

---

## Rollout Horizon $K=3$ ($N = 167$ Attack Windows)
- **LSTM Mean MAE**: $12.16$ | **Persistence Mean MAE**: $4.03$ | **Mean Delta (LSTM - Pers)**: **$+8.1275$**
- **Median Delta (LSTM - Pers)**: $+2.5217$
- **Wilcoxon Signed-Rank Statistic**: $1949.0$ ($p = 5.7855 \times 10^{-16}$)
- **Cohen's d**: $0.6678$
- **Win Breakdown**: LSTM wins on $41 / 167$ windows ($24.6\%$), Persistence wins on $126 / 167$ windows ($75.4\%$).

#### PC2 MAE Percentile Distribution ($K=3$)
| Model | p5 | p25 | p50 (Median) | p75 | p95 | Max |
|---|---|---|---|---|---|---|
| **LSTM** | 0.16 | 1.01 | 5.75 | 16.01 | 34.19 | 43.80 |
| **Persistence** | 0.06 | 0.36 | 0.91 | 2.05 | 9.00 | 31.62 |

---

## Verification & Execution Details

- **Script Executed**: `scripts/test_pc2_attack_significance.py`
- **Target Model**: `artifacts/lstm/gaussian_next_state_best_v2.pt`
- **Raw Command Output**:
```text
======================================================================
Significance Test: LSTM v2 World Model vs Persistence on PC2, Attack Windows
Model Checkpoint: artifacts\lstm\gaussian_next_state_best_v2.pt
======================================================================

--- K=1 (N=165 attack windows) ---
  STEP 1: Paired Significance
    Mean(LSTM-Pers) = 2.3530
    Median(LSTM-Pers) = 0.8701
    Std(LSTM-Pers) = 6.2577
    Wilcoxon stat  = 3259.0
    p-value        = 5.2642e-09
    Cohen's d      = 0.3760
    ==> Persistence advantage is STATISTICALLY SIGNIFICANT (p < 0.05).

--- K=2 (N=166 attack windows) ---
  STEP 1: Paired Significance
    Mean(LSTM-Pers) = 5.9057
    Median(LSTM-Pers) = 2.5987
    Std(LSTM-Pers) = 9.5526
    Wilcoxon stat  = 1643.0
    p-value        = 1.5211e-17
    Cohen's d      = 0.6182
    ==> Persistence advantage is STATISTICALLY SIGNIFICANT (p < 0.05).

--- K=3 (N=167 attack windows) ---
  STEP 1: Paired Significance
    Mean(LSTM-Pers) = 8.1275
    Median(LSTM-Pers) = 2.5217
    Std(LSTM-Pers) = 12.1714
    Wilcoxon stat  = 1949.0
    p-value        = 5.7855e-16
    Cohen's d      = 0.6678
    ==> Persistence advantage is STATISTICALLY SIGNIFICANT (p < 0.05).
```
