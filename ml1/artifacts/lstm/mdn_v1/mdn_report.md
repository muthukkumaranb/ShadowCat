# Mixture Density Network (MDN) World Model - Experiment Report

## 1. Architecture Details
To address the "mean-seeking" degenerate behavior observed in the `LSTMGaussianWorldModel`, we replaced the single-Gaussian output head with a Mixture Density Network (MDN) head.
- **Base Encoder**: Existing LSTM architecture (hidden=64, layers=1).
- **Mixture Components (K)**: 5 components were used.
- **Parameters**: 
  - `pi_k`: Mixture weights parameterized via a linear layer followed by softmax.
  - `mu_k`: Component means parameterized via a linear layer (same as baseline).
  - `sigma_k^2`: Component variances parameterized via softplus (ensuring strict positivity) + epsilon.
- **Loss**: The MDN negative log-likelihood loss was implemented using the `logsumexp` trick for numerical stability to prevent underflow/overflow: `-logsumexp(log(pi) + log(N(y|mu, sigma^2)))`.

## 2. Evaluation Results (Horizon Predictability)
We evaluated the MDN model across horizons $K=1..5$ utilizing the identical chronological split protocol as the baseline. The retrained stage head was fed the expected component mean: $\sum_k (\pi_k \times \mu_k)$.

| K | Model F1 | Spread | Persistence F1 | Markov F1 | Beats Baselines? |
|---|----------|--------|----------------|-----------|------------------|
| 1 | 0.5530   | 0.2256 | 0.9517         | 0.6876    | False            |
| 2 | 0.5527   | 0.2258 | 0.9348         | 0.6154    | False            |
| 3 | 0.5523   | 0.2260 | 0.9178         | 0.6063    | False            |
| 4 | 0.5520   | 0.2263 | 0.9141         | 0.6104    | False            |
| 5 | 0.5517   | 0.2265 | 0.9059         | 0.6100    | False            |

**Predictability Horizon ($H^*$) = 0**

## 3. Mixture-Weight Diagnostic
We sampled 100 known-benign windows and 100 known-attack windows and examined the average predicted mixture weights ($\pi_k$) to see if the MDN head learns to differentiate between traffic states internally, even if the final expected value collapses.

| Condition | $\pi_1$ | $\pi_2$ | $\pi_3$ | $\pi_4$ | $\pi_5$ |
|-----------|---------|---------|---------|---------|---------|
| Benign    | 23.68%  | 31.69%  | 9.67%   | 29.93%  | 5.03%   |
| Attack    | 25.49%  | 16.25%  | 3.42%   | 37.31%  | 17.53%  |

**Observation**: The mixture weights *do* demonstrate a noticeable shift between benign and attack windows. For example, Component 2 is heavily favored for benign traffic (31.7% vs 16.3%), while Components 4 and 5 see a significant increase in probability mass during attack windows (e.g., Component 5 jumps from 5% to 17.5%).

## 4. Honest Verdict
Despite utilizing a Mixture Density Network (Bishop 1994) to overcome the single-Gaussian mean-seeking limitation, the **model still wholly fails to beat the trivial persistence baseline** ($H^* = 0$). 

However, the mixture weight diagnostic reveals an important nuance: the MDN *is* internally distinguishing between attack and benign states (allocating different probability mass across the 5 modes). The failure occurs when we collapse these components down to a single expected value ($\sum \pi_k \mu_k$) to pass into the downstream stage head, which likely re-introduces the smoothing effect we tried to avoid, or the downstream head itself cannot leverage the nuanced expected representation.

**Conclusion**: Negative result for end-to-end forecasting skill. The MDN improves internal representation separation, but this does not translate to improved downstream classification accuracy when taking the expected value. Future work should explore passing the full distribution (or sampling from it) into the stage head rather than collapsing it to a mean.
