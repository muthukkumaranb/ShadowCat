# ShadowCat Model Advancement Summary

## 1. Predictability Horizon (H*) Analysis
Using a rolling-origin evaluation (5 chronologically spaced test splits) and maintaining the strict 35-window purge/embargo boundary, we tested the model's forecasting F1 skill at horizons K=1..5 against two baselines:
- **Persistence**: Forecasted stage equals current stage (no change).
- **Markov Transition**: Forecasted stage based on empirical transition probabilities observed in the training data.

**Results (per-horizon averages across rolling origins):**
| Horizon (K) | Model F1 | Spread (Std Dev) | Persistence F1 | Markov F1 | Beats Baselines? |
|---|---|---|---|---|---|
| 1 | 0.4528 | 0.0383 | 0.9517 | 0.6876 | False |
| 2 | 0.5527 | 0.2258 | 0.9348 | 0.6154 | False |
| 3 | 0.5523 | 0.2260 | 0.9178 | 0.6063 | False |
| 4 | 0.5520 | 0.2263 | 0.9141 | 0.6104 | False |
| 5 | 0.5517 | 0.2265 | 0.9059 | 0.6100 | False |

**Predictability Horizon H* = 0**. The model currently fails to outperform the trivial Persistence baseline at all tested horizons (Persistence heavily dominates due to high class imbalance/inertia). This highlights that the "K=4-5 exploratory" claims should be fully retracted, and the K=1-3 validation must be re-evaluated to account for this zero marginal skill over persistence.

## 2. Reframed False Positive Rate (Alerts per Day)
The existing 0.8% FPR metric is calculated **per-window** (not per-flow or per-episode), as confirmed by reviewing the `fp / (fp + tn)` calculation in evaluation scripts over window labels. This means the number of false positives scales directly with time and environment size.

At a stated 1-minute window size, an FPR of 0.8% translates to:
- ~11.5 false alerts per day **per monitored segment**.
- **~1,152 false alerts per day** across a realistic 100-segment enterprise deployment.

This level of noise will cause immediate alert fatigue for SOC analysts.
*Recommendation for Deck*: Replace the bare "5% FPR target / 0.8% achieved" percentage with a tradeoff curve (e.g., Alert Volume vs. Detection Rate) to accurately convey the operational cost of the chosen threshold.

## 3. Conformal Joint-Coverage
[Pending finalized summary from Person 1 - `feature/conformal-mimo` branch]

## 4. Cross-Dataset Evaluation
[Pending finalized summary from Person 2 - `feature/cross-dataset-eval` branch]

## Threat Model & Limitations
**Limitation:** A sophisticated attacker could pace their campaign actions to stay under the model's predictability horizon (which is currently functionally zero). There is currently no published defense in this class of models against an attacker deliberately stretching their lateral movement over days or weeks to evade temporal correlation window constraints. This remains a disclosed, open research gap that adversaries could exploit.
