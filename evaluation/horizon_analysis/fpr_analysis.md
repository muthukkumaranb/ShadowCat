# FPR Analysis and Reframing

## Current FPR Unit
By reviewing the evaluation scripts (e.g., `ml1/examples/evaluate_loeo.py` and other benchmarking scripts), it is clear that the False Positive Rate (FPR) is computed at the **per-window** level. The model predicts a hazard probability for each time window, and the confusion matrix (`fp / (fp + tn)`) is evaluated across these discrete windows.

## The Scaling Problem
The current claims state a "5% FPR target" ($\tau=0.15$) and an achieved FPR of 0.8% (0.0080).
Because this metric is per-window, it scales with time and the number of monitored segments.

Assuming a **1-minute window size**:
- **Windows per day per segment**: 60 minutes * 24 hours = 1,440 windows/day.
- At an **FPR of 0.8%** (0.008), the model will generate:
  - $1,440 \times 0.008 = 11.52$ false alerts per day **per monitored segment**.

In a realistic enterprise environment with, for example, **100 monitored network segments** (e.g., /24 subnets or host groups evaluated independently):
- Total windows per day: $1,440 \times 100 = 144,000$.
- Expected false alerts per day: $144,000 \times 0.008 = \mathbf{1,152}$ false alerts per day.

This volume of false alerts causes severe alert fatigue, rendering a "low" percentage like 0.8% functionally unacceptable for Security Operations Center (SOC) teams at scale.

## Proposed Replacement
Instead of presenting a bare percentage (e.g., "tau=0.15 / 5% FPR"), the presentation and documentation should frame performance using an **Alerts-per-Day vs. Detection Rate** (or Lead Time) tradeoff curve. 

**Example Proposed Slide Language/Table**:
*At 100 monitored segments (1-minute windows)*:
| Alert Threshold ($\tau$) | Expected False Alerts / Day | Detection Rate (Recall) |
|--------------------------|-----------------------------|-------------------------|
| 0.85                     | ~2                          | 72%                     |
| 0.60                     | ~15                         | 81%                     |
| 0.15 (Current target)    | ~7200 (5% FPR)              | 95%                     |
| **Current tuned model**  | **~1152 (0.8% FPR)**        | **88%**                 |

*Note: The SOC realistically can investigate only 10-20 alerts per day from this pipeline, meaning the threshold must be tuned much higher, accepting a lower detection rate, or the pipeline must aggregate alerts across time (e.g. per-episode rather than per-window).*
