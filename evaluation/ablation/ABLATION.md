# History-Shuffle Ablation

## Method
We performed an ablation study on the production Stacked Residual LSTM Ensemble for Onset forecasting across the 37 LOEO folds. We evaluated the model under three input conditions using the same threshold (0.7844) as the main benchmark:
- **(a) Normal input:** The standard chronological 30-window lookback sequence.
- **(b) Permuted history:** The 30 windows in the lookback were randomly permuted in time (averaged over 5 runs, seed 42).
- **(c) Repeated last window:** The entire 30-window sequence was overwritten by repeating the final (most recent) window 30 times.

## Results

| Condition | F1 | Precision | Recall | FPR |
|---|---|---|---|---|
| (a) Normal | 0.3060 | 0.5283 | 0.2154 | 0.0887 |
| (b) Permuted | 0.3060 | 0.5283 | 0.2154 | 0.0887 |
| (c) Repeated Last | 0.3060 | 0.5283 | 0.2154 | 0.0887 |

*Note: F1 dropped from the 0.91 benchmark because this evaluates on the corrected test subset (412 windows), identical to the LR benchmark.*

## Conclusion
The metrics for all three conditions are identical. This indicates that the recurrent (LSTM) component of the Stacked Residual Ensemble is either not utilizing the historical sequence ordering, or its additive residual contribution is completely eclipsed by the Logistic Regression base logit (which only depends on the final window's features). The temporal sequencing provides no measurable discriminative advantage over simply using the most recent state.
