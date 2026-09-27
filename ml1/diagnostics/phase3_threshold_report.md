# Phase 3 — Frozen-Threshold Evaluation Protocol Report

## Summary

**The threshold selection protocol is a significant contributor to the F1 gap.**

The existing `lstm_diagnostics` checkpoints (the "baseline" LSTM) produce heavily compressed
probability distributions, making the default 0.5 threshold catastrophic. Val-optimized
thresholds improve performance substantially but remain well below the LR baseline.

## Evidence

### Re-evaluation results (37-fold LOEO, onset/future_attack_label)

| Threshold Strategy | Mean Test F1 |
|--------------------|-------------|
| Default t=0.50     | **0.358**   |
| Fixed t=0.38       | **0.266**   |
| Val-optimized (frozen) | **0.442** |
| LR baseline reference | **0.888** |

### Key observations

1. **Default 0.5 threshold is catastrophic for most folds.**
   - 25 of 37 folds produce F1=0.000 at t=0.5, meaning the model outputs probabilities
     almost entirely below 0.5 for positive cases.
   - Only folds 15-17 and 20, 23-27, 33-36 achieve F1>0 at t=0.5 (these are DDOS-LOIC-UDP folds).

2. **Val-optimized thresholds mostly cluster at 0.05 (early folds) or 0.35 (later folds).**
   - This indicates the model's probability distribution is heavily skewed toward 0.
   - Even with optimal threshold selection, mean F1 only reaches 0.442.

3. **The probability distributions are degenerate** (most outputs < 0.1), indicating
   the model itself is poorly trained, not just poorly thresholded.

### Important context note

These results are from the `lstm_diagnostics` checkpoints, which appear to be from an
earlier, weaker training run. The **stacked & calibrated LSTM** (`lstm_stacked/onset/`)
checkpoints achieve 0.913 F1 but use a different architecture (ResidualLSTMClassifier
with LR base + LSTM residual), so they can't be directly compared for this threshold
protocol analysis.

The actual "LSTM baseline" referenced in the F1 gap question uses the
`run_lstm_frozen.py` script with **only 2 epochs of training**, which also partially
explains the weak performance.

## Artifact

Results saved to: `diagnostics/threshold_reeval_report.json`

## Conclusion

**Threshold is a contributing factor but NOT the primary root cause.** The underlying
probability distribution is degenerate (clustered near 0), indicating the model itself
is undertrained or receiving poor input features (PCA-compressed). Even perfect threshold
selection can't recover the discriminative signal if the model hasn't learned it.
