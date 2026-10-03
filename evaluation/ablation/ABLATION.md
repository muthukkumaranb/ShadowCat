# Stacked LSTM History Ablation Results

## Detection

| Mode | F1 | Precision | Recall | ROC AUC | Mean Abs Prob Change |
|---|---|---|---|---|---|
| normal | 0.9868 | 0.9739 | 1.0000 | 1.0000 | 0.000000 |
| permuted | 0.9835 | 0.9675 | 1.0000 | 1.0000 | 0.002368 |
| repeat_last | 0.9868 | 0.9739 | 1.0000 | 1.0000 | 0.001371 |
| lr_stage_only | 0.9835 | 0.9675 | 1.0000 | 1.0000 | 0.013492 |

## Onset

| Mode | F1 | Precision | Recall | ROC AUC | Mean Abs Prob Change |
|---|---|---|---|---|---|
| normal | 0.9427 | 0.9801 | 0.9080 | 0.9621 | 0.000000 |
| permuted | 0.9427 | 0.9801 | 0.9080 | 0.9622 | 0.001103 |
| repeat_last | 0.9427 | 0.9801 | 0.9080 | 0.9616 | 0.001732 |
| lr_stage_only | 0.9460 | 0.9803 | 0.9141 | 0.9620 | 0.011976 |

## Conclusion
the LSTM residual adds no measurable gain; the improvement over the LR baseline comes from the stacked model's calibrated LR stage.
