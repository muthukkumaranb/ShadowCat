# Phase 4 — Class Weighting Status Report

## Summary

**Class weighting is NOT applied despite the config saying `class_weighting: true`.**
However, adding it has no material impact on test F1.

## Part 1: Current Status

| Item | Value |
|------|-------|
| `model_config.yaml` `training.class_weighting` | `true` |
| `model_config.yaml` `training.loss` | `BCEWithLogitsLoss` |
| `trainer.py` `pos_weight` support | ✅ Supported |
| `run_lstm_frozen.py` passes `pos_weight` | ❌ **NO** |
| **Actual class weighting applied** | ❌ **NO** |

**Diagnosis**: `model_config.yaml` declares `class_weighting: true`, but
`run_lstm_frozen.py` (the script that trains the baseline LSTM) does NOT pass
`pos_weight` to `train()`. The config field is declarative only and is not read
by the training script.

**File evidence**:
- `ml1/configs/model_config.yaml`, line 32: `class_weighting: true`
- `ml1/lstm/trainer.py`, lines 312-317: `BCEWithLogitsLoss(pos_weight=...)` only when explicitly given
- `ml1/scripts/run_lstm_frozen.py`, lines 110-125: `train()` called WITHOUT `pos_weight`

## Part 2: Controlled Experiment (3 representative folds)

Train class balance is ~65% negative / 35% positive (pos_weight ≈ 1.86-1.93).

| Fold | Attack Type    | Unweighted val_F1 | Weighted val_F1 | Unweighted test_F1 | Weighted test_F1 |
|------|---------------|-------------------|-----------------|---------------------|-------------------|
| 0    | Botnet        | 0.728             | 0.697           | **0.667**           | **0.667**         |
| 10   | SSH-Bruteforce| 0.761             | 0.765           | **0.714**           | **0.714**         |
| 19   | DDOS-LOIC-UDP | 0.627             | 0.756           | **0.667**           | **0.667**         |

## Conclusion

**Class weighting is NOT the root cause.** The moderate class imbalance (65/35) is not
extreme enough for `pos_weight` to change outcomes. Both weighted and unweighted models
achieve identical test F1 on all three representative folds.

## Artifacts

- Checkpoints: `diagnostics/checkpoints/lstm_class_weighted_fold{0,10,19}.pt`
- Report: `diagnostics/phase4_class_weighting_report.json`
