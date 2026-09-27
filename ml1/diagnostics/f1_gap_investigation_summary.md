# F1 Gap Investigation Summary

**LR baseline F1**: 0.889 (onset, `future_attack_label`, 37-fold LOEO)
**LSTM baseline F1**: 0.750 (onset, `future_attack_label`, 37-fold LOEO)
**Gap**: 0.139

---

## Root Cause Ranking

### 🔴 #1 — PCA-32 Feature Compression (PRIMARY ROOT CAUSE)

**Severity**: Critical
**Evidence strength**: Conclusive

The LSTM operates on 32 PCA components while the LR operates on the full 406-D feature
space. Phase 5 proved this is catastrophic:

| Model | Features | Mean F1 |
|-------|----------|---------|
| LR (last-window) | 406-D full | **0.943** |
| LR baseline | 406-D full | **0.889** |
| LSTM baseline | 32-D PCA | **0.750** |
| LR (last-window) | 32-D PCA | **0.304** |

PCA-32 loses **68% of F1** compared to full features. DDOS-LOIC-UDP folds go from
perfect 1.000 F1 on full features to **0.000** on PCA-32. The unsupervised PCA
projection discards exactly the discriminative features that matter for classification.

**File evidence**:
- Config says PCA disabled: [`model_config.yaml:22-25`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/configs/model_config.yaml#L22-L25) (`pca.enabled: false`)
- Training script applies PCA anyway: [`run_lstm_frozen.py:61-67`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/scripts/run_lstm_frozen.py#L61-L67)
- Phase 5 results: [`diagnostics/phase5_last_window_report.json`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase5_last_window_report.json)

---

### 🟡 #2 — Insufficient Training (2 epochs in baseline)

**Severity**: Moderate
**Evidence strength**: Strong

The baseline LSTM script [`run_lstm_frozen.py:118`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/scripts/run_lstm_frozen.py#L118) trains for only **2 epochs**.
The config and the `train()` function support 30 epochs with early stopping. This
severely limits what the model can learn.

The stacked & calibrated LSTM (which trains properly with 30 epochs) achieves 0.913 F1
on the same PCA-32 features, demonstrating that adequate training partially compensates
for PCA compression — but still can't reach the full-feature LR's 0.943.

---

### 🟢 #3 — Config/Code Mismatches (Contributing factor, not causal)

**Severity**: Low (cosmetic/maintenance risk)

Two config fields don't match actual behavior:

| Config Field | Config Value | Actual Behavior |
|-------------|-------------|-----------------|
| `preprocessing.pca.enabled` | `false` | PCA IS applied in scripts |
| `training.class_weighting` | `true` | NOT applied in scripts |

**File evidence**:
- [`model_config.yaml:22-25`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/configs/model_config.yaml#L22-L25)
- [`model_config.yaml:32`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/configs/model_config.yaml#L32)

---

### ✅ #4 — Sigmoid/Loss Consistency (CONFIRMED FIXED)

**Severity**: None (no longer an issue)

The double-sigmoid bug is **fully resolved**. [`trainer.py:146`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/lstm/trainer.py#L146) correctly dispatches
to `forward_logits()` for training, keeping `BCEWithLogitsLoss` consistent with raw logits.

**File evidence**: [`diagnostics/phase1_sigmoid_loss_report.md`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase1_sigmoid_loss_report.md)

---

### ✅ #5 — Class Weighting (NOT a contributor)

**Severity**: None

Phase 4 showed that adding `pos_weight` (≈1.86-1.93) produces identical test F1 to
unweighted training on all 3 tested folds. The ~65/35 class split is not imbalanced
enough for weighting to matter.

**File evidence**: [`diagnostics/phase4_class_weighting_report.json`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase4_class_weighting_report.json)

---

### ✅ #6 — Threshold Protocol (Contributing factor, secondary)

**Severity**: Low

The baseline LSTM uses fixed t=0.38. Val-optimized thresholds improve F1 from 0.266
→ 0.442 on the diagnostics checkpoints, but even optimal thresholds can't overcome
the PCA information loss. This is a symptom, not a cause.

**File evidence**: [`diagnostics/threshold_reeval_report.json`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/threshold_reeval_report.json)

---

## Phase-by-Phase Findings

### Phase 1 — Sigmoid/Loss Consistency ✅

- `forward_logits()` returns raw logits → `BCEWithLogitsLoss` ✅
- `forward()` applies sigmoid → returns probabilities for inference ✅
- [`trainer.py:146`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/lstm/trainer.py#L146) dispatches to `forward_logits()` ✅
- **Double-sigmoid bug is fixed. NOT the root cause.**

### Phase 2 — PCA Status 🔴

- [`model_config.yaml`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/configs/model_config.yaml) says `pca.enabled: false`
- [`run_lstm_frozen.py:61-67`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/scripts/run_lstm_frozen.py#L61-L67) applies `apply_training_only_pca(purged, base_features, 32)`
- [`prepare_lstm_inputs()`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/lstm/ucs.py#L530-L542) does NOT include PCA — the canonical pipeline is PCA-free
- Model instantiated as `LSTMClassifier(input_size=32)` — locked to PCA dimensions
- **Config-code mismatch. PCA IS applied. This is the primary root cause.**

### Phase 3 — Threshold Re-evaluation 🟡

- Default t=0.50: mean F1 = 0.358 (25/37 folds at F1=0)
- Fixed t=0.38: mean F1 = 0.266
- Val-optimized: mean F1 = 0.442
- **Threshold helps but can't compensate for PCA information loss.**

### Phase 4 — Class Weighting ✅

- Config says `class_weighting: true` but `run_lstm_frozen.py` doesn't pass `pos_weight`
- Controlled experiment (3 folds): identical test F1 with and without weighting
- **NOT a contributor. Class balance (65/35) is moderate.**

### Phase 5 — Single-Window Signal 🔴 (KEY FINDING)

- Last-window LR on 406-D: **0.943 F1** (exceeds multi-window baseline!)
- Last-window LR on PCA-32: **0.304 F1** (catastrophic)
- **PCA-32 destroys discriminative signal. Most signal is per-window, not temporal.**
- DDOS-LOIC-UDP goes from F1=1.000 (406-D) to F1=0.000 (PCA-32) on all folds

### Phase 6 — GRU Architecture Swap ✅

| Fold | Attack Type | LSTM F1 | GRU F1 |
|------|------------|---------|--------|
| 0 | Botnet | 0.667 | 0.667 |
| 10 | SSH-Bruteforce | 0.714 | 0.286 |
| 19 | DDOS-LOIC-UDP | 0.667 | 0.667 |

GRU performs similarly or worse than LSTM on the same PCA-32 features. This confirms
the RNN architecture choice is not the bottleneck — the input features are.

**Checkpoints**: `diagnostics/checkpoints/gru_experiment_fold{0,10,19}.pt`
**File evidence**: [`diagnostics/phase6_gru_experiment_report.json`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase6_gru_experiment_report.json)

---

## Recommended Fix (NOT yet applied — pending review)

### Priority 1: Remove PCA from the LSTM data path

1. Modify the LSTM training script to use the full 406-D feature space instead of
   PCA-32. This means:
   - `LSTMClassifier(input_size=406)` instead of `input_size=32`
   - Remove the `apply_training_only_pca()` call
   - Use `prepare_lstm_inputs()` (the canonical function) which already omits PCA

2. Retrain with proper settings: 30 epochs, early stopping, the existing
   `BCEWithLogitsLoss` with `forward_logits()`.

3. Re-evaluate on all 37 LOEO folds with val-optimized threshold protocol.

### Priority 2: Fix config/code alignment

- Either update `model_config.yaml` to reflect reality, or make the training script
  actually read the config file. Currently the script hardcodes all parameters.

### Priority 3: Adopt val-optimized threshold protocol

- The evaluator already supports arbitrary thresholds. Standardize the val-sweep →
  freeze → test protocol across all evaluations.

### Expected outcome

Based on Phase 5, the LSTM on full 406-D features should achieve F1 ≥ 0.89 (matching
or exceeding the LR baseline), since even a simple LR on the same per-window features
reaches 0.943. The temporal modeling may provide an additional boost for certain attack
types that have temporal signatures.

---

## All Diagnostic Artifacts

| File | Description |
|------|-------------|
| [`diagnostics/phase1_sigmoid_loss_report.md`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase1_sigmoid_loss_report.md) | Sigmoid/loss consistency analysis |
| [`diagnostics/phase2_pca_report.md`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase2_pca_report.md) | PCA status in LSTM data path |
| [`diagnostics/phase3_threshold_report.md`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase3_threshold_report.md) | Threshold re-evaluation results |
| [`diagnostics/threshold_reeval_report.json`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/threshold_reeval_report.json) | Phase 3 raw data |
| [`diagnostics/phase4_class_weighting_report.md`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase4_class_weighting_report.md) | Class weighting analysis |
| [`diagnostics/phase4_class_weighting_report.json`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase4_class_weighting_report.json) | Phase 4 raw data |
| [`diagnostics/phase5_last_window_report.md`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase5_last_window_report.md) | Last-window signal analysis |
| [`diagnostics/phase5_last_window_report.json`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase5_last_window_report.json) | Phase 5 raw data |
| [`diagnostics/phase6_gru_experiment_report.json`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase6_gru_experiment_report.json) | Phase 6 GRU experiment data |
| [`diagnostics/checkpoints/`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/checkpoints) | Experiment checkpoints (class-weighted, GRU) |

## Diagnostic Scripts (new, non-destructive)

| File | Description |
|------|-------------|
| [`diagnostics/phase3_threshold_reeval.py`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase3_threshold_reeval.py) | Threshold sweep & frozen evaluation |
| [`diagnostics/phase4_class_weighting.py`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase4_class_weighting.py) | Class weighting experiment |
| [`diagnostics/phase5_last_window_baseline.py`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase5_last_window_baseline.py) | Last-window LR diagnostic |
| [`diagnostics/phase6_gru_experiment.py`](file:///c:/Users/Vicky/Documents/SIH_2026/ShadowCat/ml1/diagnostics/phase6_gru_experiment.py) | GRU architecture swap |
