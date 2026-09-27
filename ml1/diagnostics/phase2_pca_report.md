# Phase 2 — PCA Status in the LSTM Data Path

## Summary

**PCA is DISABLED in the model config but IS PRESENT in the LSTM evaluation scripts.**

This is a critical finding. The `model_config.yaml` says PCA is disabled, but the actual LSTM training/evaluation scripts (e.g., `run_lstm_frozen.py`) apply PCA with 32 components. This is a significant inconsistency.

## Evidence

### 1. `model_config.yaml` — PCA is disabled (lines 22-25)

```yaml
# ml1/configs/model_config.yaml, lines 22-25
preprocessing:
  pca:
    enabled: false
    components: null
```

### 2. `ucs.py` — `prepare_lstm_inputs()` does NOT apply PCA (lines 530-542)

The canonical pipeline function `prepare_lstm_inputs()` does validation, purge/embargo, robust-scaling, and sequence building — but **no PCA step**:

```python
# ml1/lstm/ucs.py, lines 530-542
def prepare_lstm_inputs(windows, *, features=None, config=UCSConfig()):
    columns = validate_ucs_windows(windows, features=features, config=config)
    purged = purge_and_embargo(windows, config=config)
    scaler = UCSRobustScaler(columns).fit(purged.loc[purged["split"] == "train"])
    normalized = scaler.transform_frame(purged)
    sequences = build_lstm_sequences(normalized, features=columns, config=config)
    return sequences, columns, scaler, normalized
```

### 3. `run_lstm_frozen.py` — APPLIES PCA with 32 components (lines 61-67)

```python
# ml1/scripts/run_lstm_frozen.py, lines 61-67
transformed, pca = apply_training_only_pca(
    purged,
    base_features,
    32,
    random_state=42,
)
features = [f"pca_{index}" for index in range(32)]
```

And the model is instantiated with `input_size=32`:

```python
# ml1/scripts/run_lstm_frozen.py, line 106
model = LSTMClassifier(input_size=32, hidden_size=64, num_layers=1, dropout=0.20)
```

### 4. Impact assessment

- The LSTM receives 32 PCA components instead of the full 406 features.
- The LR baseline operates on the full 406-D feature space (confirmed from verification scripts).
- PCA collapses 406 dimensions to 32 — a **12.7x** reduction.
- The `models/pca_32.pkl` (128 KB) file exists, suggesting this PCA is also used in production.
- The previous concern about a degenerate PCA with ~98.7% variance on one component relates to `models/pca_32_old_degenerate.pkl`.

### 5. `preprocessing.py` — No PCA present

The `preprocessing.py` module contains only `StandardScaler` and validation functions. No PCA.

### 6. `data.py` — No PCA present

The `data.py` module handles tensor conversion and dataloading. No PCA.

## Conclusion

**PCA is actively in the LSTM's data path** via the training/evaluation scripts, even though `model_config.yaml` says `pca.enabled: false`. The LSTM sees a compressed 32-D representation while the LR sees the full 406 features.

**This is a strong candidate contributor to the F1 gap.** While 32 PCA components may retain most of the global variance, PCA is an unsupervised linear transform that may discard precisely the discriminative signal that matters for classification. The LR operating on the full feature space can learn its own linear projection optimized for the classification objective — PCA pre-projecting into a different subspace may lose useful signal.

However, this alone may not explain the entire gap. Phases 3-5 will provide further evidence.
