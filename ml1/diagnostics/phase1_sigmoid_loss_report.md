# Phase 1 — Sigmoid/Loss Consistency Report

## Summary

**The double-sigmoid bug is FIXED.** The trainer correctly uses `forward_logits()` (raw logits) paired with `BCEWithLogitsLoss`, which is internally consistent. There is no double-sigmoid.

## Evidence

### 1. `model.py` — LSTMClassifier.forward() applies sigmoid (line 265)

```python
# ml1/lstm/model.py, line 260-265
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Return positive-class probabilities for compatibility."""
    return self.sigmoid(self.forward_logits(x))
```

- `forward()` calls `forward_logits()` → produces raw logits → then applies `self.sigmoid`.
- `forward_logits()` (lines 186-258) does NOT apply sigmoid — it returns raw linear output.
- **`forward()` output: probabilities in [0, 1].**

### 2. `trainer.py` — Loss criterion is `BCEWithLogitsLoss` (line 313)

```python
# ml1/lstm/trainer.py, line 311-317
if classification_task:
    if pos_weight is None:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(pos_weight), device=device),
        )
```

### 3. `trainer.py` — Training loop calls `forward_logits()`, NOT `forward()` (line 146)

```python
# ml1/lstm/trainer.py, line 146
logits = model.forward_logits(X_batch) if hasattr(model, "forward_logits") else model(X_batch)
```

This is the critical fix. The `train_one_epoch()` and `evaluate_loss()` functions (lines 146, 186) both check for `forward_logits` and use it when available, bypassing the sigmoid in `forward()`. This means:

- **Training path**: `forward_logits()` → raw logits → `BCEWithLogitsLoss` ✅ (single sigmoid inside loss)
- **Inference path**: `forward()` → `forward_logits()` + sigmoid → probabilities ✅

### 4. `model_config.yaml` — Config says `BCEWithLogitsLoss` (line 39)

```yaml
# ml1/configs/model_config.yaml, line 39
loss: BCEWithLogitsLoss
```

This matches what `trainer.py` actually instantiates.

### 5. Validation metrics computation is also correct (line 369-370)

```python
# ml1/lstm/trainer.py, line 369-370
logits = model.forward_logits(X_batch.to(device)) if hasattr(model, "forward_logits") else model(X_batch.to(device))
validation_probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
```

Logits are correctly sigmoided once for metric computation.

## Conclusion

The double-sigmoid bug described in the background is **already fixed**. The architecture cleanly separates:
- `forward_logits()` → raw logits for training with `BCEWithLogitsLoss`
- `forward()` → probabilities for inference via explicit `sigmoid`

The trainer correctly dispatches to `forward_logits()`. **This is not the root cause of the F1 gap.**
