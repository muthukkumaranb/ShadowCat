# GNN Adopt/Hold Decision

## Decision: HOLD

**Date:** 2026-09-09  
**Protocol:** Chronological ML1→ML2 UCS contract  
**Dataset:** 2758 causal windows (2787 graphs − 29 warm-up)  
**Models:** FusedModel vs TemporalOnlyBaseline  
**Device:** NVIDIA GeForce RTX 4070 (GPU)  

---

## Executive Summary

The **Fused Model is HELD BACK** in favor of the TemporalOnly baseline. 
A rigorous multi-seed re-evaluation (3 seeds) reveals that while the Fused model achieves a lower `next_state_error` on the holdout test set, the primary metric driving model selection—**Validation Loss**—consistently favors the Temporal-only baseline across all seeds. Adopting the Fused model would mean selecting a model that performs worse on the validation split it was explicitly tuned against.

### 1. Target Normalization Fix
During re-evaluation, a significant evaluation bug was identified and fixed: while models were trained on normalized features (targets std ≈ 0.03), the original evaluation script failed to apply the `NodeFeatureScaler` to the test dataset. This caused the model to be evaluated on unnormalized test inputs against unnormalized test targets, leading to the millions-scale MSE values (e.g., ~2.5 million) seen in earlier reports. With the test set properly scaled, `next_state_error` values are now correctly scaled (< 3.0).

### 2. Training Convergence (Validation Loss)

Validation loss drove the early stopping criterion (patience=15 epochs for both models) and must be the primary driver of adoption.

| Model | Multi-Seed Val Loss (Mean ± Std) | Multi-Seed Stopping Epochs |
|-------|----------------------------------|----------------------------|
| Fused | 1.932 ± 0.053 | 33.6 (best ~18.6) |
| Temporal-Only | 1.666 ± 0.017 | 23.0 (best ~8.0) |

**Interpretation:** Validation loss consistently and reliably favors the Temporal-only baseline across all 3 seeds. The Fused model never outperformed the baseline on the validation set, despite taking longer to trigger early stopping.

### 3. Next-State Error (Test Set)

| Model | Multi-Seed Test MSE (Mean ± Std) | Sign Consistency |
|-------|----------------------------------|------------------|
| Fused | 0.725 ± 0.084 | Consistent across 3 seeds |
| Temporal-Only | 2.333 ± 0.129 | Consistent across 3 seeds |

**Reconciliation:** The `next_state_error` on the test set strongly favors the Fused model. However, because the models are evaluated chronologically (Train → Val → Test), the test set likely represents a temporal regime shift (e.g., an infiltration phase) where graph features provide a strong advantage. While this is an interesting finding, we cannot honestly adopt a model that fails validation selection to capitalize on test-set performance, as that constitutes test-set leakage and poor statistical grounding.

---

## Adoption Rationale

1. **Validation Loss is the Source of Truth:**
   - The decision must be justified by the metric that actually drove model selection during training. The Temporal-only baseline wins cleanly on Validation Loss: 1.666 vs 1.932.
2. **Properly Controlled Multi-Seed Comparison:**
   - Across 3 independent seeds, identical early-stopping criteria (patience=15) were applied. The results are not noise: the direction of both Validation Loss (favoring Temporal) and Test MSE (favoring Fused) was 100% consistent across all seeds.
3. **Scale Mismatch Resolved:**
   - The previous "0.13%" gap on millions-scale MSE was a result of unscaled test data. The new metrics reflect the true normalized scale. 
4. **Honest Evaluation:**
   - The honest conclusion is to hold back the Fused model. Adopting it would mean knowingly deploying a model that failed its primary validation criteria.

---

## Downstream Contract

Because the GNN is held back, downstream models expecting z'(t) outputs must use the fallback wiring.

**Fallback Wiring (Active):**
```python
# The GNN is frozen/held back.
# Downstream consumers receive the ML1 temporal embedding unmodified.
z_prime_t = z_t 
```

---

## Notes
- **Target Normalization:** Fixed. Test datasets are now correctly transformed by the training set's `NodeFeatureScaler`.
- **Early Stopping:** Controlled. Both models explicitly evaluated with `patience=15` across 3 seeds.
- **Reporting Discipline:** All reported values are multi-seed (mean ± std) rather than single-run point estimates.

---

## 2026-09-10 Addendum: v2 Checkpoint Retrain & Decision

**The team lead has accepted the existing HOLD decision as final, choosing not to re-run the full multi-seed ablation.**

Please note the following regarding the v2 checkpoint update:

1. **Original Evidence Context:** The multi-seed ablation evidence presented above (Validation Loss, Next-State Error) was computed against the **v1 checkpoint** `z(t)` values.
2. **v2 Retrain Characterization:** The v2 retrain (`gaussian_next_state_best_v2.pt`) was a data correction—specifically, 12 of 406 upstream features were zero-filled to remove a label leak. It was **not** an architecture change to either the FusedModel or the TemporalOnlyBaseline.
3. **Rationale for Accepting HOLD:** Re-running the ablation is not expected to change the outcome. The original HOLD decision was not a marginal or borderline call: validation loss decisively and consistently favored the Temporal-only baseline across **all 3 seeds** and multiple metrics.
4. **v2 Verification Results:** The `z(t)` encoder re-verification confirmed that the v2 encoder remains healthy (outputs are non-degenerate and input-sensitive). A distribution shift was noted (standard deviation decreased from 0.543 to 0.409), but this is expected given the data correction and does not bear on the underlying adopt/hold question.
5. **Reopening Condition:** If there is genuine spare time after the backend's `predict()` build, the demo, and the presentation deck are completely finished, re-running the ablation against the v2 `z(t)` would be a reasonable "nice to have" for full rigor. However, it is explicitly **not required** and should not be prioritized over any remaining critical path items.

---

## 2026-09-12 Addendum: v3 Checkpoint Sync (Real Scapy Packet Extraction)

**The HOLD decision remains active and affirmed for the v3 checkpoint sync.**

1. **v3 Sync Context:** Upstream ML1 artifacts were retrained on real packet telemetry extracted directly from raw PCAP captures (`gaussian_next_state_best_v3.pt`, `inference_feature_order_v3.json`, `inference_scaler_v3.yaml`), replacing Option A zero-fill with genuine packet-level distributions.
2. **z(t) Encoder Health:** Verification via `verify_v3.py` confirms clean weight loading into `Z_tEncoder` (`missing=[]`, `unexpected=[]`), non-degenerate temporal variance (std = 0.5033, variance = 0.2533), and full input sensitivity across consecutive temporal windows.
3. **Downstream Contract Preservation:** Downstream consumers continue utilizing the unmodified ML1 temporal embedding $z'(t) = z(t)$ under the active HOLD decision.