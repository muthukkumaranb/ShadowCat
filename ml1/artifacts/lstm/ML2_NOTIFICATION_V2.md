# Official ML2 Notification: v2 World Model Checkpoint & Contract Available

**Date**: 2026-09-10  
**Sender**: ML1 Lead / Coding Agent  
**Recipient**: ML2 Team & System Architect  

---

## Executive Summary

1. **v2 Checkpoint Available**: The retrained probabilistic LSTM World Model checkpoint (`v2`) is officially available at:
   - `artifacts/lstm/gaussian_next_state_best_v2.pt`
   - `artifacts/lstm/probabilistic_world_model/gaussian_next_state_best_v2.pt`
2. **v2 Inference Contract Available**:
   - `artifacts/lstm/inference_scaler_v2.yaml`
   - `artifacts/lstm/inference_feature_order_v2.json`
3. **Data Correction Applied**: Option A (Corrected-but-simulated: 12 fabricated packet-level columns zeroed out and `mask_has_packet_level_features = 0.0` for 14-02-2018 per the project's standing "missing != zero" rule).

---

## Action Items & Status for ML2

> [!WARNING]
> **ML2 Existing HOLD Decision Notice**:
> ML2's `z_t_encoder.py` previously copied the OLD checkpoint (`gaussian_next_state_best.pt`) into their repository and built their multi-seed ablation and GNN adopt/hold decision (currently: **HOLD**) on $z(t)$ state representations derived from it.
> 
> Because the World Model has now been retrained on the corrected 12-column packet data:
> - **State Stale Flag**: The existing $z(t)$ representations and multi-seed ablation results generated under the OLD checkpoint are technically **stale**.
> - **Handoff Decision**: It is at the Team Lead's discretion whether ML2 should re-run the multi-seed ablation pipeline on the `v2` checkpoint's $z(t)$ state vectors or accept the old HOLD decision as a documented approximation (given that the retrain reflects a data correction rather than an architectural modification).
