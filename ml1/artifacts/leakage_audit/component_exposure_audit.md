# Phase 1: Component Exposure Audit — Packet-Level Feature Leakage

**Repository**: `LSTM-type-model-CICIDS2018-UCS`  
**Audit Date**: 2026-09-09  
**Target Scope**: Evaluation of 9 fabricated packet-level feature columns (`pkt_ttl_min`, `pkt_ttl_max`, `pkt_ttl_std`, `pkt_ttl_mode`, `pkt_payload_size_p25`, `pkt_payload_size_p50`, `pkt_payload_size_p75`, `pkt_payload_size_p95`, `pkt_tcp_retrans_count`) derived from label fallbacks on `14-02-2018`.

---

## Component Exposure Summary Table

| Trained Component | Live Training Input Exposed? | Evidence File / Location | Exact Feature Entry / Code Mechanism |
| :--- | :---: | :--- | :--- |
| **Logistic Regression Baseline** | **YES** | [`scripts/run_lr_frozen.py:126`](file:///c:/Users/Vicky/Documents/SIH_2026/LSTM-type-model-CICIDS2018-UCS/scripts/run_lr_frozen.py#L126)<br>[`artifacts/lr_set_a/set_a_features.json`](file:///c:/Users/Vicky/Documents/SIH_2026/LSTM-type-model-CICIDS2018-UCS/artifacts/lr_set_a/set_a_features.json) | Features loaded via `set_a_features.json` (lines 386–394 contain all 9 leaking packet columns). Passed directly to `x_train = train_df[feature_columns].to_numpy()` at line 65. |
| **LSTM World Model** | **YES** | [`lstm/ucs.py:83-106`](file:///c:/Users/Vicky/Documents/SIH_2026/LSTM-type-model-CICIDS2018-UCS/lstm/ucs.py#L83-L106)<br>[`artifacts/lstm/inference_feature_order_v1.json`](file:///c:/Users/Vicky/Documents/SIH_2026/LSTM-type-model-CICIDS2018-UCS/artifacts/lstm/inference_feature_order_v1.json#L395-L405) | `validate_ucs_windows` includes all 406 numerical columns. `inference_feature_order_v1.json` explicitly lists all 9 leaking packet columns in the 406-dimensional input tensor fed to `build_lstm_sequences`. |
| **Hazard Head** | **YES** | [`scripts/train_hazard_head.py:89,134`](file:///c:/Users/Vicky/Documents/SIH_2026/LSTM-type-model-CICIDS2018-UCS/scripts/train_hazard_head.py#L89-L134) | Line 89 invokes `validate_ucs_windows` (406 features). Line 134 passes all 406 features (including the 9 leaking packet columns) into `apply_training_only_pca(purged, features, 32)` to produce input features for H=1, H=2, and H=5 hazard heads. |
| **Stage Head** | **YES** | [`scripts/train_stage_head.py:80,115`](file:///c:/Users/Vicky/Documents/SIH_2026/LSTM-type-model-CICIDS2018-UCS/scripts/train_stage_head.py#L80-L115) | Line 80 invokes `validate_ucs_windows` (406 features). Line 115 builds 406-dimensional sequence tensors `build_next_state_sequences(purged, features=features)`. The input sequences containing the 9 leaking features are passed to the World Model to infer $\hat{S}_{t+1}$ next-state vectors. |

---

## Detailed Evidence by Component

### 1. Logistic Regression Baseline (`run_lr_frozen.py`)
- **Status**: Exposed (**YES**)
- **Mechanism**: Reads `artifacts/lr_set_a/set_a_features.json` at line 126.
- **Verification**: `set_a_features.json` contains:
  - `"pkt_ttl_min"`
  - `"pkt_ttl_max"`
  - `"pkt_ttl_std"`
  - `"pkt_ttl_mode"`
  - `"pkt_payload_size_p25"`
  - `"pkt_payload_size_p50"`
  - `"pkt_payload_size_p75"`
  - `"pkt_payload_size_p95"`
  - `"pkt_tcp_retrans_count"`
- Slices `train_df` and `test_df` directly by `feature_columns` without masking or dropping packet-level columns.

### 2. LSTM World Model (`lstm/ucs.py`)
- **Status**: Exposed (**YES**)
- **Mechanism**: `validate_ucs_windows()` extracts all numeric columns by default.
- **Verification**: `artifacts/lstm/inference_feature_order_v1.json` defines the official input feature contract for the 406-dimensional input layer of the LSTM. Entries 395 through 405 contain all 9 leaking packet-level columns.

### 3. Hazard Head (`train_hazard_head.py`)
- **Status**: Exposed (**YES**)
- **Mechanism**: Line 89 extracts full 406 feature list from `validate_ucs_windows()`. Line 134 fits a 32-component PCA model (`apply_training_only_pca`) over these 406 features.
- **Verification**: The 9 fabricated packet columns contributed to the training linear combination of the 32 PCA dimensions fed to the hazard classification head.

### 4. Stage Head (`train_stage_head.py`)
- **Status**: Exposed (**YES**)
- **Mechanism**: Line 80 extracts 406 features from `validate_ucs_windows()`. Line 115 calls `build_next_state_sequences(purged, features=features)`.
- **Verification**: Sequence inputs $X \in \mathbb{R}^{B \times 30 \times 406}$ contain the 9 leaking packet features, which are fed into the World Model to derive predicted state $\hat{S}_{t+1}$ for stage labeling.
