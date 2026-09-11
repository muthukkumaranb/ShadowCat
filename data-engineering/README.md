
# CSE-CIC-IDS2018 → Unified Cyber State ($S_t$) Data Pipeline
**SIH26153 — Cyber World Model Architecture (Data Engineer Track)**

This repository provides the production data engineering pipeline that processes raw CSE-CIC-IDS2018 CICFlowMeter flow CSVs into the **Unified Cyber State ($S_t$)** — a time-windowed, leakage-safe, dual-format state representation (flat temporal feature vector + graph topology edge list) designed to feed downstream LSTM/GRU and GraphSAGE world models.

---

## 🚀 Key Features

1. **Robust Ingestion**: Latin-1 fallback encoding, automated whitespace stripping, duplicate header removal (`Label == 'Label'`), and provenance tracking (`source_day`, `source_file`).
2. **Canonical 80-to-UCS Mapping**: Maps all flow-level features, timing/IAT distributions, all 8 TCP flags, subflows, bidirectional down/up ratios, and packet-level features (`Init_Win_bytes_forward`, `Init_Win_bytes_backward`, `fwd_seg_size_min`).
3. **Data Quality & Integrity**:
   - Zero infinities: Replaces division-by-zero rate infinities with safe medians.
   - Zero corrupted timestamps: Filters out epoch 1970 artifacts.
   - Negative timing handling: Identifies and audits negative durations.
   - Deduplication: Drops exact duplicate flows and logs near-duplicate signatures.
4. **1-Minute Temporal Windows**: Aggregates statistical summaries (`mean`, `std`, `sum`, `min`, `max`) per 1-minute window.
5. **Feature-Presence Masks**: Explicit mask indicators (`mask_has_traffic_volume_features`, `mask_has_flow_timing_features`, `mask_has_packet_level_features`, `mask_has_tcp_flags`, `mask_has_graph_topology`, `mask_has_identity_auth`).
6. **Graph Construction per Window**: Directed interaction topology per window with integer node anonymization and lookup mapping (`node_lookup.parquet`).
7. **Two-Phase Infiltration Segmentation**: Distinguishes `Infiltration-Compromise` (initial malware delivery) and `Infiltration-Portscan` (internal lateral discovery) on March 1.
8. **Multi-Horizon Future Forecasting ($H=5$ min Primary)**: Derives target `future_attack_label` strictly by backward shifting target masks with zero future feature leakage. Primary validated horizon is $H=5$ windows (5 minutes lead time, validated through Gate 0 LOEO). Horizons $H=10$ and $H=15$ are secondary sensitivity-analysis horizons only.
9. **Temporal Leakage Protection (Purge + Embargo)**: Enforces $W = L + H = 30 + 5 = 35$ window purge/embargo zone at both Train→Val and Val→Test boundaries (dropping 140 windows total). Ensures zero future-horizon or past-lookback overlap across partitions.
10. **Boundary-Safe LSTM Sequence Construction**: Constructs input sequences of length $L=30$ windows (`src/sequence_builder.py`). Rejects any sequence crossing split boundaries.
11. **LR/LSTM Protocol Parity**: Guarantees that both tabular Logistic Regression (flat windows) and LSTM (sequences) consume the exact same purged/embargoed window sets and training-fit scaler parameters.
12. **Leakage-Free Normalization**: Fits `RobustScaler` (median & IQR) and `Log1p` transforms exclusively on the post-purge training split (2,013 windows) and applies them to validation (369 windows) and test (405 windows) partitions.

---

## 📁 Repository Structure

```
├── configs/
│   ├── pipeline_config.yaml           # Master pipeline settings, split ratios, horizons, LSTM lookback
│   ├── canonical_mapping.yaml          # Audit-ready 80-column canonical mapping
│   └── attack_timelines.yaml           # Ground truth Table 2 attack timelines
├── src/
│   ├── __init__.py
│   ├── ingestion.py                   # Stage 1: Robust CSV reader
│   ├── canonical_mapper.py            # Stage 2: Canonical schema mapper
│   ├── cleaner.py                     # Stage 3: Data cleaning & UTC normalization
│   ├── window_aggregator.py           # Stage 4: 1-min windowing & feature masks
│   ├── graph_builder.py               # Stage 5: Per-window graph topology
│   ├── labeler_and_splits.py          # Stage 6: Attack labeling, chronological splits & purge+embargo
│   ├── normalizer.py                  # Stage 7: Leakage-free RobustScaler (post-purge train fit)
│   ├── sequence_builder.py            # Stage 8: Boundary-safe LSTM sequence construction & LR parity
│   └── pipeline_runner.py             # Master pipeline runner & audit generator
├── tests/
│   └── test_pipeline.py               # Unit & regression test suite (10/10 passing)
├── data/
│   ├── raw/                           # Raw CSE-CIC-IDS2018 CSV files
│   ├── intermediate/                  # Staged cleaned parquet files per day
│   └── ucs/                           # Final Unified Cyber State (S_t) artifacts
│       ├── ucs_windows.parquet        # Flat temporal feature state S(t) (purged)
│       ├── ucs_graph_edgelists.parquet# Graph topology edge lists (unmodified)
│       ├── node_lookup.parquet        # Anonymized node ID mapping (unmodified)
│       ├── scaler_params.yaml         # Fitted normalization parameters (post-purge)
│       ├── SCHEMA.md                  # Comprehensive column documentation
│       └── VALIDATION_REPORT.md       # Audit statistics & quality gates
└── README.md
```

---

## 🛠️ How to Run

### 1. Run Unit Tests
```bash
python tests/test_pipeline.py
```

### 2. Execute Full Pipeline
```bash
python src/pipeline_runner.py
```

### 3. Run UCSExtractor & Validator Tests
```bash
python -m unittest discover tests -v
```

All output datasets and audit reports are written directly to `data/ucs/`.

---

## ⚡ Runtime Extraction Engine (`UCSExtractor`)

The `UCSExtractor` runtime module ([`src/ucs_extractor.py`](file:///e:/SIH%202026%20-%20UCS%20Ingestion%20Pipeline%20(Main)/src/ucs_extractor.py)) packages the batch pipeline into a lightweight, high-performance module for streaming and real-time backend/inference integration.

### Contract & Architecture (Schema `v3.0`)
- **Total Output Shape**: Exactly **410 columns** per window:
  - **4 Window Identifiers**: `window_id`, `window_start_utc`, `window_end_utc`, `source_day`
  - **6 Presence Masks**: `mask_has_traffic_volume_features`, `mask_has_flow_timing_features`, `mask_has_packet_level_features`, `mask_has_tcp_flags`, `mask_has_graph_topology`, `mask_has_identity_auth`
  - **400 Scaled Features**: Normalized via frozen `data/ucs/scaler_params.yaml` fitted on the 2,013 post-purge training windows.
- **Frozen Imputation**: Missing values are imputed using frozen raw-scale training medians ([`data/ucs/imputation_params.yaml`](file:///e:/SIH%202026%20-%20UCS%20Ingestion%20Pipeline%20(Main)/data/ucs/imputation_params.yaml)), ensuring zero cross-sample leakage and centering neutral values at `0.0`.
- **Reproducibility**: Bit-exact reproduction (0 bit mismatch) against the batch ingestion pipeline.

### Class Constants & ML1 Alignment
`UCSExtractor` exposes explicit class-level constants for seamless downstream consumption:

| Constant | Column Count | Description & Usage |
|:---|:---|:---|
| `UCSExtractor.WINDOW_ID_COLUMNS` | 4 | Window provenance identifiers |
| `UCSExtractor.MASK_COLUMNS` | 6 | Presence masks indicating feature availability |
| `UCSExtractor.MODEL_FEATURE_COLUMNS` | 400 | Scaled feature columns in sequence_builder order |
| `UCSExtractor.OUTPUT_COLUMNS` | 410 | Full contract: `WINDOW_ID_COLUMNS + MASK_COLUMNS + MODEL_FEATURE_COLUMNS` |
| `UCSExtractor.MODEL_INPUT_COLUMNS` | 406 | **Direct ML1/LSTM input tensor order** (388 flow features + 6 masks + 12 packet features) matching `inference_feature_order_v1.json` |

### Backend & ML1 Integration Example

```python
import pandas as pd
from src.ucs_extractor import UCSExtractor

# 1. Initialize extractor (loads frozen scaler and imputation artifacts)
extractor = UCSExtractor(schema_version="v3.0")

# 2. Option A (Recommended): Direct 2D model tensor for backend.predict()
# Returns np.ndarray shape (N_windows, 406), float32 in exact positional LSTM order
model_tensor = extractor.extract_model_tensor(raw_csv_df, source_type="csv")

# 3. Option B: Full 410-column DataFrame with window IDs and provenance
ucs_df = extractor.extract(raw_csv_df, source_type="csv")
assert ucs_df.shape[1] == 410
model_tensor = ucs_df[UCSExtractor.MODEL_INPUT_COLUMNS].to_numpy(dtype=np.float32)
```

### Contract Verification & Diff Engine
To guarantee 100% positional and numerical parity between `UCSExtractor` and ML1's inference contract:
```powershell
# Run programmatic positional diff check
.venv\Scripts\python.exe scripts/diff_ucs_ml1_contract.py --version v1 --check-only

# View the committed audit report:
# data/ucs/CONTRACT_DIFF_REPORT.md
```
*(Once ML1 delivers `_v2` contract files, execute `.venv\Scripts\python.exe scripts/diff_ucs_ml1_contract.py --version v2` to re-validate and update the committed report).*

---

## 🛡️ Input Validation (`CICFlowMeterValidator`)

The `CICFlowMeterValidator` module ([`src/csv_validator.py`](file:///e:/SIH%202026%20-%20UCS%20Ingestion%20Pipeline%20(Main)/src/csv_validator.py)) verifies and cleans raw CSV inputs before extraction:
- **`EXACT_MATCH`**: All 80 raw CICFlowMeter columns are present with standard casing.
- **`MAPPED_VARIANT`**: Recognizable column variations (case differences, underscores, canonical names) are automatically mapped and decisions are recorded in `variant_mappings`.
- **`REJECTED`**: Missing critical fields (destination port, timestamp, duration, packet counts) or non-numeric corrupted fields are rejected with detailed diagnostics.

---

## ⚠️ Limitations

- **Contiguous Episode Granularity**: Pulsing/intermittent C2 traffic can be split into multiple single-window episodes under the strict contiguity rule; this is documented and does not affect leakage boundaries.
- **Packet-Level Feature Scope**: Raw PCAP packet extraction is currently scoped to Wednesday-14-02-2018 (`SSH-Bruteforce`); all remaining days use flow-level telemetry and have `mask_has_packet_level_features = 0.0`.


