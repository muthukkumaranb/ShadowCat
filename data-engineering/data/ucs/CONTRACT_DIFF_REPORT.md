# UCSExtractor vs. ML1 Inference Contract Diff Report

**Evaluation Status**: 🟢 **PASS**  
**Evaluated Contract Version**: `v2`  
**Timestamp (UTC)**: `2026-09-10T08:45:07.826683+00:00`  
**Feature Order Artifact**: `E:\SIH 2026 - UCS Ingestion Pipeline (Main)\scratch\ml1_repo\artifacts\lstm\inference_feature_order_v2.json`  
**Scaler Artifact**: `E:\SIH 2026 - UCS Ingestion Pipeline (Main)\scratch\ml1_repo\artifacts\lstm\inference_scaler_v2.yaml`  

> [!IMPORTANT]
> **INFERENCE GATE: UNLOCKED (UNCONDITIONAL / CONFIRMED ON v2 CHECKPOINT)**
> Exact index-by-index positional order and numerical scaler parameters match 100% across all 406 model dimensions.
> Unconditionally unlocked against the retrained v2 checkpoint (`gaussian_next_state_best_v2.pt`) under Option A (zero-filled packet telemetry & absent packet mask).
> Downstream backend services are authorized to wire [`UCSExtractor.extract()`](file:///e:/SIH%202026%20-%20UCS%20Ingestion%20Pipeline%20(Main)/src/ucs_extractor.py) and [`UCSExtractor.extract_model_tensor()`](file:///e:/SIH%202026%20-%20UCS%20Ingestion%20Pipeline%20(Main)/src/ucs_extractor.py) into `backend.predict()` with full confidence.

## 📌 Dimension Specification & PCAP Packet Count Clarification

This report explicitly confirms that **12 PCAP packet-level features** are included in the UCS contract (indices 394–405).
> [!NOTE]
> **Superseding Note**: Earlier preliminary exploratory leakage audit notes mentioned 9 packet features. This committed report formalizes the complete, authoritative specification of **12 PCAP packet-level features**, officially superseding the earlier partial count across all repository documentation.

### Dimension Verification Table

| Component | Expected (ML1 / Spec) | Observed (UCSExtractor) | Status |
|:---|:---:|:---:|:---:|
| Total LSTM Model Input Columns | 406 | 406 | PASS |
| Scaled Feature Columns (Flow + Packet) | 400 | 400 | PASS |
| Presence Masks (`mask_has_*`) | 6 | 6 | PASS |
| Total Extractor Output (IDs + Masks + Features) | 410 | 410 | PASS |
| Flow Aggregation Features | 388 | 388 | PASS |
| PCAP Packet-Level Features | 12 | 12 | PASS |

## 🔬 Positional Segment Breakdown

The LSTM tensor layout expects features in three strictly sequential contiguous blocks:

```
┌───────────────────────────┬──────────────────────┬────────────────────────────┐
│ Indices 0 .. 387          │ Indices 388 .. 393   │ Indices 394 .. 405         │
│ 388 Flow Features         │ 6 Presence Masks     │ 12 PCAP Packet Features    │
└───────────────────────────┴──────────────────────┴────────────────────────────┘
```

### Segment 2: 6 Presence Masks (Indices 388–393)

| Tensor Index | Mask Column Name | Status |
|:---:|:---|:---:|
| 388 | `mask_has_traffic_volume_features` | MATCH |
| 389 | `mask_has_flow_timing_features` | MATCH |
| 390 | `mask_has_packet_level_features` | MATCH |
| 391 | `mask_has_tcp_flags` | MATCH |
| 392 | `mask_has_graph_topology` | MATCH |
| 393 | `mask_has_identity_auth` | MATCH |

### Segment 3: 12 PCAP Packet Features (Indices 394–405)

| Tensor Index | Feature Column Name | Category | Status |
|:---:|:---|:---|:---:|
| 394 | `pkt_ttl_min` | IP TTL Statistics | MATCH |
| 395 | `pkt_ttl_max` | IP TTL Statistics | MATCH |
| 396 | `pkt_ttl_std` | IP TTL Statistics | MATCH |
| 397 | `pkt_ttl_mode` | IP TTL Statistics | MATCH |
| 398 | `pkt_frag_mf_count` | IP Fragmentation | MATCH |
| 399 | `pkt_frag_df_count` | IP Fragmentation | MATCH |
| 400 | `pkt_payload_size_p25` | Payload Quantiles | MATCH |
| 401 | `pkt_payload_size_p50` | Payload Quantiles | MATCH |
| 402 | `pkt_payload_size_p75` | Payload Quantiles | MATCH |
| 403 | `pkt_payload_size_p95` | Payload Quantiles | MATCH |
| 404 | `pkt_tcp_retrans_count` | TCP Anomalies | MATCH |
| 405 | `pkt_port_scan_seq_score` | Heuristic Scan Score | MATCH |

## 📋 Positional Diff Findings

✅ **Zero positional mismatches detected.** All 406 model input columns are in 100% exact identical order between `UCSExtractor.MODEL_INPUT_COLUMNS` and ML1's contract.

## ⚖️ Scaler Parameter Parity Audit

- **Features Evaluated**: 400 / 400 features
- **Maximum Parameter Deviation**: `0.00000000e+00`
- **Parameter Mismatches**: 0

✅ **All 400 scaled feature parameters match bit-exactly** (`median`, `scale`, `is_log1p`, `q25`, `q75`).

## 🚀 Backend Downstream Integration Guidance

For real-time streaming and inference consumption in Backend's `predict()` pipeline:

```python
from src.ucs_extractor import UCSExtractor

# Initialize extractor once during backend startup
extractor = UCSExtractor(schema_version="v3.0")

# Method 1: Extract direct 2D model tensor (shape: (N_windows, 406), dtype: float32)
model_tensor = extractor.extract_model_tensor(raw_df, source_type="csv")

# Method 2: Extract full 410-column DataFrame (provenance + masks + features)
ucs_df = extractor.extract(raw_df, source_type="csv")
model_tensor = ucs_df[UCSExtractor.MODEL_INPUT_COLUMNS].to_numpy(dtype=np.float32)
```

---
*Report automatically generated by `scripts/diff_ucs_ml1_contract.py` on 2026-09-10T08:45:07.826683+00:00.*