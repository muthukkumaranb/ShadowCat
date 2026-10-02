# Unified Cyber State ($S_t$) Schema Documentation
**SIH26153 — Cyber World Model Architecture**

The Unified Cyber State ($S_t$) is a dual-format data representation designed to feed the downstream temporal (LSTM/GRU) and topological (GraphSAGE) world model encoder.

---

## 1. Flat Temporal Feature Tensor: `ucs_windows.parquet`

One row per 1-minute time window.

| Column Name | Category | Data Type | Transformation Applied | Description |
| :--- | :--- | :--- | :--- | :--- |
| `window_id` | Metadata | `string` | Formatting | Unique window identifier (`W_{day}_{timestamp}`) |
| `window_start_utc` | Metadata | `datetime64[ns, UTC]` | Floor (60s) | UTC start timestamp of the 1-minute window |
| `window_end_utc` | Metadata | `datetime64[ns, UTC]` | Add (60s) | UTC end timestamp of the 1-minute window |
| `source_day` | Metadata | `string` | Ingestion | Day identifier from source CSV filename |
| `episode_id` | Metadata | `string` | Contiguous Run Segmentation | Granular episode identifier (`{source_day}_{attack_type}_{run_index}`) |
| `split` | Metadata | `string` | Chronological Partition | Dataset partition: `train`, `val`, `test` |
| `label_binary` | Target | `int64` | Ground Truth Alignment | 0 for benign, 1 for attack present in window |
| `label_attack_type` | Target | `string` | Canonical Mapping | Fine-grained attack type or benign |
| `future_attack_label` | Target | `int64` | Target Backward Shift | 1 if attack occurs within next $H=5$ min |
| `flow_count` | Feature | `float64` | Log1p + RobustScaler | Number of flows recorded in window |
| `unique_dst_ports_count` | Feature | `float64` | RobustScaler | Unique destination ports targeted |
| `unique_protocols_count` | Feature | `float64` | RobustScaler | Distinct transport protocols active |
| `mask_has_traffic_volume_features` | Mask | `float64` | Fixed Flag | 1.0 (Presence mask for traffic volume group) |
| `mask_has_flow_timing_features` | Mask | `float64` | Fixed Flag | 1.0 (Presence mask for flow timing group) |
| `mask_has_packet_level_features` | Mask | `float64` | Fixed Flag | 1.0 (Presence mask for packet length / win bytes) |
| `mask_has_tcp_flags` | Mask | `float64` | Fixed Flag | 1.0 (Presence mask for TCP flags) |
| `mask_has_graph_topology` | Mask | `float64` | Fixed Flag | 1.0 (Presence mask for graph edge availability) |
| `mask_has_identity_auth` | Mask | `float64` | Fixed Flag | 0.0 (Identity/Auth out-of-scope in this build) |
| `byte_count_fwd_mean` / `std` / `sum` / `min` / `max` | Feature | `float64` | Log1p + RobustScaler | Forward byte volume summary statistics |
| `byte_count_bwd_mean` / `std` / `sum` / `min` / `max` | Feature | `float64` | Log1p + RobustScaler | Backward byte volume summary statistics |
| `packet_count_fwd_mean` / `std` / `sum` / `min` / `max` | Feature | `float64` | Log1p + RobustScaler | Forward packet count statistics |
| `packet_count_bwd_mean` / `std` / `sum` / `min` / `max` | Feature | `float64` | Log1p + RobustScaler | Backward packet count statistics |
| `bytes_per_sec_mean` / `std` / `sum` / `min` / `max` | Feature | `float64` | Log1p + RobustScaler | Flow byte rate summary statistics |
| `packets_per_sec_mean` / `std` / `sum` / `min` / `max` | Feature | `float64` | Log1p + RobustScaler | Flow packet rate summary statistics |
| `duration_sec_mean` / `std` / `sum` / `min` / `max` | Feature | `float64` | Log1p + RobustScaler | Flow duration summary statistics |
| `flow_iat_*` / `fwd_iat_*` / `bwd_iat_*` | Feature | `float64` | RobustScaler | Inter-arrival time statistics |
| `fin_flag_cnt_*` ... `ece_flag_cnt_*` | Feature | `float64` | RobustScaler | All 8 TCP flag distribution statistics |
| `down_up_ratio_mean` / `std` / `sum` / `min` / `max` | Feature | `float64` | RobustScaler | Bidirectional Down/Up ratio statistics |
| `window_size_fwd_*` / `window_size_bwd_*` | Feature | `float64` | RobustScaler | TCP Initial Window byte statistics |
| `fwd_seg_size_min_*` / `fwd_seg_size_avg_*` | Feature | `float64` | RobustScaler | Segment size statistics |
| `active_*` / `idle_*` | Feature | `float64` | RobustScaler | Active/Idle burst timing statistics |

---

## 2. Graph Topology Edge List: `ucs_graph_edgelists.parquet`

One row per directed interaction edge per 1-minute window.

| Column Name | Category | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `window_id` | Foreign Key | `string` | Maps directly to `ucs_windows.parquet` `window_id` |
| `window_start_utc` | Metadata | `datetime64[ns, UTC]` | 1-minute window start time |
| `source_day` | Metadata | `string` | Day provenance |
| `src_node_id` | Graph Node | `int64` | Integer identifier for source endpoint |
| `dst_node_id` | Graph Node | `int64` | Integer identifier for destination service endpoint |
| `flow_count` | Edge Attribute | `int64` | Number of flows sharing this edge in the window |
| `byte_count_sum` | Edge Attribute | `float64` | Total bytes transmitted over edge in window |
| `packet_count_sum` | Edge Attribute | `float64` | Total packets transmitted over edge in window |
| `duration_mean_sec` | Edge Attribute | `float64` | Average flow duration for interactions on edge |
| `protocol_mode` | Edge Attribute | `int64` | Dominant transport protocol on edge |

---

## 3. Node Identifier Lookup: `node_lookup.parquet`

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `node_id` | `int64` | Anonymized integer node index |
| `endpoint_identifier` | `string` | Original endpoint / service signature string |

---

## 4. Fitted Normalization Parameters: `scaler_params.yaml`

Contains exact $Q_{25}, Q_{50}, Q_{75}$, scale, and `is_log1p` flags fitted strictly on the training partition for full pipeline reproducibility.
