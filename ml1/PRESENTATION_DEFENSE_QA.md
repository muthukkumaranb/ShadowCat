# Presentation Defense Q&A: Packet-Level Telemetry & Contract v2

**SIH26153 — Cyber World Model Architecture**

---

## Defense Q&A Brief

### Q1: How does the model handle packet-level network telemetry?
**Answer**:
The Cyber World Model architecture defines 6 presence mask flags (`mask_has_*`) to handle heterogeneously available telemetry streams. Under **Option A (Corrected-but-simulated)**, packet-level telemetry (`pkt_*`) is marked absent (`mask_has_packet_level_features = 0.0`) and zero-filled across all 2,787 windows. This guarantees zero label leakage while preserving the 406-dimensional input contract structure for live PCAP ingestion post-MVP.

### Q2: What was the scope of the packet-level feature remediation?
**Answer**:
Initial audit identified 9 packet features; subsequent complete inventory expanded the scope to **12 packet-level features** (`pkt_ttl_min/max/std/mode`, `pkt_frag_mf/df_count`, `pkt_payload_size_p25/p50/p75/p95`, `pkt_tcp_retrans_count`, `pkt_port_scan_seq_score`). All 12 were zero-filled, the scaler parameters were re-fit on post-purge training split (`median=0.0, scale=1.0`), and the World Model was fully retrained to deliver **v2 inference contract files** (`inference_scaler_v2.yaml`, `inference_feature_order_v2.json`).

### Q3: Did removing the fabricated packet features degrade downstream detection or state forecasting?
**Answer**:
No. Retraining and re-evaluating across all downstream heads confirmed:
1. **LR Baseline LOEO Detection**: Held steady at F1 = 0.8889 (0.0000 delta across all 9 SSH-Bruteforce evaluation folds).
2. **Stage Classification Head**: Maintains strong ATT&CK stage separation across Initial Access, Discovery, Command & Control, and Impact.
3. **PC2 Significance Test**: Statistical advantage over persistence baseline remains significant ($p < 0.05$ at $K=1, 2$).
