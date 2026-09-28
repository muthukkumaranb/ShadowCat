# Presentation Defense Q&A: Packet-Level Telemetry & Contract v2

**SIH26153 — Cyber World Model Architecture**

---

## Defense Q&A Brief

### Q1: How does the model handle packet-level network telemetry?
**Answer**:
The Cyber World Model architecture defines 6 presence mask flags (`mask_has_*`) to handle heterogeneously available telemetry streams. Real, Scapy/binary-parsed packet-level telemetry (`pkt_*`) is now extracted and verified for all 3 headline attack types — SSH-Bruteforce (14-02-2018), Botnet (02-03-2018), and DDOS-LOIC-UDP (21-02-2018) — with `mask_has_packet_level_features = 1.0` for every window in those three days. Windows outside those days remain honestly presence-masked and zero-filled; the mask mechanism is a real, numerically-weighted model input, not documentation-only.

### Q2: What was the scope of the packet-level feature remediation?
**Answer**:
Initial audit identified fabricated packet features; subsequent complete inventory expanded the scope to **12 packet-level features** (`pkt_ttl_min/max/std/mode`, `pkt_frag_mf/df_count`, `pkt_payload_size_p25/p50/p75/p95`, `pkt_tcp_retrans_count`, `pkt_port_scan_seq_score`). This was resolved by downloading the original raw PCAPs and extracting genuine packet telemetry for all 3 headline attacks. The RobustScaler was globally refit across these real distributions (`inference_scaler_v4.yaml`, `inference_feature_order_v4.json`), and the World Model, Stage Head, and Stacked Ensemble were fully retrained on the resulting matrix.

### Q3: Did restoring genuine packet features improve downstream detection or state forecasting?
**Answer**:
Yes. Retraining and re-evaluating across all downstream heads confirmed:
1. **LR Baseline LOEO Detection**: Held steady at F1 = 0.8889 on SSH-Bruteforce across all 9 evaluation folds (full 37-fold LOEO aggregate is F1 = 0.9730 for detection and F1 = 0.8880 for onset forecasting). The production model itself achieved an F1 of 0.9962 for detection and 0.9127 for onset forecasting.
2. **Stage Classification Head**: Accuracy is now **0.8627** (up from the earlier 0.8368 v2 citation) — see `ml1/artifacts/lstm/stage_head_v3/reeval_packetcov/` and `ml1/artifacts/lstm/stage_head_v3/NOTE.md` for the full evaluation history. Validated for high-fidelity discrimination on *Credential Access / Brute Force* vs. background (Precision = 1.0000, Recall = 0.5854 in 37-fold LOEO; 0 false positives out of 285 background windows). Multi-stage classification across Discovery, Command & Control, and Impact is exploratory / not yet validated at production quality: Discovery and Command & Control have zero test support in LOEO splits, while Impact-stage attacks (19 test windows) are not detected by the current Stage Head - 0% recall across every evaluation to date; the model defaults these to Unknown/Other.
3. **PC2 Significance Test**: Statistical advantage over persistence baseline remains significant ($p < 0.05$ at $K=1, 2$).


**Note**: The final deployed hazard threshold is globally calibrated to 0.15.
