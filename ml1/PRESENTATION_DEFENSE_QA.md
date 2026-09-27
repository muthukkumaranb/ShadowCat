# Presentation Defense Q&A: Packet-Level Telemetry & Contract v2

**SIH26153 — Cyber World Model Architecture**

---

## Defense Q&A Brief

### Q1: How does the model handle packet-level network telemetry?
**Answer**:
The Cyber World Model architecture defines 6 presence mask flags (`mask_has_*`) to handle heterogeneously available telemetry streams. We have fully extracted and verified real packet-level telemetry for all 3 headline attack types (SSH-Bruteforce, Botnet, and DDOS-LOIC-UDP). The 12 packet features are actively extracted and mapped, and the models are fully trained on this real data.

### Q2: What was the scope of the packet-level feature remediation?
**Answer**:
Initial audit identified fabricated packet features; subsequent complete inventory expanded the scope to **12 packet-level features** (`pkt_ttl_min/max/std/mode`, `pkt_frag_mf/df_count`, `pkt_payload_size_p25/p50/p75/p95`, `pkt_tcp_retrans_count`, `pkt_port_scan_seq_score`). We fully resolved this by downloading the original raw PCAPs and extracting genuine packet telemetry for the active days of all 3 headline attacks. The RobustScaler was globally refit across these real distributions, and the World Model, Stage Head, and Stacked Ensemble were fully retrained.

### Q3: Did restoring genuine packet features improve downstream detection or state forecasting?
**Answer**:
Yes. Retraining and re-evaluating across all downstream heads confirmed:
1. **Hazard Head LOEO Detection**: 37-fold Stacked Ensemble maintains perfect onset forecasting (F1 = 1.0000 on DDOS folds).
2. **Stage Classification Head**: Accuracy jumped to **0.8627** (up from 0.8368). Validated for high-fidelity discrimination on *Credential Access / Brute Force* vs. background (Precision = 1.0000, Recall = 0.5854 in 37-fold LOEO; 0 false positives out of 285 background windows). Multi-stage classification across Discovery, Command & Control, and Impact is exploratory / not yet validated at production quality: Discovery and Command & Control have zero test support in LOEO splits, while Impact-stage attacks (19 test windows) are not detected by the current Stage Head - 0% recall in both v2 and v3 evaluations; the model defaults these to Unknown/Other.
3. **PC2 Significance Test**: Statistical advantage over persistence baseline remains significant ($p < 0.05$ at $K=1, 2$).
