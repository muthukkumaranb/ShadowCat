# Project Handbook: Cyber World Model Architecture

**SIH26153 — Data Engineer & ML1 Tracks**

---

## 1. Feature Representation & Masking Rules

- **Total State Dimensions**: 406 input dimensions (400 scaled numerical features + 6 unscaled presence mask flags).
- **Missing Value Handling Rule**: *"Missing != Zero; Never Fabricate Baselines."*
- **Packet-Level Features (`pkt_*`)**:
  - Per Option A (Corrected-but-simulated), all 12 packet-level extraction features (`pkt_ttl_min`, `pkt_ttl_max`, `pkt_ttl_std`, `pkt_ttl_mode`, `pkt_frag_mf_count`, `pkt_frag_df_count`, `pkt_payload_size_p25`, `pkt_payload_size_p50`, `pkt_payload_size_p75`, `pkt_payload_size_p95`, `pkt_tcp_retrans_count`, `pkt_port_scan_seq_score`) are zero-filled (`0.0`) across all windows.
  - The presence flag `mask_has_packet_level_features` is set to `0.0` across all windows.
  - Real PCAP packet-level extraction is scoped post-MVP.

---

## 2. Canonical Contracts & Checkpoints

| Asset | Version | File Path | Status |
| :--- | :---: | :--- | :--- |
| **World Model Checkpoint** | **v2** | `artifacts/lstm/gaussian_next_state_best_v2.pt` | **Canonical Current** |
| **World Model Checkpoint** | v1 | `artifacts/lstm/probabilistic_world_model/gaussian_next_state_best.pt` | Preserved for baseline |
| **Inference Scaler Contract** | **v2** | `artifacts/lstm/inference_scaler_v2.yaml` | **Canonical Current** |
| **Inference Feature Contract** | **v2** | `artifacts/lstm/inference_feature_order_v2.json` | **Canonical Current** |
