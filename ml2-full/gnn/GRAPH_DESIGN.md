# Graph Design Document — Phase 0 Feasibility Study

## Dataset Summary

| Property | Value |
|----------|-------|
| Dataset | UGR'16 (University of Granada) |
| File | `august_week5.parquet` |
| Total flows | 40,289,595 |
| Time span | ~9.33 hours (2016-08-29 00:07:33 to 09:27:42 UTC) |
| Unique source IPs | 706,107 |
| Unique destination IPs | 4,063,943 |
| Total unique IPs | 4,361,240 |
| IPs appearing as both src & dst | 408,810 |
| Verified attacks | 1 (`anomaly-sshscan` at 01:08:01 UTC) |
| Attack flow count | 1 out of 40.3M (0.0000025%) |

## 1. Node Definition

**Nodes represent IP addresses** (both source and destination).

| Property | Value |
|----------|-------|
| Node identity | IP address string (e.g. `42.219.153.62`) |
| Node ID mapping | Per-snapshot integer mapping (0..N-1) |
| ID persistence | IPs recur across snapshots; same GNN weights produce consistent embeddings |
| Unseen nodes | Assigned features based on observed traffic; no pre-trained embedding needed |
| Missing identifiers | None — all flows have valid src_ip and dst_ip |

**Justification**: IP addresses are the only stable entity identifiers in this dataset. No host names, device IDs, or user IDs are available. The `src_ip` and `dst_ip` fields map to canonical `source_id` and `destination_id` after cleaning.

**Per-snapshot mapping chosen over global mapping** because:
- 4.3M global IPs would create enormous sparse feature matrices
- Each 1-minute snapshot uses only ~18K IPs
- GNN weights are shared; same features → same embedding regardless of integer ID
- Per-snapshot mapping is memory-efficient and scales naturally

## 2. Edge Definition

**Edges represent directed communication** from source_id → destination_id.

| Property | Decision | Rationale |
|----------|----------|-----------|
| Direction | **Directed** | Source→destination captures attacker→victim patterns |
| Aggregation | Multiple flows between same pair aggregated into one edge per snapshot | Reduces noise, creates meaningful edge features |
| Self-loops | **Excluded** | src_ip == dst_ip flows are localhost traffic, not inter-host communication |
| Duplicate handling | All flows between same (src, dst) pair within a snapshot are aggregated |

## 3. Node Features (Per Snapshot)

Each node receives features computed from its traffic within the snapshot:

| Feature | Description | Source |
|---------|-------------|--------|
| `out_degree` | Number of unique destinations contacted | Computed from edges |
| `in_degree` | Number of unique sources connecting in | Computed from edges |
| `out_flow_count` | Number of outbound flows | Aggregated from raw flows |
| `in_flow_count` | Number of inbound flows | Aggregated from raw flows |
| `out_bytes` | Total bytes sent | sum(byte_count) for outbound |
| `in_bytes` | Total bytes received | sum(byte_count) for inbound |
| `out_packets` | Total packets sent | sum(packet_count) for outbound |
| `in_packets` | Total packets received | sum(packet_count) for inbound |
| `out_tcp_ratio` | Fraction of outbound flows using TCP | protocol == 6 |
| `out_udp_ratio` | Fraction of outbound flows using UDP | protocol == 17 |
| `unique_dst_ports` | Number of distinct destination ports used (outbound) | nunique(dst_port) |
| `unique_src_ports` | Number of distinct source ports used (inbound) | nunique(src_port) |

**Total**: 12 node features

All features are computed from the current snapshot only — no future information.

## 4. Edge Features (Per Snapshot)

Each aggregated edge (src → dst pair) receives:

| Feature | Description |
|---------|-------------|
| `flow_count` | Number of flows between this pair |
| `total_bytes` | Sum of bytes transferred |
| `total_packets` | Sum of packets transferred |
| `mean_duration` | Average flow duration |
| `tcp_ratio` | Fraction of flows using TCP |
| `udp_ratio` | Fraction of flows using UDP |

**Total**: 6 edge features

## 5. Snapshot Frequency

**Decision: 1-minute windows**

| Interval | Snapshots | Avg Nodes | Avg Edges | Positive Labels |
|----------|-----------|-----------|-----------|-----------------|
| 1 min | 561 | 18,429 | 28,348 | 10 (1.78%) |
| 5 min | 113 | ~82,324 | N/A | ~2 (1.77%) |
| 10 min | 57 | ~139,670 | N/A | ~1 (1.75%) |

**Rationale**:
- 1-minute windows produce manageable graph sizes (~18K nodes fits comfortably on 8GB GPU)
- 561 samples is small but sufficient for a proof-of-concept model with sequence-based augmentation
- 5-min/10-min windows produce fewer samples with much larger graphs — worse tradeoff
- 1-minute aligns with existing windowing pipeline (`build_windows.py`)
- Attack duration is short; 1-minute resolution captures temporal structure around the event

## 6. Graph Evolution Strategy

**Observations from data**:
- Nodes appear/disappear between snapshots as IPs become active/inactive
- Average ~18K nodes persist per minute; total pool is 4.3M
- Graph density is very low (~0.00011) — typical for internet traffic
- Edge count varies ±12% across snapshots
- No zero-flow gaps observed (all 561 expected minutes present)

**Strategy**: Each snapshot is an independent graph. Temporal relationships are captured by:
1. Sequencing consecutive graph embeddings for temporal models
2. The GNN produces a per-snapshot embedding; temporal aggregation happens downstream

## 7. Current Attack Label Definition

For **detection** (not forecasting):
- `attack_at_t = 1` if the snapshot at time `t` contains any malicious flow
- Only 1 snapshot (01:08:00) is positive

## 8. Future Attack Forecasting Label Definition

For **forecasting** (the primary objective):
- `future_attack(t, H) = 1` if any attack flow occurs within `[t, t + H]`
- Horizon `H = 600 seconds` (10 minutes) per `feature_configs.yaml`
- Produces 10 positive snapshots (00:59:00 through 01:08:00 inclusive)

**No future leakage**: The label looks forward, but the **features** only use data available at time `t`.

## 9. Temporal Graph Strategy

For temporal forecasting:
```
Sequence: [G(t-W+1), G(t-W+2), ..., G(t)]
Label: future_attack(t, H)
```
- Window size `W = 30` (30 minutes of history)
- Horizon `H = 10 minutes`
- Each graph produces a fixed-size embedding via the GNN
- Temporal patterns captured by averaging or a lightweight sequence model

## 10. Leakage Prevention Strategy

| Mechanism | Implementation |
|-----------|---------------|
| Feature computation | Only uses flows within current snapshot `[t, t+60s)` |
| Labels | Forward-looking but never used as input features |
| Train/Val/Test split | Chronological only (70/15/15) with purge + embargo |
| Purge | Train rows whose label horizon crosses into val are excluded |
| Embargo | Val/test rows whose lookback reaches into prior split are excluded |
| Normalization | Fit scalers on training data only; transform val/test |
| Node mapping | Per-snapshot, no future node information |

## 11. Graph Feasibility Conclusion

### Evidence FOR graph usefulness:
1. **Structural signal exists**: Attack window shows 10% more nodes, 18% more edges, 28% more flows than benign average
2. **Degree distribution shifts**: Attack window has higher mean out-degree (3.28 vs 2.89)
3. **Rich communication structure**: ~18K nodes with ~28K directed edges per snapshot — sufficient topology for message passing
4. **Fan-out pattern**: 706K sources → 4M destinations indicates directional asymmetry that GNN can exploit

### Evidence AGAINST strong graph claims:
1. **Single attack event**: Only 1 attack, producing 10 positive labels out of 561. Statistical power is severely limited.
2. **Structural differences are modest**: Attack/benign ratios range from 0.88x to 1.28x — not dramatic
3. **Density is very low**: 0.00011 — most node pairs never communicate. Information propagation via message passing is limited.
4. **Volume dominates**: The attack window's differences may simply reflect higher traffic volume, not structural change

### Conclusion
**Proceed with caution.** The graph representation IS meaningful — network communication is inherently relational, and the GNN architecture is appropriate. However, with only 1 attack event, we cannot definitively prove that graph structure provides predictive value beyond aggregate traffic statistics. The GNN pipeline will be built as a valid architectural foundation, with honest evaluation comparing against graph-statistical baselines.

## 12. Known Limitations

1. **Extreme class imbalance**: 10/561 = 1.78% positive rate
2. **Single attack type**: Only SSH scan — no generalization testing possible
3. **Short time span**: 9.33 hours — no diurnal patterns, no long-term baselines
4. **Z-score features unavailable**: 24h rolling window requires more data than available
5. **No reverse traffic**: `packets_rev` and `bytes_rev` are 100% empty — cannot compute bidirectional features
6. **Large node space**: 4.3M unique IPs prevents global node embeddings
