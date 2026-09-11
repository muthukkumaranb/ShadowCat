# Feature Dictionary — SIH26153 UCS V1 P0 (Section 43)

Covers all model-input features and the label column. Identifier/metadata
columns (timestamp, dataset, scope_id, window_size_sec) are intentionally
excluded — they identify a row, they don't describe network behavior.

---

### flow_count
- **Type:** integer
- **Unit:** count (flows)
- **Definition:** Number of individual network flows observed within the 1-minute window.
- **Formula:** `count of flows grouped by floor-to-minute(timestamp)`
- **Source:** derived from row count per window (raw field: timestamp, used only for grouping)
- **Lookback:** 0 (current window only)
- **Missing-value behavior:** Never missing — every window has at least 1 flow by construction (0 of 561 windows had zero flows in this file).

### packet_count
- **Type:** integer
- **Unit:** count (packets)
- **Definition:** Total number of packets across all flows in the window.
- **Formula:** `sum(packet_count)` within window
- **Source:** raw field `packet_count`
- **Lookback:** 0
- **Missing-value behavior:** Never missing.

### bytes_in
- **Type:** integer
- **Unit:** bytes
- **Definition:** Total inbound byte volume across all flows in the window.
- **Formula:** `sum(byte_count)` within window
- **Source:** raw field `byte_count`
- **Lookback:** 0
- **Missing-value behavior:** Never missing. Note: there is no corresponding `bytes_out` feature — this dataset has no reverse-direction traffic data (`packets_rev`/`bytes_rev` are 100% empty in the raw file), so outbound byte volume is not computable and is not included anywhere in this feature set, rather than being silently zero-filled.

### unique_src
- **Type:** integer
- **Unit:** count (distinct hosts)
- **Definition:** Number of distinct source hosts active in the window.
- **Formula:** `nunique(source_id)` within window
- **Source:** raw field `source_id`
- **Lookback:** 0
- **Missing-value behavior:** Never missing.

### unique_dst
- **Type:** integer
- **Unit:** count (distinct hosts)
- **Definition:** Number of distinct destination hosts contacted in the window.
- **Formula:** `nunique(destination_id)` within window
- **Source:** raw field `destination_id`
- **Lookback:** 0
- **Missing-value behavior:** Never missing.

### peer_count
- **Type:** integer
- **Unit:** count (distinct src-dst pairs)
- **Definition:** Number of distinct communicating (source, destination) pairs in the window — a stronger signal than unique_src/unique_dst alone, since it captures fan-out/fan-in patterns (e.g. one host scanning many destinations shows up here even if unique_src is low).
- **Formula:** `drop_duplicates` on (source_id, destination_id) together, then count rows
- **Source:** raw fields `source_id`, `destination_id`
- **Lookback:** 0
- **Missing-value behavior:** Never missing.

### tcp_ratio
- **Type:** float
- **Unit:** ratio (0-1)
- **Definition:** Fraction of flows in the window using TCP.
- **Formula:** `count(protocol == 6) / flow_count`
- **Source:** raw field `protocol`
- **Lookback:** 0
- **Missing-value behavior:** Never missing. Note: tcp_ratio + udp_ratio + icmp_ratio do not necessarily sum to 1 — other protocols (e.g. GRE, ESP) count toward flow_count in the denominator but aren't tracked as a numerator for any of the three ratios. This is intentional, not a bug.

### udp_ratio
- **Type:** float
- **Unit:** ratio (0-1)
- **Definition:** Fraction of flows in the window using UDP.
- **Formula:** `count(protocol == 17) / flow_count`
- **Source:** raw field `protocol`
- **Lookback:** 0
- **Missing-value behavior:** Never missing.

### icmp_ratio
- **Type:** float
- **Unit:** ratio (0-1)
- **Definition:** Fraction of flows in the window using ICMP.
- **Formula:** `count(protocol == 1) / flow_count`
- **Source:** raw field `protocol`
- **Lookback:** 0
- **Missing-value behavior:** Never missing.

### connection_rate
- **Type:** float
- **Unit:** flows per second
- **Definition:** Rate of new flow establishment during the window — a proxy for connection churn/scanning intensity.
- **Formula:** `flow_count / 60`
- **Source:** derived from flow_count
- **Lookback:** 0
- **Missing-value behavior:** Never missing.

### traffic_zscore
- **Type:** float
- **Unit:** standard deviations (dimensionless)
- **Definition:** How anomalous the window's bytes_in is relative to the preceding 24 hours of traffic for this scope.
- **Formula:** `(bytes_in - rolling_24h_mean_past) / (rolling_24h_std_past + 1e-6)`, rolling window computed with `closed="left"` so the current row is never included in its own baseline
- **Source:** derived from bytes_in
- **Lookback:** 24 hours
- **Missing-value behavior:** A row is eligible for a real z-score only if (a) at least 24 hours have elapsed since the first timestamp in the file, AND (b) at least 2 historical points exist to compute a meaningful standard deviation. When eligible, missing rows are imputed with a training-derived median (never invented). **In this specific file, 0 of 561 rows are eligible** (the file spans only ~9.33 hours), so all 561 rows are NaN and flagged via `traffic_zscore_missing = 1`. This is a confirmed, frozen team-lead decision — the 24-hour window is never shortened to accommodate shorter files. Any model consuming this feature must check the `_missing` flag rather than assume 0 or another default means "no anomaly."

### connection_zscore
- **Type:** float
- **Unit:** standard deviations (dimensionless)
- **Definition:** How anomalous the window's connection_rate is relative to the preceding 24 hours.
- **Formula:** same as traffic_zscore, applied to connection_rate
- **Source:** derived from connection_rate
- **Lookback:** 24 hours
- **Missing-value behavior:** Same eligibility rule and same result as traffic_zscore — 0/561 eligible in this file, flagged via `connection_zscore_missing = 1`.

### peer_zscore
- **Type:** float
- **Unit:** standard deviations (dimensionless)
- **Definition:** How anomalous the window's peer_count is relative to the preceding 24 hours — flags unusual fan-out/fan-in behavior.
- **Formula:** same as traffic_zscore, applied to peer_count
- **Source:** derived from peer_count
- **Lookback:** 24 hours
- **Missing-value behavior:** Same eligibility rule and same result as traffic_zscore — 0/561 eligible in this file, flagged via `peer_zscore_missing = 1`.

### future_attack (label, not a feature)
- **Type:** boolean
- **Unit:** N/A
- **Definition:** Whether a verified attack occurs within the forecast horizon following this window's start.
- **Formula:** `True` if the verified attack timestamp falls within `[t, t+horizon_sec]`, where `horizon_sec = 600` (10 minutes) per `feature_config.yaml`
- **Source:** raw fields `label_binary`, `label_attack` (verified attack: `anomaly-sshscan` at 2016-08-29 01:08:01 UTC)
- **Lookback:** N/A — this is forward-looking by design (it is the prediction target, not an input feature). It must never be included in the feature matrix passed to a model; verified in leakage_audit.md item 3.
- **Missing-value behavior:** Never missing — every window gets a definite True/False. In this file, 10 of 561 windows (00:59:00 through 01:08:00 inclusive) are True; the rest are False, reflecting the single verified attack and extreme class imbalance in this dataset.