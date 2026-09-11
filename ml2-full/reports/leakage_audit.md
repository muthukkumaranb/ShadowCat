## Per-Feature Leakage Table (Section 41 format)

Feature: flow_count
Source field: timestamp (count of rows)
Calculation: count of flows grouped into window t
Lookback: 0 (current window only)
Future dependency: NO
Decision: KEEP
Reason: Computed entirely from window t's own flows; verified by perturbation test (item 1) — corrupting all data after t left this value unchanged.

Feature: packet_count
Source field: packet_count
Calculation: sum(packet_count) within window t
Lookback: 0
Future dependency: NO
Decision: KEEP
Reason: Same as flow_count — perturbation-verified.

Feature: bytes_in
Source field: byte_count
Calculation: sum(byte_count) within window t
Lookback: 0
Future dependency: NO
Decision: KEEP
Reason: Perturbation-verified. Note: bytes_out is NOT computable from this dataset (packets_rev/bytes_rev are 100% empty) — documented as a known limitation, never zero-filled.

Feature: unique_src
Source field: source_id
Calculation: nunique(source_id) within window t
Lookback: 0
Future dependency: NO
Decision: KEEP
Reason: Perturbation-verified.

Feature: unique_dst
Source field: destination_id
Calculation: nunique(destination_id) within window t
Lookback: 0
Future dependency: NO
Decision: KEEP
Reason: Perturbation-verified.

Feature: peer_count
Source field: source_id, destination_id
Calculation: count of distinct (source_id, destination_id) pairs within window t, via drop_duplicates on both columns together (not string concatenation)
Lookback: 0
Future dependency: NO
Decision: KEEP
Reason: Perturbation-verified.

Feature: tcp_ratio / udp_ratio / icmp_ratio
Source field: protocol
Calculation: count of flows with protocol == 6 / 17 / 1 respectively, divided by flow_count, within window t. Other protocols (e.g. GRE/ESP) count toward flow_count but not toward any of the three ratios — ratios do not sum to 1 by design.
Lookback: 0
Future dependency: NO
Decision: KEEP
Reason: Perturbation-verified.

Feature: connection_rate
Source field: flow_count (derived)
Calculation: flow_count / 60 (flows per second, not per minute)
Lookback: 0
Future dependency: NO
Decision: KEEP
Reason: Perturbation-verified.

Feature: traffic_zscore
Source field: bytes_in
Calculation: (bytes_in - rolling_24h_mean) / (rolling_24h_std + 1e-6), rolling window computed with closed="left" (excludes current row)
Lookback: 24 hours
Future dependency: NO
Decision: KEEP
Reason: Past-only by construction (closed="left" + explicit eligibility check requiring hours_elapsed >= 24 and count_past >= 2). Structurally sound but not yet empirically exercised — 0 of 561 rows in this file are eligible, since the file only spans ~9.33 hours. This is a confirmed, team-lead-approved frozen decision: the 24h window is never shortened to fit available data. Values left as NaN (flagged via traffic_zscore_missing) rather than imputed with an invented number, since no eligible rows exist to derive a training median from.

Feature: connection_zscore
Source field: connection_rate
Calculation: same rolling z-score logic as traffic_zscore, applied to connection_rate
Lookback: 24 hours
Future dependency: NO
Decision: KEEP
Reason: Same as traffic_zscore.

Feature: peer_zscore
Source field: peer_count
Calculation: same rolling z-score logic as traffic_zscore, applied to peer_count
Lookback: 24 hours
Future dependency: NO
Decision: KEEP
Reason: Same as traffic_zscore.

Feature: future_attack (label, not a model input feature)
Source field: label_binary, label_attack (verified attack timestamp: 2016-08-29 01:08:01 UTC)
Calculation: True if the verified attack timestamp falls within [t, t+600s], where 600s is