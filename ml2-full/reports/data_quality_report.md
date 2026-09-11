# Data Quality Report — UGR16 (Scope: UGR16:TEST)

## Baseline Availability (24-Hour Rolling Z-Score)

Dataset: UGR16
Scope: UGR16:TEST

Available history: approximately 9 hours 20 minutes
(2016-08-29 00:07:33 UTC to 2016-08-29 09:27:42 UTC)

Required baseline history: 24 hours (rolling, past-only, per configs/feature_config.yaml)

traffic_zscore available: NO
connection_zscore available: NO
peer_zscore available: NO

Percentage of rows with insufficient baseline: 100.0000% (561 of 561 windows)

Reason: Source file (august_week5.parquet) does not contain sufficient
historical observations to compute the required 24-hour past-only rolling
baseline. The file spans only ~9.33 hours; no timestamp in the file has
24 real hours of preceding history available within this dataset.

Imputation status: No training-derived median could be calculated for
traffic_zscore, connection_zscore, or peer_zscore, because zero rows in
this file are eligible to produce a real baseline value. All three
z-score columns are currently NaN, pending a dataset with sufficient
historical coverage. This is a documented, expected limitation of the
current source file — not a pipeline defect.

## Flow Validation Summary
Total rows loaded: 40,289,595
Valid flows: 40,289,595 (100.0000%)
Dropped flows: 0 (0.0000%)

## Windowing Summary
Total 1-minute windows generated: 561
Expected minute-slots in time range: 561
Missing minutes (zero-flow gaps): 0

## Class Imbalance (label_binary)
unknown: 40,289,594 rows (99.9999975%)
malicious: 1 row (0.0000025%)
Verified attack: anomaly-sshscan at approximately 2016-08-29 01:10:00 UTC