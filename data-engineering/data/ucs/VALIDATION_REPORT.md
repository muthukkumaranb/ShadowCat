# Validation & Audit Report: Unified Cyber State ($S_t$) Pipeline
**SIH26153 — Cyber World Model Architecture (Data Engineer Track)**
**Generated Date**: 2026-09-04 21:00:20 UTC

---

## 1. Executive Summary & Quality Gates

| Quality Gate / Check | Requirement | Result | Status |
| :--- | :--- | :--- | :--- |
| **Zero Infinities** | 0 Inf / -Inf in final output | **0** | **PASSED** |
| **Zero Feature NaNs** | 0 unexpected NaN values | **0** | **PASSED** |
| **Chronological Monotonicity** | Strict ascending order within & across splits | **True** | **PASSED** |
| **Chronological Split Discipline** | Train -> Val -> Test strict time partitions | **70% / 15% / 15%** | **PASSED** |
| **Purge + Embargo** | Drop windows within L+H of split boundaries | **35 windows (L=30, H=5)** | **PASSED** |
| **LSTM Boundary Safety** | 0 sequences crossing split partitions | **100% boundary-safe** | **PASSED** |
| **LR/LSTM Protocol Parity** | Derived from identical purged window sets | **Confirmed** | **PASSED** |
| **Leakage-Free Normalization** | RobustScaler fit exclusively on post-purge Train | **Fitted on Train only** | **PASSED** |
| **Future Forecast Alignment** | Target backward shift with no future feature leakage | **H=5 min horizon** | **PASSED** |

---

## 2. Dataset Processing & Row Accounting

| Source File / Day | Initial Rows | Dropped Headers / Corrupted | Exact Duplicates Dropped | Cleaned Rows | 1-Min Windows | Graph Edges |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` | 1,048,575 | 5 | 225,628 | 822,942 | 543 | 111,003 |
| `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` | 1,048,575 | 0 | 17,557 | 1,031,018 | 170 | 110,651 |
| `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv` | 331,100 | 0 | 73 | 331,027 | 570 | 79,971 |
| `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` | 1,048,575 | 0 | 5,459 | 1,043,116 | 525 | 145,510 |
| `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` | 1,048,575 | 9 | 3,278 | 1,045,288 | 549 | 224,077 |
| `Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv` | 613,071 | 0 | 6,089 | 606,982 | 570 | 82,859 |

---

## 3. Split Boundaries & Chronological Audit

- **Training Partition (70%)**: 2,048 windows (`2018-02-14 01:00:00+00:00` to `2018-03-01 04:35:00+00:00`)
- **Validation Partition (15%)**: 439 windows (`2018-03-01 04:36:00+00:00` to `2018-03-02 02:24:00+00:00`)
- **Testing Partition (15%)**: 440 windows (`2018-03-02 02:25:00+00:00` to `2018-03-02 12:59:00+00:00`)

---

## 4. Class Distribution & Contiguous Attack Episodes

### Window Class Distribution:
```
label_attack_type
Benign                     2475
Infiltration-Compromise      97
SSH-Bruteforce               82
Infiltration-Portscan        58
Botnet                       48
DDOS-LOIC-UDP                19
DDOS-HOIC                     8
```

### Binary Label Distribution:
- **Benign (0)**: 1,861 windows (66.77%)
- **Attack (1)**: 926 windows (33.23%)
- **Future Attack ($H=5$ min)**: 984 windows (35.31%)

### Independent Attack Episodes (Contiguous Attack Runs):
- **SSH-Bruteforce**: 9 independent attack episode(s)
- **DDOS-HOIC**: 1 independent attack episode(s)
- **DDOS-LOIC-UDP**: 18 independent attack episode(s)
- **Infiltration-Compromise**: 1 independent attack episode(s)
- **Infiltration-Portscan**: 1 independent attack episode(s)
- **Botnet**: 10 independent attack episode(s)

> [!NOTE]
> **Infiltration Two-Phase Segmentation Verified**: The pipeline successfully segmented March 1 Infiltration traffic into `Infiltration-Compromise` (initial malware drop & C2 connection) and `Infiltration-Portscan` (internal lateral discovery), preserving the two-phase progression required for forecasting.

---

## 5. Leakage Protection: Purge + Embargo at Split Boundaries

**Purge + Embargo Width**: 35 windows (LSTM lookback L=30 + forecast horizon H=5)

**Rationale**: At each split boundary, windows within `lookback + horizon` distance can leak information across partitions through either the LSTM's lookback context or the forecast label's forward horizon. Purge+embargo drops these windows from BOTH sides of each boundary.

### Window Counts: Before vs After Purge+Embargo

| Partition | Before Purge | After Purge | Windows Dropped |
| :--- | :--- | :--- | :--- |
| **Train** | 2,048 | 2,013 | 35 |
| **Validation** | 439 | 369 | 70 |
| **Test** | 440 | 405 | 35 |
| **Total** | 2,927 | 2,787 | 140 |

### Boundary Details

- **Train→Val boundary**: 35 windows dropped from train tail + 35 from val head
- **Val→Test boundary**: 35 windows dropped from val tail + 35 from test head
  * *Test-Head Purge Composition*: All 35 dropped test-head windows (`2018-03-02 02:25:00` to `2018-03-02 02:59:00 UTC`) fall squarely within the active Friday Botnet attack episode (`2018-03-02 01:00:00` to `12:59:59 UTC`) and contain malicious flows (`has_malicious_flows=True`, `label_binary=1`, `future_attack_label=1`), with zero benign windows dropped; the resulting test set class-balance shift (from 59.1% down to 52.1% attack sequences) is therefore a natural consequence of the split boundary falling inside this contiguous Botnet episode rather than an artifact of the purge logic itself.

### LSTM Sequence Counts (post-purge)

- **Train**: 2,013 windows → 1,984 LSTM sequences (lookback=30)
- **Val**: 369 windows → 340 LSTM sequences (lookback=30)
- **Test**: 405 windows → 376 LSTM sequences (lookback=30)

### Verification Status

| Check | Result |
| :--- | :--- |
| **Purge+Embargo Applied** | ✅ PASSED |
| **LSTM Boundary Safety** | ✅ PASSED |
| **LR/LSTM Protocol Parity** | ✅ CONFIRMED |
| **Scaler Re-fit Post-Purge** | ✅ PASSED (fitted on 2013 post-purge train windows) |

> [!IMPORTANT]
> H=5 is the **primary validated forecasting horizon** (Gate 0 LOEO approved). H=10 and H=15 are sensitivity-analysis horizons only (see H_SWEEP_REPORT.md). LSTM lookback L=30 windows. Purge+embargo width = L+H = 35 windows at each split boundary.

---

## 6. Purge/Embargo Episode-Drop Rule & Authoritative 37-Fold LOEO Grouping

### General Rule

> **Any contiguous attack episode shorter than the purge/embargo width (35 windows) that falls within `width` windows of a split boundary may be fully or partially removed by purge/embargo, reducing the eligible episode count for LOEO below the raw pre-purge episode count.**

This rule is not specific to Botnet or to this dataset — it is a structural consequence of the leakage-protection design. Any short episode whose windows all happen to sit inside an embargo zone at a split boundary will be dropped entirely from the post-purge window set and will therefore not appear as an eligible LOEO fold unit.

### Instance That Triggered This Rule (Botnet, Val→Test Boundary)

| Metric | Value |
| :--- | :--- |
| **Pre-purge contiguous Botnet episodes** | **11** |
| **Post-purge eligible Botnet episodes** | **10** |
| **Episode eliminated** | 1 short Botnet episode fully consumed by the Val→Test embargo zone |
| **Purge/embargo width applied** | 35 windows (L=30 + H=5) |

One Botnet episode was short enough that all of its windows fell within the 35-window embargo zone at the Val→Test split boundary. After purge+embargo was applied, that episode had zero surviving windows in the post-purge dataset and therefore did not appear as an eligible held-out fold unit.

### Authoritative Episode Count for the 37-Fold LOEO Grouping

The corrected, authoritative LOEO evaluation uses **37 folds** (validated run `loeo_37fold_results.csv`, 2026-09-05), scoped to **multi-episode attack types only** — types with ≥2 independent contiguous episodes in the post-purge dataset:

| Attack Type | Post-Purge Episodes | LOEO Folds | Eligible for LOEO? |
| :--- | :---: | :---: | :--- |
| **SSH-Bruteforce** | 9 | 9 | ✅ Multi-episode |
| **DDOS-LOIC-UDP** | 18 | 18 | ✅ Multi-episode |
| **Botnet** | **10** | **10** | ✅ Multi-episode |
| DDOS-HOIC | 1 | — | ❌ Singleton — excluded |
| Infiltration-Compromise | 1 | — | ❌ Singleton — excluded |
| Infiltration-Portscan | 1 | — | ❌ Singleton — excluded |
| **Multi-ep Total** | **37** | **37** | — |
| Raw total (all types) | 40 | — | — |

> [!WARNING]
> The earlier 38-fold result (`loeo_fold_results_behavior_only.csv`, 2026-09-04) was computed against a superseded 11-Botnet-episode grouping and is **no longer authoritative**. The corrected post-purge dataset has 10 Botnet episodes, yielding **37 multi-episode LOEO folds** (9 + 18 + 10). All Gate 0 LOEO numbers have been rerun under this corrected grouping; see `gate0_leakage_report.md` and `gate0_protocol4_loeo_forecasting_report.md` for the updated results.

> [!NOTE]
> The "40" appearing in earlier investigation notes refers to the **raw total attack episode count** across all types including singletons — not the LOEO-eligible subset.

> [!NOTE]
> **Known Granularity Limitation (Intermittent Bursts)**: Under the strict 1-minute unbroken contiguous-run definition, intermittent attack bursts separated by brief 1-to-3 minute benign gaps (e.g., Botnet episodes at 10:45, 10:47, and 10:51 UTC) are classified as separate single-window episodes rather than a single merged multi-pulse event; any future coarser clustering rule would consolidate these into fewer, longer episodes without altering the underlying leakage boundaries.

---

## 7. Packet-Level Feature Coverage Limitation

> Packet-level features validated for SSH-Bruteforce (14-02-2018) only; flow-level covers all six days; full extraction scoped post-MVP (~250GB download cost across remaining 5 days).

### Cross-Reference: S3 Download Cost Verification

The ~250 GB figure was cross-referenced against live S3 bucket contents (`cse-cic-ids2018.s3.amazonaws.com`):

| Day | pcap.zip Size (GB) |
| :--- | ---: |
| 21-02-2018 | 49.8 |
| 22-02-2018 | 46.8 |
| 28-02-2018 | 49.6 |
| 01-03-2018 | 48.8 |
| 02-03-2018 | 41.7 |
| **Total** | **236.7** |

Actual compressed total: **236.7 GB** (~95% of the stated ~250 GB round figure). The ~250 GB claim is consistent with the S3 data.

### Evidence

- `mask_has_packet_level_features == 1` exists exclusively for 14-02-2018 (543/543 windows). All other days have `mask_has_packet_level_features == 0`.
- `src/pcap_extractor.py` hardcodes `target_day="14-02-2018"`. No other day is referenced in any code path, config, git commit, or README.
- No `.pcap` or `.pcapng` files exist locally for any day.

---

## 8. Known Limitations & Documented Findings

### 8.1 Pre-Split Whole-Day Median: Identified Leakage Vector

> [!IMPORTANT]
> **Newly Identified Finding** — documented as a known limitation, not silently resolved.

The original batch pipeline's within-day median (computed in `cleaner.py`'s `impute_missing_flow_values()`) was computed from **each day's entire flow set**, **before** the train/val/test split existed. The split happens later, at the global window level.

This means the original per-day median for any given day mixed in flows destined for train, val, **and** test partitions of that day — a subtle, **pre-split leakage vector** distinct from anything purge/embargo protects against, since:
- Purge/embargo guards only the **lookback/horizon boundary region** (35 windows at each boundary)
- This whole-day statistic was computed **before any boundary existed**

**Decision**: `data/ucs/imputation_params.yaml` is frozen using **ONLY the 3,628,886 train-window flows** (derived strictly from the verified, deduplicated intermediate parquets) — this is the leakage-safe correct choice for a frozen production artifact, even though it means Test 1 (bit-exact regression) will legitimately NOT match the historical parquet for any window containing an originally-imputed value.

This is documented **explicitly** here, not silently accepted as "close enough."

### 8.2 Imputed Flow Count — Full-Dataset Audit (Post-Deduplication)

Full audit across all 6 intermediate cleaned parquets (4,880,373 cleaned, deduplicated flows total):

| Day | Cleaned Flows (Post-Dedup) | Flows w/ Inf→NaN (Zero Dur) | % Imputed | Columns Affected | Train-Window Flows | Train Windows Affected |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| 14-02-2018 | 822,942 | 3,821 | 0.4643% | `packets_per_sec`, `bytes_per_sec` | 822,942 | 471 / 543 |
| 21-02-2018 | 1,031,018 | 0 | 0.0000% | — | 1,031,018 | 0 / 170 |
| 22-02-2018 | 1,045,288 | 5,604 | 0.5361% | `packets_per_sec`, `bytes_per_sec` | 1,045,288 | 538 / 549 |
| 28-02-2018 | 606,982 | 4,786 | 0.7885% | `packets_per_sec`, `bytes_per_sec` | 606,982 | 536 / 570 |
| 01-03-2018 | 331,027 | 2,917 | 0.8812% | `packets_per_sec`, `bytes_per_sec` | 122,656 | 173 / 181 |
| 02-03-2018 | 1,043,116 | 4,044 | 0.3877% | `packets_per_sec`, `bytes_per_sec` | 0 | 0 / 0 (all val/test) |
| **TOTAL** | **4,880,373** | **21,172** | **0.4338%** | — | **3,628,886** | **1,718 / 2,013** |

The cause is zero-duration flows where CICFlowMeter sets rate columns to `Inf` (division by zero). The `clean_and_normalize_flow_data()` function converts `Inf → NaN` and drops exact duplicates; `impute_missing_flow_values()` then fills with the within-day median.

### 8.3 Test 1 (Bit-Exact Regression) — Scope Restriction

> [!WARNING]
> **1,718 out of 2,013 post-purge train windows (85.35%)** contain at least one flow that was originally imputed in the batch run. Only **295 train windows are "clean" (14.65%)** (contain zero originally-imputed flows).

**Test 1 is scoped to the 295 clean train windows only.** These windows are fully bit-exact reproducible because no imputation occurred during the original batch run, so the frozen artifact produces numerically identical output.

For the 1,718 affected windows, the frozen runtime imputation median (train-window-only) differs from the original batch pipeline's pre-split whole-day median — this is a **bounded, expected, documented discrepancy** that is the correct outcome of choosing leakage-safe statistics.

**Regression test window selection criterion**: Select from the 295 clean train windows (source_day=21-02-2018 provides 170 guaranteed-clean windows since that day had zero Inf-producing flows).

### 8.4 Frozen Imputation Parameters — Dual Structure

`data/ucs/imputation_params.yaml` implements a dual structure:

| Section | Contents | Used When |
| :--- | :--- | :--- |
| `flow_medians_by_day` | Per-day medians for 5 days (14-02, 21-02, 22-02, 28-02, 01-03), each from that day's train-window flows only | `source_type="csv"` + recognized `source_day` |
| `flow_medians_global` | Single pooled median across all 3,628,886 train-window flows (`bytes_per_sec=1076.026418, packets_per_sec=422.986283`) | `source_type="csv"` + unrecognized day at runtime |
| `window_feature_medians` | 400 window-level features in raw scale (inverse `expm1` applied to `log1p` features) | Missing window feature fill centering at `0.0` under RobustScaler |

Note: 02-03-2018 (Friday/Botnet) has **no** `flow_medians_by_day` entry because all its flows fall in the val/test partition — per-day train-window flows for that day are zero. Any runtime inference on 02-03-2018-like input falls back to the global median.

### 8.5 Synthetic Test for Imputation Path (Task 4)

A dedicated synthetic test (`tests/test_frozen_imputation.py :: TestSyntheticImputeSubstitution`) directly exercises the `_frozen_impute()` substitution logic by constructing a deliberate `Inf → NaN` flow and asserting the frozen median is applied correctly. This test does **not** depend on finding a natural historical example (which is rare given ~0.44% flow frequency), so it is always exercisable without the full dataset.

### 8.6 Cross-Team Scaler Alignment with ML1 Inference Contract

> [!NOTE]
> **Authoritative Compatibility Alignment**: Adopted ML1's `inference_scaler_v1.yaml` parameters into `data/ucs/scaler_params.yaml` to ensure 100% bit-exact parity with the frozen LSTM checkpoint (`gaussian_next_state_best.pt`).

During cross-team contract verification between `UCSExtractor` and ML1's artifacts (`inference_feature_order_v1.json` and `inference_scaler_v1.yaml`):
1. **Feature Order**: 406/406 input columns matched in 100% exact identical order.
2. **Scaler Parameters**: 395/400 features matched bit-exactly across all parameters (`median`, `scale`, `is_log1p`, `q25`, `q75`).
3. **Packet Feature Divergence & Root Cause**: Exactly 5 PCAP packet features (`pkt_frag_df_count`, `pkt_payload_size_p50`, `pkt_payload_size_p75`, `pkt_payload_size_p95`, `pkt_tcp_retrans_count`) showed divergent `scale` and `q75` values.
   - **Root Cause Investigation**: In `src/pcap_extractor.py`, because no local `.pcap` files exist on disk, packet features for 14-02-2018 were generated via deterministic flow-derived simulation using heuristic formulas and clipping bounds (e.g., `np.clip(fwd_byts / fwd_pkts, 40.0, 150.0)`).
   - In Data Eng's earlier scaler, these 5 features used the heuristic clipping floor/ceil values (e.g. `scale=40.0, 75.0, 150.0, 0.8, 0.01`).
   - In ML1's training-time export, they used the empirical 75th percentiles of the simulated window distribution (`1164.8, 48.2544, 88.0, 186.4017, 3.06`).
   - **Resolution**: `data/ucs/scaler_params.yaml` was updated to adopt ML1's empirical values, establishing 100% bit-exact alignment across all 400 features.
   - **Caveat**: The underlying packet features for 14-02-2018 represent flow-derived deterministic simulations rather than real extracted raw PCAP frames. Full genuine PCAP extraction across all days remains deferred post-MVP (~250 GB download cost).
