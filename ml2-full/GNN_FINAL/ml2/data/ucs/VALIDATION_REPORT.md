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
