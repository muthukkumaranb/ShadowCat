# Validation & Audit Report: Unified Cyber State ($S_t$) Pipeline
**SIH26153 — Cyber World Model Architecture (Data Engineer Track)**
**Generated Date**: 2026-09-04 06:29:13 UTC

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

> [!NOTE]
> **Packet-Level Feature Remediation & Contract v2**: Per Option A (Corrected-but-simulated), all 12 fabricated packet-level columns (`pkt_ttl_*`, `pkt_frag_*`, `pkt_payload_size_*`, `pkt_tcp_retrans_count`, `pkt_port_scan_seq_score`) for `14-02-2018` have been zero-filled (`0.0`) and `mask_has_packet_level_features` set to `0.0` across all windows. The probabilistic LSTM World Model has been retrained (`gaussian_next_state_best_v2.pt`) and inference contract v2 delivered (`inference_scaler_v2.yaml`, `inference_feature_order_v2.json`), superseding the prior 9-column finding. Real PCAP extraction remains scoped post-MVP.

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
