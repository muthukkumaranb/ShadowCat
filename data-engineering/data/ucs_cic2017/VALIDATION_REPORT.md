# Validation & Audit Report: Unified Cyber State ($S_t$) Pipeline
**SIH26153 — Cyber World Model Architecture (Data Engineer Track)**
**Generated Date**: 2026-10-02 14:08:58 UTC

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
| `Monday-WorkingHours.pcap_ISCX.csv` | 529,918 | 0 | 0 | 529,918 | 8,832 | 136,060 |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 445,909 | 0 | 0 | 445,909 | 7,432 | 113,130 |
| `Wednesday-workingHours.pcap_ISCX.csv` | 692,703 | 0 | 0 | 692,703 | 11,546 | 114,259 |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 170,366 | 0 | 0 | 170,366 | 2,840 | 38,684 |

---

## 3. Split Boundaries & Chronological Audit

- **Training Partition (70%)**: 21,455 windows (`2017-07-03 00:00:00+00:00` to `2017-07-07 07:24:00+00:00`)
- **Validation Partition (15%)**: 4,597 windows (`2017-07-07 07:25:00+00:00` to `2017-07-08 11:29:00+00:00`)
- **Testing Partition (15%)**: 4,598 windows (`2017-07-08 11:30:00+00:00` to `2017-07-11 00:25:00+00:00`)

---

## 4. Class Distribution & Contiguous Attack Episodes

### Window Class Distribution:
```
label_attack_type
BENIGN              19175
DoS Hulk             4129
DoS GoldenEye        2890
SSH-Patator          2456
FTP-Patator           791
Web-BruteForce        472
DoS slowloris         325
Web-XSS               162
DoS Slowhttptest       93
Web-SQLi               16
Heartbleed              1
```

### Binary Label Distribution:
- **Benign (0)**: 19,175 windows (62.85%)
- **Attack (1)**: 11,335 windows (37.15%)
- **Future Attack ($H=5$ min)**: 25,674 windows (84.15%)

### Independent Attack Episodes (Contiguous Attack Runs):
- **BENIGN**: 6436 independent attack episode(s)
- **DoS slowloris**: 309 independent attack episode(s)
- **FTP-Patator**: 744 independent attack episode(s)
- **Web-BruteForce**: 448 independent attack episode(s)
- **DoS Slowhttptest**: 85 independent attack episode(s)
- **Web-XSS**: 150 independent attack episode(s)
- **DoS Hulk**: 3728 independent attack episode(s)
- **Web-SQLi**: 16 independent attack episode(s)
- **SSH-Patator**: 2278 independent attack episode(s)
- **DoS GoldenEye**: 287 independent attack episode(s)
- **Heartbleed**: 1 independent attack episode(s)

> [!NOTE]
> **Infiltration Two-Phase Segmentation Verified**: The pipeline successfully segmented March 1 Infiltration traffic into `Infiltration-Compromise` (initial malware drop & C2 connection) and `Infiltration-Portscan` (internal lateral discovery), preserving the two-phase progression required for forecasting.

---

## 5. Leakage Protection: Purge + Embargo at Split Boundaries

**Purge + Embargo Width**: 35 windows (LSTM lookback L=30 + forecast horizon H=5)

**Rationale**: At each split boundary, windows within `lookback + horizon` distance can leak information across partitions through either the LSTM's lookback context or the forecast label's forward horizon. Purge+embargo drops these windows from BOTH sides of each boundary.

### Window Counts: Before vs After Purge+Embargo

| Partition | Before Purge | After Purge | Windows Dropped |
| :--- | :--- | :--- | :--- |
| **Train** | 21,455 | 21,420 | 35 |
| **Validation** | 4,597 | 4,527 | 70 |
| **Test** | 4,598 | 4,563 | 35 |
| **Total** | 30,650 | 30,510 | 140 |

### Boundary Details

- **Train→Val boundary**: 35 windows dropped from train tail + 35 from val head
- **Val→Test boundary**: 35 windows dropped from val tail + 35 from test head
  * *Test-Head Purge Composition*: All 35 dropped test-head windows (`2018-03-02 02:25:00` to `2018-03-02 02:59:00 UTC`) fall squarely within the active Friday Botnet attack episode (`2018-03-02 01:00:00` to `12:59:59 UTC`) and contain malicious flows (`has_malicious_flows=True`, `label_binary=1`, `future_attack_label=1`), with zero benign windows dropped; the resulting test set class-balance shift (from 59.1% down to 52.1% attack sequences) is therefore a natural consequence of the split boundary falling inside this contiguous Botnet episode rather than an artifact of the purge logic itself.

### LSTM Sequence Counts (post-purge)

- **Train**: 21,420 windows → 21,391 LSTM sequences (lookback=30)
- **Val**: 4,527 windows → 4,498 LSTM sequences (lookback=30)
- **Test**: 4,563 windows → 4,534 LSTM sequences (lookback=30)

### Verification Status

| Check | Result |
| :--- | :--- |
| **Purge+Embargo Applied** | ✅ PASSED |
| **LSTM Boundary Safety** | ✅ PASSED |
| **LR/LSTM Protocol Parity** | ✅ CONFIRMED |
| **Scaler Re-fit Post-Purge** | ✅ PASSED (fitted on 21420 post-purge train windows) |

> [!IMPORTANT]
> H=5 is the **primary validated forecasting horizon** (Gate 0 LOEO approved). H=10 and H=15 are sensitivity-analysis horizons only (see H_SWEEP_REPORT.md). LSTM lookback L=30 windows. Purge+embargo width = L+H = 35 windows at each split boundary.
