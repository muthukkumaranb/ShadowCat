# Validation & Audit Report: Unified Cyber State ($S_t$) Pipeline
**SIH26153 — Cyber World Model Architecture (Data Engineer Track)**
**Generated Date**: 2026-10-02 14:03:48 UTC

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
| `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` | 1,048,575 | 5 | 225,628 | 129,905 | 271 | 1,084 |
| `Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv` | 1,046,154 | 0 | 0 | 1,046,154 | 572 | 166,351 |
| `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv` | 900,988 | 0 | 0 | 900,988 | 191 | 95,311 |
| `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` | 4,769,495 | 0 | 0 | 4,769,495 | 720 | 720 |
| `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` | 1,048,575 | 0 | 17,557 | 625,312 | 161 | 46,896 |
| `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` | 1,048,575 | 9 | 3,278 | 757,872 | 549 | 170,454 |
| `Friday-23-02-2018_TrafficForML_CICFlowMeter.csv` | 1,045,957 | 0 | 0 | 1,045,957 | 556 | 233,677 |
| `Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv` | 613,071 | 0 | 6,089 | 606,982 | 570 | 82,859 |
| `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv` | 331,100 | 0 | 73 | 331,027 | 570 | 79,971 |
| `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` | 1,048,575 | 0 | 5,459 | 1,043,116 | 525 | 145,510 |

---

## 3. Split Boundaries & Chronological Audit

- **Training Partition (70%)**: 3,279 windows (`2018-02-14 01:01:00+00:00` to `2018-02-28 05:18:00+00:00`)
- **Validation Partition (15%)**: 702 windows (`2018-02-28 05:19:00+00:00` to `2018-03-01 10:00:00+00:00`)
- **Testing Partition (15%)**: 704 windows (`2018-03-01 10:01:00+00:00` to `2018-03-02 12:59:00+00:00`)

---

## 4. Class Distribution & Contiguous Attack Episodes

### Window Class Distribution:
```
label_attack_type
Benign                      3285
Botnet                       342
Web-BruteForce               133
Infiltration                 133
Web-XSS                      108
Infiltration-Compromise       97
FTP-BruteForce                91
DDoS attacks-LOIC-HTTP        80
SSH-Bruteforce                76
DoS attacks-SlowHTTPTest      47
DoS attacks-Slowloris         43
DDOS-LOIC-UDP                 36
DoS attacks-GoldenEye         19
Infiltration-Portscan         19
Web-SQLi                      18
DDOS-HOIC                     14
DoS attacks-Hulk               4
```

### Binary Label Distribution:
- **Benign (0)**: 3,285 windows (72.28%)
- **Attack (1)**: 1,260 windows (27.72%)
- **Future Attack ($H=5$ min)**: 1,373 windows (30.21%)

### Independent Attack Episodes (Contiguous Attack Runs):
- **SSH-Bruteforce**: 1 independent attack episode(s)
- **FTP-BruteForce**: 1 independent attack episode(s)
- **DoS attacks-GoldenEye**: 4 independent attack episode(s)
- **DoS attacks-Slowloris**: 1 independent attack episode(s)
- **DoS attacks-Hulk**: 1 independent attack episode(s)
- **DoS attacks-SlowHTTPTest**: 1 independent attack episode(s)
- **DDoS attacks-LOIC-HTTP**: 2 independent attack episode(s)
- **DDOS-HOIC**: 1 independent attack episode(s)
- **DDOS-LOIC-UDP**: 1 independent attack episode(s)
- **Web-XSS**: 2 independent attack episode(s)
- **Web-SQLi**: 9 independent attack episode(s)
- **Web-BruteForce**: 5 independent attack episode(s)
- **Infiltration**: 2 independent attack episode(s)
- **Infiltration-Compromise**: 1 independent attack episode(s)
- **Infiltration-Portscan**: 1 independent attack episode(s)
- **Botnet**: 2 independent attack episode(s)

> [!NOTE]
> **Infiltration Two-Phase Segmentation Verified**: The pipeline successfully segmented March 1 Infiltration traffic into `Infiltration-Compromise` (initial malware drop & C2 connection) and `Infiltration-Portscan` (internal lateral discovery), preserving the two-phase progression required for forecasting.

---

## 5. Leakage Protection: Purge + Embargo at Split Boundaries

**Purge + Embargo Width**: 35 windows (LSTM lookback L=30 + forecast horizon H=5)

**Rationale**: At each split boundary, windows within `lookback + horizon` distance can leak information across partitions through either the LSTM's lookback context or the forecast label's forward horizon. Purge+embargo drops these windows from BOTH sides of each boundary.

### Window Counts: Before vs After Purge+Embargo

| Partition | Before Purge | After Purge | Windows Dropped |
| :--- | :--- | :--- | :--- |
| **Train** | 3,279 | 3,244 | 35 |
| **Validation** | 702 | 632 | 70 |
| **Test** | 704 | 669 | 35 |
| **Total** | 4,685 | 4,545 | 140 |

### Boundary Details

- **Train→Val boundary**: 35 windows dropped from train tail + 35 from val head
- **Val→Test boundary**: 35 windows dropped from val tail + 35 from test head
  * *Test-Head Purge Composition*: All 35 dropped test-head windows (`2018-03-02 02:25:00` to `2018-03-02 02:59:00 UTC`) fall squarely within the active Friday Botnet attack episode (`2018-03-02 01:00:00` to `12:59:59 UTC`) and contain malicious flows (`has_malicious_flows=True`, `label_binary=1`, `future_attack_label=1`), with zero benign windows dropped; the resulting test set class-balance shift (from 59.1% down to 52.1% attack sequences) is therefore a natural consequence of the split boundary falling inside this contiguous Botnet episode rather than an artifact of the purge logic itself.

### LSTM Sequence Counts (post-purge)

- **Train**: 3,244 windows → 3,215 LSTM sequences (lookback=30)
- **Val**: 632 windows → 603 LSTM sequences (lookback=30)
- **Test**: 669 windows → 640 LSTM sequences (lookback=30)

### Verification Status

| Check | Result |
| :--- | :--- |
| **Purge+Embargo Applied** | ✅ PASSED |
| **LSTM Boundary Safety** | ✅ PASSED |
| **LR/LSTM Protocol Parity** | ✅ CONFIRMED |
| **Scaler Re-fit Post-Purge** | ✅ PASSED (fitted on 3244 post-purge train windows) |

> [!IMPORTANT]
> H=5 is the **primary validated forecasting horizon** (Gate 0 LOEO approved). H=10 and H=15 are sensitivity-analysis horizons only (see H_SWEEP_REPORT.md). LSTM lookback L=30 windows. Purge+embargo width = L+H = 35 windows at each split boundary.
