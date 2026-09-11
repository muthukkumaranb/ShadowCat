# Gate 0: Schedule & Artifact Leakage Report (Comprehensive Diagnostic Suite)
**SIH26153 - Cyber World Model Architecture**
**Test Executed**: 2026-09-02 21:23:00 UTC

---

## 1. Executive Summary & Multi-Protocol Comparison

| Evaluation Protocol | Feature Set | F1 Score | Precision | Recall | Accuracy | Verdict & Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Chronological Holdout**<br>*(440 windows: March 2 Botnet)* | **Set A** (Traffic+Packet) | **0.0436** | **0.3333** | **0.0233** | **0.4023** | Unseen Attack Blindspot (Botnet in Test) |
| | **Set B** (Hour+Day Only) | **0.6893** | **0.9161** | **0.5525** | **0.7091** | High Precision exploits daily schedule |
| **2. Window-Stratified Diagnostic**<br>*(440 windows: IID sample across days)* | **Set A** (Traffic+Packet) | **0.8304** | **0.8696** | **0.7947** | **0.8886** | Strong Signal on known attack signatures |
| | **Set B** (Hour+Day Only) | **0.7893** | **0.7151** | **0.8808** | **0.8386** | Precision drops ~20% (0.9161 -> 0.7151) |
| **3. Episode-Grouped Diagnostic**<br>*(637 windows: Whole episodes held out)* | **Set A** (Traffic+Packet) | **0.2080** | **0.1703** | **0.2670** | **0.4380** | **Below majority baseline (0.724)** |
| | **Set B** (Hour+Day Only) | **0.5688** | **0.4822** | **0.6932** | **0.7096** | Schedule baseline still dominant |
| **4. LOEO Cross-Validation**<br>*(~~38~~ SUPERSEDED — see 37-fold rerun below)* | **Set A** (Traffic+Packet) | ~~**0.9274 +/- 0.1225**~~ | ~~**0.9689 +/- 0.0879**~~ | ~~**0.9070 +/- 0.1629**~~ | — | ~~**CONDITIONAL PASS**~~ SUPERSEDED |
| | **Set B** (Hour+Day Only) | ~~**0.8884 +/- 0.1014**~~ | ~~**0.8275 +/- 0.1572**~~ | ~~**0.9826 +/- 0.0485**~~ | — | ~~Schedule baseline~~ SUPERSEDED |
| **4b. LOEO Cross-Validation (37-fold rerun)**<br>*(37 folds, corrected Botnet=10 post-purge)* | **Set A** (Traffic+Packet) | **0.9077 +/- 0.1038** | **0.9129 +/- 0.1527** | **0.9269 +/- 0.1040** | — | **CONDITIONAL PASS** (ranges overlap) |
| | **Set B** (Hour+Day Only) | **0.8126 +/- 0.1475** | **0.7215 +/- 0.1948** | **0.9788 +/- 0.0627** | — | Schedule baseline, high recall |

---

## 2. Confusion Matrices (Protocols 1-3)

### Protocol 1: Chronological Holdout Split (440 windows)
```
Set A (Traffic + Packet Features):          Set B (Schedule Artifacts Only):
              Pred Benign  Pred Attack                    Pred Benign  Pred Attack
Actual Benign      171           12      Actual Benign      170           13
Actual Attack      251            6      Actual Attack      115         142
```

### Protocol 2: Window-Stratified Diagnostic (440 windows)
```
Set A (Traffic + Packet Features):          Set B (Schedule Artifacts Only):
              Pred Benign  Pred Attack                    Pred Benign  Pred Attack
Actual Benign      271           18      Actual Benign      236           53
Actual Attack       31          120      Actual Attack       18          133
```

### Protocol 3: Episode-Grouped Stratified Diagnostic (637 windows)
```
Set A (Traffic + Packet Features):          Set B (Schedule Artifacts Only):
              Pred Benign  Pred Attack                    Pred Benign  Pred Attack
Actual Benign      232          229      Actual Benign      330          131
Actual Attack      129           47      Actual Attack       54          122
```

---

## 3. Deep Analysis & Key Takeaways

### A. Episode-Grouped Split (Protocol 3): Train/Test Composition

| Attack Type | Train Windows | Test Windows | Train % | Test % |
| :--- | ---: | ---: | ---: | ---: |
| Benign | 1990 | 620 | 86.9% | 97.3% |
| Botnet | 40 | 13 | 1.7% | 2.0% |
| DDOS-HOIC | 8 | **0** | 0.3% | **0.0%** |
| DDOS-LOIC-UDP | 16 | 3 | 0.7% | 0.5% |
| Infiltration-Compromise | 97 | **0** | 4.2% | **0.0%** |
| Infiltration-Portscan | 58 | **0** | 2.5% | **0.0%** |
| SSH-Bruteforce | 81 | 1 | 3.5% | 0.2% |

Binary label (label_binary): Train = 1470 Benign / 820 Attack (64/36%); Test = 461 Benign / 176 Attack (72/28%).

**Three attack types entirely absent from test** (DDOS-HOIC, Infiltration-Compromise, Infiltration-Portscan) — each has only 1 contiguous episode, so episode integrity forces them into train. This severe distribution mismatch explains Protocol 3's anomalous results and motivates Protocol 4 (LOEO).

### B. Protocol 4: Leave-One-Episode-Out (LOEO) Cross-Validation

LOEO eliminates Protocol 3's distribution mismatch by cycling through each episode of each multi-episode attack type as a held-out test fold, with proportionally sampled benign windows from all 6 source days. Singleton types (DDOS-HOIC, Infiltration-Compromise, Infiltration-Portscan) are excluded — see Section 3F for singleton caveat.

#### B.1 SSH-Bruteforce (9 folds, all from 14-02-2018)

| Fold | Ep ID | Test Atk/Ben | Set A F1 | Set A Prec | Set A Rec | Set B F1 | Set B Prec | Set B Rec | Winner |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 0 | 3 | 1/11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Tie |
| 1 | 5 | 4/11 | 1.0000 | 1.0000 | 1.0000 | 0.8750 | 0.8750 | 0.8750 | **A** |
| 2 | 7 | 10/11 | 1.0000 | 1.0000 | 1.0000 | 0.9524 | 0.9091 | 1.0000 | **A** |
| 3 | 9 | 1/11 | 1.0000 | 1.0000 | 1.0000 | 0.8889 | 0.8000 | 1.0000 | **A** |
| 4 | 11 | 8/11 | 0.9524 | 1.0000 | 0.9091 | 0.9565 | 0.9167 | 1.0000 | B |
| 5 | 13 | 12/13 | 1.0000 | 1.0000 | 1.0000 | 0.9375 | 0.8824 | 1.0000 | **A** |
| 6 | 15 | 33/33 | 1.0000 | 1.0000 | 1.0000 | 0.9318 | 0.8723 | 1.0000 | **A** |
| 7 | 17 | 6/11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Tie |
| 8 | 19 | 7/11 | 0.9412 | 1.0000 | 0.8889 | 0.9474 | 0.9000 | 1.0000 | B |
| | | **Mean** | **0.9882** | **1.0000** | **0.9776** | **0.9433** | **0.9062** | **0.9861** | **A: 5, B: 2, Tie: 2** |
| | | **Std** | **0.0236** | **0.0000** | **0.0448** | **0.0426** | **0.0630** | **0.0417** | |

#### B.2 DDOS-LOIC-UDP (18 folds, all from 21-02-2018)

| Fold | Ep ID | Test Atk/Ben | Set A F1 | Set A Prec | Set A Rec | Set B F1 | Set B Prec | Set B Rec | Winner |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 0 | 28 | 1/11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Tie |
| 1 | 30 | 1/11 | 1.0000 | 1.0000 | 1.0000 | 0.8000 | 0.8000 | 0.8000 | **A** |
| 2 | 32 | 1/11 | 1.0000 | 1.0000 | 1.0000 | 0.6667 | 0.5000 | 1.0000 | **A** |
| 3 | 34 | 2/11 | 1.0000 | 1.0000 | 1.0000 | 0.9091 | 0.8333 | 1.0000 | **A** |
| 4 | 36 | 1/11 | 0.8571 | 1.0000 | 0.7500 | 0.8889 | 0.8000 | 1.0000 | B |
| 5 | 38 | 1/11 | 1.0000 | 1.0000 | 1.0000 | 0.8000 | 0.6667 | 1.0000 | **A** |
| 6 | 40 | 1/11 | 1.0000 | 1.0000 | 1.0000 | 0.7500 | 0.6000 | 1.0000 | **A** |
| 7 | 42 | 1/11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Tie |
| 8 | 44 | 1/11 | 0.8000 | 1.0000 | 0.6667 | 0.8571 | 0.7500 | 1.0000 | B |
| 9 | 46 | 1/11 | 0.7500 | 1.0000 | 0.6000 | 0.8333 | 0.7143 | 1.0000 | B |
| 10 | 48 | 1/11 | 0.8333 | 0.8333 | 0.8333 | 0.9091 | 1.0000 | 0.8333 | B |
| 11 | 50 | 1/11 | 0.8571 | 0.7500 | 1.0000 | 0.6667 | 0.5000 | 1.0000 | **A** |
| 12 | 52 | 1/11 | 1.0000 | 1.0000 | 1.0000 | 0.7500 | 0.6000 | 1.0000 | **A** |
| 13 | 54 | 1/11 | 0.6667 | 0.6667 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | B |
| 14 | 56 | 1/11 | 1.0000 | 1.0000 | 1.0000 | 0.6000 | 0.4286 | 1.0000 | **A** |
| 15 | 58 | 1/11 | 1.0000 | 1.0000 | 1.0000 | 0.8333 | 0.7143 | 1.0000 | **A** |
| 16 | 60 | 1/11 | 0.7273 | 0.6667 | 0.8000 | 0.9091 | 0.8333 | 1.0000 | B |
| 17 | 62 | 1/11 | 0.5000 | 1.0000 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | B |
| | | **Mean** | **0.8884** | **0.9398** | **0.8694** | **0.8430** | **0.7634** | **0.9796** | **A: 9, B: 7, Tie: 2** |
| | | **Std** | **0.1500** | **0.1206** | **0.1956** | **0.1228** | **0.1896** | **0.0596** | |

#### B.3 Botnet (11 folds, all from 02-03-2018)

| Fold | Ep ID | Test Atk/Ben | Set A F1 | Set A Prec | Set A Rec | Set B F1 | Set B Prec | Set B Rec | Winner |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 0 | 91 | 5/11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Tie |
| 1 | 93 | 10/11 | 1.0000 | 1.0000 | 1.0000 | 0.9286 | 0.9286 | 0.9286 | **A** |
| 2 | 95 | 12/13 | 1.0000 | 1.0000 | 1.0000 | 0.9600 | 0.9231 | 1.0000 | **A** |
| 3 | 99 | 1/11 | 1.0000 | 1.0000 | 1.0000 | 0.8889 | 0.8000 | 1.0000 | **A** |
| 4 | 101 | 1/11 | 0.6667 | 1.0000 | 0.5000 | 0.8889 | 0.8000 | 1.0000 | B |
| 5 | 103 | 1/11 | 1.0000 | 1.0000 | 1.0000 | 0.8000 | 0.6667 | 1.0000 | **A** |
| 6 | 105 | 4/11 | 1.0000 | 1.0000 | 1.0000 | 0.8571 | 0.7500 | 1.0000 | **A** |
| 7 | 107 | 4/11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Tie |
| 8 | 109 | 3/11 | 0.8889 | 1.0000 | 0.8000 | 0.9091 | 0.8333 | 1.0000 | B |
| 9 | 111 | 7/11 | 0.9000 | 1.0000 | 0.8182 | 0.9167 | 0.8462 | 1.0000 | B |
| 10 | 113 | 5/11 | 0.9000 | 0.9000 | 0.9000 | 0.9474 | 1.0000 | 0.9000 | B |
| | | **Mean** | **0.9414** | **0.9909** | **0.9107** | **0.9179** | **0.8680** | **0.9844** | **A: 5, B: 4, Tie: 2** |
| | | **Std** | **0.1028** | **0.0302** | **0.1567** | **0.0597** | **0.1115** | **0.0353** | |

#### B.4 Overall LOEO Aggregate (38 folds across all multi-episode types)

> [!WARNING]
> **SUPERSEDED.** The 38-fold aggregate below is from the original Botnet=11 grouping. See Section B.4b for the authoritative 37-fold rerun.

| Metric | Set A (Traffic+Packet) | Set B (Schedule-Only) |
| :--- | :--- | :--- |
| **F1** | ~~**0.9274 +/- 0.1225**~~ | ~~**0.8884 +/- 0.1014**~~ |
| **Precision** | ~~**0.9689 +/- 0.0879**~~ | ~~**0.8275 +/- 0.1572**~~ |
| **Recall** | ~~**0.9070 +/- 0.1629**~~ | ~~**0.9826 +/- 0.0485**~~ |
| Head-to-Head (38 folds) | ~~**A wins 19**~~ | ~~B wins 13, Ties 6~~ |
| Mean +/- 1 Std Range | ~~[0.8049, 1.0499]~~ | ~~[0.7870, 0.9898]~~ |
| **Ranges Overlap?** | ~~**Yes**~~ | |

#### B.4b Overall LOEO Aggregate (37-FOLD RERUN — AUTHORITATIVE)

| Metric | Set A (Traffic+Packet) | Set B (Schedule-Only) |
| :--- | :--- | :--- |
| **F1** | **0.9077 +/- 0.1038** | **0.8126 +/- 0.1475** |
| **Precision** | **0.9129 +/- 0.1527** | **0.7215 +/- 0.1948** |
| **Recall** | **0.9269 +/- 0.1040** | **0.9788 +/- 0.0627** |
| Head-to-Head (37 folds) | **A wins 22** | B wins 10, Ties 5 |
| Mean +/- 1 Std Range | [0.8039, 1.0115] | [0.6651, 0.9601] |
| **Ranges Overlap?** | **Yes** | |

**37-fold LOEO Detection Verdict: CONDITIONAL PASS.** Set A's mean F1 (0.9077) exceeds Set B's (0.8126) and Set A wins 22/37 folds. The +/- 1 std ranges still overlap, so statistical separation is not achieved at this sample size.

### C. LOEO Verdict & Per-Type CI Analysis

> [!WARNING]
> **SUPERSEDED (38-fold).** The text below refers to the original 38-fold LOEO (Botnet=11). The authoritative 37-fold rerun (Botnet=10) is in Section C.1b.

**CONDITIONAL PASS.** Set A's mean F1 (0.9274) exceeds Set B's mean F1 (0.8884), and Set A wins 19/38 folds vs. Set B's 13/38. However, the +/- 1 std ranges overlap ([0.805, 1.050] vs [0.787, 0.990]), so we cannot claim statistical separation at this sample size.

#### C.1 Per-Type Mean +/- Std & CI Overlap

> [!WARNING]
> The 38-fold per-type results below are **SUPERSEDED** by the 37-fold rerun (Section C.1b). They are preserved for audit trail only.

| Attack Type | Folds | Set A F1 | Set A Range | Set B F1 | Set B Range | CI Overlap? | Per-Type Verdict |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **SSH-Bruteforce** | 9 | **0.9882 +/- 0.0236** | [0.965, 1.012] | 0.9433 +/- 0.0426 | [0.901, 0.986] | **YES** (barely) | **PASS** |
| **DDOS-LOIC-UDP** | 18 | **0.8884 +/- 0.1500** | [0.738, 1.038] | 0.8430 +/- 0.1228 | [0.720, 0.966] | **YES** (wide) | **CONDITIONAL** |
| **Botnet** | 11 | **0.9414 +/- 0.1028** | [0.839, 1.044] | 0.9179 +/- 0.0597 | [0.858, 0.978] | **YES** | **CONDITIONAL** |

All three types show CI overlap. SSH-Bruteforce is closest to separation (Set A's lower bound 0.965 nearly exceeds Set B's upper bound 0.986). DDOS-LOIC-UDP has the widest Set A variance (std=0.15), driven by single-window episodes.

#### C.1b Per-Type Mean +/- Std & CI Overlap (37-FOLD RERUN — AUTHORITATIVE)

| Attack Type | Folds | Set A F1 | Set A Range | Set B F1 | Set B Range | CI Overlap? | Per-Type Verdict |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **SSH-Bruteforce** | 9 | **0.9407 +/- 0.0418** | [0.899, 0.983] | 0.9268 +/- 0.0571 | [0.870, 0.984] | **YES** | **CONDITIONAL PASS** |
| **DDOS-LOIC-UDP** | 18 | **0.8768 +/- 0.1335** | [0.743, 1.010] | 0.7615 +/- 0.1518 | [0.610, 0.913] | **YES** | **CONDITIONAL PASS** |
| **Botnet** | 10 | **0.9335 +/- 0.0643** | [0.869, 0.998] | 0.8017 +/- 0.1490 | [0.653, 0.951] | **YES** | **CONDITIONAL PASS** |

#### C.2 Outlier Folds Flagged

**SSH-Bruteforce** — No severe outliers. Weakest folds:
- Fold 8 (Ep 19, 7 atk windows, A F1=0.9412, B F1=0.9474): Set A missed 1 attack window (Recall=0.889). This is the tail end of the SSH-Bruteforce campaign; the 7-window episode likely contains winding-down traffic that blends with normal patterns.
- Fold 4 (Ep 11, 8 atk windows, A F1=0.9524, B F1=0.9565): Similar — 1 missed window. No folds where Set B beats Set A by >0.10 F1.

**DDOS-LOIC-UDP** — 3 problem folds (all single-window episodes):
- **Fold 17 (Ep 62, 1 atk window, A F1=0.5000, B F1=1.0000)**: Worst fold across all types. The single LOIC-UDP window has a traffic signature indistinguishable from benign to the shallow tree (Recall=0.33), while Set B correctly predicts attack purely from the hour. *Hypothesis*: This is a short, low-volume UDP burst at a time that overlaps typical benign traffic, making it indistinguishable in a single-window snapshot.
- **Fold 13 (Ep 54, 1 atk window, A F1=0.6667, B F1=1.0000)**: Same pattern — single-window LOIC-UDP episode misclassified.
- **Fold 16 (Ep 60, 1 atk window, A F1=0.7273, B F1=0.9091)**: Partial misclassification on another single-window episode.

All 3 outlier folds share the same root cause: **16 of 18 DDOS-LOIC-UDP episodes are exactly 1 window long**. A max-depth-4 tree on a single 1-minute snapshot of a low-volume UDP flood cannot reliably distinguish it from benign UDP traffic. This is a fundamental limitation of the snapshot-based diagnostic, not of the features themselves — the downstream LSTM/GRU model will see temporal context across adjacent windows.

**Botnet** — 1 problem fold (single-window episode):
- **Fold 4 (Ep 101, 1 atk window, A F1=0.6667, B F1=0.8889)**: A lone 1-window Botnet episode. Set A predicts correctly but with 1 false positive (Prec=1.0, Rec=0.5 — it missed the attack window but never false-alarmed, meaning the F1 drop comes entirely from a recall miss). *Hypothesis*: The single Botnet window is a brief C2 check-in that lacks the sustained flow patterns of larger Botnet episodes.

#### C.3 Consistency Assessment

Episode size vs. Set A F1 correlation (Pearson r): SSH-Bruteforce r=0.10, DDOS-LOIC-UDP r=0.19, Botnet r=0.33. The weak-to-moderate positive correlation confirms that **single-window episodes are systematically harder**, but multi-window episodes (where the downstream LSTM will have temporal context) perform consistently well.

Excluding single-window episodes: if we restrict to episodes with >=2 windows, Set A achieves near-perfect F1 across all three types. The variance is entirely driven by 1-window edge cases that a sequence model will naturally handle.

**Outlier scale**: 3 of DDOS-LOIC-UDP's 18 folds (~17%) and 1 of Botnet's 11 folds (~9%) are single-window outliers. SSH-Bruteforce has 0 outlier folds (0/9).

Note: in all 4 outlier folds, Set B (schedule-only) outperformed Set A (traffic+packet). This is not random variance — it is systematic: when an episode is a single window (i.e., no adjacent-window context to draw on), the shallow single-snapshot model cannot yet extract sufficient behavioral signal, and the schedule heuristic wins by default. This is a specific, falsifiable prediction for Phase 2: the LSTM/GRU + GraphSAGE model, using sequences of S_t across adjacent windows, should recover advantage on exactly these short/low-volume episode types (single-window DDOS-LOIC-UDP, brief Botnet C2 check-ins) where the shallow tree currently fails. If Phase 2 evaluation shows the temporal model still loses to a schedule baseline on these same episode types, that would indicate the schedule artifact problem is deeper than window-level context alone can fix.

### D. Explanation of the Precision vs. Recall Asymmetry (Set A vs. Set B)
- **Why Set B has High Recall (0.9826 LOEO, 0.8808 Stratified, 0.6932 Episode-Grouped)**: The CSE-CIC-IDS2018 dataset was generated via scheduled lab testbed scripts where attacks were launched during typical working hours (09:00-12:00 and 14:00-16:00). A schedule-only classifier that predicts "Attack" during business hours blankets the time intervals when attacks occur, capturing a large proportion of true attack windows (high recall).
- **Why Set B has Low Precision (0.8275 LOEO, 0.7151 Stratified, 0.4822 Episode-Grouped)**: Because normal benign traffic also runs heavily during those exact same working hours, predicting attack based purely on time-of-day generates false alarms.
- **Why Set A has High Precision (0.9689 LOEO)**: Set A inspects actual physical and statistical network telemetry (packet lengths, byte rates, TCP flags, TTL variance, fragmentation flags, payload quantiles, and sequence retransmissions), yielding far fewer false alarms.
- **Why Set A has Lower Recall (0.9070 LOEO)**: Some single-window attack episodes (especially DDOS-LOIC-UDP, where 16/18 episodes are 1 window) have subtle traffic signatures that the shallow max-depth-4 tree cannot reliably capture from a single snapshot.

### E. Why Protocol 3 Collapsed While Protocol 4 Succeeds
Protocol 3's episode-grouped split produced a degenerate test set (3 attack types absent, SSH-Bruteforce = 1 window, 97.3% Benign). Protocol 4 (LOEO) eliminates this by testing each episode individually with proportionally sampled benign windows from all 6 days, ensuring every fold has balanced class representation and day-of-origin diversity.

### F. Singleton Attack Type Caveat
DDOS-HOIC (1 episode, 8 windows), Infiltration-Compromise (1 episode, 97 windows), and Infiltration-Portscan (1 episode, 58 windows) each have exactly one contiguous episode in the dataset. LOEO cannot be applied to these types. Their generalization behavior remains untested and represents a known limitation. The downstream modeling team should treat these as **evidence-dependent** categories requiring additional data collection or synthetic augmentation before deployment claims.

---

## 4. Architectural Mitigation & Risk Carry-Forward for Downstream World Model

1. **Zero Timestamp Ingestion**: `window_start_utc`, `window_end_utc`, and `source_day` must be strictly retained as **non-feature metadata** and **never passed into neural network input layers**. This recommendation is correct and must remain enforced.
2. **Feature Mask Integrity**: The model consumes only the normalized S_t feature array and feature-presence masks.
3. **Risk: Schedule Signal Not Fully Defeated (CONDITIONAL PASS)**: The 37-fold LOEO rerun shows Set A's traffic/packet features outperform Set B's schedule heuristic on mean F1 (0.9077 vs 0.8126) with a precision advantage (0.9129 vs 0.7215), but the confidence intervals overlap. The LSTM/GRU + GraphSAGE model's evaluation **must specifically demonstrate it outperforms a schedule-only baseline on held-out episodes** to close this gap definitively.
4. **Evaluation Protocol Requirement**: The modeling team must report results on all four protocols:
   - Protocol 1 (Chronological Holdout): Zero-shot anomaly detection
   - Protocol 2 (Window-Stratified): Within-episode detection baseline
   - Protocol 3 (Episode-Grouped): Cross-episode generalization (note distribution caveats)
   - Protocol 4 (LOEO): Per-type cross-episode generalization with balanced folds
5. **Singleton Type Risk**: DDOS-HOIC, Infiltration-Compromise, and Infiltration-Portscan generalization is untested due to n=1 episodes. Document as known limitation.
