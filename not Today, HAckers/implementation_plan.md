# SIH26153 — Frontend Implementation Plan
## Derived from PPT Word-by-Word Analysis
**Team:** Not Today, Hackers · **PS:** SIH26153 · **Date:** 2026-09-06

---

## PPT Feature Audit — What Did We Promise?

I read every word on all 6 slides. Here is **every feature and claim** from the PPT, mapped to whether the frontend needs to display it.

### Slide 2 — Proposed Solution (The Core Promise)

| PPT Claim | Frontend Must Show | How |
|-----------|-------------------|-----|
| **Observed Network Behaviour:** S(t-2) → S(t-1) → S(t) | ✅ Current state S(t) display | State component |
| **Learned World Model** — learns how network states evolve | ✅ Label the model source | Text label on predictions |
| **Future State Forecast (Simulate):** S(t+1) → S(t+2) → S(t+k) | ✅ **Hero trajectory chart** | Plotly forecast chart |
| **Future Attack Risk:** Probability, Uncertainty, Stage | ✅ All three must be visible | 3 separate UI elements |
| **Probability** — likelihood of attack | ✅ Per-step probability on trajectory | Chart + card |
| **Uncertainty** — confidence in prediction | ✅ **Confidence/uncertainty display** | New component needed |
| **Stage** — expected attack stage | ✅ Predicted ATT&CK stage | Stage card + stepper |
| **From detection → future-state forecasting** | ✅ Visual comparison | Detection vs Forecasting section |
| **Early Forecasting** — predicts risk before attack occurs | ✅ Lead time display | Predicted stage card |
| **Future-State Prediction** — forecasts how behaviour will evolve | ✅ Multi-step trajectory | Forecast page |
| **Attack Progression** — predicts next attack stage (K-steps) | ✅ K-step rollout visualization | Trajectory + stepper |
| **Uncertainty-Aware** — reports probability AND confidence | ✅ Both probability AND confidence | Two separate values |
| **Behaviour-Based** — uses traffic behaviour, not signatures | ✅ Mention in system description | Header/about text |
| **Leakage-tested at every layer** | ✅ Mention in validation page | Validation section |
| **One model, two signals** — forecasts attacks + flags unseen behaviour | ✅ **Novelty/anomaly flagging** | Novelty score component |
| **Counterfactual defence simulation** — trajectory changes when defender acts | ✅ **Defence simulation section** | New component needed |

### Slide 3 — Technical Approach (Architecture Diagram)

| PPT Element | Frontend Must Show | How |
|-------------|-------------------|-----|
| Input: PCAP (LIVE/DEMO) or CSV (CICFlowMeter) | ✅ Input source label | Status indicator |
| Feature Extraction: Flow + Packet (Scapy) | ❌ Backend only | — |
| Unified Cyber State S(t): flow + packet features + masks | ✅ State ID + feature summary | State component |
| LSTM Encoder Z(t), L=30 | ❌ Backend only | — |
| GraphSAGE Encoder g(t), 2-layer, fused with z(t) | ❌ Backend only | — |
| Z'(t) = [Z(t) : g(t)] | ❌ Backend only | — |
| Dynamics Head — Next-State Prediction + K-Step Rollout | ✅ Label as data source | "Source: dynamics head" |
| **Hazard + Stage → Risk Score + ATT&CK Stage + Uncertainty Bands** | ✅ All three outputs | 3 UI elements |
| **Explainable Dashboard** (Streamlit-offline) | ✅ This IS the dashboard | Entire app |
| **Probability Trajectory** | ✅ Probability over K steps | Plotly line chart |
| **Stage Trajectory** | ✅ Stage progression over K steps | Stepper + chart |
| **Novelty Score** | ✅ Anomaly/deviation metric | Novelty component |
| **Flagged Flows** | ✅ Suspicious flow evidence | Evidence table |
| Demo: Streamlit (offline, single process) | ✅ Architecture constraint | No REST API |
| Explainability: Feature-group ablation + deletion validation | ✅ Attribution chart | "Why this forecast?" |

### Slide 4 — Feasibility & Viability

| PPT Element | Frontend Must Show | How |
|-------------|-------------------|-----|
| Open Data & Open-Source | ✅ Dataset source label | Sidebar/footer |
| Offline Deployment — air-gapped | ✅ "OFFLINE ANALYSIS" badge | Header status |
| Efficient Inference — lightweight runtime | ❌ Backend concern | — |
| Dataset Scheduling Artifacts (viability risk) | ✅ Mention in validation | Validation page |
| Rare Multi-Stage Attack Examples (viability risk) | ✅ Mention in validation | Validation page |
| Autoregressive Error Growth (viability risk) | ✅ **Show uncertainty growing with K** | Uncertainty bands on chart |
| Schedule Control Testing (strategy) | ✅ Validation section | Validation page |
| Case-Study Validation (strategy) | ✅ Validation section | Validation page |
| Validated Forecast Horizons (strategy) | ✅ **Show which horizons are reliable** | Validation page |

### Slide 5 — Impact & Benefits

| PPT Claim | Frontend Must Show | How |
|-----------|-------------------|-----|
| Forecast attacks before major compromise | ✅ Lead time display | Predicted stage card |
| Estimate potential operational impact | ✅ Impact description | Attack info section |
| Prioritize threats using predicted risk | ✅ Risk score ranking | Trajectory chart |
| Identify features driving each prediction | ✅ **Feature attribution** | "Why this forecast?" chart |
| Track changes in network behaviour over time | ✅ State comparison | State component |
| Observe predicted future network states | ✅ K-step forecast states | Forecast page |
| Anticipate threats to essential services | ✅ System description text | About/header |
| Compare risk across future attack trajectories | ✅ **Multi-trajectory comparison** | Forecast page |
| Support risk-based security decisions | ✅ Actionable output format | Predicted stage card |

---

## Features Missing From Previous Plan

Cross-referencing the PPT against my earlier plan, **3 features were missing:**

> [!IMPORTANT]
> ### 1. Uncertainty / Confidence Display
> The PPT explicitly promises **"Uncertainty — Confidence in prediction"** as a core output alongside Probability and Stage. The previous plan only showed probability. The frontend **must** show a confidence/uncertainty value for each forecast step.

> [!IMPORTANT]
> ### 2. Counterfactual Defence Simulation
> Slide 2 USP: *"Counterfactual defence simulation tests how the future attack trajectory changes when the defender acts."* The frontend needs a section showing how the trajectory **could change** if a defensive action is taken. This can be mock data with a "What if?" comparison.

> [!IMPORTANT]
> ### 3. Autoregressive Error Growth Visualization
> Slide 4: *"Prediction uncertainty compounds over longer rollouts."* The frontend should visually show that uncertainty **grows** at higher K steps. This aligns with the uncertainty bands mentioned in the architecture output.

---

## Updated Mock Data Contract

```python
DEMO_DATA = {
    "analysis": {
        "status": "Analysis Ready",
        "source": "CSE-CIC-IDS2018",
        "window": "10:30:00 – 10:31:00",
        "flows_analyzed": 12480,
        "packets_analyzed": 84216,
    },

    "current_state": {
        "state_id": "S(t)",
        "dominant_behavior": "Suspicious connection probing",
        "novelty_score": 0.23,          # "One model, two signals" — anomaly flagging
    },

    "forecast": [
        {"horizon": "t+1", "time_ahead": "1 min", "stage": "Reconnaissance",
         "probability": 0.21, "uncertainty": 0.06},       # ← NEW: uncertainty per step
        {"horizon": "t+2", "time_ahead": "2 min", "stage": "Initial Access",
         "probability": 0.38, "uncertainty": 0.11},
        {"horizon": "t+3", "time_ahead": "3 min", "stage": "Lateral Movement",
         "probability": 0.67, "uncertainty": 0.18},       # uncertainty grows with K
        {"horizon": "t+4", "time_ahead": "4 min", "stage": "Impact",
         "probability": 0.81, "uncertainty": 0.27},       # highest uncertainty at K=4
    ],

    "explanation": [
        {"feature": "SYN rate", "contribution": 0.34},
        {"feature": "Connection rate", "contribution": 0.28},
        {"feature": "TTL variance", "contribution": 0.21},
        {"feature": "Packet timing", "contribution": 0.12},
    ],

    "attack": {
        "type": "SSH-Bruteforce",
        "predicted_stage": "Lateral Movement",
        "confidence": 0.67,
        "lead_time": "~3–4 min",
    },

    "mitre": [
        "Reconnaissance",
        "Initial Access",
        "Lateral Movement",
        "Impact",
    ],

    "flagged_flows": [
        {"source": "10.0.2.15", "destination": "10.0.4.21",
         "protocol": "TCP", "reason": "SYN burst"},
        {"source": "10.0.2.18", "destination": "10.0.4.21",
         "protocol": "TCP", "reason": "Repeated connection attempts"},
        {"source": "10.0.2.31", "destination": "10.0.4.10",
         "protocol": "SSH", "reason": "Repeated authentication attempts"},
    ],

    "baseline": {
        "world_model": 0.67,
        "logistic_regression": 0.38,
        "lagged_logistic_regression": 0.41,
    },

    # ── NEW: Counterfactual Defence Simulation (Slide 2 USP) ──
    "counterfactual": {
        "action": "Block source 10.0.2.15 at firewall",
        "original_trajectory": [0.21, 0.38, 0.67, 0.81],   # without defence
        "mitigated_trajectory": [0.21, 0.25, 0.30, 0.34],   # with defence
        "risk_reduction": "58%",
    },
}
```

**Changes from previous plan:**
- Added `uncertainty` field to each forecast step (growing with K)
- Added `counterfactual` block for defence simulation
- `novelty_score` was already present (covers "one model, two signals")

---

## File Structure (Unchanged)

```
not-today-hackers/
│
├── app.py                      ← Entry point / Overview page
├── mock_data.py                ← ALL demo values (single source)
├── styles.py                   ← Theme CSS + sidebar helper
│
├── components/
│   ├── __init__.py
│   ├── header.py               ← Banner + offline status
│   ├── state.py                ← Current network state S(t)
│   ├── forecast.py             ← Hero trajectory + predicted stage
│   ├── explanation.py          ← Attribution + novelty + ATT&CK stepper
│   └── evidence.py             ← Flagged flows table
│
├── pages/
│   ├── 01_Forecast.py          ← Deep forecast + defence simulation
│   ├── 02_Evidence.py          ← Full evidence + baseline
│   └── 03_Validation.py        ← Validation placeholders
│
├── .streamlit/
│   └── config.toml             ← Dark theme
│
└── requirements.txt
```

---

## Page Layouts (Updated with PPT Features)

### `app.py` — Overview (Main Page)

| # | Section | PPT Feature Covered | Size |
|---|---------|-------------------|------|
| 1 | **Header:** NOT TODAY, HACKERS + OFFLINE ANALYSIS · READY | Slide 1 branding, Slide 4 offline | — |
| 2 | **Current Network State — S(t):** flows, packets, window, behaviour, deviation | Slide 2: Observed Network Behaviour, S(t) | compact |
| 3 | **Future Attack Trajectory** (hero Plotly chart with **uncertainty bands**) | Slide 2: Future State Forecast + Probability + Uncertainty + Stage | **largest** |
| 4 | **Predicted Future Stage:** stage name + probability + confidence + lead time | Slide 2: Stage + Probability + Uncertainty | medium |
| 5 | **Why This Forecast?:** feature attribution bars | Slide 3: Explainability, Slide 5: "Identify features driving each prediction" | medium |
| 6 | **State Deviation:** novelty score | Slide 2: "One model, two signals — flags unseen behaviour", Slide 3: Novelty Score | compact |
| 7 | **Predicted ATT&CK Tactic Trajectory:** stepper | Slide 3: ATT&CK Stage | compact |
| 8 | **Flagged Flow Evidence:** top 3 flows | Slide 3: Flagged Flows | supporting |

### `01_Forecast.py` — Multi-Step Forecast

| # | Section | PPT Feature Covered |
|---|---------|-------------------|
| 1 | **Forecast trajectory with uncertainty bands** | Slide 2: K-step rollout, Slide 4: autoregressive error growth |
| 2 | **Horizon selector:** [t+1] [t+2] [t+3] [t+4] | Slide 2: S(t+1)…S(t+k), Slide 4: validated forecast horizons |
| 3 | **Selected step detail:** stage, probability, uncertainty, time | Slide 2: Probability + Uncertainty + Stage |
| 4 | **Probability trajectory:** bar chart per step | Slide 3: Probability Trajectory |
| 5 | **Stage trajectory:** sequential stage display | Slide 3: Stage Trajectory |
| 6 | **Counterfactual Defence Simulation** | Slide 2 USP: "tests how trajectory changes when defender acts" |
| 7 | **Baseline comparison:** World Model vs LR vs Lagged LR | Slide 3: scikit-learn baselines |

### `02_Evidence.py` — Forecast Evidence

| # | Section | PPT Feature Covered |
|---|---------|-------------------|
| 1 | **Why This Forecast?** — full attribution chart | Slide 3: Feature-group ablation |
| 2 | **ATT&CK Tactic Trajectory** — full stepper | Slide 3: ATT&CK Stage |
| 3 | **Flagged Flows** — complete table | Slide 3: Flagged Flows |
| 4 | **State Deviation** — novelty detail | Slide 2: "flags unseen behaviour" |
| 5 | **Baseline Comparison** — model vs baselines | Slide 3: scikit-learn baselines |
| 6 | **Detection vs Forecasting** — conceptual comparison | Slide 2: "From detection to future-state forecasting" |

### `03_Validation.py` — Model Validation (Placeholders)

| # | Section | PPT Feature Covered | Status |
|---|---------|-------------------|--------|
| 1 | Classification metrics (P, R, F1, PR-AUC, FPR) | Slide 3: scikit-learn metrics | Placeholder |
| 2 | LOEO validation | Slide 3: Custom LOEO | Placeholder |
| 3 | Chronological evaluation | Slide 3: chronological split | Placeholder |
| 4 | **Forecast horizon reliability** | Slide 4: "Validated Forecast Horizons" | Placeholder |
| 5 | **Schedule control testing** | Slide 4: "Schedule Control Testing" | Placeholder |
| 6 | **Case-study validation** | Slide 4: "Case-Study Validation" | Placeholder |
| 7 | **Rollout error vs K** | Slide 4: "Autoregressive Error Growth" | Placeholder |
| 8 | **Leakage testing report** | Slide 2: "Leakage-tested at every layer" | Placeholder |
| 9 | Per-episode lead time | Slide 2: "Early Forecasting" | Placeholder |

---

## Hero Chart Update — Uncertainty Bands

The forecast trajectory chart now includes **uncertainty bands** (the PPT promises this):

```
                    CURRENT          FORECAST →
Probability 1.0 ┤
                │                              ╱███ ← uncertainty band
            0.8 ┤                           ●─╱
                │                          ╱
            0.6 ┤                     ●───╱
                │                    ╱ ██ ← band grows with K
            0.4 ┤               ●──╱
                │              ╱█
            0.2 ┤         ●───╱
                │   ◆    ╱
            0.0 ┤───┼───┼─────────────────────
                   Now  t+1   t+2   t+3   t+4
```

- **Bands = probability ± uncertainty** at each step
- Bands **grow wider** at higher K (autoregressive error growth from Slide 4)
- Implemented as Plotly `fill='tonexty'` between upper and lower bounds

---

## Counterfactual Defence Simulation Section

New section on the **Forecast page** (maps to Slide 2 USP):

```
┌─────────────────────────────────────────────────────────────┐
│  COUNTERFACTUAL DEFENCE SIMULATION                          │
│                                                             │
│  What if: Block source 10.0.2.15 at firewall                │
│                                                             │
│  ───── Original trajectory (no action)                       │
│  - - - Mitigated trajectory (with defence)                   │
│                                                             │
│    1.0 ┤                                                     │
│        │                           ●───── 81%                │
│    0.8 ┤                     ●────╱                          │
│        │                    ╱                                │
│    0.6 ┤               ●──╱                                  │
│        │              ╱                                      │
│    0.4 ┤         ●───╱                                       │
│        │        ╱         ○- - - -○ 34%                      │
│    0.2 ┤   ●───╱    ○- - -○                                  │
│        │  ╱   ○- - -○                                        │
│    0.0 ┤─────────────────────────                            │
│          t+1   t+2   t+3   t+4                               │
│                                                             │
│  Estimated risk reduction: 58%                               │
│                                                             │
│  ⚠ Demo values only — real simulation requires backend       │
└─────────────────────────────────────────────────────────────┘
```

Two lines on the same Plotly chart:
- **Solid line:** original trajectory (without defence)
- **Dashed line:** mitigated trajectory (with defence)
- Shows the **risk reduction percentage**
- Clear disclaimer: demo data only

---

## Design System (Unchanged)

| Token | Hex | Usage |
|-------|-----|-------|
| Background | `#0D1117` | Page bg |
| Surface | `#161B22` | Cards, chart bg |
| Border | `#21262D` | Subtle borders |
| Text primary | `#C9D1D9` | Body text |
| Text secondary | `#8B949E` | Labels |
| Accent | `#3FB9B2` | Teal — trajectory, active elements |
| Safe | `#7EE787` | < 25% probability |
| Caution | `#D29922` | 25–50% |
| Warning | `#F85149` | 50–75% |
| Critical | `#FF7B72` | > 75% |
| Uncertainty band | `rgba(63,185,178,0.15)` | Shaded area on chart |

---

## Backend Integration Point (Unchanged)

```python
# Before (mock):
from mock_data import DEMO_DATA
data = DEMO_DATA

# After (real):
data = predict(input_data)
```

The frontend consumes `data["forecast"]`, `data["counterfactual"]`, etc. It does NOT care about LSTM/GNN/PyTorch internals.

---

## Build Phases

### Phase 1 — Foundation (5 files)
`.streamlit/config.toml`, `requirements.txt`, `mock_data.py`, `styles.py`, `components/__init__.py`

### Phase 2 — Components (5 files)
`header.py`, `state.py`, `forecast.py` (now with uncertainty bands), `explanation.py`, `evidence.py`

### Phase 3 — Pages (4 files)
`app.py` (overview), `01_Forecast.py` (now with defence simulation), `02_Evidence.py`, `03_Validation.py` (now with 9 validation sections)

### Phase 4 — Verify
`pip install -r requirements.txt` → `streamlit run app.py` → verify all pages → verify offline

---

## Complete PPT ↔ Frontend Traceability

Every feature from all 6 slides is now accounted for:

| PPT Feature | Frontend Location | Status |
|-------------|------------------|--------|
| S(t) observed state | Overview → State component | ✅ |
| S(t+1)…S(t+k) forecast | Overview → Hero chart | ✅ |
| Probability per step | Hero chart + stage card | ✅ |
| **Uncertainty per step** | Hero chart bands + stage card | ✅ NEW |
| Stage per step | Stepper + chart labels | ✅ |
| Feature attribution | "Why this forecast?" | ✅ |
| Novelty score / anomaly flagging | State deviation component | ✅ |
| ATT&CK tactic trajectory | Stepper visualization | ✅ |
| Flagged flows | Evidence table | ✅ |
| Baseline comparison | Evidence page | ✅ |
| **Counterfactual defence simulation** | Forecast page | ✅ NEW |
| **Autoregressive error growth** | Uncertainty bands growing with K | ✅ NEW |
| Leakage testing | Validation page placeholder | ✅ |
| Schedule control testing | Validation page placeholder | ✅ |
| Case-study validation | Validation page placeholder | ✅ |
| Validated forecast horizons | Validation page placeholder | ✅ |
| LOEO + chronological split | Validation page placeholder | ✅ |
| Offline / air-gapped deployment | Status badge + no network calls | ✅ |
| Detection vs Forecasting comparison | Evidence page section | ✅ |
| Open data source labels | Sidebar + footer | ✅ |
| Lead time display | Predicted stage card | ✅ |
