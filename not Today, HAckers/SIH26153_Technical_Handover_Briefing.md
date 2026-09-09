# SIH26153 — Technical Handover Briefing

> **⚠️ SUPERSEDED (2026-09-01):** The architecture in this document (direct
> `X(t-k..t) → P(attack)` forecaster, enrichment-only ATT&CK role, UGR'16-primary
> dataset strategy, full product layer) has been revised after two independent
> reviews. The authoritative plan is now
> **`SIH26153_Final_Gap_Analysis_and_Plan.md`**. This document remains valid for:
> leakage rules (§8.7), splitting/purge/embargo (§9.8), UCS concept (§9.5),
> feature-engineering principles (§9.6), and the positioning-honesty table (§15).

## AI-Based Network Attack Forecasting from Network Traffic Data

**Organization:** NTRO (National Technical Research Organisation)
**Theme:** Blockchain & Cybersecurity
**Category:** Software

---

## 1. Project Overview

We are building an AI-based system that forecasts the likelihood of a future network attack, rather than only detecting attacks that are already in progress. The system observes historical and current network/cybersecurity activity and estimates the probability that an attack-related state will emerge within a defined future window.

**Core question the system answers:**

> Given everything observable up to time *t*, what is the probability that an attack will occur during the future horizon [t, t+H]?

The system is not designed to claim it can detect every attack or predict the exact moment of occurrence. It estimates attack **likelihood** over a **forecast horizon**, and backs that estimate with behavioural evidence, threat-intelligence context, and an explainable risk score.

---

## 2. Problem Statement Interpretation

Traditional Intrusion Detection Systems (IDS) answer a narrow question:

```
Current Traffic
      ↓
Benign / Attack
```

Our system is built around a fundamentally different question:

```
Historical Cyber Behaviour
          ↓
    Current Cyber State
          ↓
    Behaviour Evolution
          ↓
  Future Attack Probability
```

Formally, given observations up to time *t*:

```
X(t-k), ..., X(t-1), X(t)  →  P(attack occurs in [t, t+H])
```

This differs from conventional detection, expressed as `X(t) → Y(t)`, because the model must use only past information to predict a **future** event — a strict constraint that shapes almost every downstream design decision (see Leakage Control, Section 8.7).

---

## 3. Core Insight

A traffic record can look completely legitimate in isolation. A *sequence* of individually legitimate actions, however, can indicate an emerging attack when viewed together:

```
Normal authentication
        ↓
Normal service access
        ↓
New host communication
        ↓
Increasing connection activity
        ↓
Unusual peer relationship
        ↓
Traffic behaviour changes
        ↓
Potential attack
```

**Guiding principle:** *A legitimate action performed in an abnormal sequence or relationship can indicate an emerging attack.*

This is why the architecture combines temporal modelling (what is changing) with relational/graph modelling (how entities are interacting), rather than relying on either alone.

---

## 4. Background Research

Before settling on an architecture, we reviewed the major existing approaches:

| Approach | Strengths | Limitations |
|---|---|---|
| **Signature-based detection** | Fast, interpretable, effective on known attacks | Poor against unknown attacks; requires constant signature updates; no forecasting |
| **Anomaly detection** | Detects previously unseen behaviour | High false-positive rate; anomaly ≠ attack; no inherent future forecasting |
| **Supervised intrusion classification** | Strong baseline performance on labelled data | Classifies current state only; doesn't demonstrate forecasting |
| **Time-series modelling (LSTM/GRU/Transformer)** | Learns how behaviour evolves over time | Requires well-structured temporal data |
| **Graph Neural Networks (GNN)** | Captures relationships between entities | Requires meaningful graph structure; not all datasets provide it naturally |

Our architecture combines the last two (temporal + graph) rather than relying on classification alone.

---

## 5. Data Sources

### 5.1 Network / Intrusion Datasets Referenced by the Problem Statement
- **CIC-IDS2017** — benign + multiple attack types; good for classification/benchmarking, but limited temporal coverage for long-horizon forecasting.
- **CIC-IDS2018** — larger, multi-day dataset; stronger candidate for temporal modelling and a practical backup dataset.
- **UNSW-NB15** — flow-level features, useful for benchmarking; temporal structure needs care before use in forecasting.
- **CTU-13** — botnet + background traffic; useful for graph/relational modelling.
- **CICIoT2023** — IoT-focused; useful for device-interaction and IoT-specific forecasting experiments.
- **LANL Authentication Dataset** — user/host authentication events rather than flow data; strong candidate for graph construction and behavioural sequence modelling.
- **DARPA Intrusion Detection datasets** — historical benchmarks; useful for research grounding but dated relative to modern network environments.

### 5.2 Cybersecurity Knowledge Bases
- **MITRE ATT&CK** — tactics, techniques, and technique relationships.
- **CAPEC** — attack pattern context.
- **CVE / NVD** — vulnerability identifiers, descriptions, severity, and affected-product mappings where reliable.

**Design decision:** Network datasets drive *behaviour/forecasting*. Knowledge bases drive *context/enrichment* after a forecast is made. We do not treat threat-intelligence data as if it can be appended row-by-row to traffic data — that would be an architecturally incorrect assumption.

---

## 6. Dataset Research & Audit

We audited every SIH-listed dataset against forecasting-specific criteria — not just detection-suitability criteria:

- Temporal coverage and timestamp availability/ordering
- Data granularity and attack timeline structure
- Attack onset clarity and label structure
- Missing values, duplicates, feature availability
- Flow/event semantics and recorded-activity duration
- Ability to construct a genuine *future* attack-window target

**Key finding:**

> A dataset being suitable for intrusion detection does not automatically make it suitable for attack forecasting.

A dataset can have excellent attack labels but still be unsuitable for forecasting if it has short time coverage, artificially separated attack sessions, or insufficient historical context before attack onset.

**Resulting dataset strategy:**

```
SIH-recommended datasets
          ↓
    Dataset audit
          ↓
Temporal suitability analysis
          ↓
  Select appropriate dataset(s)
          ↓
      Convert into UCS
          ↓
       Forecasting
```

We do **not** blindly concatenate every SIH-listed dataset into one model. The pipeline is built to support heterogeneous datasets architecturally, but any given forecasting experiment uses only datasets that genuinely support the forecasting formulation.

**Primary candidates identified:**
- **UGR'16** — investigated specifically for its long chronological network-flow coverage, which is valuable for studying behavioural evolution over time. Treated as a strong research candidate, not an automatic final choice.
- **CSE-CIC-IDS2018** — practical, tractable backup with multiple days of activity.

---

## 7. Forecasting Target Design

The attack label is redefined around a future window rather than the current instant:

```
Y(t) = 1   if attack activity occurs during [t, t+H]
Y(t) = 0   otherwise
```

Possible extensions once the baseline is validated:
- Binary attack probability → multi-class attack category
- Attack onset window prediction
- Multiple simultaneous forecast horizons

---

## 8. System Architecture

### 8.1 End-to-End Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                 │
├──────────────────────────────────────────────────────────────────────┤
│ CIC-IDS2017 │ CIC-IDS2018 │ UNSW-NB15 │ CTU-13 │ CICIoT2023         │
│ LANL        │ DARPA       │ Other Open Cybersecurity Resources      │
└──────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  DATA INGESTION & NORMALIZATION                      │
├──────────────────────────────────────────────────────────────────────┤
│ Schema Validation │ Cleaning │ Deduplication │ Timestamp Normalizing │
│ Canonical Mapping │ Missing Value Handling │ Protocol Normalization  │
└──────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    UNIFIED CYBER STATE (UCS)                         │
├──────────────────────────────────────────────────────────────────────┤
│ Time Windows │ Canonical Features │ Behavioural Statistics           │
│ Traffic State │ Entity State │ Scope / Dataset Metadata              │
│ Leakage Validation │ Temporal Alignment                              │
└───────────────┬──────────────────────────────┬───────────────────────┘
                ▼                              ▼
┌─────────────────────────────┐    ┌──────────────────────────────────┐
│   TEMPORAL FEATURE STREAM   │    │      CYBER RELATION GRAPH        │
├─────────────────────────────┤    ├──────────────────────────────────┤
│ Historical UCS Sequences    │    │ Users / Hosts / IPs / Services   │
│ Behaviour Evolution         │    │ Communication Relationships      │
│ Traffic Evolution           │    │ Authentication Relationships     │
└──────────────┬──────────────┘    └────────────────┬─────────────────┘
               ▼                                    ▼
┌─────────────────────────────┐    ┌──────────────────────────────────┐
│  LSTM / GRU / Transformer   │    │          GNN ENGINE               │
├─────────────────────────────┤    ├──────────────────────────────────┤
│ Temporal Representation     │    │ Graph Representation             │
│ Behaviour Trends            │    │ Relational Behaviour             │
│ Sequential Dependencies     │    │ Node/Edge Interactions           │
└──────────────┬──────────────┘    └────────────────┬─────────────────┘
               └────────────────┬───────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    MULTI-MODAL FUSION LAYER                          │
├──────────────────────────────────────────────────────────────────────┤
│ Temporal Embedding + Graph Embedding + Cyber-State Context           │
│ Attention / Feature Fusion / Representation Combination              │
└──────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     FORECASTING ENGINE                               │
├──────────────────────────────────────────────────────────────────────┤
│ Attack Probability │ Forecast Horizon │ Attack Onset Probability     │
│ Multi-class / Threat Category Prediction                             │
└──────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  THREAT INTELLIGENCE ENRICHMENT                      │
├──────────────────────────────────────────────────────────────────────┤
│ MITRE ATT&CK │ CAPEC │ CVE/NVD │ Attack Technique Mapping            │
│ Vulnerability Context │ Threat Knowledge                             │
└──────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    RISK & DECISION ENGINE                            │
├──────────────────────────────────────────────────────────────────────┤
│ Forecast Probability │ Threat Severity │ Asset Criticality           │
│ Confidence │ Historical Behaviour │ Threat Context                  │
└──────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 EXPLAINABILITY & INVESTIGATION                       │
├──────────────────────────────────────────────────────────────────────┤
│ SHAP / Feature Importance │ Temporal Evidence                        │
│ Graph Evidence │ Prediction Explanation │ Attack Context             │
└──────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   ALERT & RESPONSE LAYER                             │
├──────────────────────────────────────────────────────────────────────┤
│ Risk Alerts │ Early Warning │ Prioritization │ Investigation Context │
└──────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 SECURITY OPERATIONS DASHBOARD                        │
├──────────────────────────────────────────────────────────────────────┤
│ Risk Timeline │ Forecasts │ Affected Entities │ Attack Graph         │
│ Alerts │ Explanations │ Threat Context │ Historical Trends           │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.2 Application / Backend Architecture

```
                Frontend
                   │
                   ▼
             API Gateway
                   │
       ┌───────────┼────────────┐
       ▼           ▼            ▼
   Risk API    Forecast API   Alert API
       │           │            │
       └───────────┼────────────┘
                   ▼
             Inference Layer
                   │
       ┌───────────┴────────────┐
       ▼                        ▼
 Temporal Model              GNN Model
       │                        │
       └────────────┬───────────┘
                    ▼
               Fusion Layer
                    │
                    ▼
              Risk Engine
                    │
                    ▼
             Alert Generator
```

---

## 9. Component Breakdown

### 9.1 Data Ingestion
CSV/Parquet ingestion, schema validation, type validation, dataset identification, timestamp extraction, column mapping, duplicate detection, corrupt-record handling, dataset version tracking. Every dataset carries a source identifier for traceability through the pipeline.

### 9.2 Cleaning & Normalization
- **Timestamps:** raw → timezone normalization → canonical UTC.
- **Network identifiers:** source/destination IP, ports, protocol → consistent representation.
- **Missing values:** handled per feature semantics, not blanket zero-filling (a missing value must not be silently read as "no activity").

### 9.3 Canonical Feature Mapping
Dataset-specific column names (e.g., `Tot Fwd Pkts`, `Total Forward Packets`, `fwd_pkt_count`) are mapped to a single canonical vocabulary (e.g., `forward_packet_count`), giving the ML layer one stable feature contract regardless of source dataset. Features with no meaningful cross-dataset equivalent are either kept as optional extensions or excluded from the shared representation.

### 9.4 Temporal Windowing
Raw events/flows are aggregated into fixed time windows (1-minute windows explored as an initial baseline; window size remains an experimental parameter, not a fixed assumption). Each window aggregates flow count, packet/byte counts and rates, unique source/destination counts, protocol distribution, and behavioural statistics.

### 9.5 Unified Cyber State (UCS)
The UCS is the **central contract between data engineering and ML**. It standardizes heterogeneous datasets (different schemas, timestamp formats, granularities, labels) into one representation per time window, containing:

```
timestamp
dataset
scope_id
window_size_sec
traffic / connection / protocol / behavioural / peer-activity features
```

Metadata (timestamp, dataset, scope_id) is explicitly kept separate from model-input features so identifiers are never accidentally treated as predictive signals.

### 9.6 Feature Engineering
Grouped into four families:
- **Traffic:** flow/packet/byte counts and rates
- **Protocol:** TCP/UDP/ICMP activity, port behaviour, protocol distribution
- **Behavioural:** connection frequency, new peers, repeated communication, authentication behaviour, activity deviation
- **Statistical:** rolling statistics, historical baselines, z-score/deviation features, trends

**Important lesson:** statistical baseline features (e.g., z-scores) must not be fabricated when there is insufficient historical data. If a baseline can't be legitimately computed, the value is left missing rather than manufactured — a principle that applies broadly across the pipeline.

### 9.7 Leakage Control
For a prediction made at time *t*:
- **Allowed:** information timestamped ≤ t
- **Not allowed:** information from t+1, from the future attack window, or future-derived statistics

Without this discipline, a model can appear highly accurate while not actually forecasting anything.

### 9.8 Temporal Splitting, Purging & Embargo
We use chronological (not random) splits — earlier data for training, later for validation, latest for testing. Because overlapping windows/sequences can still leak information across a chronological boundary, we additionally apply:
- **Purging** — removing samples near split boundaries with overlapping information
- **Embargo** — leaving a temporal gap between train/validation/test

```
TRAIN ████████████████  ░PURGE/EMBARGO░  VALIDATION ████████  ░PURGE/EMBARGO░  TEST ████████████
```

### 9.9 Temporal Model (LSTM/GRU)
Input: `S(t-k) ... S(t)` → Output: `P(Attack in future horizon)`. Selected as the initial baseline for being well understood, lightweight, appropriate for sequential data, faster to train, and realistic for a hackathon timeline. A Transformer variant remains a possible later upgrade if data scale and compute justify it.

### 9.10 Graph Model (GNN)
Represents the environment as a graph of users, hosts, IPs, services, and devices, connected by edges such as *communicates with*, *authenticates to*, *accesses*, *connects to*. Where the temporal model answers "what is changing," the graph model answers "how are entities interacting" — important because attack progression can involve movement or unusual relationships across entities that look individually benign.

### 9.11 Fusion
Temporal embedding + graph embedding + current cyber-state context are combined in a fusion layer before the forecasting head. **We are explicitly not assuming fusion or the GNN automatically improves results** — this must be demonstrated empirically against the temporal-only baseline before being adopted as part of the final pipeline.

### 9.12 Forecasting Engine
Primary output: `P(attack within future horizon)`, e.g. `0.82`. Future extensions (not required for MVP): attack category, attack onset window, multi-horizon forecasting.

### 9.13 Threat Intelligence Enrichment
MITRE ATT&CK, CAPEC, and CVE/NVD are used to attach technique/vulnerability context to a forecast after it is generated — enrichment, not training labels.

### 9.14 Risk Engine
Converts a raw probability into an operational risk score:

```
Forecast Probability + Asset Criticality + Threat Context + Behavioural Evidence + Model Confidence → Risk Score
```

The scoring formula is to be calibrated experimentally, not assigned arbitrary fixed weights.

### 9.15 Explainability
The system should never present a bare number. Target output style:

> "Attack probability increased because of abnormal traffic growth, unusual peer relationships, and a recent behavioural deviation" — not just "Attack probability = 87%."

Candidate techniques: SHAP, feature importance, temporal attribution, graph/node-level explanations.

### 9.16 Alerting
High-risk forecasts generate structured early-warning alerts containing risk level, probability, forecast horizon, affected entity, supporting evidence, and threat context.

### 9.17 Security Dashboard
Risk overview, forecast timeline, entity view, attack graph, and alert-investigation view (prediction, evidence, risk score, timeline, threat context, explanation).

---

## 10. Evaluation Strategy

Accuracy alone is not sufficient for a forecasting claim.

**Classification metrics:** Precision, Recall, F1-score, PR-AUC, ROC-AUC (where appropriate).

**Forecasting-specific metric (most important): Lead Time** — how early the system provides a warning before actual attack onset.

```
Prediction ──── Lead Time ────► Attack Onset
```

A model that only flags an attack after it starts is not demonstrating meaningful forecasting, regardless of its classification scores.

**Additional metrics:** false alarms per unit time, forecast coverage, calibration, performance by attack category, performance across datasets/scopes.

---

## 11. Model Lifecycle

```
Dataset → Dataset Audit → UCS Construction → Temporal Split → Baseline →
LSTM/GRU → GNN → Fusion → Evaluation → Explainability → Model Registry →
Deployment → Monitoring → Retraining
```

Every trained model version should record: dataset version, UCS version, feature version, architecture, training configuration, and evaluation metrics.

---

## 12. Team Structure & Responsibilities

| Role | Responsibilities |
|---|---|
| **Team Lead / Solution Architect** | Overall architecture, technical decisions, integration, evaluation strategy, MVP prioritization, backend/system coordination, final presentation |
| **Data Engineering** | Dataset ingestion, cleaning, canonical mapping, timestamp normalization, windowing, feature engineering, UCS generation, leakage validation, dataset restructuring |
| **ML Engineer 1 — Temporal** | UCS sequence generation, temporal baseline, LSTM/GRU, forecast target, training, evaluation, lead-time metrics |
| **ML Engineer 2 — Graph** | Graph construction, node/edge definitions, graph features, GNN model, graph evaluation, comparison against temporal model |
| **Backend** | Inference APIs, risk engine, alert APIs, model integration, data serving, backend testing |
| **Frontend** | Dashboard, forecast visualization, risk visualization, alert display, entity/graph visualization, backend integration |

---

## 13. Current Status

### Completed / Established
- ✅ Problem understanding — SIH26153 analysed; detection-vs-forecasting distinction established
- ✅ Background research across signature, anomaly, supervised, temporal, and graph-based approaches
- ✅ Dataset research and audit across all SIH-listed datasets
- ✅ Dataset strategy (audit → temporal suitability → selection, no blind concatenation)
- ✅ Forecasting formulation and future-window target design
- ✅ UCS V1 design (canonical schema, metadata/feature separation)
- ✅ Data engineering pipeline design (cleaning, mapping, normalization, windowing, feature engineering)
- ✅ Leakage strategy (chronological splitting, purge/embargo)
- ✅ Feature engineering definition (traffic/protocol/behavioural/statistical)
- ✅ Temporal model design (LSTM/GRU selected as baseline)
- ✅ MITRE ATT&CK / CAPEC / CVE-NVD integration concept
- ✅ Explainability layer and risk engine concept
- ✅ Backend API architecture and frontend/dashboard architecture direction
- ✅ Core documentation: API, architecture, ML lifecycle, telemetry, UCS/data contracts, feature dictionary, dataset decisions

### In Progress
- 🔄 GNN design and graph construction
- 🔄 Temporal + graph fusion design
- 🔄 Forecasting evaluation and lead-time evaluation implementation
- 🔄 Threat intelligence integration (implementation)
- 🔄 Explainability (implementation)
- 🔄 Risk engine (implementation)
- 🔄 Backend integration
- 🔄 Frontend integration
- 🔄 End-to-end demo
- 🔄 Final PPT / submission materials

**Note:** The data engineering member has completed the major pipeline design work and is currently restructuring/cleaning the implementation.

---

## 14. Execution Plan — Phased Roadmap

### Phase 1 — Foundation
- Finalize dataset audit and dataset selection
- Build/validate the data engineering pipeline
- Finalize UCS V1 and leakage validation
- Produce a reproducible, versioned dataset

### Phase 2 — Forecasting Baseline
- Implement LSTM/GRU temporal model
- Finalize forecast target construction
- Run temporal evaluation
- Implement lead-time evaluation (the key forecasting proof metric)

### Phase 3 — Advanced Intelligence
- Graph construction from available entity relationships
- Implement GNN
- Implement temporal + graph fusion
- Comparative evaluation: does fusion measurably outperform the temporal-only baseline?

### Phase 4 — Security Intelligence
- MITRE ATT&CK / CAPEC / CVE-NVD enrichment integration
- Risk engine implementation and calibration
- Explainability implementation (SHAP / feature / temporal / graph evidence)

### Phase 5 — Product Layer
- Backend inference, risk, and alert APIs
- Alert engine
- Security dashboard (risk overview, forecast timeline, entity view, attack graph, investigation view)
- End-to-end demo assembly

### Phase 6 — Advanced Research Extensions (post-MVP)
- Transformer-based temporal model comparison
- Continuous learning / analyst feedback loop
- Model monitoring and retraining pipeline
- Multi-dataset generalization studies

**Immediate priority sequence:**

```
Finish/validate UCS
        ↓
LSTM/GRU forecasting baseline
        ↓
Reliable evaluation + lead-time metric
        ↓
GNN branch
        ↓
Fusion — only if it demonstrates measurable improvement
        ↓
Integrate risk + explanation + dashboard
        ↓
End-to-end demo
```

This sequencing keeps the project scientifically defensible at every stage while still allowing the final system to demonstrate the complete proposed architecture. The **MVP** (UCS → leakage-free sequences → LSTM/GRU → future attack probability → lead-time evaluation → basic API/dashboard) is a genuine vertical slice of the full system, not a throwaway prototype — every later phase extends it rather than replacing it.

---

## 15. Positioning Principles — Claims We Deliberately Avoid

| Overstated claim | What we say instead |
|---|---|
| "We detect every attack." | "The system forecasts attack likelihood under evaluated conditions." |
| "We combine all SIH datasets into one model." | "Our pipeline supports heterogeneous datasets through UCS; datasets are selected based on temporal and forecasting suitability." |
| "GNN automatically improves the model." | "We evaluate whether graph-based relational information measurably improves forecasting performance." |
| "The system predicts exactly when an attack will happen." | "The system estimates the probability of attack activity occurring within a defined future horizon." |
| "MITRE ATT&CK predicts attacks." | "MITRE ATT&CK provides threat-intelligence context and attack-technique mapping." |

---

## 16. One-Paragraph Summary

SIH26153 proposes an AI-based network attack forecasting system that learns the temporal and relational evolution of cybersecurity activity to estimate the likelihood of future attacks. The architecture converts heterogeneous public cybersecurity datasets (CIC-IDS2017/2018, UNSW-NB15, CTU-13, CICIoT2023, LANL Authentication, DARPA) into a standardized, leakage-controlled Unified Cyber State (UCS). A temporal model (LSTM/GRU) learns behavioural evolution across historical cyber states, while a GNN models relationships between entities such as users, hosts, services, and communication endpoints. Their representations are fused to forecast future attack probability over a defined horizon. The prediction is then enriched with MITRE ATT&CK, CAPEC, and CVE/NVD context, converted into an operational risk score, explained with feature/temporal/graph evidence, and surfaced through an alerting and security dashboard layer. Execution prioritizes a reproducible UCS and a strong temporal forecasting baseline first, followed by the GNN, fusion (only if it proves valuable), threat-intelligence enrichment, explainability, and operational visualization.

## One-Line Pitch

*We are building an AI system that learns how network behaviour evolves and forecasts when that behaviour is likely to enter an attack state — combining temporal patterns, cyber relationships, and threat intelligence to provide explainable early warning rather than simply detecting attacks after they occur.*
