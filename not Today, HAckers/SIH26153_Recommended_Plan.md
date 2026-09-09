# SIH26153 — Recommended Plan (Revised)

> **⚠️ SUPERSEDED (2026-09-01):** The Codex independent review found four material
> defects in this plan (residual-timing conflation, class≠stage label shortcut,
> overstated CIC-IDS2018 progression ground truth, underspecified probabilistic
> dynamics), all verified and corrected. Authoritative version:
> **`SIH26153_Final_Gap_Analysis_and_Plan.md`**.
## World-Model-Aligned Architecture, Dataset Strategy, Evaluation & Roadmap

**Companion to:** `SIH26153_Assessment_Gap_Analysis.md` (why these changes are needed)
**Date:** 2026-08-31 · **Idea-submission deadline:** 2026-09-20

---

## 1. Revised Core Formulation

Replace the discriminative formulation with a genuine dynamics model plus a pragmatic
companion head.

### 1.1 What changes

| | Current (briefing) | Revised (required) |
|---|---|---|
| Core model | `X(t-k..t) → P(attack in [t,t+H])` | `P(S_t+1 \| S_t-k..t)` — next-**state** prediction |
| Inference | Single classification | **K-step autoregressive rollout**: `Ŝ(t+1), Ŝ(t+2), … Ŝ(t+K)` |
| Attack probability | Direct classifier output | Derived from the predicted trajectory (stage/attack head applied to each `Ŝ`), giving a **time-series probability score** over the next K windows |
| Unseen attacks | Not addressed | **Deviation score**: divergence between predicted `Ŝ(t)` and observed `S(t)` flags novel behaviour |

### 1.2 Two-head architecture (pragmatic hybrid)

Train one temporal backbone (LSTM/GRU baseline; Transformer variant later) with two heads:

1. **Dynamics head** — predicts the next UCS state (regression over the state vector,
   or over a learned latent). This is the World Model. Used for K-step rollout,
   trajectory-based infiltration probability, and deviation-based novelty detection.
2. **Direct forecast head** — predicts `P(attack in [t, t+H])` directly. This is the
   reliable metric-producing branch and the honest fallback if rollouts degrade.

Present both openly: "the dynamics head demonstrates the world-model requirement; the
direct head anchors benchmark comparability." This is scientifically defensible and
matches the briefing's existing honesty culture.

### 1.3 Rollout-error mitigations (know these for Q&A)

- Predict in a **latent space** (encoder → latent dynamics → decoder), not raw
  high-dimensional features — compounding error is smaller in latent space.
- **Scheduled sampling** during training (feed model's own predictions back in with
  increasing probability) so rollouts don't collapse at inference.
- Direct multi-horizon head (above) as companion, not replacement.

### 1.4 Outputs per inference (matches PS deliverable list exactly)

1. Time-series infiltration probability for the next K windows
2. Predicted **MITRE ATT&CK stage** (Reconnaissance / Initial Access / Lateral
   Movement / C2 / Exfiltration) derived from predicted future state
3. **Driving features** — top contributors via attention weights + prediction-residual
   attribution (which features diverged from forecast)
4. Deviation/novelty score for unseen-attack signalling

---

## 2. Revised Dataset Strategy

### 2.1 Priority flip

| Priority | Dataset | Role | Why |
|---|---|---|---|
| **Primary** | **CSE-CIC-IDS2018** | Train + evaluate everything | On the SIH list; PS names it explicitly as expected input; ships **raw PCAPs + flow CSVs**; multi-day; attack-timeline annotations give ground-truth state transitions for supervised dynamics learning |
| Secondary | CIC-IDS2017 | Smaller-scale dev/iteration set | Same feature family; faster experiments |
| Graph track | CTU-13 | GNN/relational experiments | PS names it; botnet + background traffic suits communication graphs |
| Generalization study | UGR'16 | Post-MVP cross-dataset study only | Long chronological coverage is valuable research, but it's unlisted and NetFlow-only (no PCAPs) — cannot carry the mandatory packet-level requirement |

Keep the dataset-audit narrative (it's a strength) — the audit now *concludes* with
CIC-IDS2018 as primary instead of UGR'16.

### 2.2 New pipeline branch: PCAP / packet-level feature extraction

Add to data engineering scope (this is mandatory, not optional):

- Parser: **Scapy or PyShark** over CIC-IDS2018 PCAPs (a curated subset is fine —
  full PCAP volume is huge; select the attack days you evaluate on).
- Per-window packet-level aggregates joined into the UCS alongside flow features:
  - TTL values + variance across sessions
  - TCP window size statistics
  - IP fragment flag counts
  - Payload size distribution (histogram/quantiles per window)
  - Retransmission counts
  - Port-scan signatures: sequential vs. randomized port-access patterns per source
- The demo app must accept a **raw PCAP or CSV** as input and run this extractor live.

UCS schema change: extend the canonical vocabulary with a `packet.*` feature family;
same missing-not-fabricated rule applies when a window has flow data but no PCAP
coverage.

---

## 3. MITRE ATT&CK — Dual Role

1. **Stage prediction head (new, required):** map dataset attack labels to ATT&CK
   tactics and train a stage classifier applied to (predicted) states:

   | CIC-IDS2018 label | ATT&CK stage (PS's five + Impact) |
   |---|---|
   | Port scan / probing | Reconnaissance |
   | Brute force (SSH/FTP) | Initial Access / Credential Access |
   | Web attacks (XSS, SQLi) | Initial Access |
   | Infiltration | Lateral Movement |
   | Botnet (Zeus/Ares) | Command & Control |
   | DoS / DDoS | Impact (map to nearest of the PS's five: C2/Exfiltration adjacency — document the choice) |

2. **Enrichment layer (keep from briefing):** CAPEC pattern context + CVE/NVD
   vulnerability context attached *after* the stage prediction. The briefing's
   "enrichment, not training labels" principle still holds for CAPEC/CVE — only the
   ATT&CK *stage* becomes a supervised target.

---

## 4. Revised Evaluation Plan

### 4.1 Required benchmark (PS-mandated)

- **Logistic regression baseline trained on the same UCS features** (current-window,
  no temporal context).
- Report **F1, precision, recall, false-positive rate** for LR vs. world model —
  the delta *is* the "temporal dynamics learning provides measurable improvement" claim.
- SHAP on the LR baseline (trivial) doubles as an explainability artifact.

### 4.2 Forecasting-specific (keep + tighten)

- **Lead time** per attack episode (keep as headline metric).
- **Event-based evaluation:** score per attack episode, not per window; CIC-IDS2018
  has few independent onsets, so report per-episode numbers honestly rather than
  aggregate significance claims.
- **Exclude in-attack windows from forecast credit** — once an attack has started,
  predicting "attack in next H" is trivial autocorrelation. Only pre-onset windows
  count toward forecasting claims.
- **False alarms per unit time** (keep — the SOC-real metric).
- **Calibration:** reliability diagram + Brier score (cheap, substantiates "probability").
- PR-AUC over ROC-AUC given extreme class imbalance.

### 4.3 World-model-specific (new)

- **State-prediction quality:** per-feature error of `Ŝ(t+1)` vs `S(t+1)` on benign
  and pre-attack windows; error growth curve over rollout depth K.
- **Deviation-score ROC:** does predicted-vs-observed divergence separate attack
  windows from benign ones?

### 4.4 Unseen-attack generalization (new, PS-mandated)

- **Leave-one-attack-type-out:** train with one attack family's days held out
  (e.g. infiltration); test whether the deviation score + stage head still flag it.
- Report honestly per held-out family — partial success here is still a differentiator,
  since most teams won't attempt it at all.

---

## 5. Explainability Plan (refined)

| Technique | Where | Cost |
|---|---|---|
| **Attention weights** | Temporal backbone (PS explicitly accepts attention) | Free with architecture |
| **Prediction-residual attribution** | World model native: rank features by divergence between predicted and observed state — "TTL variance and SYN rate departed from forecast" | Near-free, honest, unique to world models |
| **SHAP** | LR baseline (fast, exact-ish) | Low |
| Graph/node evidence | Only if GNN branch survives its empirical gate | Deferred |

Keep the briefing's principle: never a bare number — every alert carries evidence.
The residual attribution makes this concrete without SHAP-on-LSTM pain.

---

## 6. Scope Cuts (product layer)

The required demo is: **Streamlit (or Flask/CLI) app, fully offline, accepts PCAP or
CSV, displays infiltration probability timeline + flagged flows + attack-stage
annotations + explanations + LR-baseline comparison.**

| Cut / simplify | Replacement |
|---|---|
| API gateway + 3 separate APIs | One inference service behind the Streamlit app (or direct function calls) |
| Risk engine w/ asset criticality | Simple calibrated risk banding of forecast probability (asset criticality of anonymized dataset IPs isn't computable — say so proudly, it's honest) |
| Alert engine as separate layer | Alert = dashboard row + structured JSON export |
| Model registry / retraining / monitoring | A versioned `configs/` + `weights/` directory (PS requires reproducible config + weights anyway) |

The dashboard views worth keeping: risk timeline, K-step forecast trajectory,
per-window flagged flows, ATT&CK stage annotation, explanation panel, LR-vs-model
comparison chart.

---

## 7. Revised Roadmap (replaces briefing §14 phases)

### Phase 1 — Foundation (reworked)
- Flip primary dataset to CIC-IDS2018; re-run audit conclusion
- **Add PCAP/packet-feature extraction branch** (Scapy/PyShark, curated PCAP subset)
- Extend UCS with `packet.*` family; finalize leakage validation
- Produce reproducible, versioned dataset

### Phase 2 — World-Model Baseline (reworked — this is the heart)
- Temporal backbone + **dynamics head** (next-state prediction, latent-space option)
- **K-step rollout engine** with error-growth measurement
- **Direct forecast head** as companion branch
- **LR baseline on same features** (PS-mandated benchmark)
- Lead-time + event-based evaluation harness (pre-onset-only credit)

### Phase 3 — Stage Prediction & Novelty (new priority, was scattered)
- ATT&CK stage head over predicted states (label mapping table §3)
- Deviation/novelty scoring from prediction residuals
- **Leave-one-attack-type-out** generalization runs
- Calibration analysis

### Phase 4 — Graph Branch (gated, unchanged philosophy)
- Communication graph per window (CIC-IDS2018 IPs; CTU-13 for richer structure)
- Prefer **graph-embeddings-inside-the-state** over a parallel fusion branch
- Adopt only if it measurably beats the temporal-only world model

### Phase 5 — Demo & Enrichment
- Offline Streamlit app: PCAP/CSV in → probability timeline, stages, flagged flows,
  explanations, LR comparison
- CAPEC/CVE enrichment on stage predictions
- Attention + residual-attribution explanation panel

### Phase 6 — Submission Package (PS-explicit deliverables)
- Repo: source + README + **model weights + reproducible training config**
- **2-page architecture document** — World-Model vocabulary front and center:
  `P(S_t+1|S_t)`, forward simulation, rollout, transition dynamics
- **2-minute demo video** — lead with a rollout catching an attack pre-onset
- **5-slide deck** (suggested: 1 problem/insight, 2 world-model architecture,
  3 leakage-free methodology + datasets, 4 results incl. lead time + LR delta +
  unseen-attack run, 5 demo + honest limitations)

### Immediate priority sequence

```
Flip dataset → add PCAP branch → UCS v2
        ↓
Dynamics head + K-step rollout  (the world model)
        ↓
Direct head + LR baseline + lead-time eval
        ↓
ATT&CK stage head + deviation scoring + leave-one-out
        ↓
Offline demo app (PCAP in → timeline out)
        ↓
Graph branch (only if it earns its place)
        ↓
Submission package (2-pager, video, 5 slides)
```

---

## 8. Language Alignment Cheat-Sheet (for the 2-pager, slides, and Q&A)

Use the PS's own vocabulary everywhere:

- "world model" · "state-transition dynamics `P(S_t+1 | S_t)`" · "forward simulation"
- "K-step rollout" · "trajectory converges to an infiltration state"
- "before the attacker completes the kill chain"
- "supervised dynamics learning from attack-timeline annotations"
- "not a static input-output classifier" (say it about your own model, with the
  dynamics head as proof)
- "generalises to unseen attack patterns" (backed by leave-one-out results)

Keep the briefing's honesty table (§15) verbatim — add one row:

| Overstated claim | What we say instead |
|---|---|
| "The rollout is accurate K steps ahead." | "Rollout error grows with depth; we measure it, mitigate it in latent space, and pair the rollout with a direct-horizon head." |

---

## 9. What Carries Over Unchanged from the Briefing

- Leakage control rules (§8.7) — unchanged, showcase
- Chronological splits + purging + embargo (§9.8) — unchanged, showcase
- UCS concept + canonical mapping + metadata/feature separation — extended, not replaced
- Missing-not-fabricated statistics rule — extended to packet features
- Fusion-only-if-proven principle — unchanged
- Team structure — unchanged; the delta lands mostly on Data Engineering (PCAP branch)
  and ML Engineer 1 (dynamics head + rollout), with ML Engineer 2's graph work
  reframed as state-embedding rather than parallel branch
