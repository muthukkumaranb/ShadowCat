# SIH26153 — Architecture Diagram Review
## `SIH26153_Architecture.pdf` checked against the v3 Final Plan and the Problem Statement

**Reviewed artifact:** `SIH26153_Architecture.pdf` ("Cyber World Model Architecture", single-page diagram)
**Checked against:** `SIH26153_Final_Gap_Analysis_and_Plan.md` (v3, authoritative) and the PS text in `problem-statement-scrape-data.json`
**Date:** 2026-09-03

---

## 1. Verdict

The diagram is a faithful translation of the v3 plan at the block level. Every major
v3 component is present: Gate 0, project-owned flow + packet extraction, UCS with
masks and flow index, latent encoder, probabilistic dynamics head, K-step rollout,
hazard head with the coherent cumulative formula, stage head with the mapping
manifest, the three separated evidence types, the dual input contract, two evaluation
tracks, the five-baseline set, and the ablation grid.

**Where it falls short is wiring and omission, not concept.** Five gaps are high
priority because they either recreate a risk the Codex review explicitly flagged
(the "checkbox world model") or drop a PS-mandated item a judge will scan for:

1. The **runtime path from rollout → hazard → UI is not drawn**; the UI is fed via
   the ablation grid, so it is ambiguous which branch produces the displayed
   probability.
2. **No named explainability technique.** The PS says "SHAP values or model
   attention weights" and "black-box outputs … are not acceptable." The diagram
   names evidence *types* but no method.
3. **Chronological split, purge/embargo, and train-only normalization fit are
   absent**, and "feature normalize" is drawn *before* windowing and splitting.
4. **No training objective** (one-step + explicit multi-step rollout loss) — the
   thing that makes the rollout real rather than decorative.
5. **No target/label construction step** — the PS's "supervised dynamics learning,
   where ground-truth state transitions are derived from the attack timeline
   annotations" has no corresponding block.

All are fixable with edge additions, a few labels, and text in the accompanying
2-page document. Details below.

---

## 2. Alignment Checklist

| Requirement (PS / v3) | In diagram? | Note |
|---|---|---|
| Gate 0 dataset feasibility | ✓ | Arrow only points at PCAP; should gate the whole pipeline |
| Flow-level features (PS list) | ✓ | Complete |
| Packet-level features (PS list) | ✓ | Complete; "scan sequencing" covers port-scan signatures |
| Project-owned extraction, corrected CICFlowMeter | ✓ | Good — visible judge differentiator |
| UCS with feature-presence masks + flow-to-window index | ✓ | |
| Latent state transition `p(z(t+1) \| history)` | ✓ | Gaussian / quantiles / ensemble options shown |
| K-step latent rollout with uncertainty growth | ✓ | |
| Direct forecast head labeled as baseline/safety net | ✓ | |
| Hazard head with `P(event ≤ K) = 1 − Π(1 − h_k)` | ✓ | "event" undefined across the two tracks (§3.7) |
| Stage head from predicted state, manifest, DoS→Impact | ✓ | |
| Three evidence types, separated | ✓ | Method names missing (§3.2) |
| Ablation grid with source-labeled results | ✓ | Misplaced as a runtime component (§3.12) |
| Dual input contract (PCAP full / CSV flow-only) | ✓ | Schema-shift caveat (§3.8) |
| Offline demo UI | ✓ | Missing LR comparison, uncertainty, novelty (§3.6) |
| Two evaluation tracks (onset / progression) | ✓ | |
| Five baselines incl. lagged LR + persistence | ✓ | |
| PS-mandated metrics F1/P/R/FPR + lead time | ✓ | Missing false alarms per unit time, per-episode CIs (§3.13) |
| Unseen-attack: risk / novelty / stage-where-supported | ✓ | Leave-one-family-out protocol not named |
| Rollout → hazard → UI runtime wiring | ✗ | §3.1 — high |
| Named XAI method (SHAP / attention / IG) | ✗ | §3.2 — high |
| Chronological split + purge/embargo | ✗ | §3.3 — high |
| Normalization fitted on train only | ✗ (order implies otherwise) | §3.3 — high |
| One-step + multi-step rollout loss | ✗ | §3.4 — high |
| Target/label construction from timeline annotations | ✗ | §3.5 — high |
| Model artifacts: weights, config, data manifest, seeds | ✗ | §3.11 — PS-explicit deliverable |
| Training-vs-inference separation | ✗ | §3.12 |
| Graph construction component | ✗ (state says "+ graph", nothing builds it) | §3.10 |

---

## 3. Gaps and Misalignments (by severity)

### HIGH

#### 3.1 Runtime path is not wired; producing branch is ambiguous
- **Diagram shows:** Direct head → Ablation grid; Hazard head → Ablation grid; Stage
  head → Explainability; Ablation grid → Demo UI. There is no edge from the hazard
  head (probability trajectory) or the rollout (future trajectory) to the UI.
- **v3 requires (C1 non-negotiables):** the rollout must be *executed, evaluated,
  and used* for the displayed trajectory/stage outputs; every displayed result
  states its producing branch. Codex flagged exactly this: if the dashboard number
  comes from the direct head, the system is v1 plus a reconstruction task.
- **Fix:** draw the runtime edges explicitly —
  `K-step rollout → Hazard head → "probability trajectory" → UI`,
  `K-step rollout → "future state trajectory" → UI`,
  `Stage head → "stage trajectory" → UI`,
  `Direct head → UI (labeled "baseline comparison")`.
  Remove the Ablation grid → UI edge (ablation is evaluation, not runtime; see 3.12).

#### 3.2 No named explainability technique
- **Diagram shows:** "Forecast explanation — attribution over observed history";
  "Trajectory evidence"; "Current novelty". Types only.
- **PS requires:** "explainability output for each prediction — using SHAP values
  or model attention weights"; "Black-box outputs without interpretability are not
  acceptable." A judge scanning for those terms finds neither.
- **v3 (C2):** feature ablation / integrated gradients validated by deletion tests;
  attention shown as model internals only; standardized coefficients for LR.
- **Fix:** label the box "Integrated Gradients / feature ablation (SHAP-family
  attribution), validated by deletion tests; attention weights shown as model
  internals". The 2-page text should say in one sentence why IG/ablation is used
  as the primary attribution and how it relates to SHAP.

#### 3.3 Split discipline and normalization order are missing or wrong
- **Diagram shows:** Data engineering: clean → canonical map → timestamp normalize →
  missing-value → **feature normalize → 1-min windowing → timeline alignment →
  leakage prevention**. No chronological split, purge, or embargo anywhere.
- **v3 / v1 §9.8 require:** chronological train/val/test with purge + embargo
  ≥ max lookback + horizon; normalization statistics **fitted on the training split
  only**; thresholds selected on validation only. This is the v1 briefing's
  strongest asset and it has vanished from the picture.
- **Fix:** add a block "Chronological split → purge/embargo → fit normalizers on
  train only" *after* windowing and *before* any model. Reorder so normalization
  follows the split. Expand "leakage prevention" into its concrete rules (≤ t
  only; no future-derived statistics).

#### 3.4 Training objective absent
- **Diagram shows:** heads and rollout, no losses.
- **v3 (C1):** one-step loss + **explicit multi-step rollout loss**; scheduled
  sampling only as an ablation. Without the multi-step loss the rollout is a
  by-product, not a trained capability — the "decorative dynamics head" failure.
- **Fix:** annotate the dynamics head / rollout with "trained with one-step NLL +
  K-step rollout loss"; list scheduled sampling under the ablation grid, not as a
  default.

#### 3.5 Target and label construction step absent
- **Diagram shows:** data engineering produces S(t); nothing produces the targets.
- **PS requires:** "trained on labelled open-source datasets using supervised
  dynamics learning, where ground-truth state transitions are derived from the
  attack timeline annotations in the dataset."
- **v3 Phase 1:** define onset and progression targets separately; build the
  ATT&CK mapping manifest (source_label · interval · telemetry · tactic(s) ·
  confidence · rationale).
- **Fix:** add a "Target construction" block fed by the timeline annotations:
  onset target `y_onset(t)`, progression target `y_stage(t+k)`, next-state target
  `S(t+1)`, and the mapping manifest as the stage-label source. Draw it feeding
  the hazard and stage heads.

### MEDIUM

#### 3.6 Demo UI is missing three required outputs
- **Diagram shows:** probability timeline · attack stage · driving features ·
  flagged flows · future trajectory.
- **v3 Phase 4 / solution slide / PS:** also **LR-baseline comparison in the same
  interface** (PS requires benchmark results; the original solution slide promised
  it on the dashboard), **uncertainty bands** on the trajectory, and the **current
  novelty score**.
- **Fix:** add the three items to the UI box.

#### 3.7 Hazard head "event" is undefined across two tracks
- **Diagram shows:** one hazard head, `P(event ≤ K)`; one direct head,
  `P(event within H)`.
- **v3 (C1, E1):** the hazard head estimates first-onset *or* next-stage
  transition; evaluation has two tracks with different cutoffs and targets.
- **Fix:** either two hazard targets (`h_k^onset`, `h_k^stage`) or one head
  parameterized by track; label which the demo displays. Same for the direct head.

#### 3.8 CSV contract vs. corrected-CICFlowMeter schema mismatch
- **Diagram shows:** training uses corrected CICFlowMeter / relabeled data; the
  CSV demo input is "NetFlow / IPFIX / CSV" routed to a flow-only model.
- **Risk:** Engelen's corrections change feature *values* (flow termination, TCP
  handling), so a raw CICFlowMeter-v3 CSV from a user is a different schema than
  the corrected features the flow-only model was trained on — a modality shift
  inside the CSV path, the exact thing the dual contract was meant to prevent.
- **Fix:** state the CSV contract precisely: "CSV must match the corrected-
  CICFlowMeter feature schema (documented); other CSVs are re-derived through the
  canonical mapper or rejected." Evaluate the flow-only model on that schema.

#### 3.9 Identity / authentication path reintroduces cross-dataset fusion
- **Diagram shows:** "Authentication logs (optional, needs LANL dataset)" →
  identity features → into S(t).
- **Problem:** LANL is a different network with no packet/flow telemetry; its
  auth events cannot be time-aligned with CIC-IDS2018 windows. The v1 briefing
  itself warned against blind concatenation, and v3 removed identity from the MVP
  entirely. Drawing it invites a judge question with no good answer.
- **Fix:** remove from the MVP diagram, or footnote as "future work on a separate
  dataset with its own state schema" — not as an optional input to the same S(t).

#### 3.10 Graph appears twice and is built nowhere
- **Diagram shows:** "+ graph" inside S(t) **and** a "Graph path (optional):
  GraphSAGE over network topology" inside the encoder; no component constructs a
  graph.
- **v3 (C1 / Part D):** prefer graph embeddings *inside the state*; GNN is
  post-MVP and gated on measurable improvement; the graph is a **per-window
  communication graph** (src/dst IP–port edges), not a static "network topology".
- **Fix:** pick one placement (inside S(t) is v3's preference), add a
  "communication-graph builder (per window)" block in feature extraction, rename
  "network topology" → "per-window communication graph", and keep the whole
  thing dashed/optional.

#### 3.11 Model artifacts / reproducibility not represented
- **PS requires:** "Training scripts, model weights, and a reproducible training
  configuration must be included."
- **Fix:** add a small "Artifacts: configs · weights · data manifest · seeds ·
  env lock" block hanging off the world-model box. It signals compliance in one
  glance.

#### 3.12 Training-time and runtime are mixed; ablation grid is misplaced
- **Diagram shows:** the ablation grid sits inside the world-model box and feeds
  the UI.
- **Fix:** move the ablation grid into the Evaluation box; separate the diagram
  (or the 2-page text) into a **training/evaluation pipeline** and an
  **inference/demo path**. A judge reading a 2-page doc should see in seconds
  what runs when a PCAP is dropped in vs. what ran offline to produce the tables.

### LOW

#### 3.13 Metrics list incomplete
- Missing vs v3 E3: **false alarms per unit time** (the SOC-real metric from v1),
  **per-episode reporting with CIs** (overlapping windows are not independent),
  **deviation-score ROC**, **predictive-interval coverage**. Add as a second line.

#### 3.14 Window size hardcoded
- "1-min windowing" — v1 §9.4 treated it as an experimental parameter. Label
  "default 1-min (tunable)".

#### 3.15 Leave-one-family-out protocol not named
- "Unseen-attack generalization" lists the three claims but not the protocol that
  produces them. Add "leave-one-attack-family-out; thresholds fit without the
  held-out family".

#### 3.16 CAPEC / CVE enrichment absent
- Acceptable — v3 makes it optional. A one-line dashed "optional post-stage
  enrichment" note would show awareness of the PS's knowledge-base list without
  committing to it.

#### 3.17 Gate 0 arrow only targets PCAP
- Cosmetic: Gate 0 gates the whole pipeline (it freezes the primary claim). Point
  it at the Data engineering block.

---

## 4. Explicit Wiring Spec for the Corrected Diagram

Runtime (inference/demo) path — solid edges:

```
PCAP ──► flow+packet extraction ──► UCS S(t) ──► encoder z(t)
CSV  ──► schema validation (corrected-CICFlowMeter schema) ──► flow-only model
z(t) ──► dynamics head ──► K-step rollout ẑ(t+1..t+K)
rollout ──► hazard head ──► probability trajectory (+ uncertainty) ──► UI
rollout ──► stage head  ──► stage trajectory ──► UI
rollout ──► trajectory evidence ──► UI
history ──► forecast explanation (IG/ablation) ──► UI
S(t) vs Ŝ(t) from t−1 ──► current novelty ──► UI
z(t) ──► direct head ──► UI (labeled "baseline comparison")
flow index ──► flagged flows ──► UI
```

Training/evaluation path — dashed edges, separate region:

```
timeline annotations ──► target construction (onset, stage, next-state, manifest)
windowed UCS ──► chronological split ──► purge/embargo ──► fit normalizers (train only)
losses: one-step NLL + K-step rollout loss (+ hazard NLL, stage CE)
ablation grid ──► evaluation (two tracks, baselines, metrics, LOO) ──► artifacts
```

---

## 5. What the Accompanying 2-Page Text Must Carry

A diagram cannot hold everything; these belong in the prose and must not be
assumed from the picture:

1. The exact split/purge/embargo widths and the train-only normalization rule.
2. The training objective (one-step + multi-step rollout loss) and which
   rollout-error mitigations were tried.
3. The named attribution method and its validation (deletion test).
4. Definition of "event" per evaluation track and which track the demo displays.
5. The CSV input schema contract.
6. The ATT&CK mapping rules (DoS = Impact shown as its own category; brute force
   = Credential Access unless success is evidenced; infiltration annotated by phase).
7. Which branch produced every reported number.
8. Reproducibility: repo layout for configs, weights, data manifest, seeds.

---

## 6. Priority Order for the Revision

1. Wire the runtime path (3.1) and move the ablation grid (3.12) — 10 minutes,
   removes the biggest judge-facing ambiguity.
2. Name the XAI method (3.2) — one label.
3. Add split/purge/embargo + train-only normalization (3.3) and the target-
   construction block (3.5).
4. Annotate losses (3.4), define "event" per track (3.7), add artifacts block (3.11).
5. UI additions (3.6), CSV schema note (3.8), drop/footnote identity path (3.9),
   fix graph placement (3.10).
6. Low items (3.13–3.17) as time permits.

---

## 7. Resolution — `SIH26153_Architecture_v2.pdf` (2026-09-03)

A corrected diagram was produced from `SIH26153_Architecture_v2.dot` (Graphviz;
edit the `.dot` and re-run `dot -Tpdf` to regenerate). It closes every item above:

| Gap | How v2 resolves it |
|---|---|
| 3.1 runtime wiring | New **Forecast outputs** collector: hazard → probability trajectory + uncertainty, stage → stage trajectory, rollout → future state trajectory, direct → baseline comparison; collector → UI. Ablation grid no longer feeds the UI. Solid = runtime, dashed orange = training/eval (legend in title). |
| 3.2 XAI method | Forecast explanation box names **Integrated Gradients / feature ablation (SHAP-family attribution), validated by deletion tests; attention as internals only**. |
| 3.3 split & normalization | New dashed **Training-time only** cluster: chronological split → purge → embargo (≥ lookback + horizon), normalizers fit on TRAIN only, thresholds on VALIDATION only, explicit ≤ t leakage rule. Normalization removed from the pre-split data-engineering chain. |
| 3.4 training objective | Dynamics head annotated **one-step NLL + explicit K-step rollout loss**; Losses note lists hazard NLL, stage CE, direct BCE; scheduled sampling listed under ablations only. |
| 3.5 target construction | New **Target construction** block fed by timeline annotations: S(t+1), y_onset, y_stage, and the ATT&CK mapping manifest with its rules; dashed edges into dynamics, hazard, and stage heads. |
| 3.6 UI outputs | UI lists uncertainty bands, current novelty score, and LR-baseline comparison (fed by a dashed edge from Metrics). |
| 3.7 event per track | Hazard head shows `h_k^onset` and `h_k^stage` explicitly. |
| 3.8 CSV schema | CSV input labeled **corrected-CICFlowMeter schema; other CSVs → canonical mapper, else rejected**. |
| 3.9 identity path | Removed from the pipeline; shown as a dashed **Future work — NOT in MVP** note. |
| 3.10 graph | Single placement: optional **Communication-graph builder (per window)** in feature extraction → embedding inside S(t); "network topology" wording dropped; dotted throughout. |
| 3.11 artifacts | **Reproducibility artifacts** block (scripts · configs · weights · data manifest · seeds · env lock · limitations). |
| 3.12 train vs runtime | Ablation grid moved into the dashed Evaluation cluster; edge style separates the two regimes. |
| 3.13 metrics | Adds false alarms per unit time, deviation-score ROC, predictive-interval coverage, per-episode CIs. |
| 3.14 window size | "default 1-min, tunable". |
| 3.15 LOO protocol | Unseen-attack box names leave-one-attack-family-out with thresholds fit without the held-out family. |
| 3.16 enrichment | Dotted optional CAPEC / CVE-NVD note on the stage output. |
| 3.17 Gate 0 | Gate 0 now gates Data engineering ("gates the whole pipeline"). |

Also added: the one-step novelty forecast is sourced from the rollout's first step
made at t−1 (consistent with the v3 evidence-type separation), and every input to
the explainability cluster is a single labeled bus (observed S(t), history, flow
index) so the diagram cannot be read as using unobserved future data.
