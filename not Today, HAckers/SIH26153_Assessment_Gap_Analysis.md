# SIH26153 — Assessment & Gap Analysis

> **⚠️ SUPERSEDED (2026-09-01):** Adjudicated by the Codex independent review and
> online verification. Known corrections: UGR'16 is a modality risk, not a
> compliance violation; the in-attack-window exclusion rule was too broad (two
> evaluation tracks now); the DoS→"nearest stage" mapping was wrong (DoS = Impact).
> Authoritative version: **`SIH26153_Final_Gap_Analysis_and_Plan.md`**.
## Review of the Technical Handover Briefing and Proposed Solution vs. the Problem Statement

**Reviewed artifacts:**
- Problem statement (SIH portal, full scraped description)
- Proposed solution slide (`Proposed solution.jpeg`)
- `SIH26153_Technical_Handover_Briefing.md`

**Date:** 2026-08-31 · **Idea-submission deadline:** 2026-09-20

---

## 1. Verdict (TL;DR)

The research quality is genuinely above typical hackathon level — leakage discipline,
purge/embargo splits, the lead-time metric, and the "claims we avoid" table are things
most teams never think about.

**But there is one significant misalignment: the briefing's architecture is a
discriminative forecaster, while the problem statement explicitly demands a World
Model — and says so in language that disqualifies the current design.** The solution
slide already fixed this *in words* ("World Model using LSTM and GNN", "K-step forward
forecasting", "deviations between predicted and observed"); the underlying architecture
and briefing have not caught up to the slide.

The fix isn't quality — it's **aim**. The briefing solves "attack forecasting" well,
but the rubric scores "**World Models** for attack forecasting."

---

## 2. What's Strong (keep and showcase)

| Strength | Why it matters |
|---|---|
| **Leakage control + purge/embargo + chronological splits** (§8.7, §9.8) | Most competing teams will unknowingly leak future data and report inflated numbers. NTRO evaluators know this failure mode; this discipline is a differentiator. |
| **Lead time as the headline metric** (§10) | Exactly right — a "forecast" that fires after onset is just detection. Best original contribution in the briefing. |
| **Dataset-audit insight** — "detection-suitable ≠ forecasting-suitable" (§6) | Correct, non-obvious, worth a slide of its own. |
| **Empirically gating GNN/fusion** (§9.11) | Honest, defensible; avoids the "GNN automatically helps" trap. |
| **Positioning principles table** (§15) | This maturity will land well in judge Q&A. |
| **UCS canonical schema with metadata/feature separation** (§9.5) | Solid engineering; prevents identifiers from becoming fake signal. |
| **No fabricated baselines rule** (§9.6) | Missing-not-manufactured z-scores — good scientific hygiene. |

---

## 3. Critical Gaps (ordered by risk to selection)

### Gap 1 — The formulation is not a World Model, and the PS explicitly rejects what it currently is

- Briefing's core equation: `X(t-k..t) → P(attack in [t, t+H])`. That is a **sequence
  classifier with a future-shifted label**.
- The PS, verbatim: *"The core deliverable is a learned model of network state
  transition dynamics — **not a static classifier**"* and demands learning
  `P(S_t+1 | S_t)` with **K-step forward simulation**.
- Shifting the label into the future does not make a classifier a dynamics model. In
  the evaluators' framing, the current design is still input→output classification.
- The phrase "World Model" appears **zero times** in the briefing. NTRO put the concept
  in the challenge's core description. This is the single biggest selection risk.

**Required change:** the model must predict the **next network state** `S(t+1)`
(a regression/generative head over the UCS state vector or a latent), roll out
autoregressively K steps, and derive attack probability from the predicted trajectory.
See the recommended plan document for the concrete architecture.

### Gap 2 — Packet-level features are mandated and completely absent

- PS: *"Teams **must** work with both flow-level and packet-level features"* — naming
  TTL variance, TCP window size, IP fragment flags, payload size distribution,
  retransmission counts, port-scan sequencing patterns.
- PS also requires the demo to **accept a raw PCAP** parsed with Scapy or PyShark.
- The briefing's entire UCS is flow/window aggregates. No PCAP pipeline exists anywhere
  in the plan. The solution slide claims "both flow-level and packet-level features"
  but nothing in the architecture produces the packet level.
- PS rationale (worth internalizing): flow-level catches aggregate behaviour (SYN
  flood); packet-level exposes timing/sequencing (slow recon scans designed to evade
  flow thresholds).

### Gap 3 — Dataset strategy conflicts with the requirements

- UGR'16 as primary candidate is defensible on temporal-coverage grounds, **but**:
  - It is **not on the SIH-listed datasets** — a presentation/compliance risk.
  - It is **NetFlow-only — no PCAPs** — so it structurally cannot satisfy the
    packet-level requirement (Gap 2).
- The PS names **CIC-IDS2018 and CTU-13** as the expected pipeline inputs.
  CSE-CIC-IDS2018 ships raw PCAPs alongside flow CSVs, with attack-timeline
  annotations — it satisfies both feature levels and the ground-truth-transition
  requirement.
- **Flip the priority: CIC-IDS2018 primary, UGR'16 demoted to a generalization
  study.** The audit narrative survives intact; you stop fighting the rubric.

### Gap 4 — MITRE ATT&CK positioned as post-hoc enrichment; PS wants stage *prediction*

- PS: *"Predicted attack stage: mapping to MITRE ATT&CK phases — Reconnaissance,
  Initial Access, Lateral Movement, Command & Control, or Exfiltration — **based on
  the predicted future state**."*
- That is an **output head**, not an enrichment layer. The briefing (§9.13, §5.2
  design decision) explicitly scopes ATT&CK to context-after-forecast, which
  by itself fails this rubric line.
- Both can coexist: a stage classifier over predicted states **plus** CAPEC/CVE
  enrichment on top. Only the enrichment-only version is non-compliant.

### Gap 5 — No concrete plan for unseen-attack generalization

- PS requires: *"Generalise to unseen attack patterns — not merely memorize signatures."*
- Solution slide bullet 5 (deviation between predicted and observed behaviour) is the
  correct world-model-native answer — but the briefing contains nothing implementing it.
- Two cheap, persuasive additions:
  1. **Deviation/residual scoring** — when observed `S(t+1)` diverges from the model's
     prediction, the residual is itself an anomaly signal for novel attacks.
  2. **Leave-one-attack-type-out evaluation** — hold out e.g. the infiltration days at
     training time; show the system still flags them at test time.

### Gap 6 — The mandatory logistic-regression baseline is missing from the eval plan

- PS requires benchmarking against **LR trained on the same features** (F1, precision,
  recall, FPR), to demonstrate that temporal dynamics learning provides measurable
  improvement.
- The solution slide mentions it; briefing §10 does not.
- This is a gift, not a burden: LR-on-current-window vs. world-model-K-step is exactly
  the "temporal dynamics matter" story the PS asks you to tell.

---

## 4. Secondary Improvements

1. **Trim the product layers.** Risk engine with asset criticality, API gateway, three
   APIs, alert engine, model registry, retraining loops — asset criticality of
   anonymized dataset IPs isn't even computable, and none of it earns rubric points.
   The required deliverable is a **Streamlit/Flask/CLI app that takes PCAP/CSV and
   shows the infiltration probability timeline, flagged flows, and attack-stage
   annotations, fully offline**. Every hour on the API gateway is an hour not spent on
   the rollout engine.

2. **Explainability: prefer attention + prediction-residual attribution over SHAP on
   deep nets.** SHAP on an LSTM is slow and awkward; the PS explicitly accepts
   attention. A world model gives something better for free: *which features diverged
   most between predicted and observed state* ("TTL variance and SYN rate departed
   from the forecast trajectory") — honest and native. Use SHAP where it's trivially
   easy: the LR baseline.

3. **Evaluation honesty on lead time.** CIC-IDS2018 has only a handful of attack
   episodes across ~10 days:
   - Evaluate **per-episode**, report per-episode lead times, avoid aggregate
     significance claims.
   - **Exclude in-attack windows from forecast credit** — predicting "attack
     continues" after onset is trivial via autocorrelation and inflates results.
   - Add **calibration** (reliability diagram / Brier score) — cheap, and it
     substantiates the word "probability."
   - Report **false alarms per unit time** (already planned — keep; it's the SOC-real
     metric).

4. **Make the graph part of the state, not a parallel branch.** Per-window graph
   embeddings as components of `S(t)` aligns with the PS's "represent network state
   using feature vectors **or graphs**" and makes the GNN serve the world model
   instead of competing with it. (Keep the empirical gating either way.)

5. **Rollout-error mitigations to name (and defend in Q&A):** compounding
   autoregressive error in high dimensions is the classic world-model failure mode.
   Standard mitigations: predict in a **latent space** rather than raw features;
   **scheduled sampling** during training; a **direct multi-horizon head** as a
   fallback/companion branch.

6. **Deliverables compliance checklist** (PS-explicit, bake in now):
   - Source code link (GitHub/Drive) with README + setup instructions
   - **Model weights + reproducible training configuration in the repo** (explicitly required)
   - Architecture document (max **2 pages**) — this is where World-Model vocabulary must appear
   - Demo video (max **2 minutes**)
   - Technical presentation (max **5 slides**)
   - Demo runs **fully offline, no cloud APIs**

---

## 5. Alignment Scorecard

| PS requirement | Briefing | Solution slide | Status |
|---|---|---|---|
| World Model / learn `P(S_t+1 \| S_t)` | ✗ (discriminative forecaster) | ✓ (in words) | **Rework architecture** |
| K-step forward simulation | ✗ | ✓ (in words) | **Rework architecture** |
| Flow-level features | ✓ | ✓ | OK |
| Packet-level features (PCAP-derived) | ✗ | ✓ (in words) | **Add PCAP branch** |
| SIH-listed dataset as primary | ✗ (UGR'16 unlisted) | — | **Flip to CIC-IDS2018** |
| ATT&CK stage predicted from future state | ✗ (enrichment only) | ✓ (in words) | **Add stage head** |
| Unseen-attack generalization mechanism | ✗ | ✓ (deviation idea) | **Implement + LOO eval** |
| LR baseline on same features | ✗ | ✓ (mentioned) | **Add to eval plan** |
| Explainability (SHAP/attention) | ✓ (planned) | ✓ | OK — refine technique choice |
| Offline demo app (PCAP/CSV in) | partial (dashboard planned, no PCAP input) | ✓ | **Add PCAP input path** |
| Leakage-free training/eval | ✓✓ (best-in-class) | — | Showcase it |
| Lead-time metric | ✓✓ (original) | — | Showcase it |

**Pattern:** the solution slide evolved in the right direction; the briefing/architecture
must now be updated to actually deliver what the slide promises.

---

*Companion document: `SIH26153_Recommended_Plan.md` — the revised formulation,
architecture, dataset strategy, evaluation plan, and prioritized roadmap.*
