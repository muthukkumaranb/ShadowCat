# SIH26153 — Final Gap Analysis & Revised Plan (v3 — Authoritative)
## Consolidated from: Technical Handover Briefing (v1) → Claude Gap Analysis + Plan (v2) → Codex Independent Review → online verification

**Status:** This is the **single source of truth**. It supersedes:
- `SIH26153_Technical_Handover_Briefing.md` (v1 — original architecture)
- `SIH26153_Assessment_Gap_Analysis.md` (v2 — Claude review)
- `SIH26153_Recommended_Plan.md` (v2 — Claude plan)
- `SIH26153_Second_Commit_Review.md` (Codex review — adjudicated below)

**Date:** 2026-09-01 · **Idea-submission deadline:** 2026-09-20

---

## Part A — Adjudication of the Codex Review (verified online)

Every factual claim in the Codex review was checked against primary sources on
2026-09-01. Verdicts:

| Codex claim | Verification | Verdict |
|---|---|---|
| CSE-CIC-IDS2018 provides PCAPs/logs/flow CSVs + 7 scenarios, but **no ready-made five-stage progression ground truth** | UNB CIC official page + AWS Open Data registry confirm structure; independent researchers had to **inject hand-crafted APT campaigns into 2018** to obtain multi-stage scenarios | **Confirmed** |
| DoS/DDoS belongs to ATT&CK **Impact**; tactics are adversary goals, not a linear stage sequence; forcing DoS→C2/Exfiltration is a knowingly false label | MITRE ATT&CK Enterprise tactics list | **Confirmed** — v2's mapping table was wrong on this row |
| Raw attention weights are not guaranteed faithful explanations (Jain & Wallace 2019) | Paper real, correctly characterized. Nuance: Wiegreffe & Pinter's "Attention is not not Explanation" (EMNLP 2019) shows the debate is unsettled | **Confirmed with nuance** — practical guidance (attention as internals + ablation validation) stands |
| Scheduled sampling has objective-consistency concerns (Huszár 2015, arXiv:1511.05101) | Paper real; known biased-objective critique | **Confirmed** — treat as an ablation, not a default remedy |
| UGR'16 is NetFlow-only; unlisted dataset ≠ compliance violation ("such as" wording) | UGR'16 official page: NetFlow v9, anonymized, **PCAPs withheld for privacy**; PS wording verified as illustrative | **Confirmed** — v2 overstated "requirements violation"; correct framing is *delivery/modality risk* |
| Residual scoring at time t cannot use unobserved S(t+1) | Logic, not fact — reviewed | **Correct** — v2 conflated three evidence types; adopted below |
| Two-head design risks a "checkbox world model" | Logic — reviewed | **Correct** — ablations now required |
| Aggregated UCS cannot produce "flagged flows" without a preserved flow index | Logic — reviewed | **Correct** — real catch; adopted below |
| CSV-only inference creates modality shift vs PCAP-trained model | Logic — reviewed | **Correct** — dual input contract adopted |

**New finding from verification (neither review had it):**
CICIDS2017 and CSE-CIC-IDS2018 have **documented labeling and flow-construction
errors** — CICFlowMeter TCP-handling bugs, mislabeled attacks, duplicate flows,
packet misordering, undocumented capture gaps (Engelen et al. 2021; Lanvin et al.
2023). Corrected CICFlowMeter versions and relabeled datasets exist (Engelen's
CNS2022 code release). **Consequences for this project:**

1. Do **not** train on the shipped CSVs uncritically — use the corrected/relabeled
   variants where available, or better, extract features from PCAP with the
   **project-owned pipeline** (which we must build anyway for the packet branch).
2. This is a judge-Q&A differentiator: knowing the known flaws of the de-facto
   standard dataset signals real depth.
3. It reinforces the Gate 0 feasibility check (Part D): label positions and flow
   timestamps must be validated, not trusted.

---

## Part B — Final Gap Analysis (consolidated & corrected)

### B1. Confirmed critical gaps in the original briefing (v1)

1. **Not a world model.** `X(t-k..t) → P(attack in [t,t+H])` is a future-shifted
   classifier. The PS demands learned transition dynamics `P(S(t+1)|S(t))` with
   K-step forward simulation, and explicitly rejects "a static input-output
   classifier." *(All three reviews agree; highest-priority fix.)*
2. **No packet-level feature pipeline.** PS mandates both feature levels and names
   TTL variance, TCP window size, fragment flags, payload-size distribution,
   retransmissions, scan sequencing; demo must accept PCAP (Scapy/PyShark).
3. **ATT&CK as enrichment only.** PS requires *predicted attack stage from the
   predicted future state* — an output head, not post-hoc context.
4. **No unseen-attack mechanism.** PS requires generalization beyond memorized
   signatures.
5. **Mandatory LR baseline missing** from the evaluation section (F1, precision,
   recall, FPR on the same features).
6. **Product-layer overreach** (API gateway, asset criticality, registry,
   retraining) relative to the required offline demo app.

### B2. Corrections to the v2 gap analysis (per Codex, verified)

1. **UGR'16 reframed:** unlisted ≠ violation. Correct statement: *UGR'16 is a
   delivery and packet-modality risk if used alone* (NetFlow-only, no PCAPs), not a
   formal compliance failure. CIC-IDS2018-as-primary stands on pragmatic grounds.
2. **Demo PCAP handling:** "accepts PCAP or CSV" does not require synchronously
   parsing multi-GB captures in a 2-minute demo. A bundled small PCAP sample +
   cached extraction + documented live path is honest and sufficient.
3. **In-attack windows must NOT all be excluded.** v2's rule was right for onset
   forecasting but would delete the *progression* objective the PS also asks for
   ("anticipating attacker progression … before compromise is completed").
   → Two evaluation tracks (Part E).

### B3. New defects in the v2 plan (per Codex, verified/accepted)

1. **Residual timing error (critical).** At decision time t, observed S(t+1) does
   not exist. Three distinct evidence types were conflated:
   - *Current novelty:* observed S(t) vs. the forecast made at t−1 → online
     deviation alert (one step behind, honest).
   - *Forecast explanation:* attribute the hazard/stage output to observed history
     S(t−k..t) via ablation / integrated gradients.
   - *Trajectory evidence:* which **simulated** features change most across the
     rollout — predicted changes, never "residuals."
2. **Attack class ≠ ATT&CK stage (critical).** Direct class→stage mapping is not
   defensible: Infiltration ≠ automatically Lateral Movement; brute force is
   Credential Access and/or Initial Access depending on outcome; botnet *traffic*
   is not per-flow proof of C2 stage; **DoS/DDoS = Impact** — never forced into
   C2/Exfiltration. → Auditable mapping manifest (Part C3).
3. **No ready-made progression ground truth in CIC-IDS2018 (critical).** Scenario
   schedules + labels support timeline construction, not five-stage journeys.
   Scheduled, profile-generated attacks also risk the model learning **schedule
   artifacts** (day, time-of-day, IP, capture host) instead of precursors.
   → Gate 0 feasibility check before any architecture work (Part D).
4. **Dynamics head underspecified as a probability model (critical).** MSE point
   regression is a transition *predictor*, not `P(S(t+1)|S(t))`. → Probabilistic
   formulation + hazard head (Part C1). If only deterministic regression ships,
   call it a *deterministic approximation to the transition dynamics* — never
   claim a learned distribution.
5. **Checkbox-world-model risk (high).** If the dashboard number comes only from
   the direct head, the delivered system is still v1 plus a decorative
   reconstruction task. → Mandatory ablations; every displayed result labeled
   with the branch that produced it.
6. **"Flagged flows" impossible from aggregated UCS alone (high).** → Retain a
   flow-to-window index and score retained flows with a documented attribution
   rule (or say "flagged windows + supporting flows," honestly).
7. **CSV-only inference modality shift (high).** → Two input contracts with
   feature-presence masks; never silently feed an all-missing packet family to a
   model that never saw that pattern.
8. **LOO ≠ stage validation for unseen classes (high).** A supervised stage head
   cannot assign trustworthy stages to tactics absent from training. → Separate
   claims: unseen-family *risk* detection; *novelty* scoring post-observation;
   *stage* classification only where supported.
9. **Benchmark controls too weak (high).** Current-window LR is mandated but easy
   to beat. → Add lagged-feature LR, persistence/last-state baseline, direct
   temporal head; identical splits; CIs across episodes/seeds.

### B4. What survives unchanged from v1 (showcase these)

Leakage rules · chronological splits + purge/embargo · UCS canonical schema with
metadata/feature separation · missing-not-fabricated statistics · lead-time focus ·
fusion-only-if-proven · positioning-honesty table.

---

## Part C — Final Architecture

### C1. Model (probabilistic world model + controlled companion head)

```text
PCAP ────────────► project-owned flow + packet extraction ──┐
                   (corrected-CICFlowMeter-informed logic)  │
Full-schema CSV ───────────────► schema validation ─────────┤
                                                            ▼
                              windowed UCS S(t) + feature-presence masks
                                        + retained flow-to-window index
                                                            │
                              observed history S(t−L+1 … t) │
                                                            ▼
                                     encoder → latent state z(t)
                                        │                     │
                     probabilistic dynamics head        direct forecast head
                     p(z(t+1) | history)                P(event within H)
                     (Gaussian μ,σ / quantiles /        (baseline & safety
                      small ensemble)                    net — labeled as such)
                                        │
                              K-step latent rollout
                     (one-step loss + explicit multi-step loss;
                      scheduled sampling only as an ablation)
                                        │
                     per-step discrete-time hazard head  +  supported-stage head
                                        │
        cumulative onset/progression probability derived coherently from hazards:
                     P(event ≤ K) = 1 − Π(1 − h_k)
                                        │
          probability trajectory + stage trajectory + uncertainty bands

Separately, when S(t+1) actually arrives:
  predicted Ŝ(t+1) vs observed S(t+1) → current novelty/deviation alert (online)
```

**Non-negotiables:**
- The rollout must be *executed, evaluated, and used* for the displayed
  trajectory/stage outputs — otherwise the world-model claim is not made.
- Required ablations: direct head alone · rollout+hazard alone · joint · static
  LR · lagged LR · persistence. Every chart states its producing branch.

### C2. Explainability (three evidence types, never conflated)

| Evidence | Computed from | Method | When shown |
|---|---|---|---|
| Forecast explanation | Observed history S(t−k..t) | Feature ablation / integrated gradients (validated by deletion tests); attention shown only as *model internals* | With every forecast |
| Trajectory evidence | Simulated rollout | Largest predicted feature changes along Ŝ(t+1..t+K) — labeled "predicted" | With K-step view |
| Current novelty | Observed S(t) vs forecast from t−1 | Per-feature/per-entity residual | Online alerts |

LR baseline explanation: **standardized coefficients** (simpler and more faithful
than SHAP-for-presentation; SHAP optional).

### C3. ATT&CK stage label design (auditable manifest)

- Manifest per labeled interval: `source_label · time interval · supporting
  telemetry · ATT&CK tactic(s) · confidence · rationale`.
- Internally allow `Unknown/Other` and multi-label; the five PS display categories
  are a *presentation* layer over defensible labels.
- Fixed rules: DoS/DDoS → **Impact** (displayed as its own category with an
  explicit note that the PS's five-phase list omits it — a documented limitation,
  not a forced mislabel). Brute force → Credential Access, escalating to Initial
  Access only on evidence of success. Infiltration scenario → stage-annotated
  *within* the scenario (exploit phase vs. internal-scan phase) rather than
  blanket "Lateral Movement."
- CAPEC/CVE enrichment: optional, after everything required works; without host
  inventory it easily becomes generic noise.

### C4. Input contracts (modality shift prevention)

1. **PCAP input** → project pipeline extracts flow + packet features (full UCS).
2. **CSV input** → either documented full-schema UCS CSV (incl. packet
   aggregates), or a **separately trained flow-only model**. Feature-presence
   masks trained in; both modes evaluated and reported.

---

## Part D — Dataset Strategy (with feasibility gate)

| Priority | Dataset | Role | Notes |
|---|---|---|---|
| Primary (pending Gate 0) | CSE-CIC-IDS2018 | Train + evaluate | PCAPs + logs + CSVs; PS names it in the expected solution. **Use Engelen-corrected labels/CICFlowMeter or own PCAP extraction; do not trust shipped CSVs blindly** |
| Secondary | CIC-IDS2017 | Fast iteration | Same caveats (documented label errors) |
| Graph track | CTU-13 | Only if GNN branch earns its place | PS-named |
| Generalization study | UGR'16 | Post-MVP only | NetFlow-only; long temporal coverage is its one unique asset |

### Gate 0 — dataset feasibility (must pass BEFORE architecture work)

Verify on a small attack-day subset:
1. PCAP ↔ CSV timestamp and flow-identity reconciliation (known misordering/
   duplicate-flow issues — check, don't assume).
2. Count of independent attack episodes + onsets; observable pre-onset history per
   episode.
3. Whether any scenario supports ≥2 defensible progression stages (infiltration
   day is the best candidate: exploit phase → internal scan phase).
4. **Schedule-artifact leakage test:** train a trivial model with and without
   IP/day/time-of-day identifiers; if they carry the signal, the evaluation design
   must remove/neutralize them.
5. Packet-feature reproducibility from the chosen PCAP subset.

**Exit rule:** freeze the primary claim as *onset forecasting*, *progression
forecasting*, or both — based on what the data actually supports. If pre-onset
prediction is unsupported, progression-from-early-behaviour becomes the primary
claim and onset forecasting is reported as inconclusive. **Never manufacture
evidence.**

---

## Part E — Evaluation Plan (final)

### E1. Two separate tracks (replaces v2's blanket exclusion rule)

| Track | Input cutoff | Target | Earns NO credit |
|---|---|---|---|
| **Onset forecasting** | Strictly before first malicious event | Attack onset within horizon H | Any alert after onset |
| **Progression forecasting** | During a known early stage | Later stage / worsening state within K windows | Predicting mere persistence of the current stage |

### E2. Baselines & controls (identical splits, purge/embargo ≥ max lookback + horizon)

1. Current-window logistic regression *(PS-mandated)*
2. Lagged-window logistic regression *(fairness control)*
3. Persistence / last-state baseline *(for state rollout)*
4. Direct LSTM/GRU forecast head
5. Full world model (rollout + hazard)

### E3. Metrics

- PS-mandated: F1, precision, recall, FPR (vs LR)
- Forecasting: per-episode lead time · false alarms per unit time · PR-AUC ·
  calibration (reliability diagram + Brier)
- World-model-specific: per-feature next-state error (benign vs pre-attack) ·
  **rollout error growth vs depth K** · deviation-score ROC
- Uncertainty: coverage of predictive intervals (if probabilistic head ships)
- Reporting: per-episode with CIs across episodes/seeds; overlapping windows are
  **not** independent samples

### E4. Unseen-attack generalization (three separate claims)

1. Leave-one-family-out **risk detection** (does risk score fire?)
2. **Novelty scoring** once observations arrive (residual channel)
3. **Stage classification** only for tactics represented in training or covered by
   a pre-declared zero-shot rule
- Thresholds and preprocessing fitted without the held-out family.

---

## Part F — Execution Roadmap (Sep 1 → Sep 20)

| Gate/Phase | Dates | Work | Exit criterion |
|---|---|---|---|
| **Gate 0** | Sep 1–3 | Dataset feasibility (Part D); freeze primary claim | One versioned slice yields aligned states, targets, ≥1 defensible episode, no schedule leakage |
| **Phase 1** | Sep 4–7 | Flow+packet extraction from same PCAP source; UCS v2 + masks + flow index; onset & progression targets; ATT&CK manifest; frozen splits/seeds/configs | Deterministic regeneration of train/val/test tensors from one documented command |
| **Phase 2** | Sep 8–11 | LR (current + lagged) + persistence baselines; GRU/LSTM encoder + probabilistic dynamics head; K-step rollout w/ multi-step loss; hazard + supported-stage heads | Saved weights + test script → state forecasts, K-step probability trajectory, uncertainty, stage output |
| **Phase 3** | Sep 12–14 | Full ablation grid (E2); both eval tracks; ≥1 leave-one-family-out run; explanation fidelity via feature deletion | Every slide claim maps to a reproducible table/figure — including negative results |
| **Phase 4** | Sep 15–17 | Offline demo: small PCAP sample + documented CSV schema in → history, K-step risk/state trajectory, uncertainty, stage, novelty, explanations, linked flows, LR comparison. Cached extraction for the 2-min demo, reproducible live path retained | Clean offline-machine run, input → visualization, zero network access |
| **Phase 5** | Sep 18–20 | README + one-command demo; versioned configs/weights/data manifest/env lock/limitations; 2-page architecture doc; 5 slides; 2-min video; Q&A rehearsal | Submission package complete |

**Explicitly optional (must not delay the MVP):** GNN/graph embeddings ·
CAPEC/CVE enrichment · Transformer comparison · registry/retraining/monitoring/
multi-API product layer.

**Q&A rehearsal topics:** deterministic vs probabilistic dynamics · rollout error
compounding · label validity & the documented CIC dataset flaws · class-vs-stage
distinction · unseen-attack claims (which of the three) · dataset limitations ·
which branch produced which number.

---

## Part G — Language & Claims Discipline

Use PS vocabulary: *world model · state-transition dynamics `P(S(t+1)|S(t))` ·
forward simulation · K-step rollout · trajectory converges to an infiltration
state · before the attacker completes the kill chain · supervised dynamics
learning*.

Honesty table (v1's, extended):

| Never claim | Say instead |
|---|---|
| "The rollout is accurate K steps ahead." | "Rollout error grows with depth; we measure it, mitigate in latent space, and pair the rollout with a direct-horizon head." |
| "We learned the full transition distribution." *(if only point regression shipped)* | "A deterministic approximation to the transition dynamics; probabilistic extension specified." |
| "Residuals explain future predictions." | "Residuals provide online novelty evidence; forecasts are explained by attribution over observed history." |
| "The model recognizes stages of unseen attack families." | "Unseen families are flagged by risk and novelty scores; stage labels are only assigned where training evidence supports them." |
| "Attention shows why the model predicted this." | "Attention is shown as a model internal; explanations are validated by deletion/ablation tests." |

---

## Part H — References (verified 2026-09-01)

**Dataset facts**
- [CSE-CIC-IDS2018 official page (UNB CIC)](https://www.unb.ca/cic/datasets/ids-2018.html) — PCAPs/logs/CSVs, 7 scenarios, CICFlowMeter-V3
- [CSE-CIC-IDS2018 on AWS Open Data Registry](https://registry.opendata.aws/cse-cic-ids2018/)
- [UGR'16 official page (NESG, U. Granada)](https://nesg.ugr.es/nesg-ugr16/) — NetFlow v9, anonymized, PCAPs withheld for privacy

**Dataset quality issues (new finding)**
- [Engelen et al., "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study" (2021)](https://www.researchgate.net/publication/353107141_Troubleshooting_an_Intrusion_Detection_Dataset_the_CICIDS2017_Case_Study) — CICFlowMeter TCP bugs, labeling errors; corrected tool + relabeled data
- [Engelen CNS2022 corrected-dataset code](https://github.com/GintsEngelen/CNS2022_Code) — preliminary CSE-CIC-IDS2018 flow-construction errors also reported
- [Lanvin et al., "Errors in the CICIDS2017 Dataset and the Significant Differences in Detection Performances It Makes" (2023)](https://link.springer.com/chapter/10.1007/978-3-031-31108-6_2) — packet misordering, duplicate flows, capture gaps

**Method citations**
- [MITRE ATT&CK Enterprise tactics](https://attack.mitre.org/tactics/)
- [Jain & Wallace, "Attention is not Explanation" (NAACL 2019)](https://aclanthology.org/N19-1357/)
- [Wiegreffe & Pinter, "Attention is not not Explanation" (EMNLP 2019)](https://aclanthology.org/D19-1002/)
- [Huszár, "How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?" (2015)](https://arxiv.org/abs/1511.05101)
