# SIH26153 — Impact of Architecture v2 on the Team's Phase 1 → Phase 2 Progression
## Analysis of the 2026-09-02 status report against `SIH26153_Architecture_v2.pdf` and the v3 Final Plan

**Inputs:** `SIH26153_Team_Status_2026-09-02.md` (team report, verbatim) · `SIH26153_Architecture_v2.pdf` · `SIH26153_Final_Gap_Analysis_and_Plan.md`
**Date:** 2026-09-03

---

## 1. Bottom Line

**Phase 1 survives almost entirely.** Roughly 80% of the frozen artifacts map directly
onto v2, and in three places the team is *ahead* of the plan: they ran the
schedule-artifact leakage test as a proper leave-one-episode-out (LOEO) protocol,
fitted the scaler on train only, and produced the ATT&CK mapping manifest and a
CICFlowMeter sanity check without being asked.

**Phase 2 as written builds the v1 architecture.** ML1's deliverable —
`predict(window) → {risk_score, attention_weights}` from an LSTM + GNN + fusion
stack — is the discriminative classifier the problem statement explicitly rejects
("not a static input-output classifier") and that v3 replaced. There is no
next-state head, no rollout, no hazard head, no stage head. The architecture
document is scheduled to describe a "World Model, K=1 scope" — a claim the shipped
model would not support.

**The divergence is concentrated in one work item and one document**, not spread
across the team. Realignment cost is roughly: half a day of Data Engineering (forecast
targets), one day of ML1 (replace fusion work with next-state head + rollout + hazard),
and small additions for ML2 (lead time, lagged LR, stage assignment). The GNN fusion is
the natural scope cut — v2 already made it optional and the team's own fallback rule
("LSTM-only if fusion fails") points the same way.

**Timeline caveat.** The report targets submission on **Sep 5**; the SIH portal deadline
is **Sep 20**. This analysis assumes Sep 5 is an internal checkpoint (institute-level
round) and gives a two-tier plan: a minimum viable world model by Sep 5, the rest of v2
by Sep 20. If Sep 5 is the hard deadline, ship tier one and state the limitations.

---

## 2. Phase 1 Artifacts — What Stands, What Needs Adding

| Frozen artifact | v2 status | Action |
|---|---|---|
| `ucs_windows.parquet` | ✅ Matches UCS S(t) | Verify it excludes IP identity and time-of-day from model inputs (metadata columns only). Add **feature-presence masks** if CSV demo rows lack packet features (§4, CSV path). |
| `ucs_graph_edgelists.parquet` + `node_lookup.parquet` | ✅ Reusable | Serves two v2 roles: (a) the optional communication-graph builder, (b) the **flow-to-window index** needed for "flagged flows". Keep even if the GNN is cut. |
| `scaler_params.yaml` (train-only) | ✅ Exactly v2 §3.3 | None. Say so in the architecture doc — it is a judge-facing strength. |
| `attack_tactics_mapping.yaml` | ✅ Matches target-construction block | Verify the rules: DoS/DDoS → Impact (own category), brute force → Credential Access unless success evidenced, infiltration split by phase (already done: Compromise / Portscan), confidence + rationale columns, Unknown/Other allowed. |
| `cic_sanity_check.md` | ✅ Matches corrected-CICFlowMeter note | Cite Engelen 2021 / Lanvin 2023 in the doc; this is a differentiator. |
| `loeo_fold_results.csv` (38 folds) | ✅ Better than v2 asked for | Reuse the harness for the Phase 2 gate — but on **forecast targets** (§3). |
| `VALIDATION_REPORT.md`, `gate0_leakage_report.md` | ✅ | Add the Gate 0 exit decision (§3.3). |
| **Missing:** forecast targets | ✗ v2 target-construction block | Add `y_onset(t)` (onset within next H windows, defined on pre-onset windows only), `S(t+1)` (row shift), `y_stage(t+k)` from the manifest. This is a parquet shift — hours, not days. **Blocks ML1.** |
| **Missing:** purge / embargo on the chronological split | ✗ v2 §3.3 | The report mentions a chronological test set and train-only scaler; purge/embargo width (≥ lookback + horizon) is not mentioned. Verify or add. |
| **Missing:** Gate 0 exit decision | ✗ v2 Gate 0 exit rule | Freeze the primary claim (§3.3). |

---

## 3. Gate 0 — Reading the Results Under v2

### 3.1 What the team proved
Set A (traffic + packet) beats Set B (schedule-only) on LOEO mean F1 — a real, honest
result, and the "conditional pass" language is exactly right.

### 3.2 What the numbers also say
- **Schedule-only recall is 0.98.** The clock alone finds nearly every attack window.
  For *forecasting*, the clock is an even stronger shortcut: "attack in the next H
  windows" is trivially predicted by time-of-day on a scheduled dataset.
  → The Phase 2 mandatory gate (beat schedule-only on LOEO) must be run on the
  **forecast targets**, not on current-window labels. Otherwise a model that learned
  the schedule passes the gate.
- **The Gate 0 test measured detection, not forecasting.** F1 on in-attack windows
  says behaviour distinguishes attack from benign; it does not yet say behaviour
  *precedes* attacks. Pre-onset history per episode is not reported — it is the one
  Gate 0 item still open.
- **Attacker IPs are fixed in CIC-IDS2018.** If any IP-derived feature is a model
  input, the model learns attacker identity, not behaviour. Confirm exclusion in the
  leakage report.

### 3.3 The exit decision the data implies
| Track | Episodes available | Verdict |
|---|---|---|
| **Onset forecasting** | SSH-Bruteforce (multi-episode, clear pass), DDoS-LOIC-UDP, Botnet (multi-episode, conditional) | **Primary claim.** LOEO-testable; report per-episode lead time. |
| **Progression forecasting** | Infiltration only: Compromise → Portscan, **one episode** | **Case study, n = 1.** Demonstrate qualitatively, label explicitly as single-episode. Do not cross-validate or claim generalization. |
| DDoS-HOIC | 1 episode, 8 windows | Exclude from claims; document. |

Freeze this in `VALIDATION_REPORT.md` before ML1 starts.

---

## 4. Phase 2 Plan — Line-by-Line Impact

| Status-report item | v2 requirement | Impact | Action |
|---|---|---|---|
| ML1: "retrain LSTM+GNN+Fusion" | Encoder → dynamics head → K-step rollout → hazard + stage heads; GNN optional | **High.** Builds the rejected v1 classifier; fusion consumes the day that the world model needs. | Cut fusion now (their own fallback). Spend Thursday on next-state head + rollout + hazard (§5). |
| ML1: `predict(window) → {risk_score, attention_weights}` | `predict(history) → {probability trajectory (K), stage trajectory, future state trajectory, novelty, attribution}` | **High.** Backend contract changes. | Redefine the interface Thursday morning before either side codes (§5.3). |
| ML1: "sanity-check attention weights" | Attention as internals only; named attribution (IG / feature ablation) validated by deletion | **Medium.** Attention alone fails the "black-box not acceptable" bar. | Implement feature-group ablation over observed history (small). Their fallback "static feature importance" is closer to acceptable than attention. |
| Arch doc: "World Model, LOEO, K=1 scope" | Rollout K ≥ 3 with rollout-error-vs-K curve; honesty table | **High.** K=1 with no dynamics head is a classifier called a world model; a reviewer will catch it. | Ship a next-state head + K=3 rollout; call it a "deterministic approximation to the transition dynamics" (v3 honesty rule). |
| ML2: LR + persistence, F1/P/R/FPR | + lagged-LR control; + per-episode lead time; + rollout error vs K | **Low.** Additions are small. | Add lagged-LR (one more `fit`); compute lead time on pre-onset windows per episode. |
| Target: "beat 0.9274 F1 (tree) and 0.8884 (schedule) on LOEO" | Forecasting proof = lead time on pre-onset windows + schedule gate on forecast targets | **Medium.** The target is a detection target. | Keep it as the sanity floor; add "positive median lead time on SSH-Bruteforce episodes; beats schedule-only on y_onset LOEO" as the forecasting target. |
| Backend: CSV primary, PCAP minimal sample | Dual input contract; CSV = corrected-CICFlowMeter schema; flow-only model or masks if packet features absent | **Medium.** Set A was trained *with* packet features. A CIC CSV has none → modality shift at demo time. | Either demo on a full-schema UCS CSV (packet aggregates included — the team already computes them), or train a flow-only model for the CSV path. Decide Thursday. |
| Backend: dashboard shows risk score | Probability trajectory + uncertainty, stage, future trajectory, driving features, flagged flows, novelty, LR comparison | **Medium.** More panels, same data plumbing. | Prioritize: trajectory (K bars), stage label, driving features, LR comparison. Uncertainty bands only if an ensemble ships; else document. |
| ML2 owns ATT&CK mapping | Stage head over predicted state | **Low.** Manifest exists. | Minimum viable: stage classifier trained on current-window stage labels, applied to rollout states ẑ(t+k). |
| Fallback: "LSTM-only if fusion fails" | GNN optional | **None** — aligned. | Make it the default, not the fallback. |
| Fallback: "PCAP minimal sample" | Bundled small PCAP + cached extraction | **None** — aligned. | Keep the live extraction path in the repo even if the demo uses the cache. |

---

## 5. Minimum Viable World Model for Sep 3–5

Everything below fits on top of the frozen UCS with the targets from §2 added.

### 5.1 Model (ML1, Thursday)
1. **Encoder:** GRU over the last L windows → z(t). (Already planned.)
2. **Next-state head:** Ŝ(t+1) = MLP(z(t)), MSE on the scaled UCS vector.
   Deterministic — label it "deterministic approximation to p(S(t+1)|S(t))".
3. **Rollout:** feed Ŝ back through the encoder for K = 3 (5 if time permits).
   Log per-step error vs k on the test split — this single curve is the world-model
   evidence.
4. **Onset hazard head:** h_k = σ(MLP(ẑ(t+k))) on each rolled state; cumulative
   P(onset ≤ K) = 1 − Π(1 − h_k). Train on `y_onset`.
5. **Direct head (baseline):** P(onset within H) straight from z(t). Keep it, label it.
6. **Stage head:** classifier over rolled states using manifest labels (ML2 supplies
   the label column).
7. **Novelty:** |S(t) − Ŝ(t)| from the previous step's prediction — free once step 2
   exists.
8. **Attribution:** feature-group ablation over the observed history (zero a group,
   record Δ in cumulative probability). Attention weights exposed as internals only.

### 5.2 Evaluation (ML2, Thursday–Friday)
- Baselines on identical splits: LR current-window · LR lagged · persistence ·
  direct head · rollout + hazard · joint.
- Metrics: F1 / precision / recall / FPR (PS), per-episode **lead time** on the onset
  track (pre-onset windows only), PR-AUC, rollout error vs K, schedule-only gate on
  `y_onset` under LOEO.
- Progression: one qualitative infiltration case study (Compromise → Portscan
  trajectory), labeled n = 1.

### 5.3 Backend contract (agree Thursday 9 am)
```
predict(history: [L × F]) -> {
  probability_trajectory: [K],        # cumulative P(onset ≤ k)
  stage_trajectory:       [K],        # manifest labels
  future_states:          [K × F],    # Ŝ(t+1..t+K), for the trajectory panel
  direct_baseline:        float,      # labeled "baseline comparison"
  novelty_score:          float,      # |S(t) − Ŝ(t)| from previous step
  driving_features:       [(name, Δ)],# ablation over observed history
  source_branch:          str         # which head produced each number
}
```

### 5.4 Cut order if Thursday slips
GNN fusion (already cut) → K = 3 down to K = 2 → stage head to manifest lookup on the
current window (documented as limitation) → uncertainty bands → live PCAP extraction.
**Never cut:** next-state head, rollout error curve, hazard trajectory, lead-time
metric, LR comparison, source-branch labels.

---

## 6. Revised Sep 3–5 Plan

| When | DE | ML1 | ML2 | Backend | Flex1 / Flex2 |
|---|---|---|---|---|---|
| **Thu AM** | Add `y_onset`, `S(t+1)`, `y_stage` columns; confirm purge/embargo + IP/time exclusion; write Gate 0 exit decision | Agree predict() contract; drop fusion; build GRU + next-state head + rollout | Retrain LR (current + lagged) + persistence on new targets; supply stage-label column | Build panels against the new contract with mocks | Flex1: start arch doc from v2 diagram + honesty table |
| **Thu PM** | Standby; flow-only model data slice if CSV path needs it | Hazard head + direct head; rollout-error-vs-K curve; first LOEO run on `y_onset` | Lead-time computation per episode; schedule-only gate on `y_onset`; comparison table | Trajectory + stage + driving-features panels | Flex2: slide skeleton with placeholders |
| **Fri** | CSV schema doc for the demo input | Stage head; novelty; ablation attribution; hand off predict() | Ablation grid table; infiltration case study figure | Swap mocks for real predict(); LR comparison panel; flagged flows via edge list | Arch doc final (2 pp); slides with real numbers; video script |
| **Sat** | — | Freeze weights + config + seeds | Freeze tables | Offline run test on a clean machine | Record video; rehearse Q&A (§8); submit |

---

## 7. If Sep 20 Is the Real Deadline — What Sep 6–20 Adds

In v2 priority order: probabilistic dynamics head (Gaussian μ,σ or a 3-model
ensemble → uncertainty bands + interval coverage) → progression hazard `h_k^stage` →
Integrated Gradients replacing group ablation, with deletion-test validation →
leave-one-attack-family-out risk/novelty runs → flow-only model for the CSV path →
live PCAP extraction demo → GNN embedding inside S(t), adopted only if it beats the
temporal-only model on LOEO.

---

## 8. Claims Discipline for the Doc, Slides, and Q&A (given what ships Sep 5)

| Say | Do not say |
|---|---|
| "Learns a deterministic approximation to the state-transition dynamics and rolls it out K = 3 steps; rollout error vs K is reported." | "World model, K = 1." / "Learns p(S(t+1)\|S(t))." |
| "Onset forecasting is the primary claim, validated leave-one-episode-out on SSH-Bruteforce; DDoS-LOIC and Botnet are conditional." | "Generalizes across attack types." |
| "Progression is shown on the single infiltration episode as a case study." | "Predicts attack progression." |
| "Behavioural features beat a schedule-only baseline on forecast targets under LOEO — conditionally; the risk is documented." | "Schedule leakage is eliminated." |
| "Driving features via feature-group ablation over observed history; attention weights shown as model internals." | "Attention weights explain the prediction." |
| "Stage labels come from an auditable manifest; DoS/DDoS is reported as Impact." | Any DoS → C2 / Exfiltration mapping. |

The team's own "Critical Rule" — cut scope, don't extend the timeline — is
compatible with all of this, provided the cut is the GNN fusion and not the world
model itself.
