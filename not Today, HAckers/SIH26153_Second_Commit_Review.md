# SIH26153 — Independent Review of the Gap Analysis and Recommended Plan

## Review scope

This document reviews the second commit, `9a097a7` (`docs: add SIH26153 assessment gap analysis and recommended plan`), against:

- the original problem statement in commit `86adb81`;
- `SIH26153_Technical_Handover_Briefing.md`;
- `Proposed solution.jpeg`; and
- the explicit deliverables and evaluation requirements in the problem statement.

The review distinguishes between two questions:

1. Did the gap analysis correctly identify weaknesses in the first proposal?
2. Does the recommended plan turn those findings into a feasible, scientifically defensible solution?

## Executive verdict

The second commit is a substantial improvement. Its central diagnosis is correct: the first proposal describes a temporal attack classifier, while the problem statement asks for a model of network-state transitions with K-step forward simulation. It also correctly restores packet-derived features, ATT&CK-stage output, unseen-attack evaluation, the logistic-regression benchmark, and submission-scope discipline.

However, the recommended plan is not yet implementation-ready. Four issues are material:

1. It treats prediction residuals as if they can explain or forecast a future event before the future observation exists.
2. It maps dataset attack classes directly to ATT&CK stages even though attack type and attack stage are not equivalent labels.
3. It assumes CIC-IDS2018 supplies ground-truth progression transitions; it supplies attack scenarios, traffic, logs, and labels, but not a ready-made five-stage progression target.
4. Its “world model” may still be only a point-regression auxiliary head unless it specifies a state, transition distribution, rollout objective, uncertainty, and evidence that the rollout contributes to the final forecast.

Overall assessment:

| Artifact | Verdict |
|---|---|
| Gap analysis | **Strong and mostly correct, with two overstatements and one important evaluation error** |
| Recommended plan | **Directionally good, but needs architectural and label-design corrections before execution** |
| Expected result after the corrections below | **A credible, scoped hackathon plan that directly answers the problem statement** |

## Where Claude got it right

### 1. The first architecture is not the requested world model

This is the most important and most accurate finding in the second commit. The first briefing defines:

```text
X(t-k..t) -> P(attack in [t, t+H])
```

That is a useful temporal forecaster, but it does not itself learn or expose the transition dynamics `P(S(t+1) | S(t))`, nor does it produce a K-step state rollout. Adding a future-shifted label to a classifier does not make it a world model.

Claude was also right to notice that the proposed-solution slide had evolved beyond the briefing. The slide mentions a world model, future states, K-step forecasting, and predicted-versus-observed deviation, while the detailed architecture still terminates in a direct attack-probability head.

### 2. The missing packet-feature pipeline is a real compliance gap

The problem statement says the solution must work with both flow-level and packet-level features. The first briefing mentions packet counts inside aggregates but does not define extraction of the required packet-derived attributes such as TTL variance, TCP window size, fragmentation, payload-size distribution, retransmissions, or scan sequencing.

Adding a PCAP feature-extraction path and extending the UCS schema is therefore correct. Claude also made the sensible scope choice of using a curated PCAP subset rather than trying to process the entire dataset during a hackathon.

### 3. ATT&CK stage must be an output, not only enrichment

The first briefing treats ATT&CK, CAPEC, and CVE/NVD as post-forecast context. Claude correctly identifies that this is insufficient: the problem statement expects a predicted attack stage derived from a predicted future state. Enrichment may remain, but it cannot replace stage prediction.

### 4. The mandatory logistic-regression baseline was correctly restored

The problem statement explicitly asks for logistic regression on the same features and comparison using F1, precision, recall, and false-positive rate. The gap analysis is correct that this was absent from the briefing's evaluation section.

### 5. Leakage, lead time, calibration, and event-level evaluation are strong recommendations

The following should be retained:

- chronological splitting with purge and embargo;
- thresholds selected on validation data only;
- per-episode reporting rather than pretending that thousands of overlapping windows are independent attacks;
- false alarms per unit time;
- calibration and Brier score for probability claims;
- PR-AUC in the presence of imbalance; and
- rollout-error curves by forecast depth.

The recommendation to keep the original briefing's scientific restraint is also sound.

### 6. Cutting nonessential product layers is the right trade-off

Removing the API gateway, multiple micro-APIs, speculative asset-criticality scoring, model registry, and retraining loop is appropriate for the deadline. These layers do not compensate for a missing transition model, stage output, or reproducible benchmark.

### 7. Gating the GNN on evidence remains correct

A GNN is permitted, not required. Treating graph embeddings as part of state is cleaner than building a parallel product branch merely to claim an LSTM-plus-GNN architecture. Deferring it until a temporal world-model baseline works is the right priority.

## Where the gap analysis needs correction

### 1. An unlisted dataset is not a requirements violation

The gap analysis describes UGR'16 as conflicting with the requirements and marks “SIH-listed dataset as primary” as a requirement. The problem statement instead says to use publicly available datasets “such as” the named examples. That wording is illustrative, not exclusive.

Using CIC-IDS2018 as the primary dataset is still a good recommendation because it offers raw PCAPs, logs, and flow CSVs in one source and is easy to justify to evaluators. The accurate conclusion is:

> UGR'16 is a delivery and packet-modality risk if used alone, not a formal compliance violation merely because it is unlisted.

### 2. The demo requirement is “PCAP or CSV,” not necessarily live parsing of every full PCAP

The plan should support both documented input modes across the prototype, but it does not need to synchronously parse a many-gigabyte capture during a two-minute demo. A bundled small PCAP sample, cached extraction, or an already extracted packet-feature CSV can demonstrate the path honestly.

### 3. Excluding all in-attack windows would discard the progression objective

The recommendation says only pre-onset windows should count and all in-attack windows should be excluded from forecast credit. That is correct for an **attack-onset forecast**, but too broad for this problem statement, which also asks the system to anticipate attacker progression before compromise is completed.

Evaluation should have two explicitly separate tracks:

| Track | Input cutoff | Target | What does not earn credit |
|---|---|---|---|
| Onset forecasting | Before the first malicious event | Attack onset within horizon H | Alerts after onset |
| Progression forecasting | During a known early stage | A later stage or worsening state within K windows | Merely predicting persistence of the current stage |

This separation prevents detection from being mislabeled as onset forecasting without throwing away the more feasible and relevant progression task.

## Problems in the recommended plan

### Critical 1 — Residual scoring is detection after observation, not a future explanation

The plan proposes a deviation score and “prediction-residual attribution” based on the difference between predicted and observed `S(t+1)`. At decision time `t`, the observed `S(t+1)` does not exist. Therefore:

- a residual can identify unexpected behavior **after the next window arrives**;
- it can supply evidence for a current novelty alert based on `S(t) - predicted_S(t)`; but
- it cannot explain why the model currently predicts an attack in an as-yet unobserved future window.

The plan currently mixes these three meanings in its core outputs, explainability section, and novelty phase.

**Required correction:** expose two different evidence types:

1. **Current deviation evidence:** compare observed `S(t)` with the one-step forecast made at `t-1`; use this for online novelty detection.
2. **Forecast explanation:** attribute the future hazard/stage output to the observed history `S(t-k..t)` using feature ablation, integrated gradients, or another validated model-output attribution method.

For a future rollout, it is valid to show which simulated features change most, but those are predicted trajectory changes, not observed residuals.

### Critical 2 — Attack class is not the same as ATT&CK stage

The proposed table maps broad CIC-IDS2018 classes directly to stages. Several mappings are not defensible without event-level evidence:

- “Infiltration” does not automatically mean Lateral Movement.
- Brute force may support Credential Access, Initial Access, or both depending on whether access succeeds.
- Botnet traffic may support Command and Control, but the label alone does not prove every flow is a C2-stage observation.
- DoS/DDoS belongs naturally to ATT&CK Impact. Mapping it to a “nearest” C2 or Exfiltration stage would create a knowingly false label.

MITRE defines tactics as adversary goals, not a universally linear kill-chain stage sequence. The problem statement uses a simplified five-output vocabulary, but the training labels still need a documented evidence rule.

**Required correction:** create an auditable mapping manifest with `source_label`, time interval, supporting telemetry, ATT&CK tactic(s), confidence, and mapping rationale. Permit `Unknown/Other` and multi-label outputs internally. For the five required display categories, either:

- restrict the stage experiment to scenarios with defensible mappings; or
- show unsupported categories as `Unknown/Impact`, with an explicit limitation.

Do not force DDoS into C2 or Exfiltration.

### Critical 3 — CIC-IDS2018 does not provide ready-made progression ground truth

The official dataset page confirms that CSE-CIC-IDS2018 provides raw PCAPs and logs per day, flow CSVs, seven attack scenarios, and detailed scenario descriptions. That supports feature extraction and timeline construction. It does **not** by itself establish that the dataset contains enough independent, multi-stage attacker journeys to train and test a five-stage progression model.

The recommended plan's statement that attack-timeline annotations “give ground-truth state transitions” is too strong. Consecutive traffic states can be constructed from timestamps, but ATT&CK-stage transitions must be derived and validated separately. The scheduled, profile-generated attacks may also allow a model to learn time, host, or scenario artifacts instead of genuine precursors.

**Required correction:** put a dataset feasibility gate before the dataset “priority flip.” Verify:

- PCAP/CSV timestamp and flow alignment;
- number of independent attack episodes and onsets;
- observable pre-onset history for each episode;
- whether a scenario contains more than one defensible progression stage;
- whether IP address, day, time-of-day, or capture-machine identity leaks the schedule; and
- whether the required packet features can be reproduced consistently from the chosen PCAP subset.

If the dataset cannot support pre-onset prediction, make progression from early observed behavior the primary claim and report onset forecasting as inconclusive rather than manufacturing evidence.

### Critical 4 — The dynamics model is underspecified as a probability model

The problem statement asks for `P(S(t+1) | S(t))`. The recommended plan allows ordinary regression over a state vector or latent. A point estimate trained with mean-squared error is a transition predictor, but it does not represent the uncertainty or multimodality implied by a distribution over possible future states.

The plan also does not define how per-step attack probabilities become a coherent probability of infiltration within K windows.

**Required correction:** specify at least one implementable probabilistic formulation:

- an encoder maps the observed state window to latent `z(t)`;
- the dynamics head predicts a distribution for `z(t+1)` or `S(t+1)` (for example, Gaussian mean and variance, quantiles, or a small ensemble if time is limited);
- training includes one-step loss and an explicit multi-step rollout loss;
- a discrete-time hazard head estimates the conditional probability of first onset or next-stage transition at each future step; and
- cumulative event probability is derived coherently from the hazards, rather than informally combining independent per-step scores.

If only deterministic regression is feasible, call it a deterministic approximation to the transition dynamics and do not claim that it learns a full probability distribution.

### High 1 — The two-head design can become a checkbox world model

The direct forecast head is a pragmatic safety net, but it can also make the dynamics head decorative. If the dashboard's probability comes only from the direct head, the delivered system is still primarily the original discriminative forecaster plus an auxiliary reconstruction task.

Require these ablations:

- direct head alone;
- dynamics rollout plus trajectory/hazard head;
- joint model; and
- static and lagged-feature baselines.

The final claim should state which branch produced each displayed result. The world-model claim is justified only if the state rollout is actually executed, evaluated, and used for the trajectory or stage output.

### High 2 — Aggregated UCS cannot produce “flagged flows” by itself

The plan aggregates traffic into one state vector per window, then promises a demo with flagged flows. Once individual flows are reduced to counts and statistics, the model cannot identify which flow caused a prediction unless the pipeline preserves an entity/flow-level path.

Add one of the following:

- retain the raw flow rows linked to every state window and score them with the same downstream classifier;
- use a set/attention encoder that preserves per-flow contribution indices; or
- describe the output honestly as “flagged windows and supporting flows selected by a separate attribution rule.”

### High 3 — Missing packet features at CSV inference time create modality shift

The plan says absent packet features should remain missing. That is scientifically honest, but not sufficient. A model trained with packet features may fail when a flow-only CSV is supplied.

Define two supported contracts:

1. PCAP input -> both flow and packet features extracted by the project pipeline.
2. CSV input -> either a documented full UCS schema including packet aggregates, or a separately trained flow-only model.

Use feature-presence masks and evaluate both modes. Do not silently feed an all-missing packet family to a model that never saw that pattern during training.

### High 4 — Attention weights should not be presented as faithful feature attribution

Attention is not “free” with an LSTM, and raw attention weights are not guaranteed to measure causal feature importance. They may be shown as model internals, but the explanation should be validated with deletion/ablation tests and paired with a prediction-specific attribution method.

For the logistic-regression baseline, standardized coefficients are simpler and more faithful than adding SHAP merely for presentation. SHAP remains optional.

### High 5 — Leave-one-attack-type-out does not validate stage recognition of an unseen class

Holding out an attack family is a good generalization test for generic risk or novelty detection. A supervised stage head cannot be expected to assign a trustworthy stage to behavior whose relevant tactic was absent from training, especially when attack-family and stage labels are conflated.

Report separate results for:

- unseen-family risk detection;
- novelty scoring after a new observation arrives; and
- stage classification only where the stage label is supported in training or by a zero-shot rule declared in advance.

Thresholds and preprocessing must be fitted without the held-out family.

### High 6 — The benchmark needs stronger controls

The mandated current-window logistic regression is necessary but weak. A temporal model naturally has more information. Add:

- logistic regression on flattened lagged windows;
- persistence/last-state baseline for state rollout;
- direct LSTM/GRU forecast baseline; and
- dynamics rollout model.

Use identical train/validation/test time boundaries and input availability. Set purge and embargo width to cover at least the maximum lookback plus forecast horizon. Report confidence intervals across episodes or seeds without treating overlapping windows as independent samples.

### Medium — Scheduled sampling is an experiment, not a guaranteed fix

Scheduled sampling is one possible rollout mitigation, but it has known objective-consistency concerns. Keep it behind an ablation alongside teacher forcing plus multi-step loss, input-noise training, and direct multi-horizon prediction. Do not label it a required standard remedy before measuring it.

### Medium — CAPEC/CVE enrichment remains low priority

Without host software inventory, version evidence, or vulnerability telemetry, CVE/NVD attachment can easily become generic or misleading. It should remain optional after the required stage, forecast, explanation, and benchmark outputs work.

### Medium — The second commit leaves two conflicting sources of truth

The recommended plan says it replaces the briefing's roadmap, but the second commit only adds companion documents. The original briefing still presents the direct classifier, enrichment-only ATT&CK role, product-heavy architecture, and old phase order without a superseded notice. A team member can follow the repository accurately and still implement the wrong plan.

After review approval, either update the briefing to the consolidated architecture or add a prominent superseded banner linking to one authoritative plan. Maintain a single requirement-to-component-to-test matrix rather than three documents that disagree.

## Assessment of each gap in the second commit

| Claude's gap | Assessment | Required adjustment |
|---|---|---|
| World model absent | **Correct and critical** | Specify probabilistic state, rollout loss, uncertainty, and ablations |
| Packet-level branch absent | **Correct and critical** | Define PCAP/CSV modality contracts and demo latency strategy |
| Dataset strategy conflicts with requirements | **Partly correct** | CIC-IDS2018 is pragmatic, but unlisted datasets are allowed; add feasibility gate |
| ATT&CK must be predicted, not only enriched | **Correct diagnosis** | Replace class-to-stage shortcuts with evidence-backed, auditable labels |
| Unseen-attack plan absent | **Correct diagnosis** | Separate current residual detection from future forecasting and stage prediction |
| Logistic-regression benchmark absent | **Correct** | Add lagged LR and persistence controls for a fair scientific comparison |

## Improved architecture

```text
PCAP -----------------> project-owned flow + packet extraction ----+
                                                                  |
Full-schema CSV -----------------------------------------------> validation
                                                                  |
                                                                  v
                                                     windowed UCS S(t)
                                             + masks + retained flow index
                                                                  |
                                      observed history S(t-L+1 ... t)
                                                                  |
                                            encoder -> latent state z(t)
                                                     /             \
                                                    /               \
                         probabilistic dynamics p(z(t+1)|history)   direct baseline head
                                      |                             P(event in H)
                             K-step state rollout
                                      |
                         per-step hazard + supported stage head
                                      |
                   probability trajectory + stage trajectory + uncertainty

Separately, when S(t+1) arrives:
predicted S(t+1) vs observed S(t+1) -> current novelty/deviation alert

Explanations:
- forecast: ablation/integrated-gradients evidence over observed history
- novelty: observed one-step residual by feature/entity
- flagged flows: resolved through the retained flow-to-window index
```

This architecture keeps residual detection useful without pretending the future has already been observed.

## Revised execution plan for the September 20 deadline

### Gate 0 — September 1-3: prove the data can support the claim

- Download a small CIC-IDS2018 attack-day subset with its PCAP, logs, and CSVs.
- Reconcile timestamps and flow identities.
- Enumerate independent episodes, pre-onset duration, and defensible stage annotations.
- Test leakage by training simple models with and without IP/day/time identifiers.
- Freeze the primary claim as either onset forecasting, progression forecasting, or both.

**Exit criterion:** one versioned data slice can produce aligned states, future targets, and at least one defensible evaluation episode without future leakage. If not, change dataset or narrow the claim before building the model.

### Phase 1 — September 4-7: build the minimum reproducible state pipeline

- Implement flow and required packet features from the same PCAP source.
- Define UCS V2, feature masks, normalization fitted on training only, and retained flow indices.
- Define onset and progression targets separately.
- Create the ATT&CK mapping manifest with confidence and `Unknown/Other` handling.
- Freeze chronological splits, purge/embargo, dataset manifest, seeds, and configuration.

**Exit criterion:** deterministic regeneration of train/validation/test tensors and labels from a documented command.

### Phase 2 — September 8-11: baselines and world-model MVP

- Train current-window logistic regression as the mandated baseline.
- Train lagged logistic regression and persistence baselines.
- Train a small GRU/LSTM encoder with probabilistic next-state head and K-step rollout.
- Add explicit one-step and multi-step rollout losses.
- Add a discrete-time onset/progression hazard head and a supported stage head.

**Exit criterion:** saved weights and a test script produce state forecasts, a K-step probability trajectory, uncertainty, and stage output.

### Phase 3 — September 12-14: honest evaluation and explanation

- Compare static LR, lagged LR, direct temporal head, rollout branch, and joint model.
- Report mandated metrics, PR-AUC, calibration, false alarms per time, rollout error by K, and per-episode lead time.
- Evaluate onset and progression tracks separately.
- Run at least one leave-one-family-out risk/novelty experiment.
- Add forecast attribution and current-residual novelty evidence; test explanation fidelity with feature deletion.

**Exit criterion:** every claim in the slides points to a reproducible table or figure, including negative results.

### Phase 4 — September 15-17: offline demo

- Support a small PCAP sample and a documented CSV input schema.
- Display observed history, K-step state/risk trajectory, uncertainty, supported stage, current novelty score, explanations, and linked supporting flows.
- Show the logistic-regression comparison in the same interface.
- Cache expensive extraction for the two-minute demo while retaining a reproducible live path.

**Exit criterion:** a clean offline-machine run completes from input to visualization without network access.

### Phase 5 — September 18-20: submission package and buffer

- README and one-command demo instructions.
- Versioned configs, weights, data manifest, environment lock, and known limitations.
- Two-page architecture document, five-slide deck, and two-minute video.
- Rehearse Q&A around deterministic-versus-probabilistic dynamics, rollout error, label validity, unseen attacks, and dataset limitations.

### Explicitly optional after the MVP

- GNN/graph embeddings;
- CAPEC and CVE/NVD enrichment;
- Transformer comparison;
- model registry, retraining, monitoring, and multiple APIs.

Do not let any optional item delay a working K-step rollout, valid evaluation, or offline demo.

## Final recommendation

Adopt the second commit's main direction, but do not execute its roadmap verbatim. First correct the residual timing error, replace the ATT&CK shortcut with auditable label design, validate whether CIC-IDS2018 truly supports the intended onset/progression claim, and make the transition model probabilistic and outcome-relevant rather than auxiliary.

With those changes, the proposal preserves the strongest parts of both commits:

- the first commit's leakage discipline, UCS engineering, lead-time focus, and honest positioning; and
- the second commit's world-model alignment, packet branch, stage output, baseline compliance, and aggressive scope control.

That combined plan is stronger than either commit on its own.

## External references used for factual checks

- [CSE-CIC-IDS2018 official dataset page](https://www.unb.ca/cic/datasets/ids-2018.html) — confirms raw PCAP/log organization, generated flow CSVs, and seven attack scenarios.
- [MITRE ATT&CK Enterprise tactics](https://attack.mitre.org/tactics/) — defines tactics as adversary goals and includes Reconnaissance, Initial Access, Credential Access, Lateral Movement, Command and Control, Exfiltration, and Impact as distinct tactics.
- [Jain and Wallace, “Attention is not Explanation,” NAACL 2019](https://aclanthology.org/N19-1357/) — evidence against treating raw attention weights as automatically faithful explanations.
- [Huszar, “How (not) to Train your Generative Model,” 2015](https://arxiv.org/abs/1511.05101) — explains the objective-consistency concern with scheduled sampling.
