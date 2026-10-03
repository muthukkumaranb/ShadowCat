# ShadowCat Dashboard Restoration & Honesty Verification Report

## Executive Summary
The original 8-page SOC dashboard UI from `origin/main` has been fully restored on `release/judge-ready` while integrating the real, verified backend ML and audit notarization pipelines. Every original layout, section, chart, Cytoscape topology graph, KPI metric, alert view, and navigation element was preserved intact without deleting any visual component. All synthetic, fabricated, or ungrounded values have been replaced with honest data computed by the 37-fold LOEO stacked residual model, real CIC-IDS2018 demo slices, Integrated Gradients attributions, and local SQLite audit logs.

Visual parity was confirmed by side-by-side screenshots of all 8 pages captured under live execution (`origin/main` on port 8501 vs `release/judge-ready` on port 8502) with the SSH brute-force slice loaded.

---

## Visual Comparison Analysis

| Page | Main Screenshot | Release Screenshot | Visual Parity & Differences Observed |
| :--- | :--- | :--- | :--- |
| **01 Telemetry Ingestion** | [`ingestion_main.png`](restore/ingestion_main.png) | [`ingestion_release.png`](restore/ingestion_release.png) | Identical three-card layout (Live Ingestion, Ingestion Buffer, Feed Health). Replaced fake static throughput (142.5k rows/s) and sync mode (PTP v2) with real streaming ingestion metrics and offline slice parser status. |
| **02 Overview** | [`overview_main.png`](restore/overview_main.png) | [`overview_release.png`](restore/overview_release.png) | Identical top metric cards, onset probability gauge, and alert banner. Replaced static 0.816/0.842 metrics with real stacked model onset probability (0.9315 for SSH) and true alert status. |
| **03 Forecast** | [`forecast_main.png`](restore/forecast_main.png) | [`forecast_release.png`](restore/forecast_release.png) | Identical 5-step forecast chart and confidence intervals. Rather than fabricated escalating risk multipliers, displays honest onset probability P(attack within 5 min) with conformal prediction intervals and clear explanatory subtitle. |
| **04 Attack Graph** | [`attack_graph_main.png`](restore/attack_graph_main.png) | [`attack_graph_release.png`](restore/attack_graph_release.png) | Identical Cytoscape canvas and entity risk sidebar. Displays the real network communication topology from the ingested slice instead of hardcoded synthetic nodes (10.0.2.15/10.0.4.21). |
| **05 Alerts** | [`alerts_main.png`](restore/alerts_main.png) | [`alerts_release.png`](restore/alerts_release.png) | Identical triage table and entity threat summary. Populated by real flagged flows ranked by Logistic Regression attribution scores with honest severity classifications. |
| **06 Explainability** | [`explainability_main.png`](restore/explainability_main.png) | [`explainability_release.png`](restore/explainability_release.png) | Identical attribution bar chart, temporal evidence plot, and counterfactual panel. Relabeled 'SHAP' to 'Integrated Gradients' (the actual gradient-based feature attribution method implemented). |
| **07 Validation & Trust** | [`trust_main.png`](restore/trust_main.png) | [`trust_release.png`](restore/trust_release.png) | Identical benchmark comparison table, calibration plot, and LOEO distribution. All figures dynamically loaded from `loeo_37fold_results.json` and `stacked_benchmark_results.json`. |
| **08 Reports & History** | [`reports_main.png`](restore/reports_main.png) | [`reports_release.png`](restore/reports_release.png) | Identical report generation form and historical audit table. Fed by real entries from `reports.db` and the SHA-256 tamper-evident audit chain. |

---

## Detailed Page-by-Page Element & Data Source Audit

| Page Name | Visual Elements Kept | Live Data Source | Items Shown as 'Not Available' / Honesty Notes & Why |
| :--- | :--- | :--- | :--- |
| **01 Telemetry Ingestion** | Ingestion source radio, throughput KPI metrics, packet distribution chart, network interface health status | Real slice ingestion (`frontend/demo_data/`), `backend/predict.py` ingestion buffer | 'PTP v2 / Hardware Timestamping' replaced with 'NTP System Time Sync' as hardware PTP NIC is not present on host. Mock Kafka feed labeled as offline demo mode. |
| **02 Overview** | Model onset risk gauge, threat level banner, 4 top-level KPI metrics, active stage indicator, recent alerts table | `stacked_model` detection probability, `stage_predictions`, `flagged_flows` | Hardcoded lead time (+3.5 min) replaced with honest '5-minute horizon detection'. Active MITRE tactic shows 'No stage determined' when confidence is below gating threshold. |
| **03 Forecast** | Multi-step risk trajectory chart, conformal prediction bounds, stage progression timeline, forecast summary | Conformal calibrated interval from `conformal_calibration_residuals.json`, onset probability | Individual per-minute escalation removed; all steps t+1..t+5 display the constant window-level probability P(attack within next 5 min) with explicit notation explaining single-probability forecast. |
| **04 Attack Graph** | Interactive Cytoscape network graph, layout selectors, node inspection panel, host risk ranking | Real flow graph topology extracted from active flow records | Synthetic 5-host enclave story replaced with actual IP communication graph of active traffic slice. Nodes with unobserved telemetry display 'Telemetry not available for uninstrumented IP'. |
| **05 Alerts** | Filterable alerts table, severity breakdown badges, host threat ranking, MITRE tactic mapping | Backend `flagged_flows` ranked by LR attribution scores, Enterprise ATT&CK v19.2 offline database | Unmapped flows display 'No MITRE technique mapped' rather than synthetic T0000/TA0000 placeholders. Fixed risk scores replaced with dynamic flow-level anomaly scores. |
| **06 Explainability** | Top feature attributions horizontal bar chart, temporal attribution heatmap, counterfactual recommendations | Backend Integrated Gradients computation (`attributions`), feature standard deviations | Relabeled 'SHAP / Shapley values' to 'Integrated Gradients' to honestly represent the gradient path integration used. Uncalculated interaction effects marked as 'Not evaluated'. |
| **07 Validation & Trust** | Model benchmark comparison table, 37-fold LOEO F1 boxplot, conformal reliability diagram, audit chain verification status | `frontend/models/loeo_37fold_results.json`, `evaluation/benchmark/stacked_benchmark_results.json` | Replaced synthetic 0.816 baseline with verified 37-fold LOEO mean onset F1 (0.835) and benchmark metrics. Fabric notarization clearly notes 'SHA-256 Fallback' when Docker Fabric peer is unavailable. |
| **08 Reports & History** | Incident report generator, historical incident log table, export JSON/CSV buttons, blockchain notarization verification | Local SQLite database (`backend/reports.db`), runtime tamper-evident audit ledger | Fabric block transaction IDs show 'SHA-256 Notarization (Local Engine)' when Fabric daemon is not running. Historical entries reflect genuine executed predictions. |

---

## Audit Chain Check 5 Status
Verified on clean runtime directory (`$env:SHADOWCAT_RUNTIME_DIR = "$freshDir"`):
```text
======================================================================
TAMPER-EVIDENT AUDIT CHAIN VERIFICATION
Total entries in chain: 3
======================================================================
AUDIT CHAIN VALID — all entries intact, no tampering detected.
[i] Docker: not found — Fabric notarization will use SHA-256 fallback (this is fine)

  [0] model_checkpoint       | 0fadd135cdf116bc... | LSTM world model (gaussian_next_state_best_v3_packetcov.pt) loaded into predict pipeline (SHA-256 fallback)
  [1] prediction_lineage     | d5c543223f0b2aa8... | Prediction lineage lin_W_14-02-2018_20180214_014400_04289303 [LOW] (SHA-256 fallback)
  [2] forecast_payload       | 80ebaa8d4f4046be... | Inference forecast for W_14-02-2018_20180214_014400

Summary: Total entries in chain: 3 (forecast_payload entries: 1)
```
