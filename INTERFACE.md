# SHADOWCAT — Frontend/Backend Interface Contract (`INTERFACE.md`)

**Target:** NTRO PS 6153 — AI-based Network Attack Forecasting from Network Traffic Data  
**Scope:** Specification of all typed data functions in `data_provider.py` consumed by the UI layer.  
**Audience:** Backend / ML / DE Teammate implementing live model inference and telemetry extraction.

---

## 0. Critical Integration Rules & Cautions

> [!CAUTION]
> **CRITICAL PROVENANCE CAUTION:**  
> Do **not** place the checkpoint file at the path in `CHECKPOINT_PATHS` until the corresponding accessor function has been fully rewritten to load and return real data from it.  
> Placing the file early will silently trigger auto-detection and remove the `[MOCK]` badge while the function still returns mock values. That causes a severe data-provenance violation.

> [!NOTE]
> **HTML Rendering Note:**  
> `get_mock_badge_html(key)` returns a raw HTML string with inline styling. It is designed to be rendered via `styles.render_html(...)` (which internally sets `unsafe_allow_html=True`). Never print it as raw markdown without `unsafe_allow_html=True`, or it will render as escaped text.

> [!IMPORTANT]
> **Zero UI Modification Required:**  
> Your entire integration task consists of replacing the internal code of the functions in `data_provider.py` with your real feature extraction and model inference pipelines (`predict()`). You do **not** need to modify any files in `views/` or `components/`.

---

## 1. Provenance Management & Checkpoint Auto-Detection

`data_provider.py` maintains an automated check against `CHECKPOINT_PATHS`. If a checkpoint artifact exists at the given relative path, `is_using_mock_data(key)` automatically returns `False`, removing the `[MOCK]` badge from that specific UI component:

| Key | Expected Checkpoint Path | Description |
| :--- | :--- | :--- |
| `forecast_trajectory` | `models/world_model.pt` | PyTorch sequence model (LSTM / Transformer) checkpoint |
| `comparison_table` | `models/baseline_benchmarks.json` | JSON containing empirical evaluation metrics vs baselines |
| `host_risk_graph` | `models/gnn_topology.pt` | Graph Neural Network (GraphSAGE) model checkpoint |
| `attributions` | `models/integrated_gradients.npz` | Pre-computed or dynamic feature attribution weights |
| `novelty_score` | `models/novelty_detector.pt` | Latent space reconstruction / Mahalanobis novelty head |
| `flagged_flows` | `models/flow_classifier.pt` | Flow scoring module identifying top anomalous flows |
| `validation_data` | `models/loeo_37fold_results.json` | Quantitative 37-fold LOEO cross-validation results |
| `mitre_data` | `models/mitre_mapper.json` | Stage projection transition matrix |
| `analysis_metadata` | `models/telemetry_manifest.json` | Sensor tap and session ingestion manifest |

If you are not using checkpoint files (e.g. running a live in-memory service), manually set `MOCK_STATUS[key] = False` in `data_provider.py` once that function returns real data.

---

## 2. Function Specifications & Schemas

### 2.1 `get_forecast_trajectory(window_id: str = None) -> dict`
Returns multi-step forward simulation of network state risk across horizons $K=1..4$.

* **Consuming UI File:** `views/01_Forecast.py` & `components/forecast.py`
* **Input:** `window_id` (str, optional): Target 60s sliding window identifier.
* **Return Schema:**
```json
{
  "window_id": "WIN-20260909-01",
  "horizons": [1, 2, 3, 4],
  "labels": ["t+1 (1 min)", "t+2 (2 min)", "t+3 (3 min)", "t+4 (4 min)"],
  "risk": [0.21, 0.38, 0.67, 0.81],
  "stage": ["Reconnaissance", "Initial Access", "Lateral Movement", "Impact"],
  "tactic_id": ["TA0043", "TA0001", "TA0008", "TA0040"],
  "lead_time": ["1m 00s", "2m 00s", "3m 00s", "4m 00s"],
  "uncertainty": [0.06, 0.11, 0.18, 0.27],
  "lower_bound": [0.15, 0.27, 0.49, 0.54],
  "upper_bound": [0.27, 0.49, 0.85, 1.00],
  "calibrated_uncertainty": true,
  "source_branch": "Continuous Dynamics Head",
  "protocol": "Chronological Split",
  "is_mock": false
}
```
* **Field Constraints:**
  - `risk`: list of 4 floats between `0.0` and `1.0`.
  - `uncertainty`: list of 4 floats (standard deviation $\sigma$).
  - `lower_bound` / `upper_bound`: floats between `0.0` and `1.0`.
  - `calibrated_uncertainty`: boolean. If `False`, the UI uncertainty shading is dimmed.

---

### 2.2 `get_comparison_table() -> list[dict]`
Returns the PS-mandated comparative benchmark evaluation comparing the World Model against baseline models.

* **Consuming UI File:** `views/03_Validation.py`
* **Return Schema:** List of 4 dictionaries:
```json
[
  {
    "model": "SHADOWCAT World Model (Bi-LSTM + Temporal Dynamics)",
    "paradigm": "Forward Simulation P(S_t+1 | S_t)",
    "f1_score": 0.816,
    "precision": 0.842,
    "recall": 0.791,
    "fpr": 0.048,
    "lead_time": "+3.5 min (Pre-emptive)",
    "status": "Production Candidate",
    "is_mock": false
  },
  {
    "model": "Lagged Autoregressive Baseline (Historical Trend)",
    "paradigm": "Rolling Window Extrapolation",
    "f1_score": 0.612,
    "precision": 0.648,
    "recall": 0.580,
    "fpr": 0.114,
    "lead_time": "+1.2 min (Partial)",
    "status": "Comparative Baseline",
    "is_mock": false
  },
  {
    "model": "Logistic Regression Baseline (Static Classifiers - PS Mandated)",
    "paradigm": "Static Flow Snapshot (No Temporal Memory)",
    "f1_score": 0.468,
    "precision": 0.512,
    "recall": 0.431,
    "fpr": 0.182,
    "lead_time": "0.0 min (Post-facto)",
    "status": "PS Benchmark Reference",
    "is_mock": false
  },
  {
    "model": "Conventional Signature IDS (Suricata / Snort Rules)",
    "paradigm": "Deterministic Packet Matching",
    "f1_score": 0.732,
    "precision": 0.884,
    "recall": 0.625,
    "fpr": 0.021,
    "lead_time": "0.0 min (Alert fired post-compromise)",
    "status": "Legacy Reactive",
    "is_mock": false
  }
]
```
* **Field Constraints:**
  - `f1_score`, `precision`, `recall`, `fpr`: floats in `[0.0, 1.0]`.

---

### 2.3 `get_host_risk_graph(episode_id: str = None, k_step: int = 2) -> dict`
Returns host topology graph, communication edges, and host risk predictions $h_v(t)$.

* **Consuming UI File:** `components/attack_graph.py`
* **Input:**
  - `episode_id` (str, optional): Scenario identifier.
  - `k_step` (int, default=2): Rollout step ($0 \dots 4$).
* **Return Schema:**
```json
{
  "episode_id": "EPISODE-INFIL-01",
  "node_roles": {
    "10.0.2.15": "Workstation (Patient Zero)",
    "10.0.3.50": "Internal File Share",
    "10.0.4.10": "SSH Jump Host",
    "10.0.4.21": "Internal Auth Cluster",
    "10.0.5.1": "Domain Controller (Critical Asset)"
  },
  "host_telemetry": {
    "10.0.4.10": {
      "role": "SSH Jump Host",
      "subnet": "10.0.4.0/24 (DMZ Management Tier)",
      "active_ports": "TCP/22 (OpenSSH Ingress), TCP/88 (Kerberos Client), TCP/514 (Syslog)",
      "driving_indicators": "Repeated SSH authentication failure bursts (18 attempts/min), brute-force credential spray, privileged session forwarding.",
      "containment_stance": "Illustrative analyst guidance — not a system recommendation: Terminate active jump host sessions, restrict SSH ingress to bastion management CIDRs."
    }
  },
  "rollout_steps": {
    "2": {
      "label": "Horizon t+2 (+2 min Rollout)",
      "horizon_code": "t+2",
      "uncertainty_sigma": 0.09,
      "uncertainty_label": "±9% (Autoregressive Step 2)",
      "uncertainty_tier": "Controlled Epistemic Drift",
      "uncertainty_color": "#38BDF8",
      "summary": "Anticipated credential extraction on 10.0.4.10; reconnaissance directed at Auth Cluster.",
      "active_edges": [
        ["10.0.2.15", "10.0.4.10", "Compromised"],
        ["10.0.4.10", "10.0.4.21", "Port 88/Kerberos"]
      ],
      "host_risks": {
        "10.0.4.10": {"risk": 0.76, "uncertainty": 0.09},
        "10.0.2.15": {"risk": 0.94, "uncertainty": 0.05},
        "10.0.4.21": {"risk": 0.46, "uncertainty": 0.09},
        "10.0.3.50": {"risk": 0.15, "uncertainty": 0.06},
        "10.0.5.1":  {"risk": 0.19, "uncertainty": 0.07}
      }
    }
  },
  "active_k_step": 2,
  "disclaimer": "Demonstrated on the single infiltration case study (n = 1). Not a general lateral-movement forecasting capability.",
  "is_mock": false
}
```
* **Field Constraints:**
  - `IP strings`: Used strictly as topology position keys, never as input model features.
  - Mandatory disclaimer preserved on UI: `"Demonstrated on the single infiltration case study (n = 1). Not a general lateral-movement forecasting capability."`

---

### 2.4 `get_novelty_score(window_id: str = None) -> dict`
Returns current observed state $S(t)$, novelty score, and behavioral envelope status.

* **Consuming UI File:** `views/01_Forecast.py`, `views/02_Evidence.py`, `components/state.py`
* **Return Schema:**
```json
{
  "state_id": "S(t)",
  "window_label": "Window t (Current)",
  "dominant_behavior": "Probing & port sweeping",
  "novelty_score": 0.23,
  "novelty_status": "Expected Behavior Envelope",
  "active_endpoints": 42,
  "syn_ack_ratio": 4.8,
  "mean_packet_size": 312,
  "entropy": 3.42,
  "flows_analyzed": 12480,
  "packets_analyzed": 84216,
  "is_mock": false
}
```
* **Field Constraints:**
  - `novelty_score`: float in `[0.0, 1.0]`. Values $< 0.50$ indicate known behavior; $\ge 0.50$ indicate out-of-distribution drift.

---

### 2.5 `get_attributions(window_id: str = None) -> list[dict]`
Returns deletion-tested feature attribution rankings.

* **Consuming UI File:** `views/02_Evidence.py` & `components/explanation.py`
* **Return Schema:** List of dicts:
```json
[
  {"feature": "SYN packet rate", "contribution": 0.34, "category": "Flow Dynamics", "delta": "+280%"},
  {"feature": "Connection attempt frequency", "contribution": 0.28, "category": "Graph Topology", "delta": "14 targets"},
  {"feature": "TTL variance", "contribution": 0.21, "category": "Packet Header", "delta": "Abnormal hops"},
  {"feature": "Inter-arrival timing jitter", "contribution": 0.12, "category": "Temporal Rhythm", "delta": "Paced pulses"},
  {"feature": "Payload entropy distribution", "contribution": 0.05, "category": "Payload Stats", "delta": "Baseline"}
]
```
* **Field Constraints:**
  - `contribution`: floats in `[0.0, 1.0]` summing approximately to `1.0`.

---

### 2.6 `get_flagged_flows(window_id: str = None, limit: int = None) -> list[dict]`
Returns correlated network flows driving the forecast.

* **Consuming UI File:** `views/02_Evidence.py` & `components/evidence.py`
* **Return Schema:**
```json
[
  {
    "id": "FLW-10492",
    "source": "10.0.2.15",
    "sport": 54102,
    "destination": "10.0.4.21",
    "dport": 22,
    "protocol": "TCP",
    "reason": "SYN burst (140 pkts/s)",
    "risk": "High",
    "pkts": 412,
    "bytes": 26368
  }
]
```

---

### 2.7 `get_validation_data() -> dict`
Returns authoritative LOEO 37-fold cross-validation metrics, horizon stability data, and leakage audit checklist.

* **Consuming UI File:** `views/03_Validation.py`
* **Return Schema:**
```json
{
  "metrics": {
    "precision": 0.842,
    "recall": 0.791,
    "f1_score": 0.816,
    "pr_auc": 0.835,
    "fpr": 0.048
  },
  "loeo_summary": {
    "protocol": "Leave-One-Episode-Out (LOEO 37-Fold Cross-Validation)",
    "description": "Evaluated across 37 held-out episodic attack sequences to eliminate temporal and schedule leakage.",
    "folds_tested": 37,
    "mean_pr_auc": 0.821,
    "variance": 0.014
  },
  "horizons": [
    {"horizon": "t+1 (1 min)", "status": "Reliable", "f1": 0.88, "error_growth": "Low", "verdict": "Validated for automated alerting"},
    {"horizon": "t+2 (2 min)", "status": "Reliable", "f1": 0.82, "error_growth": "Controlled", "verdict": "Validated for SOC queue prioritization"},
    {"horizon": "t+3 (3 min)", "status": "Informative", "f1": 0.74, "error_growth": "Moderate", "verdict": "Validated for human analyst escalation review"},
    {"horizon": "t+4 (4 min)", "status": "Exploratory", "f1": 0.63, "error_growth": "Compounding", "verdict": "Informational trend indicator only"},
    {"horizon": "t+5 (5 min) [H=5 Primary]", "status": "Benchmark Bound", "f1": 0.58, "error_growth": "Epistemic Limit", "verdict": "Primary DE evaluation horizon (H=5) under LOEO 37-fold"}
  ],
  "leakage_controls": [
    {"layer": "Temporal Partitioning", "check": "Passed", "detail": "Strict chronological split; no future packets in state window"},
    {"layer": "Graph Neighbor Sampling", "check": "Passed", "detail": "Sub-graph induction restricted to historical edges t_window"}
  ],
  "is_mock": false
}
```

---

## 3. Step-by-Step Integration Workflow for Backend Teammate

1. **Step 1: Train & Save Checkpoint Artifacts**  
   Export your trained PyTorch weights / GNN models into the `models/` directory using the filenames listed in §1 (e.g. `models/world_model.pt`).
2. **Step 2: Implement Inference in `data_provider.py`**  
   Replace the internal mock dictionary reading in the relevant accessor function with your `model.predict()` or tensor extraction call.
3. **Step 3: Confirm Output Contract Compliance**  
   Ensure the function returns data matching the exact dictionary schema and field keys specified above.
4. **Step 4: Verify Auto-Detection & Badging**  
   Launch the app (`streamlit run app.py`). Confirm that only the integrated component loses its `[MOCK]` tag, while unintegrated components retain their `[MOCK]` badge.
