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
| `host_risk_graph` | `models/graph_topology.parquet` | Canonical network interaction topology edgelist/lookup |
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
Returns the forecast block of the active `predict()` result (empty dict if no inference ran).

* **Consuming UI File:** `views/03_Forecast.py`
* **Return Schema (keys; values come from `backend/predict.py`):**
```json
{
  "window_id": "<window id of the last input window>",
  "onset_probability": "<float, mean of the 37 stacked onset fold models>",
  "onset_probability_label": "P(attack within the next 5 minutes)",
  "risk": ["<onset_probability>"],
  "onset_alert": "<bool, onset_probability >= alert_threshold>",
  "alert_threshold": 0.5,
  "conformal_interval": ["<lower>", "<upper>"],
  "stage": ["<stage head output on world-model rollout S_hat(t+k), k=1..5>"],
  "is_mock": false
}
```
* There is no per-horizon risk: the onset model predicts one probability for t+1..t+5.

---

### 2.2 Comparison table
Removed. Model-vs-baseline numbers are part of `get_validation_data()` (§2.7).

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

* **Consuming UI File:** `views/03_Forecast.py`, `views/06_Explainability.py`, `components/state.py`
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

* **Consuming UI File:** `views/06_Explainability.py` & `components/explanation.py`
* **Return Schema:** List of dicts:
```json
[
  {"feature": "SYN packet rate", "contribution": 0.34, "category": "Flow Dynamics", "delta": "+280%"},
  {"feature": "Connection attempt frequency", "contribution": 0.28, "category": "Flow Dynamics", "delta": "14 targets"},
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

* **Consuming UI File:** `views/06_Explainability.py` & `components/evidence.py`
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
Returns the contents of `models/loeo_37fold_results.json`, generated by
`scripts/build_validation_json.py` from committed files (stacked sidecars, LR per-fold CSVs,
`evaluation/benchmark/stacked_benchmark_results.json`). Empty dict if the file is missing.

* **Consuming UI File:** `views/07_Validation_Trust.py`
* **Top-level keys:** `generated_by`, `sources`, `protocol`, `h_star_note`, `lr_results_caveat`,
  `tasks.{detection,onset}` with `stacked_mean`, `lr_mean`, `per_fold`, `matched_fpr`,
  `history_ablation_f1_at_0.5`. See the generated file for values.

---

### 2.8 `get_graph_topology() -> dict`
Returns real network graph topology (nodes and edges) constructed from flow data.

* **Consuming UI File:** `views/04_Attack_Graph.py` & `components/cytoscape_attack_graph.py`
* **Return Schema:**
```json
{
  "status": "active",
  "nodes_count": 30,
  "edges_count": 45,
  "graph_nodes": [{"id": "10.0.2.15", "name": "10.0.2.15", "ip": "10.0.2.15", "role": "Enclave Host Node"}],
  "graph_edges": [{"source": "10.0.2.15", "target": "10.0.4.21", "flow_count": 5, "byte_count": 8250.0, "packet_count": 40.0, "label": "5 flows (8.1 KB)"}]
}
```

---

### 2.9 `get_graph_traversal(max_k: int = 5) -> dict`
Returns real, explainable, weight-based attack propagation traversal over actual network graph edges.

* **Consuming UI File:** `views/04_Attack_Graph.py` & `components/cytoscape_attack_graph.py`
* **Return Schema:**
```json
{
  "status": "success",
  "status_message": "Real graph-propagation walk computed successfully.",
  "start_node": "10.0.2.15",
  "start_reason": "Alert driver: Source endpoint of flagged flow FLW-10490",
  "total_nodes": 30,
  "total_edges": 45,
  "steps": [
    {
      "k": 1,
      "frontier": ["10.0.2.15", "10.0.4.10"],
      "frontier_size": 2,
      "added_node": "10.0.4.10",
      "source_node": "10.0.2.15",
      "edge_byte_count": 7500.0,
      "edge_flow_count": 5,
      "candidate_degree": 3,
      "step_surge_bytes": 7500.0,
      "cumulative_surge_bytes": 7500.0,
      "step_surge_flows": 5,
      "cumulative_surge_flows": 5,
      "stopped_early": false,
      "stop_reason": null,
      "explanation": "Compromised 10.0.4.10 from 10.0.2.15: 7.3 KB (5 flows) on connecting edge; candidate degree = 3"
    }
  ],
  "walked_edges": [...]
}
```

---

## 3. Step-by-Step Integration Workflow for Backend Teammate

1. **Step 1: Train & Save Checkpoint Artifacts**  
   Export your trained PyTorch weights into the `models/` directory using the filenames listed in §1 (e.g. `models/world_model.pt`).
2. **Step 2: Implement Inference in `data_provider.py`**  
   Replace the internal mock dictionary reading in the relevant accessor function with your `model.predict()` or tensor extraction call.
3. **Step 3: Confirm Output Contract Compliance**  
   Ensure the function returns data matching the exact dictionary schema and field keys specified above.
4. **Step 4: Verify Auto-Detection & Badging**  
   Launch the app (`streamlit run app.py`). Confirm that only the integrated component loses its `[MOCK]` tag, while unintegrated components retain their `[MOCK]` badge.
