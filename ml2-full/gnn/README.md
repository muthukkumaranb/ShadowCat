> [!WARNING]
> **Exploratory GNN Work (Superseded)**
> This directory contains the old, superseded exploratory UGR16-based subproject. It is **NOT** part of the canonical Shadowcat pipeline and should not be merged into the submission monorepo. It is kept here only for historical reference. The verified pipeline lives under GNN_FINAL/ml2/.
# Graph-Based Cyber Attack Risk Forecasting (GNN Component)

This module implements the **Graph Representation Encoder** for the SIH26153 World Model architecture.

**Primary contract:**
```
UCS(t) → Graph Construction → GraphEncoder → 64-D Graph Embedding
```

The GNN produces a **fusion-ready graph representation**; it is NOT the final attack classifier.

---

## 1. World Model Architecture (GNN Ownership)

```
Raw Traffic / PCAP / CSV / Auth Logs
                ↓
        Data Engineering
                ↓
        Unified Cyber State (UCS)
                ↓
        ┌───────┴────────┐
        ↓                ↓
      LSTM/GRU          GNN  ← YOU ARE HERE
   Temporal State    Graph State
        │                │
        └───────┬────────┘
                ↓
             Fusion
                ↓
          Latent State z_t
                ↓
       World Model Transition
```

**GNN Ownership:**
```
UCS(t)
   ↓
Graph Construction (graph_builder.py)
   ↓
EdgeAwareSAGEConv Encoder (models/graphsage.py)
   ↓
64-D Graph Embedding
   ↓
Fusion-ready representation
```

---

## 2. Key Architecture Decision: Edge-Aware Message Passing

Standard `SAGEConv` ignores edge features. Our `EdgeAwareSAGEConv` fixes this:

```
Standard SAGEConv:
  h_v = W_self·h_v + W_neigh·AGG({h_u : u ∈ N(v)})
  ↳ edge_attr IGNORED

EdgeAwareSAGEConv:
  h_v = W_self·h_v + W_neigh·AGG({MLP_edge([h_u || e_uv]) : u ∈ N(v)})
  ↳ edge_attr concatenated with source node features, transformed via edge MLP
```

This allows edge information (flow count, bytes, packets, protocol ratios) to influence neighbor representations during message passing.

---

## 3. Directory Structure

```
gnn/
├── IMPLEMENTATION_CHECKLIST.md  # Source of truth for progress
├── GRAPH_DESIGN.md              # Feasibility study, node/edge definitions
├── GRAPH_DATASET_REPORT.md      # Statistical validation of graph snapshots
├── GNN_BASELINE_COMPARISON.md   # Statistical baselines vs GNN comparison
├── GNN_EVALUATION_REPORT.md     # Full evaluation report
├── GNN_ABLATION_REPORT.md       # Ablation study (Structure vs Node vs Edge)
├── README.md                    # This file
│
├── node_mapping.py              # Stable IP-to-index mapper (per-snapshot)
├── graph_builder.py             # Vectorized + streaming PyG Data construction
├── graph_dataset.py             # PyG Dataset + sliding temporal window sequences
├── baselines.py                 # Graph-statistical baselines (LR, RF, GBM)
├── run_experiments.py           # End-to-end master experiment runner
│
├── models/
│   ├── __init__.py
│   ├── graphsage.py             # EdgeAwareSAGEConv, GraphEncoder, SnapshotGNN, TemporalAttackGNN
│   └── gnn.py                   # Legacy (preserved for reference)
│
├── training/
│   ├── __init__.py
│   └── train_gnn.py             # Training with edge_attr, class weighting & early stopping
│
├── inference/
│   ├── __init__.py
│   └── predict.py               # GNNPredictor API: graph_embedding + explainability
│
├── configs/
│   └── gnn.yaml                 # Hyperparameters (embedding_dim, hidden_channels, etc.)
│
├── checkpoints/
│   ├── best_gnn.pt              # Snapshot encoder checkpoint + full metadata
│   └── best_temporal_gnn.pt     # Temporal forecaster checkpoint (experimental)
│
└── tests/
    ├── test_node_mapping.py     # Mapping bijective property tests
    ├── test_graph_builder.py    # Self-loop removal & feature dimension tests
    ├── test_graph_dataset.py    # Temporal sequences, chronological order, leakage tests
    ├── test_labels.py           # Purge & embargo boundary tests
    └── test_gnn.py              # 21 tests: encoder, edge features, inference, integration
```

---

## 4. Quickstart & Reproducibility

### Installation
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric pyarrow pyyaml scikit-learn pytest
```

### Running Unit Tests
```bash
python -m pytest gnn/tests/ -v
# Expected: 36 passed, 0 failed
```

### Running the Full Experiment Pipeline
```bash
python -u gnn/run_experiments.py
```

### Running Inference via CLI
```bash
python gnn/inference/predict.py --checkpoint gnn/checkpoints/best_gnn.pt
python gnn/inference/predict.py --checkpoint gnn/checkpoints/best_gnn.pt --explain
```

---

## 5. Integration API: GNN and LSTM/Fusion Contract

### Primary Interface: Get Graph Embedding

```python
from gnn.inference.predict import GNNPredictor
from gnn.graph_builder import build_graph_snapshot

# 1. Initialize predictor (loads weights, sets eval mode)
predictor = GNNPredictor("gnn/checkpoints/best_gnn.pt", device="auto")

# 2. Build graph snapshot from incoming pandas flow dataframe
graph = build_graph_snapshot(flows_df, timestamp=current_ts)

# 3. Get the 64-D graph embedding (primary output)
result = predictor.predict(graph)

# Primary output:
graph_embedding = result["graph_embedding"]  # 64-D list
# len(graph_embedding) == 64

# Optional diagnostic (NOT the World Model's final output):
snapshot_risk = result["snapshot_risk"]       # float, P(attack)
prediction = result["prediction"]             # int, 0 or 1
```

### LSTM/Fusion Integration

```python
# GNN engineer produces:
graph_embedding = predictor.get_embedding(graph)  # numpy array, shape [64]

# LSTM engineer produces:
temporal_embedding = lstm.predict(ucs_sequence)    # numpy array, shape [64]

# Fusion engineer combines:
import numpy as np
combined = np.concatenate([graph_embedding, temporal_embedding])  # shape [128]
latent_state = fusion_mlp(combined)  # → z_t
```

### Feature Ablation Explainability

```python
explanation = predictor.explain_features(graph, top_k=5)
# Returns:
# {
#     "top_influential_features": [
#         {"feature": "node:out_degree", "importance": 0.234},
#         {"feature": "edge:total_bytes", "importance": 0.189},
#         ...
#     ],
#     "node_feature_importances": {...},
#     "edge_feature_importances": {...}
# }
```

---

## 6. Model Architecture

| Component | Value |
|---|---|
| **Encoder** | EdgeAwareSAGEConv (2 layers) |
| **Edge Feature Handling** | Edge MLP: [h_source ‖ e_uv] → message |
| **Node Features** | 12 features (degree, flow, byte, packet, protocol ratio, port fan) |
| **Edge Features** | 6 features (flow_count, total_bytes, total_packets, mean_duration, tcp_ratio, udp_ratio) |
| **Pooling** | Global Mean + Global Max → Concatenate |
| **Embedding Dimension** | 64 (configurable via `gnn/configs/gnn.yaml`) |
| **Diagnostic Head** | Optional MLP classifier for training signal only |

### Expected Tensor Shapes

```
Node features:     x           → [N_t, 12]
Edge connectivity: edge_index  → [2, E_t]
Edge features:     edge_attr   → [E_t, 6]
Graph embedding:   h_graph_t   → [1, 64]   (single graph)
                                  [B, 64]   (batch of B graphs)
```

---

## 7. Checkpoint Metadata

Checkpoints contain full metadata per specification §34:

```json
{
  "ucs_version": "UCS_V1",
  "feature_version": "GRAPH_V2_EDGE_AWARE",
  "model_name": "EdgeAwareSAGEConv + GraphEncoder",
  "embedding_dim": 64,
  "node_feature_dim": 12,
  "edge_feature_dim": 6,
  "node_feature_ordering": ["out_degree", "in_degree", ...],
  "edge_feature_ordering": ["flow_count", "total_bytes", ...],
  "dataset": "UGR16",
  "training_dataset_id": "UGR16:august_week5",
  "model_version": "2.0.0"
}
```

---

## 8. Summary of Findings

- **Dataset**: UGR'16 August Week 5 (40,289,595 network flows across 9.33 hours)
- **Topology**: Average 18,428 nodes and 28,348 directed edges per 1-minute snapshot
- **Hardware**: CUDA acceleration on NVIDIA GeForce RTX 4070 Laptop GPU (8.0 GB VRAM)
- **Leakage Prevention**: Chronological splits with label purge and lookback embargo
- **Edge Features**: Now actively influence message passing (verified by test_edge_features_change_embedding)
- **Known Limitation**: Single attack event in dataset limits statistical evaluation power

