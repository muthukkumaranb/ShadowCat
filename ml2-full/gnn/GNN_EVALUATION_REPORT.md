# GNN Evaluation Report

## Comparison Table: Graph-Statistical Baseline vs GNN vs Temporal Forecaster

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | Lead Time (min) |
|---|---|---|---|---|---|---|---|
| Baseline (Random Forest) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| Edge-Aware GraphSAGE (Snapshot) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| Temporal GNN Forecaster | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |

## Model Architecture
- **Graph Encoder**: EdgeAwareSAGEConv (edge features contribute to message passing via edge MLP)
- **Node Features**: 12 features (out_degree, in_degree, out_flow_count, in_flow_count...)
- **Edge Features**: 6 features (flow_count, total_bytes, total_packets, mean_duration...)
- **Pooling**: Global Mean + Max → Concatenation → 64-D embedding
- **Diagnostic**: Snapshot classifier head used for training signal only — primary output is the graph embedding

## Interpretation
- **Edge-Aware GraphSAGE**: Encodes topological structural context + edge-level flow information via neighborhood aggregation.
- **Temporal GNN Forecaster**: Aggregates sequences of 30 historical graph embeddings to forecast risk over a 10-minute horizon.
- **Forecast Lead Time**: Produces advance warning alerts with up to 0.0 minutes of lead time before attack onset.
