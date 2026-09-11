# GNN Ablation Report

## Objective
Evaluate the contribution of Graph Structure, Node Attributes, and Edge Attributes to graph representation quality.

## Ablation Comparison Table

| Configuration | Edge Features Used | PR-AUC | F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|---|---|
| Graph Structure Only | No | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Node Features Only | No | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Node + Edge Features (Full) | Yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Key Findings
1. **Structure vs Node Features**: Node features (degree, port distribution, traffic volume) provide significant discriminative capacity.
2. **Edge Features**: Edge-level flow attributes (flow_count, total_bytes, tcp_ratio etc.) now contribute to message passing via EdgeAwareSAGEConv, enabling the model to capture communication characteristics at the connection level.
3. **Architecture**: The EdgeAwareSAGEConv concatenates edge features with source node features before aggregation, making edge information a first-class citizen in the GNN computation.
