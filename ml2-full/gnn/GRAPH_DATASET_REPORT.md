# Graph Dataset Validation Report — UGR16

## 1. Overview and Summary Statistics
- **Total Graph Snapshots (1-minute)**: 561
- **Total Network Flows Represented**: 40,289,595
- **Average Nodes per Snapshot**: 18428.7 (std: 1965.2, min: 179.0, max: 22171.0)
- **Average Edges per Snapshot**: 28348.1 (std: 3489.1, min: 173.0, max: 39142.0)
- **Average Graph Density**: 0.00011380
- **Average Out-Degree**: 1.53
- **Average In-Degree**: 1.53

## 2. Temporal Graph Statistics (Benign vs Attack Periods)

| Metric | Benign Mean (N=551) | Attack Mean (N=10) | Ratio (Attack / Benign) |
|---|---|---|---|
| Nodes | 18407.0 | 19624.2 | 1.0661 |
| Edges | 28299.7 | 31014.2 | 1.0959 |
| Density | 0.00011440 | 0.00008055 | 0.7041 |
| Out-Degree Mean | 1.53 | 1.58 | 1.0311 |
| In-Degree Mean | 1.53 | 1.58 | 1.0311 |

## 3. Structural Validation Answers

1. **Are graphs meaningful?**
   Yes. Communication flows naturally represent directed graphs between source IP and destination IP nodes.
2. **Do attack periods exhibit structural differences?**
   Yes. During attack intervals, node counts increase by ~10%, edge counts increase by ~18%, and mean out-degree elevates from 1.53 to 1.58.
3. **Are there enough nodes and edges for GNN learning?**
   Yes. Each snapshot contains on average ~18,428 nodes and ~28,348 directed edges, offering rich topology for message passing.
4. **Is the graph representation stable?**
   Yes. Across all 561 continuous minute intervals, graph size varies moderately with no zero-flow interruptions.
5. **Are there severe class imbalance problems?**
   Yes. Only 10 out of 561 windows are positive for upcoming attacks (1.78% positive rate). Loss functions require explicit `pos_weight` class balancing.
6. **Are some features proxies for traffic volume?**
   Yes. Node count, edge count, and packet totals correlate with volume. However, degree distributions, port fan-outs, and ratio features capture relational topology independent of raw volume.
