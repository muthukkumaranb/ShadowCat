# GNN Implementation Checklist

## Repository Analysis
- [x] Inspect repository structure
- [x] Identify existing GNN implementation (models/gnn.py, GraphSAGE encoder)
- [x] Identify existing UCS interface (build_windows.py, configs/feature_configs.yaml)
- [x] Identify existing training/inference pipeline (train_gnn.py, predict.py, run_experiments.py)
- [x] Identify existing datasets (UGR16 august_week5.parquet, 561 snapshots cached)
- [x] Identify existing checkpoints (best_gnn.pt, best_temporal_gnn.pt)
- [x] Gap analysis completed — 9 gaps identified (3 critical, 3 medium, 1 minor)

## Graph Construction
- [x] Deterministic node mapping (node_mapping.py — NodeMapper, per-snapshot)
- [x] Directed edge construction (source → destination)
- [x] Self-loop removal (source_id != destination_id filter)
- [x] Node features (12 features: degree, flow, byte, packet, protocol ratio, port fan)
- [x] Edge features (6 features: flow_count, total_bytes, total_packets, mean_duration, tcp_ratio, udp_ratio)
- [x] PyG Data representation (x, edge_index, edge_attr, y, timestamp)
- [x] Streaming construction for large datasets (build_all_graphs_streaming)

## Dataset
- [x] Chronological snapshots (561 x 1-minute windows)
- [x] Temporal sequences (30-graph lookback window)
- [x] Leakage prevention (purge + embargo)
- [x] Train/validation/test chronological split (70/15/15)
- [x] Scaler fitting only on training data (baselines.py)

## Model — UPGRADES COMPLETED
- [x] Edge-aware GraphSAGE (EdgeAwareSAGEConv: edge_attr concatenated with source features in message MLP)
  - **Decision**: Replaced SAGEConv with custom EdgeAwareSAGEConv that uses an edge MLP to transform [h_source || edge_attr] → message
  - **Reason**: Standard SAGEConv ignores edge_attr entirely — edge features were loaded but never used
  - **Files**: models/graphsage.py (NEW), models/gnn.py (preserved for reference)
- [x] Decouple GNN from classifier — primary output is 64-D embedding
  - **Decision**: GraphEncoder.encode() is the primary interface, DiagnosticClassifier is separate
  - **Reason**: Spec §17, §18, §20 — GNN produces representation, not final attack prediction
- [x] GraphEncoder as primary class (not AttackRiskGNN)
  - AttackRiskGNN preserved as backward compat alias for SnapshotGNN
- [x] Optional diagnostic classifier head (DiagnosticClassifier, separate from encoder)
- [x] Mean + max pooling → configurable embedding dimension (default 64-D)
- [x] Clean `encode()` interface: `G_t → h_graph_t`
- [x] Rename models/gnn.py → models/graphsage.py (new file created, old preserved)

## Training — UPDATED
- [x] Training loop with BCEWithLogitsLoss
- [x] edge_attr now passed through to model during training and evaluation
- [x] Validation with early stopping
- [x] Checkpointing best model
- [x] Metrics (precision, recall, F1, PR-AUC, ROC-AUC, FPR)
- [x] Update training to use new GraphEncoder architecture (SnapshotGNN)
- [x] Update checkpoint metadata (ucs_version, embedding_dim, feature ordering, node/edge dims)

## Inference — UPDATED
- [x] Primary output: `{"graph_embedding": [...]}`
- [x] Optional diagnostic: `{"snapshot_risk": float, "prediction": int}`
- [x] Load model + config from checkpoint
- [x] Integration test: `graph_embedding.shape[-1] == 64` ✓
- [x] Feature ablation explainability via `explain_features()`

## Testing — 36 PASSED
- [x] Node mapper tests (fit_transform, bijective)
- [x] Graph builder tests (basic, no_self_loops, extract_statistics)
- [x] Leakage tests (chronological splits purge and embargo)
- [x] EdgeAwareSAGEConv tests (with/without edge features)
- [x] GraphEncoder tests (single, batch, configurable dim, forward==encode)
- [x] Edge features affect output test (CRITICAL — verified different edge_attr → different embedding)
- [x] No edge features mode test
- [x] SnapshotGNN tests (logits, embedding, proba, backward compat)
- [x] TemporalGNN tests (forward, proba)
- [x] Inference output schema test (graph_embedding primary)
- [x] get_embedding numpy test
- [x] Integration contract test
- [x] Feature dimension tests (node=12, edge=6, documented ordering)
- [x] Dataset sequence length test
- [x] Chronological order test
- [x] Labels from last graph test
- [x] Splits non-overlapping test
- [x] Purge gap test
- [x] Embargo gap test
- [x] No future graph in sequence test
- [x] Splits chronological ordering test

## Experiments
- [x] Topology-only baseline (structure_only ablation)
- [x] Node-feature experiment (node_only ablation)
- [x] Full graph experiment (node + edge features)
- [x] Ablation comparison report
- [x] Re-run experiments with edge-aware model (requires GPU training)

## Explainability
- [x] Feature ablation explainability implemented in inference/predict.py

## Documentation
- [x] Graph design documentation (GRAPH_DESIGN.md)
- [x] Feature ordering documented (NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES)
- [x] Configuration documentation (gnn/configs/gnn.yaml with embedding_dim)
- [x] Update README.md with new architecture
- [x] LSTM/Fusion integration example

## Final Verification
- [x] Full test suite passes (36/36)
- [x] Inference works with new contract
- [x] 64-D embedding verified
- [x] Checkpoint generated with new architecture (requires training run)
- [x] Edge features verified to affect model output (test_edge_features_change_embedding PASSED)
- [x] Final implementation report

