# Retired GraphSAGE Verification Scripts

This directory preserves standalone verification scripts for the experimental GraphSAGE fusion branch (`FusedModel`):
- `verify_fused_model.py`
- `verify_graph_fusion.py`

## Retirement Notice & Architectural Rationale

**STATUS**: RETIRED / DEPRECATED (Do NOT wire into production inference).

### 37-Fold Leave-One-Episode-Out (LOEO) Findings
During rigorous 37-fold LOEO cross-validation across all CSE-CIC-IDS2018 attack episodes, the GraphSAGE fused branch was evaluated against the standalone temporal baseline:
- **DDOS-LOIC-UDP Catastrophic Failure**: GraphSAGE embeddings collapsed under dense, symmetrical volumetric attack topology, resulting in a **54.4% drop in macro F1** (dropping to 0.00 recall on LOIC-UDP bursts) due to message-passing over-smoothing.
- **Inference Latency Overhead**: Running GNN message-passing over dynamic edgelists added significant latency without delivering accuracy improvements on multi-stage lateral progression.

### Authoritative Architecture Decision
1. **NO-GO on Neural Graph Fusion**: The GraphSAGE neural fusion module was retired from the live inference path (`z'(t) = z(t)` pass-through).
2. **Decoupled Topological Propagation**: Replaced by deterministic, explainable BFS/diffusion traversal over real network topology (`backend/graph_traversal.py`), preserving pure temporal dynamics while providing transparent SOC lateral movement tracking.
