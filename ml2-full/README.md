# ShadowCat: ML2 Track (Graph Neural Network & Fusion Ablation)

This directory contains the Graph Neural Network (GNN) modeling, dynamic graph construction, and multimodal fusion ablation experiments for the SHADOWCAT platform.

---

## Directory Guide & Canonical Implementation

- **`GNN_FINAL/ml2/` (CANONICAL)**: Contains the verified, multi-seed GraphSAGE encoder and Temporal-Graph fusion ablation framework built against canonical Unified Cyber State ($S_t$) window graphs.
- **`gnn/` (SUPERSEDED / EXPLORATORY)**: Contains early exploratory graph construction scripts on legacy datasets. Kept strictly for historical provenance; do not use for new evaluations.

---

## Scientific Findings: GNN Adopt/Hold Decision

A core tenet of SHADOWCAT is empirical honesty. The GraphSAGE fusion model underwent a rigorous 3-seed ablation study comparing the multimodal `FusedModel` ($LSTM + GNN$) against the `TemporalOnlyBaseline` ($LSTM$).

### Summary Results:
| Metric | Temporal-Only Baseline | Fused (LSTM + GNN) | Verdict |
| :--- | :---: | :---: | :---: |
| **Validation Loss (Mean ± Std)** | **1.666 ± 0.017** | 1.932 ± 0.053 | **Temporal-Only Wins** |
| **Early Stopping Epoch** | **23.0** (best ~8.0) | 33.6 (best ~18.6) | **Temporal-Only Faster** |
| **Test Set MSE (Mean ± Std)** | 2.333 ± 0.129 | 0.725 ± 0.084 | *Regime-shift artifact* |

### Formal Decision: **HOLD (Temporal-Only Deployed)**
- **Primary Selection Metric**: Model selection and early stopping were governed by validation loss on chronological splits. Across all 3 seeds, the temporal baseline converged faster and achieved strictly lower validation error.
- **Scientific Integrity**: Even though the fused model demonstrated lower MSE on the holdout test set (likely due to a late-stage infiltration regime shift), adopting a model that lost on validation criteria would constitute test-set leakage.
- **Production Integration**: In accordance with the adopt/hold protocol, the production pipeline deploys the temporal LSTM world model.

For the full audit and mathematical reconciliation, see [`GNN_FINAL/ml2/results/gnn_adopt_hold_decision.md`](GNN_FINAL/ml2/results/gnn_adopt_hold_decision.md).

---

## Structure

```
ml2-full/
├── GNN_FINAL/               # Canonical multi-seed ablation workspace
│   ├── ml2/
│   │   ├── models/          # GraphSAGE encoder & FusedModel architectures
│   │   ├── training/        # Multi-seed trainer with early stopping
│   │   ├── experiments/     # Multi-seed evaluation runner
│   │   └── results/         # gnn_adopt_hold_decision.md & checkpoint logs
├── gnn/                     # [SUPERSEDED] Exploratory legacy graph research
└── README.md
```
