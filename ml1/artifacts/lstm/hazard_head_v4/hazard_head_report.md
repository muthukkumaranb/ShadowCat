# Hazard Head Evaluation Report (LOEO Protocol)

## Onset Forecasting Performance by Horizon

| Horizon | F1 Mean | Precision Mean | Recall Mean | FPR Mean | ROC-AUC Mean | PR-AUC Mean |
|---|---|---|---|---|---|---|
| **H=1** | 0.2793 | 0.2973 | 0.2703 | 0.0000 | 0.8864 | 0.8865 |
| **H=2** | 0.3177 | 0.3475 | 0.3117 | 0.0088 | 0.8739 | 0.8763 |
| **H=5** | 0.2780 | 0.2973 | 0.2748 | 0.0045 | 0.7442 | 0.7853 |

## Cumulative Hazard Trajectory Verification
- **Formula**: $P(\text{event} \le K) = 1 - \prod_{k=1}^K (1 - h_k)$
- **Properties Verified**:
  1. Boundedness: $0.0 \le P(\text{event} \le K) \le 1.0$ across all horizons.
  2. Monotonicity: $P(\text{event} \le K+1) \ge P(\text{event} \le K)$ for all sequences.

## Horizon Diagnostic Notes (Claim Ladder Rung 6 Alignment)
- **H=1 / H=2**: Short-horizon hazard predictions capture immediate onset transitions.
- **H=5 Onset vs Schedule Baseline**: Per the project's prior Gate 0 diagnostic finding, real traffic features underperform schedule-only features at H=5 (F1 ~0.095 vs ~0.253 schedule baseline) due to long temporal distance from initial probes. This is an expected, documented finding in the claim ladder.