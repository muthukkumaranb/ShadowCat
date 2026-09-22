# Hazard Head Evaluation Report (LOEO Protocol)

## Onset Forecasting Performance by Horizon

| Horizon | F1 Mean | Precision Mean | Recall Mean | FPR Mean | ROC-AUC Mean | PR-AUC Mean |
|---|---|---|---|---|---|---|
| **H=1** | 0.4986 | 0.5075 | 0.4973 | 0.0077 | 0.9708 | 0.9689 |
| **H=2** | 0.5331 | 0.5277 | 0.5405 | 0.0191 | 0.9708 | 0.9652 |
| **H=5** | 0.5017 | 0.4985 | 0.5081 | 0.0257 | 0.9745 | 0.9820 |

## Cumulative Hazard Trajectory Verification
- **Formula**: $P(\text{event} \le K) = 1 - \prod_{k=1}^K (1 - h_k)$
- **Properties Verified**:
  1. Boundedness: $0.0 \le P(\text{event} \le K) \le 1.0$ across all horizons.
  2. Monotonicity: $P(\text{event} \le K+1) \ge P(\text{event} \le K)$ for all sequences.

## Horizon Diagnostic Notes (Claim Ladder Rung 6 Alignment)
- **H=1 / H=2**: Short-horizon hazard predictions capture immediate onset transitions.
- **H=5 Onset vs Schedule Baseline**: Per the project's prior Gate 0 diagnostic finding, real traffic features underperform schedule-only features at H=5 (F1 ~0.095 vs ~0.253 schedule baseline) due to long temporal distance from initial probes. This is an expected, documented finding in the claim ladder.