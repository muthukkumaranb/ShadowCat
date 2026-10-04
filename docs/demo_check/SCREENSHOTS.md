# ShadowCat Dashboard Honest Screenshots (Phase A Verification)

This document records the visual state of the ShadowCat SOC Cockpit across both benign and SSH-Bruteforce evaluation windows from the canonical `data-engineering/data/ucs/ucs_windows_models_v1.parquet` dataset.

---

## 1. Verified Forecast Screenshots

| Evaluation Window | Screenshot | Description & Honest Behavior Observed |
| :--- | :--- | :--- |
| **Benign Baseline Window** (`benign_02-03-2018_windows.parquet`) | [`forecast_benign.png`](forecast_benign.png) | Flat constant onset probability ($P \approx 0.314$, below 5% FPR alert threshold of $0.3749$). 90% finite-sample conformal prediction interval plotted honestly without artificial escalation. World-model rollout stages labeled `(world-model rollout (not validated: H* = 0))`. Mitigation status: `Baseline Monitoring` with timing text `not crossed`. |
| **SSH-Bruteforce Window** (`ssh_14-02-2018_windows.parquet`) | [`forecast_ssh.png`](forecast_ssh.png) | High constant onset probability ($P \approx 0.932$, above alert threshold). Conformal bounds dynamically reflect calibrated finite-sample quantile from 37-fold LOEO validation. Mitigation stance: `Immediate Mitigation Required` with honest alert crossing timestamp. |

---

## 2. Key Honesty Elements Enforced

1. **No Always-Rising Curve**:
   - Replaced artificial $r_0 \dots r_5$ multiplier ladder with constant onset probability $P(\text{attack within next 5 minutes})$ across $K=1..5$.
   - Clearly annotated as a single 5-minute onset window probability rather than independent per-minute predictions.
2. **Honest World-Model Stage Predictions**:
   - Per-$K$ rollout stages explicitly labeled `world-model rollout (not validated: H* = 0)` based on rigorous state rollout evaluation ($H^* = 0$).
3. **Calibrated Finite-Sample Conformal Intervals**:
   - Dynamic uncertainty band derived from split conformal predictor calibrated on 37 LOEO held-out folds.
   - No hardcoded fallback $q = 0.12$ or fake `±5% (Conformal Bound)` labels.
4. **Honest Telemetry & Timing Text**:
   - Real telemetry metrics (flow count, active hosts, 406-dim world-model deviation) without synthetic formulas ($900 + i \times 150$).
   - Real alert threshold crossing time or `not crossed`; removed fake `Crossed at t-08m` and countdown `m remaining`.
5. **Real Host Topology**:
   - Host communication graph derived dynamically from active slice flows; hardcoded fallback IP list `["172.31.69.21", "172.31.69.1"]` removed.
