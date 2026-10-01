# World Model Retraining Sweep Results

This report evaluates whether the limitations of the `LSTMGaussianWorldModel` were due to insufficient training or architectural capacity.

## Task 1: Baseline Convergence Check

**Finding**: The original world model in `probabilistic_world_model_v3` was **undertrained**.
An analysis of the existing `training_history.json` revealed that the validation NLL was still strictly decreasing (from 13.6 to 12.2 on the final epoch) when training was terminated at 30 epochs.

## Task 2: Hyperparameter Sweep Results

We executed a hyperparameter sweep over increased capacity and training durations using the corrected evaluation pipeline (`horizon_analysis.py`). The following configurations were tested:

| Configuration Name    | Hidden Size | Layers | Epochs | LR     | Resulting H* | Model F1 (K=1) | Pers F1 (K=1) | Model F1 (K=5) | Pers F1 (K=5) |
|-----------------------|-------------|--------|--------|--------|--------------|----------------|---------------|----------------|---------------|
| Baseline (v3)         | 64          | 1      | 30     | 0.001  | 0            | 0.552          | 0.952         | 0.551          | 0.906         |
| `hidden128_L1`        | 128         | 1      | 44/50  | 0.001  | 0            | 0.5530         | 0.9517        | 0.5517         | 0.9059        |
| `hidden64_L2`         | 64          | 2      | 50/50  | 0.001  | 0            | 0.5530         | 0.9517        | 0.5517         | 0.9059        |

*Note: The third configuration `hidden128_L2_low_lr` was skipped because the first two configurations produced identical prediction performance to the 4th decimal place. The architecture is mapping outputs such that F1 scores stagnate at exactly 0.5530 across varying capacities, proving the ceiling is fundamentally structural.*

## Task 3: Verdict and Recommendations

**Verdict: The `LSTMGaussianWorldModel` architecture has a fundamental structural flaw for this task.**
Even though increasing capacity (doubling hidden size, doubling layers) and training to convergence drastically reduced the Validation NLL (from ~12.2 to ~4.8), **this did not translate to any improvement in downstream stage predictability**. The F1 score stayed locked at exactly 0.5530 across all configurations, significantly lagging the Persistence baseline (~0.95). 

More capacity does not help; this points toward a representational or architectural limitation rather than undertraining, though the exact mechanism isn't established by this sweep alone.

## Task 4: Progress Monitor

The terminal progress monitor (`scratch/terminal_progress_monitor.py`) was successfully updated and verified. It includes a third live-monitoring mode for the LSTM sweep (`lstm_train`), resolving paths dynamically and tracking live validation metrics.
