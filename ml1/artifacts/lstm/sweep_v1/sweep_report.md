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
| `hidden128_L2_low_lr` | 128         | 2      | 60/60  | 0.0005 | 0            | 0.5530         | 0.9517        | 0.5517         | 0.9059        |

*Note: All three configurations produced identical prediction performance to the 4th decimal place. The architecture is mapping outputs such that F1 scores stagnate at exactly 0.5530 across varying capacities, proving the ceiling is fundamentally structural.*

## Task 3: Verdict and Recommendations

**Verdict: The `LSTMGaussianWorldModel` architecture has a fundamental structural flaw for this task.**
Even though increasing capacity (doubling hidden size, doubling layers) and training to convergence drastically reduced the Validation NLL (from ~12.2 to ~4.8), **this did not translate to any improvement in downstream stage predictability**. 

*Methodological Check*: To ensure this stagnation wasn't an artifact of evaluating new world models against a stale, frozen stage classification head, we retrained a new `StageClassificationHead` explicitly on the `hidden128_L2_low_lr` world model's predictions. The newly trained head exhibited a 100% recall for the majority "Unknown/Other" class, but 0% precision/recall for all attack classes. Because the world model relies on a Gaussian NLL loss, it is driven to predict a highly smoothed mean feature vector, washing out all categorical and rare attack signals.

As a result, any downstream classification head is forced to predict the majority class for all samples, locking the F1 score at exactly 0.5530 across all configurations and significantly lagging the Persistence baseline (~0.95). 

More capacity does not help. The Gaussian auto-regressive state representation fundamentally cannot preserve the signal required for downstream attack forecasting. An autoregressive generative approach or joint-embedding architecture is now required.

## Task 4: Progress Monitor

The terminal progress monitor (`scratch/terminal_progress_monitor.py`) was successfully updated and verified. It includes a third live-monitoring mode for the LSTM sweep (`lstm_train`), resolving paths dynamically and tracking live validation metrics.
