# Stacked Ensemble PCA Representation Note

## Finding
Before s2b-t4, the stacked ensemble in `backend/predict.py` projected the 30-window history with a
**single** 32-dim PCA, `ml1/artifacts/lstm/lstm_stacked/pca_32_stacked.pkl`, for all 37 fold models.
Each fold model was trained with **its own** PCA, stored in its checkpoint as `pca_components` /
`pca_mean` (fitted on that fold's training rows). `evaluation/benchmark/reproduce_stacked_benchmark.py`
uses the per-fold PCA and reproduces every fold's sidecar `test_f1`.

## Measurement
Script: `evaluation/benchmark/measure_stacked_pca.py`
Output: `docs/demo_check/stacked_pca_measurement.json`

20 windows of `data-engineering/data/ucs/ucs_windows_models_v1.parquet` (RandomState(42), windows with a
full 30-window history). For each window the 37-fold mean probability is computed with the single PCA
and with each fold's own PCA; everything else (LR stage, scaler, temperature, inputs) is identical.

| Ensemble | Mean abs diff | Max abs diff | Pearson r |
|---|---|---|---|
| Detection | 0.0068 | 0.0258 | 0.9995 |
| Onset | 0.0065 | 0.0171 | 0.9995 |

Per-window values are in the JSON (`windows`).

## Consequence
The single-PCA shortcut moves the displayed probability by up to about 0.026 on these windows: small, but
not the computation that was validated. s2b-t4 switches the deployed ensemble to the per-fold PCA so the
dashboard probability equals `reproduce_stacked_benchmark.py`.
