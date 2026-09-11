# LSTM + UCS for CSE-CIC-IDS2018

This repository connects the canonical Unified Cyber State (UCS) representation
to the existing PyTorch LSTM. The LSTM consumes UCS windows, never raw flow
records. The canonical ingestion and windowing implementation is maintained in
the [UCS reference repository](https://github.com/div007x/SIH2026---UCS-INGESTION-PIPELINE).

## Pipeline

```text
CSE-CIC-IDS2018 CSV
    -> canonical UCS ingestion, cleaning, mapping, and 1-minute windows
    -> future_attack_label (H = 5 windows)
    -> chronological purge + embargo (L + H = 35 windows)
    -> train-only log1p + robust scaling
    -> 30-window sequences (30, F)
    -> existing causal LSTM
    -> future attack probability and metrics
```

UCS input is `data/ucs/ucs_windows.parquet`. Its feature columns are inferred
from numeric UCS columns while excluding metadata, `label_binary`,
`label_attack_type`, and `future_attack_label`. The input feature count `F` is
therefore determined by the actual UCS table and passed to `LSTMClassifier`.

The authoritative settings are in [configs/model_config.yaml](configs/model_config.yaml):
1-minute windows, a 30-minute lookback, a 5-minute forecast horizon, and
70/15/15 chronological splits. Sequences are built independently within each
split. The last window in each sequence supplies the future label and endpoint
timestamp. Purge and embargo remove 35 windows around each split boundary.

Normalization is fit only on purged training windows. The canonical transform
uses `log1p` for traffic-volume families and `(x - Q50) / (Q75 - Q25)` for the
robust scale; validation and test data are transformed with those saved
training statistics.

The world-model experiment adds a deterministic next-state head on the causal
LSTM, approximating `p(S(t+1)|S(t))`, plus persistence and lagged-linear
baselines. Next-state and rollout metrics use the `chronological split`
protocol. Detection evaluation uses retrained episode-wise `LOEO`; the
available UCS artifact determines the number of evaluable episodes.

The current-data probabilistic experiment can be run with:

```powershell
python examples/train_probabilistic.py data/ucs/ucs_windows.parquet
```

It writes Gaussian next-state metrics, prediction intervals, recursive rollout
metrics, feature-group attribution/deletion results, deviation summaries, and
reproducibility metadata under `artifacts/experiments/world_model_20260904/probabilistic`.
The run also saves per-window NLL distributions, sigma diagnostics, top-N NLL
outliers, and `rollout_error_vs_k.png`. The lagged-linear rollout uses the
existing 128-component PCA experiment configuration, fitted on purged training
windows only and inverted before original-space error calculation. All results
use the `chronological split` protocol. Episode-level LOEO, onset, B2, and
unseen-episode novelty claims remain deferred because episode IDs are
unavailable; `source_day` is not used as an episode ID.

## Python API

```python
from lstm.ucs import UCSConfig, load_ucs_windows, prepare_lstm_inputs

windows = load_ucs_windows("data/ucs/ucs_windows.parquet")
sets, features, scaler, normalized = prepare_lstm_inputs(
    windows,
    config=UCSConfig(),
)
```

`sets["train"]`, `sets["val"]`, and `sets["test"]` each expose `X`, `y`,
and `timestamps`, ready for the existing trainer, evaluator, and model.

## Reproducibility and validation

The model configuration, feature list, scaler parameters, seed, checkpoint,
predictions, and metrics should be saved as experiment artifacts by the
training entry point. Raw CICIDS2018 data and generated datasets are ignored
by Git. Run the included tests with:

```powershell
$env:PYTHONPATH = "."
pytest -q
```

The repository includes a synthetic end-to-end test because the raw dataset is
not bundled. Full-dataset metrics are intentionally not reported here until a
real CICIDS2018 UCS artifact is supplied and the experiment is run.

## Standalone components

The generic data, preprocessing, training, evaluation, checkpoint, and
inference APIs remain available for non-UCS sequence datasets. The included
`examples/train_example.py` continues to demonstrate that standalone path.

