# Stacked Ensemble PCA Representation Note

## Summary
In `backend/predict.py`, the stacked ensemble inference uses a **single shared 32-dim PCA transform** (`pca_32_stacked.pkl`, derived from Fold 0) across all 37 fold models.
However, during Leave-One-Episode-Out (LOEO) training, each fold model was trained with its own fold-specific PCA transform stored in the checkpoint (`ckpt["pca_components"]` and `ckpt["pca_mean"]`).

This note documents this architectural simplification and provides numerical measurements comparing the dashboard single-PCA stacked probabilities against the rigorous per-fold-PCA probabilities across 20 representative windows from `ucs_windows_models_v1.parquet`.

## Measurement Results (20 Sample Windows)

| Window Index | UTC Timestamp | Attack Type | Dash Det P | Per-Fold Det P | Det Diff | Dash Onset P | Per-Fold Onset P | Onset Diff |
|---|---|---|---|---|---|---|---|---|
| 383 | 2018-02-14 10:20:00+00:00 | Benign | 0.0114 | 0.0109 | 0.0005 | 0.0211 | 0.0214 | 0.0003 |
| 437 | 2018-02-14 11:14:00+00:00 | Benign | 0.9951 | 0.9957 | 0.0005 | 0.9785 | 0.9797 | 0.0012 |
| 493 | 2018-02-14 12:10:00+00:00 | Benign | 0.8617 | 0.8832 | 0.0215 | 0.7468 | 0.7705 | 0.0236 |
| 515 | 2018-02-14 12:32:00+00:00 | Benign | 0.0008 | 0.0007 | 0.0001 | 0.0052 | 0.0048 | 0.0004 |
| 561 | 2018-02-21 02:13:00+00:00 | DDOS-HOIC | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| 729 | 2018-02-22 01:16:00+00:00 | Benign | 0.0241 | 0.0247 | 0.0006 | 0.0492 | 0.0545 | 0.0053 |
| 773 | 2018-02-22 02:00:00+00:00 | Benign | 0.4057 | 0.4532 | 0.0475 | 0.4304 | 0.4888 | 0.0584 |
| 1007 | 2018-02-22 08:45:00+00:00 | Benign | 0.0785 | 0.0935 | 0.0150 | 0.1443 | 0.1759 | 0.0316 |
| 1073 | 2018-02-22 09:51:00+00:00 | Benign | 0.0198 | 0.0237 | 0.0040 | 0.0247 | 0.0316 | 0.0070 |
| 1354 | 2018-02-28 02:32:00+00:00 | Benign | 0.9785 | 0.9787 | 0.0001 | 0.9095 | 0.9137 | 0.0041 |
| 1386 | 2018-02-28 03:04:00+00:00 | Benign | 0.2098 | 0.1684 | 0.0414 | 0.2023 | 0.1721 | 0.0302 |
| 1582 | 2018-02-28 08:50:00+00:00 | Benign | 0.0097 | 0.0072 | 0.0025 | 0.0261 | 0.0211 | 0.0050 |
| 1806 | 2018-02-28 12:34:00+00:00 | Benign | 0.0156 | 0.0114 | 0.0042 | 0.0413 | 0.0344 | 0.0070 |
| 1844 | 2018-03-01 01:12:00+00:00 | Benign | 0.0990 | 0.0767 | 0.0223 | 0.1307 | 0.1111 | 0.0196 |
| 2252 | 2018-03-01 11:40:00+00:00 | Benign | 0.0149 | 0.0117 | 0.0031 | 0.0547 | 0.0475 | 0.0072 |
| 2274 | 2018-03-01 12:02:00+00:00 | Benign | 0.0151 | 0.0116 | 0.0035 | 0.0420 | 0.0356 | 0.0064 |
| 2367 | 2018-03-02 01:35:00+00:00 | Benign | 0.9525 | 0.9526 | 0.0002 | 0.9486 | 0.9498 | 0.0012 |
| 2497 | 2018-03-02 04:55:00+00:00 | Benign | 0.0226 | 0.0231 | 0.0006 | 0.0307 | 0.0338 | 0.0032 |
| 2598 | 2018-03-02 09:51:00+00:00 | Benign | 0.0041 | 0.0040 | 0.0002 | 0.0295 | 0.0274 | 0.0021 |
| 2667 | 2018-03-02 11:00:00+00:00 | Benign | 0.9696 | 0.9700 | 0.0004 | 0.9596 | 0.9607 | 0.0011 |

## Summary Metrics

- **Detection Ensemble**:
  - Mean Absolute Difference: `0.008399`
  - Max Absolute Difference: `0.047474`
  - Pearson Correlation: `0.999298`

- **Onset Ensemble**:
  - Mean Absolute Difference: `0.010748`
  - Max Absolute Difference: `0.058425`
  - Pearson Correlation: `0.999035`

## Technical Finding
1. The shared PCA projection maintains extremely high correlation with individual per-fold PCAs (r = 0.9993 detection, r = 0.9990 onset) because the top 32 principal components across 10-day network telemetry capture macroscopic traffic variance modes common to all folds.
2. The slight numerical divergence arises because individual LOEO training folds excluded one attack episode when fitting their PCA basis.
3. For exact LOEO benchmark replication matching `stacked_benchmark_results.json` to 6 decimal places, evaluate via `evaluation/benchmark/reproduce_stacked_benchmark.py` which applies per-fold PCA. The dashboard runtime uses `pca_32_stacked.pkl` for single-pass vectorization efficiency.
