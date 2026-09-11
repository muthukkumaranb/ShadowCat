# Slot Normalization Compatibility

Protocol: chronological ML2 canonical window population. The chronological
ML1 export covers 2275/2787 windows overall and 837/837 validation+test windows.

## Own graph slots 0-9

These summaries are over active nodes in the covered canonical windows:

| Slot | Mean | Std | Min | Median | Max | P01 | P99 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.875552 | 1.16515 | 0 | 1 | 66 | 0 | 3 |
| 1 | 0.875552 | 33.5548 | 0 | 0 | 7222 | 0 | 11 |
| 2 | 5.49284 | 182.984 | 0 | 1 | 47576 | 0 | 75 |
| 3 | 5.49284 | 187.657 | 0 | 0 | 23975 | 0 | 38 |
| 4 | 11611.5 | 1477460 | 0 | 0 | 304696000 | 0 | 17553 |
| 5 | 11611.5 | 1481010 | 0 | 0 | 304697000 | 0 | 9804.18 |
| 6 | 313.498 | 45819.7 | 0 | 2 | 9521760 | 0 | 404 |
| 7 | 313.498 | 45823.1 | 0 | 0 | 9521760 | 0 | 165 |
| 8 | 14.2645 | 31.8504 | 0 | 0.000104 | 119.998 | 0 | 88.3851 |
| 9 | 4.37492 | 18.8825 | 0 | 0 | 120 | 0 | 101.167 |

## Slot 10

Slot 10 is ML1's `normalized_deviation_score`, already transformed by log1p
and robust normalization per chronological training chunk. Its observed range
is approximately `-1.0305` to `94.1063`, with median approximately `-0.02884`.
Because this is a normalized score, applying another `log1p` is invalid for
values below `-1` and produces NaNs. The ML2 sanity check now reports slot 10
directly.

## Compatibility conclusion

Slot 10 is numerically on a different scale from the raw graph aggregation
slots 0-9, whose byte and packet features have very large heavy-tailed ranges.
It is nevertheless compatible as a pre-normalized fusion input: it must remain
untouched while slots 0-9 use the training-only `NodeFeatureScaler`. Any future
rescaling must be fitted on the chronological training portion only.