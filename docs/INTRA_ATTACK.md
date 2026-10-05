# Intra-Attack Forecast

## Run Statistics
intra-attack forecasting not evaluable on this dataset
Counts: len>=6: 12, in_attack_windows_ending_within_5: 104, n_run_endings: 40

## Results
H*_intra: not evaluated (stop rule: 12 runs of length >= 6, 40 runs total)

## Limitations
- scripted CIC-IDS2018 attacks
- many very short runs (e.g. DDOS-LOIC-UDP)
- single-episode families
- 6 days

## Summary
- Intra-attack forecasting needs longer attack runs than this dataset contains (evaluation/intra_attack/runs.json:len_ge_6=12)
