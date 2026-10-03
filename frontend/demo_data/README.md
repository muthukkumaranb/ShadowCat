# Demo slices (real CSE-CIC-IDS2018 data)

Built by `frontend/scripts/build_demo_slices.py`; all numbers below are in `slices.json`.
The script reads the raw CICFlowMeter CSVs from `paths.raw_data_dir` in
`data-engineering/configs/pipeline_config.yaml` (override with `--raw-dir`).

| Slice | Source file | Raw rows | Time range (UTC, 32 one-minute windows) | Window labels (dataset) |
|---|---|---|---|---|
| `benign_02-03-2018` | `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` | 47,714 | 2018-03-02 04:55 - 05:26 | 32 Benign |
| `botnet_02-03-2018` | `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` | 66,026 | 2018-03-02 03:04 - 03:35 | 27 Benign, then 5 Botnet (episode `02-03-2018_Botnet_0`, onset 03:31) |
| `ssh_14-02-2018` | `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` | 66,513 | 2018-02-14 02:17 - 02:48 | 28 SSH-Bruteforce, 4 Benign; ends 5 windows into episode `14-02-2018_SSH-Bruteforce_6` (onset 02:44) |

Each slice has two files:

- `<slice>.csv.gz`: the raw CICFlowMeter rows for those minutes, copied unmodified (80 columns, including the raw `Label`).
- `<slice>_windows.parquet`: the 32 matching rows of `data-engineering/data/ucs/ucs_windows_models_v1.parquet`,
  the UCS windows the deployed models were trained and validated on. The dashboard's demo buttons run these
  (`source_type="windows"`).

Notes:

- SSH-Bruteforce traffic exists only on 14-02-2018, so that slice comes from 14-02-2018, not 02-03-2018.
- These windows are part of the LOEO dataset: each was in the training split of most of the 37 fold models.
  The demo shows the pipeline on real traffic; it is not a held-out result.
- `csv_path_parity` in `slices.json` compares the live CSV extractor with the training windows. On 02-03-2018
  all 388 flow features match; on 14-02-2018 only 110 of 388 do. On every slice the CSV route has no
  packet-level features (`mask_has_packet_level_features = 0`, while the training windows have 1), and
  `model_output.csv` shows near-zero probabilities for all three slices.
