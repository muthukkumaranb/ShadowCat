"""s2b-t2: build the three real demo slices for the dashboard.

Each slice covers 32 consecutive 1-minute windows (30-window lookback + 2) and is written twice:
  <name>.csv.gz          raw CICFlowMeter rows copied unmodified (all 80 columns as text) from the
                         CSE-CIC-IDS2018 "TrafficForML_CICFlowMeter" CSV for those 32 minutes
  <name>_windows.parquet the 32 matching rows of data-engineering/data/ucs/ucs_windows_models_v1.parquet,
                         the UCS windows the deployed models were trained and validated on

Slices (episodes as labelled in ucs_windows_models_v1.parquet):
  benign_02-03-2018  02-03-2018: first 32 minutes with no attack-labelled raw row, at least 60
                     minutes from any attack-labelled raw row; every window label_binary = 0
                     and future_attack_label = 0
  botnet_02-03-2018  02-03-2018: 27 windows before episode 02-03-2018_Botnet_0 + its first 5 windows
  ssh_14-02-2018     14-02-2018: 27 windows before episode 14-02-2018_SSH-Bruteforce_6 + its first
                     5 windows (02-03-2018 has no SSH-Bruteforce traffic)

slices.json records source files, rows, time range, raw and window labels, and a parity check of
the live CSV ingestion path (UCSExtractor, source_type="csv") against the training windows.

The raw directory defaults to paths.raw_data_dir in data-engineering/configs/pipeline_config.yaml
(relative to data-engineering/); pass --raw-dir to point at another checkout's raw data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data-engineering"))
from src.ucs_extractor import UCSExtractor  # noqa: E402

OUT = REPO / "frontend" / "demo_data"
WINDOWS = REPO / "data-engineering/data/ucs/ucs_windows_models_v1.parquet"
TS_FMT = "%d/%m/%Y %H:%M:%S"
SLICE_MIN = 32
LEAD_MIN = 27
CHUNK = 200_000
ONE = pd.Timedelta(minutes=1)


def raw_index(path):
    parts = []
    for ch in pd.read_csv(path, usecols=["Timestamp", "Label"], dtype=str, chunksize=CHUNK):
        parts.append(ch[ch["Timestamp"] != "Timestamp"])  # some raw files repeat the header row
    df = pd.concat(parts, ignore_index=True)
    df["minute"] = pd.to_datetime(df["Timestamp"], format=TS_FMT, errors="coerce").dt.floor("min")
    return df


def benign_start(idx, win_day):
    attack = pd.Series(sorted(idx.loc[idx["Label"] != "Benign", "minute"].dropna().unique()))
    present = set(idx["minute"].dropna().unique())
    clean = set(win_day.loc[(win_day.label_binary == 0) & (win_day.future_attack_label == 0), "t"].dt.tz_localize(None))
    for start in sorted(present):
        start = pd.Timestamp(start)
        mins = [start + k * ONE for k in range(SLICE_MIN)]
        if ((attack >= start - 60 * ONE) & (attack < start + (SLICE_MIN + 60) * ONE)).any():
            continue
        if all(m in present and m in clean for m in mins):
            return start
    raise SystemExit("no benign range found")


def episode_start(win_day, episode):
    onset = win_day.loc[win_day.episode_id == episode, "t"].min().tz_localize(None)
    return onset - LEAD_MIN * ONE, onset


def build(name, raw_path, start, onset, windows, extractor):
    end = start + SLICE_MIN * ONE
    kept = []
    for ch in pd.read_csv(raw_path, dtype=str, keep_default_na=False, chunksize=CHUNK):
        ch = ch[ch["Timestamp"] != "Timestamp"]
        m = pd.to_datetime(ch["Timestamp"], format=TS_FMT, errors="coerce").dt.floor("min")
        kept.append(ch[(m >= start) & (m < end)])
    raw = pd.concat(kept)
    raw.to_csv(OUT / f"{name}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})

    t = windows["t"].dt.tz_localize(None)
    win = windows[(t >= start) & (t < end)].drop(columns=["t"])
    assert len(win) == SLICE_MIN, f"{name}: expected {SLICE_MIN} windows, got {len(win)}"
    win.to_parquet(OUT / f"{name}_windows.parquet", index=False)

    raw_minutes = pd.to_datetime(raw["Timestamp"], format=TS_FMT).dt.floor("min").nunique()
    csv_win = extractor.extract(raw, source_type="csv")
    csv_win.index = pd.to_datetime(csv_win["window_start_utc"], utc=True)
    ref = win.copy()
    ref.index = pd.to_datetime(ref["window_start_utc"], utc=True)
    common = ref.index.intersection(csv_win.index)
    flow_cols = [c for c in UCSExtractor.MODEL_FEATURE_COLUMNS if c not in UCSExtractor.PCAP_PACKET_COLUMNS]
    diff = (csv_win.loc[common, flow_cols] - ref.loc[common, flow_cols]).abs()
    return {
        "raw_csv": f"frontend/demo_data/{name}.csv.gz",
        "windows": f"frontend/demo_data/{name}_windows.parquet",
        "source_file": raw_path.name,
        "rows": int(len(raw)),
        "raw_minutes_present": int(raw_minutes),
        "time_range_utc": [str(start), str(end - ONE)],
        "episode_onset_utc": str(onset) if onset is not None else None,
        "raw_label_counts": {k: int(v) for k, v in raw["Label"].value_counts().items()},
        "window_label_counts": {k: int(v) for k, v in win["label_attack_type"].value_counts().items()},
        "last_window": {k: (int(win.iloc[-1][k]) if k != "label_attack_type" else str(win.iloc[-1][k]))
                        for k in ("label_binary", "future_attack_label", "label_attack_type")},
        "csv_path_parity": {
            "windows_compared": int(len(common)),
            "flow_feature_columns": len(flow_cols),
            "flow_columns_within_1e-3_on_all_windows": int((diff < 1e-3).all().sum()),
            "max_abs_diff_scaled_units": float(np.nanmax(diff.to_numpy())),
            "mask_has_packet_level_features": {"csv_path": float(csv_win["mask_has_packet_level_features"].iloc[-1]),
                                               "training_windows": float(win["mask_has_packet_level_features"].iloc[-1])},
        },
    }


def main():
    cfg = yaml.safe_load((REPO / "data-engineering/configs/pipeline_config.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(REPO / "data-engineering" / cfg["paths"]["raw_data_dir"]))
    args = ap.parse_args()
    raw_dir = Path(args.raw_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    windows = pd.read_parquet(WINDOWS)
    windows["t"] = pd.to_datetime(windows["window_start_utc"], utc=True)
    extractor = UCSExtractor(schema_version="v3.0", config_dir=str(REPO / "data-engineering/configs"),
                             data_dir=str(REPO / "data-engineering/data/ucs"))
    meta = {"generated_by": "frontend/scripts/build_demo_slices.py", "dataset": "CSE-CIC-IDS2018",
            "windows_source": str(WINDOWS.relative_to(REPO).as_posix()), "slices": {}}

    day = raw_dir / "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
    w_day = windows[windows.source_day == "02-03-2018"]
    meta["slices"]["benign"] = build("benign_02-03-2018", day, benign_start(raw_index(day), w_day), None, w_day, extractor)
    start, onset = episode_start(w_day, "02-03-2018_Botnet_0")
    meta["slices"]["botnet"] = build("botnet_02-03-2018", day, start, onset, w_day, extractor)

    day = raw_dir / "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
    w_day = windows[windows.source_day == "14-02-2018"]
    start, onset = episode_start(w_day, "14-02-2018_SSH-Bruteforce_6")
    meta["slices"]["ssh"] = build("ssh_14-02-2018", day, start, onset, w_day, extractor)

    # Deployed predict() on both inputs of every slice
    sys.path.insert(0, str(REPO / "frontend"))
    from data_provider import run_core_ml_inference
    for s in meta["slices"].values():
        out = {}
        for kind, path, reader in (("windows", s["windows"], pd.read_parquet), ("csv", s["raw_csv"], pd.read_csv)):
            res = run_core_ml_inference(reader(REPO / path), source_type=kind)
            out[kind] = {"onset_probability": res.get("forecast_trajectory", {}).get("onset_probability"),
                         "detection_probability": res.get("detection_probability"),
                         "error": res.get("_inference_error")}
        s["model_output"] = out

    (OUT / "slices.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
