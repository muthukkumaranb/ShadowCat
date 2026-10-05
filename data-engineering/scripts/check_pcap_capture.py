"""
End-to-end check of the PCAP upload path on a real CSE-CIC-IDS2018 capture.

The dashboard's PCAP upload runs: capture -> CICFlowMeter-schema flows + 12 packet features per minute
(src/pcap_ingest.py) -> UCS windows -> predict(). This script runs the same path over a capture slice, one
1-minute window at a time (each prediction sees only the windows up to that minute, as a live feed would),
and lines the result up against the published dataset for the same minutes:

  * the dataset's labels (attack now, attack within 5 minutes), and
  * the onset probability the same models give on the dataset's own windows (built from the CIC-IDS2018 CSVs).

Usage (after downloading the 14-02-2018 capture and cutting the demo slice):
    python data-engineering/scripts/download_pcap_14022018.py
    python data-engineering/scripts/make_pcap_demo_slice.py --preset ssh
    python data-engineering/scripts/check_pcap_capture.py
    python data-engineering/scripts/check_pcap_capture.py --pcap path/to/slice.pcap --out runtime/pcap_check.csv

Capture clock vs dataset clock: the CIC-IDS2018 CSV timestamps are offset from the capture clock by +16 h
(CSV time before 10:00) or +4 h (from 10:00), as documented in src/pcap_extractor.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "data-engineering", REPO / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATASET = REPO / "data-engineering" / "data" / "ucs" / "ucs_windows_models_v1.parquet"
DEFAULT_PCAP = REPO / "data-engineering" / "data" / "raw_pcap" / "demo_ssh_slice.pcap"


def capture_to_dataset_minute(ts: pd.Timestamp, dataset_minutes: set) -> pd.Timestamp | None:
    """Map a capture-clock minute to the dataset (CSV) clock with the documented offsets: capture = CSV + 16 h for
    CSV times before 10:00, CSV + 4 h from 10:00. Of the candidates that obey that rule, the one present in the
    dataset is used; None if no dataset window matches."""
    cands = [c for c, ok in ((ts - pd.Timedelta(hours=16), lambda c: c.hour < 10),
                             (ts - pd.Timedelta(hours=4), lambda c: c.hour >= 10)) if ok(c)]
    hits = [c for c in cands if c in dataset_minutes]
    return hits[0] if len(hits) == 1 else None


def onset_probability(pred: dict) -> float:
    return float((pred.get("forecast_trajectory", {}).get("risk") or [np.nan])[0])


def run(pcap: Path, out: Path | None, min_history: int = 1) -> pd.DataFrame:
    os.environ.setdefault("SHADOWCAT_RUNTIME_DIR", str(REPO / "runtime" / "pcap_check"))
    from src.pcap_ingest import ingest_pcap
    from src.ucs_extractor import UCSExtractor
    from backend.predict import predict

    print(f"Ingesting {pcap} ...", flush=True)
    flows = ingest_pcap(str(pcap), progress=True)
    windows = UCSExtractor().extract(flows, source_type="pcap")
    windows["window_start_utc"] = pd.to_datetime(windows["window_start_utc"], utc=True)
    windows = windows.sort_values("window_start_utc").reset_index(drop=True)
    n_flows = pd.to_datetime(flows["Timestamp"], format="%d/%m/%Y %H:%M:%S", utc=True).dt.floor("min").value_counts()
    print(f"{len(flows):,} flows, {len(windows)} one-minute windows "
          f"({windows.window_start_utc.iloc[0]} .. {windows.window_start_utc.iloc[-1]} capture clock)", flush=True)

    ds = pd.read_parquet(DATASET)
    ds["window_start_utc"] = pd.to_datetime(ds["window_start_utc"], utc=True)
    ds = ds.sort_values("window_start_utc").reset_index(drop=True)
    ds_minutes = set(ds.window_start_utc)

    rows = []
    for i in range(min_history - 1, len(windows)):
        t = windows.window_start_utc.iloc[i]
        p_pcap = onset_probability(predict(windows.iloc[: i + 1].copy(), source_type="windows"))
        d_t = capture_to_dataset_minute(t, ds_minutes)
        row = {"capture_minute_utc": t, "flows": int(n_flows.get(t, 0)), "p_onset_from_pcap": round(p_pcap, 4),
               "dataset_minute": d_t, "label_attack_now": np.nan, "label_attack_within_5min": np.nan,
               "p_onset_from_dataset": np.nan}
        if d_t is not None:
            hit = ds.index[ds.window_start_utc == d_t]
            if len(hit):
                j = int(hit[0])
                day = ds.window_start_utc.dt.date == d_t.date()
                hist = ds[day & (ds.index <= j)]
                row.update(label_attack_now=int(ds.at[j, "label_binary"]),
                           label_attack_within_5min=int(ds.at[j, "future_attack_label"]),
                           p_onset_from_dataset=round(onset_probability(predict(hist.copy(), source_type="windows")), 4))
        rows.append(row)
        print(f"  {t:%H:%M}  flows={row['flows']:>6}  P(onset) PCAP={row['p_onset_from_pcap']:.3f}"
              f"  dataset={row['p_onset_from_dataset'] if not pd.isna(row['p_onset_from_dataset']) else '   -'}"
              f"  label now/≤5min={row['label_attack_now']}/{row['label_attack_within_5min']}", flush=True)

    res = pd.DataFrame(rows)
    summarise(res)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out, index=False)
        print(f"\nwrote {out}")
    return res


def summarise(res: pd.DataFrame) -> None:
    from sklearn.metrics import roc_auc_score

    m = res.dropna(subset=["label_attack_within_5min", "p_onset_from_dataset"])
    print("\nSUMMARY")
    print(f"  windows checked: {len(res)}; matched to dataset minutes: {len(m)}")
    if len(m):
        diff = (m.p_onset_from_pcap - m.p_onset_from_dataset).abs()
        print(f"  PCAP path vs dataset path, onset probability: mean |diff| {diff.mean():.3f}, max {diff.max():.3f}")
        y = m.label_attack_within_5min.astype(int)
        if y.nunique() == 2:
            print(f"  onset ROC-AUC on this slice: PCAP path {roc_auc_score(y, m.p_onset_from_pcap):.3f}, "
                  f"dataset path {roc_auc_score(y, m.p_onset_from_dataset):.3f}  (n={len(m)}, {int(y.sum())} positive)")
        else:
            print("  onset ROC-AUC not defined on this slice (only one label present)")
        first = m[m.label_attack_now == 1]
        if len(first):
            t0 = first.capture_minute_utc.iloc[0]
            pre = m[(m.capture_minute_utc < t0)]
            print(f"  attack starts at {t0:%H:%M} capture clock; mean P(onset) before it "
                  f"{pre.p_onset_from_pcap.mean():.3f} (PCAP), after {first.p_onset_from_pcap.mean():.3f} (PCAP)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcap", default=str(DEFAULT_PCAP))
    ap.add_argument("--out", default=str(REPO / "runtime" / "pcap_check.csv"))
    a = ap.parse_args()
    pcap = Path(a.pcap)
    if not pcap.exists():
        raise SystemExit(f"{pcap} not found. Download the capture and cut a slice first:\n"
                         "  python data-engineering/scripts/download_pcap_14022018.py\n"
                         "  python data-engineering/scripts/make_pcap_demo_slice.py --preset ssh")
    run(pcap, Path(a.out) if a.out else None)


if __name__ == "__main__":
    main()
