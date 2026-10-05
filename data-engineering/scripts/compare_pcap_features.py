"""
Where do the windows built from a raw capture differ from the published dataset's windows for the same minutes?

check_pcap_capture.py showed that the PCAP upload path and the dataset path give different onset probabilities on
the same minutes. This script finds which of the 406 model features cause it. For every minute of the slice it
builds the model input from the capture (exactly as the upload path does) and from the dataset, then reports:

  * per feature: the mean difference in model units (standardised), separately for benign and attack minutes;
  * per feature: the difference in its contribution to the onset model's LR-stage logit (coefficient x
    standardised value, averaged over the 37 onset folds). This ranks features by how much they move the score;
  * the same, summed by feature family (flow timing, sizes, flags, packet-level, masks, ...).

Usage (same slice as check_pcap_capture.py):
    python data-engineering/scripts/compare_pcap_features.py
    python data-engineering/scripts/compare_pcap_features.py --pcap path/to/slice.pcap --top 25

Writes runtime/pcap_feature_diff.csv (one row per feature) and runtime/pcap_feature_diff_by_minute.csv.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_pcap_capture as cpc  # noqa: E402  (shared paths, clock mapping and quiet mode)

REPO = cpc.REPO

FAMILIES = [
    ("masks", r"^mask_"),
    ("packet-level (raw PCAP parser)", r"^pkt_"),
    ("flow timing: IAT", r"iat"),
    ("flow timing: active/idle", r"active|idle"),
    ("flow timing: duration", r"duration"),
    ("bulk / subflow", r"_b_avg|blk|subflow"),
    ("rates", r"per_sec|_rate|byts_s|pkts_s"),
    ("TCP flags", r"(^|_)(flag|flags|syn|fin|rst|psh|ack|urg|cwe|ece)(_|$)"),
    ("window size / header", r"window|header|hdr|seg_size"),
    ("packet length / bytes", r"len|byts|bytes|size|payload"),
    ("packet counts", r"count|pkts|packets"),
    ("ports / protocol / ratio", r"port|proto|ratio|down_up"),
]


def family(name: str) -> str:
    for fam, pat in FAMILIES:
        if re.search(pat, name, flags=re.I):
            return fam
    return "other"


def build(pcap: Path):
    os.environ["SHADOWCAT_RUNTIME_DIR"] = os.environ.get("SHADOWCAT_CHECK_RUNTIME_DIR", str(REPO / "runtime" / "pcap_check"))
    from src.pcap_ingest import ingest_pcap
    from src.ucs_extractor import UCSExtractor
    from backend.predict import get_pipeline

    print(f"Ingesting {pcap} ...", flush=True)
    flows = ingest_pcap(str(pcap), progress=True)
    ex = UCSExtractor()
    with cpc.quiet():
        win = ex.extract(flows, source_type="pcap")
        pipe = get_pipeline()
    win["window_start_utc"] = pd.to_datetime(win["window_start_utc"], utc=True)
    win = win.sort_values("window_start_utc").reset_index(drop=True)
    with cpc.quiet():
        x_pcap = ex.extract_model_tensor(win, source_type="windows").astype(np.float64)

    ds = pd.read_parquet(cpc.DATASET)
    ds["window_start_utc"] = pd.to_datetime(ds["window_start_utc"], utc=True)
    ds = ds.sort_values("window_start_utc").reset_index(drop=True)
    ds_minutes = set(ds.window_start_utc)
    pairs = []
    for i, t in enumerate(win.window_start_utc):
        d_t = cpc.capture_to_dataset_minute(t, ds_minutes)
        if d_t is not None:
            pairs.append((i, int(ds.index[ds.window_start_utc == d_t][0]), t))
    if not pairs:
        raise SystemExit("No capture minute matches a dataset minute; is this a CIC-IDS2018 capture slice?")
    rows_ds = ds.loc[[j for _, j, _ in pairs]]
    with cpc.quiet():
        x_ds = ex.extract_model_tensor(rows_ds, source_type="windows").astype(np.float64)
    x_pc = x_pcap[[i for i, _, _ in pairs]]
    attack = rows_ds["future_attack_label"].to_numpy().astype(int)
    contrib = lambda X: np.stack([pipe._lr_contributions(x) for x in X])  # noqa: E731
    return x_pc, x_ds, contrib(x_pc), contrib(x_ds), attack, [t for _, _, t in pairs]


def report(x_pc, x_ds, c_pc, c_ds, attack, minutes, top: int, out_dir: Path):
    from src.ucs_extractor import UCSExtractor
    cols = list(UCSExtractor.MODEL_INPUT_COLUMNS)
    d, dc = x_pc - x_ds, c_pc - c_ds
    ben, att = attack == 0, attack == 1

    def m(a, mask):
        return a[mask].mean(axis=0) if mask.any() else np.full(a.shape[1], np.nan)

    feat = pd.DataFrame({
        "feature": cols,
        "family": [family(c) for c in cols],
        "mean_value_pcap": x_pc.mean(axis=0), "mean_value_dataset": x_ds.mean(axis=0),
        "mean_diff_benign": m(d, ben), "mean_diff_attack": m(d, att),
        "mean_abs_diff": np.abs(d).mean(axis=0),
        "logit_contrib_diff_benign": m(dc, ben), "logit_contrib_diff_attack": m(dc, att),
    })
    feat["abs_logit_contrib_diff"] = np.abs(dc).mean(axis=0)
    feat = feat.sort_values("abs_logit_contrib_diff", ascending=False).reset_index(drop=True)

    lg_pc, lg_ds = c_pc.sum(axis=1), c_ds.sum(axis=1)
    print(f"\nMinutes compared: {len(attack)} ({int(att.sum())} attack-within-5-min, {int(ben.sum())} benign)")
    print(f"LR-stage logit, mean over minutes: PCAP {lg_pc.mean():+.2f} vs dataset {lg_ds.mean():+.2f}"
          f"  (benign {lg_pc[ben].mean():+.2f} vs {lg_ds[ben].mean():+.2f}; attack {lg_pc[att].mean():+.2f} vs {lg_ds[att].mean():+.2f})")

    fam = feat.groupby("family").agg(
        features=("feature", "size"),
        logit_diff_benign=("logit_contrib_diff_benign", "sum"),
        logit_diff_attack=("logit_contrib_diff_attack", "sum"),
        mean_abs_value_diff=("mean_abs_diff", "mean"),
    ).sort_values("logit_diff_attack", key=lambda s: s.abs(), ascending=False)
    print("\nBY FEATURE FAMILY (logit difference = PCAP minus dataset, summed over the family)")
    print(fam.round(3).to_string())

    print(f"\nTOP {top} FEATURES by |difference in contribution to the onset logit|")
    show = feat.head(top)[["feature", "family", "mean_value_pcap", "mean_value_dataset",
                           "logit_contrib_diff_benign", "logit_contrib_diff_attack"]]
    with pd.option_context("display.width", 200, "display.max_colwidth", 48):
        print(show.round(3).to_string(index=False))

    flat = [c for c in cols if np.ptp(x_pc[:, cols.index(c)]) == 0 and np.ptp(x_ds[:, cols.index(c)]) > 0]
    print(f"\nFeatures constant across all minutes in the PCAP windows but varying in the dataset: {len(flat)}")
    if flat:
        print("  " + ", ".join(flat[:30]) + (" ..." if len(flat) > 30 else ""))

    out_dir.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out_dir / "pcap_feature_diff.csv", index=False)
    pd.DataFrame({"capture_minute_utc": minutes, "attack_within_5min": attack,
                  "lr_logit_pcap": lg_pc.round(4), "lr_logit_dataset": lg_ds.round(4)}).to_csv(
        out_dir / "pcap_feature_diff_by_minute.csv", index=False)
    print(f"\nwrote {out_dir / 'pcap_feature_diff.csv'} and {out_dir / 'pcap_feature_diff_by_minute.csv'}")
    return feat, fam


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcap", default=str(cpc.DEFAULT_PCAP))
    ap.add_argument("--out-dir", default=str(REPO / "runtime"))
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    cpc.VERBOSE = a.verbose
    pcap = Path(a.pcap)
    if not pcap.exists():
        raise SystemExit(f"{pcap} not found (see check_pcap_capture.py for how to make the slice)")
    report(*build(pcap), top=a.top, out_dir=Path(a.out_dir))


if __name__ == "__main__":
    main()
