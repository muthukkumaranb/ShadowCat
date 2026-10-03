"""s2b-t10: run the deployed backend.predict() over every window of the models' dataset.

For each window i of data-engineering/data/ucs/ucs_windows_models_v1.parquet (sorted by
window_start_utc) that has 29 predecessors, predict() is called on rows i-29..i with
source_type="windows" (the same 30-row history the LOEO benchmark uses). Per-window outputs go
to evaluation/demo_sweep_windows.csv; the summary goes to evaluation/demo_sweep_results.json:
share of windows alerting, onset/detection probability distribution for benign vs attack
windows, and stage distribution. Progress is resumable (rows already in the CSV are skipped).

Note: every window here belongs to the LOEO dataset, so 36 of the 37 fold models saw it in
training; this sweep checks dashboard behaviour, it is not a held-out evaluation.
Set SHADOWCAT_RUNTIME_DIR to keep the per-call alert/lineage records out of runtime/.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backend"))

DATA = "data-engineering/data/ucs/ucs_windows_models_v1.parquet"
ROWS = REPO / "evaluation/demo_sweep_windows.csv"
OUT = REPO / "evaluation/demo_sweep_results.json"
LOOKBACK = 30


def dist(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {"n": 0}
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(x)), "mean": float(x.mean()), "p05": float(q[0]), "p25": float(q[1]),
            "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4]),
            "share_ge_0.5": float((x >= 0.5).mean())}


PARTS = REPO / "runtime/demo_sweep_parts"  # gitignored scratch for parallel shards


def all_rows():
    files = ([ROWS] if ROWS.exists() else []) + sorted(PARTS.glob("part*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if not args.summarize_only:
        sweep(args.shard, args.num_shards)
    if args.num_shards == 1 or args.summarize_only:
        summarize()


def sweep(shard, num_shards):
    from backend.predict import get_pipeline

    df = pd.read_parquet(REPO / DATA).sort_values("window_start_utc").reset_index(drop=True)
    prev = all_rows()
    done = set(prev["window_index"]) if len(prev) else set()
    out_file = ROWS if num_shards == 1 else PARTS / f"part{shard}.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pipe = get_pipeline()
    buf = []
    for i in range(LOOKBACK - 1, len(df)):
        if i in done or i % num_shards != shard:
            continue
        res = pipe.predict(df.iloc[i - LOOKBACK + 1: i + 1], source_type="windows")
        fc = res["forecast_trajectory"]
        md = fc["mitre_details"][0]
        buf.append({
            "window_index": i,
            "window_id": df.loc[i, "window_id"],
            "label_binary": int(df.loc[i, "label_binary"]),
            "future_attack_label": int(df.loc[i, "future_attack_label"]),
            "label_attack_type": df.loc[i, "label_attack_type"],
            "onset_probability": fc["onset_probability"],
            "onset_alert": bool(fc["onset_alert"]),
            "detection_probability": res["detection_probability"],
            "stage_t1": fc["stage"][0],
            "stage_head_top_class_t1": md["stage_head_top_class"],
            "stage_confidence_t1": md["stage_confidence"],
        })
        if len(buf) >= 20:
            pd.DataFrame(buf).to_csv(out_file, mode="a", header=not out_file.exists(), index=False)
            buf = []
            print(f"shard {shard}: window {i}/{len(df) - 1}", flush=True)
    if buf:
        pd.DataFrame(buf).to_csv(out_file, mode="a", header=not out_file.exists(), index=False)


def summarize():
    r = all_rows().sort_values("window_index").drop_duplicates("window_index")
    r.to_csv(ROWS, index=False)
    benign = r[r.label_binary == 0]
    attack = r[r.label_binary == 1]
    clean = r[(r.label_binary == 0) & (r.future_attack_label == 0)]
    summary = {
        "data": DATA,
        "windows_swept": int(len(r)),
        "note": "all windows are LOEO dataset windows (in training for 36 of 37 fold models); not a held-out result",
        "alert_threshold": 0.5,
        "share_alerting_all": float(r.onset_alert.mean()),
        "share_alerting_benign_label_binary_0": float(benign.onset_alert.mean()),
        "share_alerting_benign_and_no_attack_within_5min": float(clean.onset_alert.mean()),
        "share_alerting_attack_label_binary_1": float(attack.onset_alert.mean()),
        "benign_windows_all_alert": bool(len(benign) > 0 and benign.onset_alert.all()),
        "onset_probability": {"benign": dist(benign.onset_probability), "attack": dist(attack.onset_probability),
                              "benign_and_no_attack_within_5min": dist(clean.onset_probability)},
        "detection_probability": {"benign": dist(benign.detection_probability), "attack": dist(attack.detection_probability)},
        "stage_shown_t1": {k: int(v) for k, v in r.stage_t1.value_counts().items()},
        "stage_head_top_class_t1": {k: int(v) for k, v in r.stage_head_top_class_t1.value_counts().items()},
        "stage_shown_t1_by_attack_type": {
            t: {k: int(v) for k, v in g.stage_t1.value_counts().items()} for t, g in r.groupby("label_attack_type")
        },
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("windows_swept", "share_alerting_all", "share_alerting_benign_label_binary_0",
                                               "benign_windows_all_alert")}, indent=2))


if __name__ == "__main__":
    main()
