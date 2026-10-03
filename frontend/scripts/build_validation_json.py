"""s2b-t1: build frontend/models/loeo_37fold_results.json for the Validation page.

Every number in the output is copied from a committed results file:
  - ml1/artifacts/lstm/lstm_stacked/{detection,onset}/sidecar_fold_*.json   (stacked model, per fold)
  - ml1/artifacts/lr_set_a_corrected/lr_{detection,onset}_loeo_per_fold.csv (LR baseline, per fold)
  - evaluation/benchmark/stacked_benchmark_results.json                     (matched-FPR, ablation)
Means are plain means over the 37 folds. Run from anywhere:
    python frontend/scripts/build_validation_json.py
"""
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STACKED = REPO / "ml1/artifacts/lstm/lstm_stacked"
LR_DIR = REPO / "ml1/artifacts/lr_set_a_corrected"
BENCH = REPO / "evaluation/benchmark/stacked_benchmark_results.json"
OUT = REPO / "frontend/models/loeo_37fold_results.json"
METRICS = ("f1", "precision", "recall", "fpr", "roc_auc")


def rel(p):
    return p.relative_to(REPO).as_posix()


def stacked_folds(task):
    rows = []
    for f in sorted((STACKED / task).glob("sidecar_fold_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        s = json.loads(f.read_text())
        rows.append({
            "fold_id": s["fold_id"],
            "held_out_episode_id": s["held_out_episode_id"],
            "attack_type": s["attack_type"],
            **{m: s[f"test_{m}"] for m in METRICS},
        })
    return rows


def lr_folds(task):
    with open(LR_DIR / f"lr_{task}_loeo_per_fold.csv", newline="") as fh:
        return {int(r["fold_id"]): {m: (float(r[m]) if r[m] != "" else None) for m in METRICS} | {"n_test": int(r["n_test"])}
                for r in csv.DictReader(fh)}


def mean(rows, key):
    """Mean over folds where the metric is defined (ROC-AUC is undefined on single-class folds)."""
    vals = [r[key] for r in rows if r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    bench = json.loads(BENCH.read_text())
    out = {
        "generated_by": "frontend/scripts/build_validation_json.py",
        "sources": {
            "stacked_per_fold": rel(STACKED) + "/{detection,onset}/sidecar_fold_*.json",
            "lr_per_fold": rel(LR_DIR) + "/lr_{detection,onset}_loeo_per_fold.csv",
            "matched_fpr_and_ablation": rel(BENCH),
            "benchmark_data_file": bench["data_file"],
        },
        "protocol": "Leave-One-Episode-Out, 37 folds (ml1/artifacts/loeo/corrected_37fold_manifest.json); "
                    "threshold 0.5 for per-fold metrics",
        "tasks": {},
        "h_star_note": (
            "The world model's multi-step rollout does not beat a persistence baseline at any K=1..5 "
            "(predictability horizon H* = 0); see evaluation/horizon_analysis/advancement_summary.md."
        ),
        "lr_results_caveat": (
            "ml1/artifacts/lr_set_a_corrected/ also contains SUPERSEDED_INVALID_PROTOCOL.md, committed in the "
            "same commit as these LR results (5d3641f)."
        ),
    }
    for task, target in (("detection", "label_binary"), ("onset", "future_attack_label")):
        st = stacked_folds(task)
        lr = lr_folds(task)
        per_fold = []
        for r in st:
            l = lr[r["fold_id"]]
            per_fold.append({
                "fold_id": r["fold_id"], "held_out_episode_id": r["held_out_episode_id"],
                "attack_type": r["attack_type"], "n_test": l["n_test"],
                "stacked_f1": r["f1"], "lr_f1": l["f1"],
            })
        lr_rows = list(lr.values())
        b = bench[task]
        out["tasks"][task] = {
            "target": target,
            "folds": len(st),
            "stacked_mean": {m: mean(st, m) for m in METRICS},
            "roc_auc_defined_folds": {"stacked": sum(1 for r in st if r["roc_auc"] is not None),
                                      "lr": sum(1 for r in lr_rows if r["roc_auc"] is not None)},
            "lr_mean": {m: mean(lr_rows, m) for m in METRICS},
            "lr_folds_f1_eq_1": sum(1 for r in lr_rows if r["f1"] == 1.0),
            "lr_folds_f1_eq_0": [fid for fid, r in sorted(lr.items()) if r["f1"] == 0.0],
            "per_fold": per_fold,
            "matched_fpr": {
                "test_windows": b["test_windows"],
                "positives": b["positives"],
                "stacked": {k: b["matched_fpr_5pct"]["stacked_ensemble"][k] for k in ("threshold", "f1", "precision", "recall", "fpr")},
                "lr": {k: b["matched_fpr_5pct"]["lr_baseline"][k] for k in ("threshold", "f1", "precision", "recall", "fpr")},
                "note": b["matched_fpr_5pct"]["stacked_ensemble"]["note"],
            },
            "history_ablation_f1_at_0.5": {k: v["f1"] for k, v in b["history_ablation_at_0.5"].items()},
        }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {rel(OUT)}")


if __name__ == "__main__":
    main()
