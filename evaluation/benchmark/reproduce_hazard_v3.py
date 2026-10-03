"""s2-fix1b: reproduce the reported hazard_head_v3 LOEO ROC-AUC (0.789 / 0.843 / 0.770 at H=1/2/5)
from the committed fold models in ml1/artifacts/lstm/hazard_head_v3/H{1,2,5}/model_fold_*.pt.

Two input representations are evaluated, because the history is ambiguous:

  pca32        models/pca_32.pkl + its StandardScaler (what backend/predict.py feeds the hazard heads;
               this is the procedure in the CURRENT ml1/scripts/train_hazard_head.py).
  refit_train  a fresh UCSPCA(32, random_state=42) fitted per fold on that fold's training rows only,
               unscaled -- the procedure in ml1/scripts/train_hazard_head.py at commit 371b855, which is
               the commit that added the hazard_head_v3 checkpoints (models/pca_32.pkl was first committed
               10 days later, in ce166a2).

Data can be a repo-relative parquet (--data) or a parquet blob read from git history (--data-rev),
so the 371b855 training file can be used without committing another copy of it.

Targets, folds, sequence building and metrics follow train_hazard_head.py exactly:
  H=1/H=2 = label_binary shifted by -1/-2 within source_day, H=5 = future_attack_label,
  per-fold ROC-AUC on the fold's test windows, mean over folds with both classes present.

Writes evaluation/benchmark/hazard_v3_results.json.
"""
import argparse
import io
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ml1"))

from lstm.model import LSTMClassifier  # noqa: E402
from lstm.ucs import UCSConfig, apply_training_only_pca, validate_ucs_windows  # noqa: E402

REPORTED_ROC = {1: 0.7893, 2: 0.8432, 5: 0.7701}
TOLERANCE = 0.01
HAZARD_DIR = REPO / "ml1/artifacts/lstm/hazard_head_v3"
PCA_COLS = [f"pca_{i}" for i in range(32)]


def load_windows(data, data_rev):
    if data_rev:
        blob = subprocess.run(
            ["git", "show", f"{data_rev}:data-engineering/data/ucs/ucs_windows.parquet"],
            cwd=REPO, capture_output=True, check=True,
        ).stdout
        df = pd.read_parquet(io.BytesIO(blob))
        source = f"git:{data_rev}:data-engineering/data/ucs/ucs_windows.parquet"
    else:
        df = pd.read_parquet(REPO / data)
        source = data
    return df.sort_values("window_start_utc").reset_index(drop=True), source


def load_model(path):
    m = LSTMClassifier(input_size=32, hidden_size=64, num_layers=1, dropout=0.2)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data-engineering/data/ucs/ucs_windows_models_v1.parquet")
    ap.add_argument("--data-rev", default=None, help="read ucs_windows.parquet from this git revision instead")
    ap.add_argument("--pca-mode", choices=["pca32", "refit_train"], required=True)
    ap.add_argument("--out", default="evaluation/benchmark/hazard_v3_results.json")
    args = ap.parse_args()

    windows, source = load_windows(args.data, args.data_rev)
    manifest = json.loads((REPO / "ml1/artifacts/loeo/corrected_37fold_manifest.json").read_text())
    config = UCSConfig()
    features = validate_ucs_windows(windows, config=config)
    windows["target_h1"] = windows.groupby("source_day")["label_binary"].shift(-1).fillna(0)
    windows["target_h2"] = windows.groupby("source_day")["label_binary"].shift(-2).fillna(0)
    windows["target_h5"] = windows["future_attack_label"]
    committed = pd.read_csv(HAZARD_DIR / "hazard_head_all_horizons_loeo.csv")

    if args.pca_mode == "pca32":
        with open(REPO / "models/pca_32.pkl", "rb") as f:
            art = pickle.load(f)
        scaled = pd.DataFrame(art["scaler"].transform(windows[features].to_numpy(np.float64)),
                              columns=features, index=windows.index)
        shared = art["pca"].transform(scaled)[PCA_COLS].to_numpy(np.float32)

    per_h = {}
    for h in (1, 2, 5):
        target = windows[f"target_h{h}"].to_numpy(np.float32)
        folds = []
        for fold in manifest["folds"]:
            fid = fold["fold_id"]
            train_idx = np.array(fold["train_indices"])
            test_idx = np.array(fold["test_indices"])
            if args.pca_mode == "pca32":
                z = shared
            else:
                frame = windows.copy()
                frame["split"] = "none"
                frame.loc[test_idx, "split"] = "test"
                tr = frame.loc[train_idx].sort_values("window_start_utc")
                cut = int(len(tr) * 0.8)
                frame.loc[tr.index[:cut], "split"] = "train"
                frame.loc[tr.index[cut:], "split"] = "val"
                purged = frame[frame["split"].isin(["train", "val", "test"])].copy()
                _, pca = apply_training_only_pca(purged, features, 32, random_state=42)
                z = pca.transform(frame)[PCA_COLS].to_numpy(np.float32)
            keep = [i for i in test_idx if i - config.lookback_windows + 1 >= 0]
            X = np.stack([z[i - config.lookback_windows + 1: i + 1] for i in keep]).astype(np.float32)
            y = target[keep].astype(int)
            with torch.no_grad():
                p = load_model(HAZARD_DIR / f"H{h}" / f"model_fold_{fid}.pt").predict_proba(
                    torch.as_tensor(X)).cpu().numpy().reshape(-1)
            two = len(np.unique(y)) > 1
            roc = float(roc_auc_score(y, p)) if two else None
            ref = committed[(committed.horizon == h) & (committed.fold_id == fid)]["roc_auc"]
            folds.append({
                "fold_id": fid, "n_test": int(len(y)), "positives": int(y.sum()),
                "roc_auc": roc, "pr_auc": float(average_precision_score(y, p)) if two else None,
                "committed_csv_roc_auc": None if ref.empty or pd.isna(ref.iloc[0]) else float(ref.iloc[0]),
            })
        rocs = [f["roc_auc"] for f in folds if f["roc_auc"] is not None]
        diffs = [abs(f["roc_auc"] - f["committed_csv_roc_auc"]) for f in folds
                 if f["roc_auc"] is not None and f["committed_csv_roc_auc"] is not None]
        mean_roc = float(np.mean(rocs))
        per_h[f"H{h}"] = {
            "reported_roc_auc_mean": REPORTED_ROC[h],
            "reproduced_roc_auc_mean": mean_roc,
            "abs_diff_vs_reported": abs(mean_roc - REPORTED_ROC[h]),
            "folds_with_both_classes": len(rocs),
            "max_abs_fold_diff_vs_committed_csv": float(max(diffs)) if diffs else None,
            "reproduces_within_tolerance": abs(mean_roc - REPORTED_ROC[h]) <= TOLERANCE,
            "folds": folds,
        }

    out_path = REPO / args.out
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results["tolerance"] = TOLERANCE
    results.setdefault("runs", {})
    results["runs"][f"{args.pca_mode}@{source}"] = {
        "pca_mode": args.pca_mode,
        "data": source,
        "all_horizons_reproduce": all(v["reproduces_within_tolerance"] for v in per_h.values()),
        "horizons": per_h,
    }
    out_path.write_text(json.dumps(results, indent=2))
    for k, v in per_h.items():
        print(f"{args.pca_mode} {source} {k}: reproduced {v['reproduced_roc_auc_mean']:.4f} "
              f"vs reported {v['reported_roc_auc_mean']:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
