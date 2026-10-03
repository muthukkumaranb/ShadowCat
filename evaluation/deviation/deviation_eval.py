"""s4-t1: how well does the world model's next-state deviation separate attack from benign windows?

Reuses the procedure of ml1/scripts/generate_loeo_deviation.py (the script that produced
ml1/artifacts/lstm/loeo_world_model/deviation_predictions.csv): per fold, a 32-dim PCA fitted
on that fold's training rows (apply_training_only_pca, random_state 42), an
LSTMGaussianWorldModel(32 -> 64 -> 32) trained with train_gaussian (epochs 2, batch 64, lr 1e-3,
weight decay 1e-4, patience 5, seed 42, last 20% of training rows chronologically = validation),
and deviation = deviation_scores(predicted mean, actual next PCA state) for each test window,
using the preceding 30 windows as history. Unlike the original, the score of each window is
compared with that window's own label_binary (attack vs benign).

Two protocols:
  LOEO      one fold per attack episode, built with the rule of
            ml1/scripts/build_and_verify_frozen_folds.py (episode windows + the same number,
            at least 5, of nearest same-day windows with label_binary = 0 and
            future_attack_label = 0; training rows within 35 minutes of any test row removed),
            for every attack type in the dataset.
  family-out for each attack family F: test = all days on which F occurs, train = every other
            day (so F is never seen in training); scored on F's attack windows vs the benign
            windows of those days (other families' attack windows on those days are excluded).

A family is "not evaluable" if it has fewer than 2 episodes.
Writes evaluation/deviation/deviation_results.json and evaluation/deviation/DEVIATION.md.
"""
import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ml1"))

from lstm.model import LSTMGaussianWorldModel  # noqa: E402
from lstm.probabilistic import deviation_scores, train_gaussian  # noqa: E402
from lstm.ucs import UCSConfig, apply_training_only_pca, build_next_state_sequences, validate_ucs_windows  # noqa: E402

OUT_DIR = REPO / "evaluation/deviation"
PCA_COLS = [f"pca_{i}" for i in range(32)]
PURGE_SECONDS = 2100.0


def load(data, rev):
    if rev:
        blob = subprocess.run(["git", "show", f"{rev}:{data}"], cwd=REPO, capture_output=True, check=True).stdout
        df = pd.read_parquet(io.BytesIO(blob))
    else:
        df = pd.read_parquet(REPO / data)
    df["window_start_utc"] = pd.to_datetime(df["window_start_utc"], utc=True)
    return df.sort_values("window_start_utc").reset_index(drop=True)


def loeo_folds(df):
    """Rule of ml1/scripts/build_and_verify_frozen_folds.py, applied to every attack type."""
    meta = df.groupby("episode_id")["label_attack_type"].agg(lambda v: sorted(set(v.astype(str))))
    episodes = [ep for ep, types in meta.items() if types[0] != "Benign"]
    times = df["window_start_utc"].values
    idx = df.index.values
    folds = []
    for ep in episodes:
        ep_mask = (df["episode_id"] == ep) | (df.get("forecast_episode_id", df["episode_id"]) == ep)
        ep_idx = idx[ep_mask]
        day = df.loc[ep_idx[0], "source_day"]
        n_pos = int(((df.loc[ep_idx, "label_binary"] == 1) | (df.loc[ep_idx, "future_attack_label"] == 1)).sum())
        cand = idx[(df["source_day"] == day) & (df["label_binary"] == 0) & (df["future_attack_label"] == 0) & ~ep_mask]
        start, end = df.loc[ep_idx, "window_start_utc"].min(), df.loc[ep_idx, "window_start_utc"].max()
        ct = df.loc[cand, "window_start_utc"]
        dist = np.where(ct < start, (start - ct).dt.total_seconds(), np.where(ct > end, (ct - end).dt.total_seconds(), 0.0))
        benign = cand[np.argsort(dist)[: max(n_pos, 5)]]
        test = np.concatenate([ep_idx, benign])
        train = np.setdiff1d(idx, test)
        d = np.abs((times[train][:, None] - times[test][None, :]).astype("timedelta64[s]").astype(float)).min(axis=1)
        folds.append({"name": ep, "family": meta[ep][0], "train": train[d > PURGE_SECONDS], "test": test})
    return folds


def family_out_folds(df):
    folds = []
    for fam in sorted(t for t in df["label_attack_type"].astype(str).unique() if t != "Benign"):
        days = df.loc[df["label_attack_type"] == fam, "source_day"].unique()
        test_mask = df["source_day"].isin(days)
        score_mask = test_mask & ((df["label_attack_type"] == fam) | (df["label_binary"] == 0))
        folds.append({"name": fam, "family": fam, "train": df.index[~test_mask].values,
                      "test": df.index[score_mask].values, "test_days": sorted(map(str, days))})
    return folds


def run_fold(df, features, fold, config, device, epochs):
    frame = df.copy()
    frame["split"] = "none"
    frame.loc[fold["test"], "split"] = "test"
    tr = frame.loc[fold["train"]].sort_values("window_start_utc")
    cut = int(len(tr) * 0.8)
    frame.loc[tr.index[:cut], "split"] = "train"
    frame.loc[tr.index[cut:], "split"] = "val"
    purged = frame[frame["split"].isin(["train", "val", "test"])].copy()
    transformed, pca = apply_training_only_pca(purged, features, 32, random_state=42)
    seqs = build_next_state_sequences(transformed, features=PCA_COLS, config=config)
    z = pca.transform(frame)[PCA_COLS].to_numpy(np.float32)
    L = config.lookback_windows
    keep = [i for i in fold["test"] if i - L >= 0]
    X = np.stack([z[i - L: i] for i in keep]).astype(np.float32)
    y_next = z[keep]
    model = LSTMGaussianWorldModel(input_size=32, hidden_size=64, state_dim=32, num_layers=1, dropout=0.2)
    train_gaussian(model, seqs["train"].X, seqs["train"].y, seqs["val"].X, seqs["val"].y, epochs=epochs,
                   batch_size=64, learning_rate=0.001, weight_decay=0.0001, patience=5, min_delta=0.001,
                   seed=42, device=device, checkpoint_path=None)
    model.eval()
    with torch.no_grad():
        mean, _ = model(torch.as_tensor(X).to(device))
    dev = deviation_scores(mean.cpu().numpy(), y_next)
    return pd.DataFrame({"window_index": keep, "fold": fold["name"], "family": fold["family"],
                         "label_binary": df.loc[keep, "label_binary"].to_numpy(int),
                         "label_attack_type": df.loc[keep, "label_attack_type"].astype(str).to_numpy(),
                         "deviation": dev})


def metrics(rows):
    y, s = rows["label_binary"].to_numpy(), rows["deviation"].to_numpy()
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "pr_auc": None, "n": int(len(y)), "positives": int(y.sum())}
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
            "n": int(len(y)), "positives": int(y.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data-engineering/data/ucs/ucs_windows.parquet")
    ap.add_argument("--data-rev", default=None, help="read --data from this git revision")
    ap.add_argument("--label", default="", help="text added to the report title, e.g. '(6-day dataset)'")
    ap.add_argument("--epochs", type=int, default=2, help="2 = generate_loeo_deviation.py's setting")
    args = ap.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load(args.data, args.data_rev)
    config = UCSConfig()
    features = validate_ucs_windows(df, config=config)

    loeo = pd.concat([run_fold(df, features, f, config, device, args.epochs) for f in loeo_folds(df)], ignore_index=True)
    fam_folds = family_out_folds(df)
    famout = pd.concat([run_fold(df, features, f, config, device, args.epochs).assign(test_days=",".join(f["test_days"]))
                        for f in fam_folds], ignore_index=True)

    episodes = df[df.label_binary == 1].groupby("label_attack_type")["episode_id"].nunique()
    windows = df[df.label_binary == 1].groupby("label_attack_type").size()
    rows = []
    unlabeled_attack = int(((df.label_binary == 1) & (df.label_attack_type.astype(str) == "Benign")).sum())
    for fam in sorted(f for f in episodes.index.astype(str) if f != "Benign"):
        n_ep = int(episodes[fam])
        row = {"family": fam, "episodes": n_ep, "attack_windows": int(windows[fam])}
        if n_ep < 2:
            row["status"] = f"not evaluable: {n_ep} episode(s)"
        else:
            row["status"] = "evaluated"
            row["loeo"] = metrics(loeo[loeo.family == fam])
            row["family_out"] = metrics(famout[famout.family == fam])
        rows.append(row)

    out = {"generated_by": "evaluation/deviation/deviation_eval.py",
           "data": f"{args.data_rev + ':' if args.data_rev else ''}{args.data}", "label": args.label,
           "epochs": args.epochs, "loeo_folds": int(loeo["fold"].nunique()),
           "loeo_pooled": metrics(loeo), "family_out_pooled": metrics(famout), "families": rows,
           "label_binary_1_with_attack_type_benign": unlabeled_attack}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "deviation_results.json").write_text(json.dumps(out, indent=2) + "\n")

    def f(m, k):
        return "—" if m is None or m.get(k) is None else f"{m[k]:.3f}"

    lines = [f"# World-model deviation as attack evidence {args.label}".rstrip(), "",
             f"Generated by `evaluation/deviation/deviation_eval.py` from `{out['data']}`; all numbers are in "
             "`deviation_results.json`.", "",
             "Deviation = mean |actual next PCA state - world-model predicted mean| (the procedure of "
             "`ml1/scripts/generate_loeo_deviation.py`, epochs = %d). Scored against each window's label_binary." % args.epochs, "",
             f"Pooled LOEO ({out['loeo_folds']} folds): ROC-AUC {f(out['loeo_pooled'], 'roc_auc')}, "
             f"PR-AUC {f(out['loeo_pooled'], 'pr_auc')}.  Pooled family-out: ROC-AUC {f(out['family_out_pooled'], 'roc_auc')}, "
             f"PR-AUC {f(out['family_out_pooled'], 'pr_auc')}.", "",
             "| Family | Episodes | Attack windows | LOEO ROC-AUC | LOEO PR-AUC | Family-out ROC-AUC | Family-out PR-AUC |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "evaluated":
            lines.append(f"| {r['family']} | {r['episodes']} | {r['attack_windows']} | {r['status']} | | | |")
        else:
            lines.append(f"| {r['family']} | {r['episodes']} | {r['attack_windows']} | {f(r['loeo'], 'roc_auc')} | "
                         f"{f(r['loeo'], 'pr_auc')} | {f(r['family_out'], 'roc_auc')} | {f(r['family_out'], 'pr_auc')} |")
    lines += ["", "Family-out: the family's days are held out entirely, so the world model never saw that family in training. "
              "ROC-AUC 0.5 = no separation.",
              "", f"The dataset also has {unlabeled_attack} windows with label_binary = 1 but label_attack_type = Benign "
              "(dataset labels); they are not an attack family. They count as attack windows in the pooled LOEO figure "
              "when a fold's test set contains them, and are excluded from family-out scoring."]
    (OUT_DIR / "DEVIATION.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
