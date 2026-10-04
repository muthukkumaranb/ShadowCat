"""Two leakage checks on the headline detection / onset results.

Check A: threshold chosen without test labels (37-fold LOEO, existing models)
    The headline "matched 5% FPR" numbers (evaluation/benchmark/stacked_benchmark_results.json) pick the
    threshold on the pooled TEST predictions. Here each fold picks its threshold on data it was allowed to
    see, the threshold is applied to that fold's test windows, and the realised test FPR / F1 are reported.
      - Stacked model: the fold checkpoint's own validation predictions (chronological last 20% of the
        training windows, base logit = out-of-fold LR logit, temperature-calibrated).
      - LR baseline: out-of-fold LR predictions on the training windows (5-fold KFold over training
        episodes, same scheme the stacked script uses for its LR stage); the full-train LR is refit and
        must reproduce the committed test predictions.

Check B: leave-one-DAY-out (6 folds)
    Train on 5 capture days, test on the 6th, with a 36-window purge on both sides of the held-out day
    (the same purge as the 37-fold manifest), so no training window's 30-window history or 5-window
    onset target overlaps the test day. The stacked model is retrained with the UNCHANGED training
    function ml1/scripts/run_stacked_calibrated_lstm.py::run_stacked_target (epochs=10, as in its main());
    the LR baseline is retrained with its committed protocol (StandardScaler + liblinear LR).
    Note: in CIC-IDS2018 each day holds essentially one attack family, so leave-one-day-out is ALSO
    leave-one-family-out for most days. It is the hardest available generalisation test.

Run from the repo root:
    python evaluation/leakage_checks/run_leakage_checks.py
Outputs: evaluation/leakage_checks/leakage_results.json, LEAKAGE_CHECKS.md, lodo_manifest.json.
Retrained day-out checkpoints go to a temporary directory and are not committed.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ml1"))
sys.path.insert(0, str(REPO / "ml1" / "scripts"))
from run_stacked_calibrated_lstm import ResidualLSTMClassifier as ResidualLSTM, run_stacked_target  # noqa: E402

OUT = REPO / "evaluation" / "leakage_checks"
DATA = REPO / "data-engineering/data/ucs/ucs_windows_models_v1.parquet"
TASKS = [("detection", "label_binary"), ("onset", "future_attack_label")]
LOOKBACK, PURGE, TARGET_FPR, SEED = 30, 36, 0.05, 42


# ----------------------------------------------------------------------------- helpers
def threshold_at_fpr(y, p, target=TARGET_FPR):
    """Lowest threshold whose FPR on (y, p) is <= target (same rule as reproduce_stacked_benchmark.matched_fpr)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    neg = (y == 0).sum()
    if neg == 0:
        return float(np.max(p))
    best = float(np.max(p)) + 1e-12
    for t in np.unique(p)[::-1]:
        if ((p >= t) & (y == 0)).sum() / neg <= target:
            best = float(t)
        else:
            break
    return best


def metrics_at(y, p, thr):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = (p >= thr).astype(int)
    neg = (y == 0).sum()
    return {"f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "fpr": round(float(((pred == 1) & (y == 0)).sum() / neg), 4) if neg else None,
            "n": int(len(y)), "positives": int(y.sum())}


def ranking(y, p):
    y = np.asarray(y, int)
    if 0 < y.sum() < len(y):
        return {"roc_auc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4)}
    return {"roc_auc": None, "pr_auc": None}


def lr_oof_and_full(df, train_idx, test_idx, feats, target):
    """LR baseline: out-of-fold probabilities on training windows + full-train probabilities on test windows."""
    tr = df.iloc[train_idx]
    eps = tr["episode_id"].unique()
    oof = np.full(len(tr), np.nan)
    for a, b in KFold(n_splits=5, shuffle=True, random_state=SEED).split(eps):
        m_tr, m_va = tr["episode_id"].isin(set(eps[a])).to_numpy(), tr["episode_id"].isin(set(eps[b])).to_numpy()
        y_in = tr.loc[m_tr, target].to_numpy(int)
        if len(np.unique(y_in)) < 2:
            continue
        sc = StandardScaler().fit(tr.loc[m_tr, feats].to_numpy(float))
        clf = LogisticRegression(solver="liblinear", max_iter=2000, random_state=SEED).fit(sc.transform(tr.loc[m_tr, feats].to_numpy(float)), y_in)
        oof[m_va] = clf.predict_proba(sc.transform(tr.loc[m_va, feats].to_numpy(float)))[:, 1]
    sc = StandardScaler().fit(tr[feats].to_numpy(float))
    clf = LogisticRegression(solver="liblinear", max_iter=2000, random_state=SEED).fit(sc.transform(tr[feats].to_numpy(float)), tr[target].to_numpy(int))
    p_test = clf.predict_proba(sc.transform(df.iloc[test_idx][feats].to_numpy(float)))[:, 1]
    ok = ~np.isnan(oof)
    return oof[ok], tr[target].to_numpy(int)[ok], p_test


def stacked_test_probs(df, ckpt, test_idx):
    """Same inference as evaluation/benchmark/reproduce_stacked_benchmark.py."""
    model = ResidualLSTM(input_size=32)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    x = df.loc[test_idx, ckpt["lr_features"]].to_numpy(float)
    z = ((x - np.array(ckpt["scaler_mean"])) / np.array(ckpt["scaler_scale"])) @ np.array(ckpt["lr_coef"]).reshape(-1) + np.array(ckpt["lr_intercept"]).reshape(-1)[0]
    pca = (df[ckpt["pca_columns"]].to_numpy(float) - np.array(ckpt["pca_mean"])) @ np.array(ckpt["pca_components"]).T
    seqs = np.array([pca[i - LOOKBACK + 1 : i + 1] for i in test_idx], dtype=np.float32)
    with torch.no_grad():
        logits = model.forward_logits(torch.tensor(seqs), torch.tensor(z, dtype=torch.float32)).numpy()
    return 1.0 / (1.0 + np.exp(-logits / ckpt["temperature"]))


# ----------------------------------------------------------------------------- check A
def check_a(df, feats, manifest):
    res = {}
    for task, target in TASKS:
        st = pd.read_csv(REPO / f"ml1/artifacts/lstm/lstm_stacked/lstm_{task}_loeo_test_predictions.csv")
        lr = pd.read_csv(REPO / f"ml1/artifacts/lr_set_a_corrected/lr_{task}_loeo_predictions.csv")
        st["ts"], lr["ts"] = pd.to_datetime(st["timestamp"], utc=True), pd.to_datetime(lr["timestamp"], utc=True)
        ys, ps, ps_raw, yl, pl, pl_raw, per_fold, max_dev = [], [], [], [], [], [], [], 0.0
        for fold in manifest["folds"]:
            fid = fold["fold_id"]
            side = json.loads((REPO / f"ml1/artifacts/lstm/lstm_stacked/{task}/sidecar_fold_{fid}.json").read_text())
            thr_s = threshold_at_fpr(side["val_targets"], side["val_predictions"])
            s = st[st.fold_id == fid]
            test_idx = [i for i in fold["test_indices"] if i >= LOOKBACK - 1]
            oof_p, oof_y, p_lr = lr_oof_and_full(df, fold["train_indices"], test_idx, feats, target)
            thr_l = threshold_at_fpr(oof_y, oof_p)
            lf = lr[lr.fold_id == fid].set_index("ts")
            committed = lf.loc[pd.to_datetime(df.loc[test_idx, "window_start_utc"], utc=True), "y_probability"].to_numpy()
            max_dev = max(max_dev, float(np.max(np.abs(committed - p_lr))))
            ys.append(s["y_true"].to_numpy(int)); ps.append((s["y_probability"].to_numpy() >= thr_s).astype(float))
            ps_raw.append(s["y_probability"].to_numpy())
            yl.append(df.loc[test_idx, target].to_numpy(int)); pl.append((p_lr >= thr_l).astype(float)); pl_raw.append(p_lr)
            per_fold.append({"fold_id": fid, "stacked_threshold": round(thr_s, 4), "lr_threshold": round(thr_l, 4)})
        y_s, d_s, r_s, y_l, d_l, r_l = map(np.concatenate, (ys, ps, ps_raw, yl, pl, pl_raw))
        lr_test_thr = threshold_at_fpr(y_l, r_l)  # same test-chosen rule as the headline, for a like-for-like comparison
        res[task] = {
            "stacked_ensemble": {**metrics_at(y_s, d_s, 0.5), **ranking(y_s, r_s)},
            "lr_baseline_refit_on_same_dataset": {**metrics_at(y_l, d_l, 0.5), **ranking(y_l, r_l)},
            "lr_baseline_refit_test_chosen_threshold": {**metrics_at(y_l, r_l, lr_test_thr), "threshold": round(lr_test_thr, 4)},
            "lr_refit_max_abs_diff_vs_committed_lr_predictions": round(max_dev, 4),
            "per_fold_thresholds": per_fold}
        print(f"[A] {task}: stacked {res[task]['stacked_ensemble']}  lr {res[task]['lr_baseline_refit_on_same_dataset']}  committed-LR dev {max_dev:.3f}", flush=True)
    return res


# ----------------------------------------------------------------------------- check B
def lodo_manifest(df):
    day = pd.to_datetime(df["window_start_utc"], utc=True).dt.date.to_numpy()
    folds = []
    for k, d in enumerate(sorted(set(day))):
        test = np.where(day == d)[0]
        lo, hi = test.min(), test.max()
        purge = set(range(max(0, lo - PURGE), lo)) | set(range(hi + 1, min(len(df), hi + 1 + PURGE)))
        train = [i for i in range(len(df)) if day[i] != d and i not in purge]
        att = df.iloc[test]
        held = att.loc[att["label_binary"] == 1, "episode_id"]
        folds.append({"fold_id": k, "held_out_day": str(d), "held_out_episode_id": str(held.iloc[0] if len(held) else att["episode_id"].iloc[0]),
                      "train_indices": [int(i) for i in train], "test_indices": [int(i) for i in test], "purged": len(purge)})
    return {"description": "leave-one-day-out, 36-window purge on both sides of the held-out day", "folds": folds}


def check_b(df, feats, man):
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for task, target in TASKS:
            run_stacked_target(df=df, manifest_folds=man["folds"], lr_features=feats, target_col=target,
                               output_dir=tmp, target_name=task, epochs=10)
            rows, pooled = [], {"y": [], "s": [], "s_val": [], "l": [], "l_val": []}
            for fold in man["folds"]:
                fid = fold["fold_id"]
                ckpt = torch.load(tmp / "lstm_stacked" / task / f"model_fold_{fid}.pt", map_location="cpu", weights_only=False)
                side = json.loads((tmp / "lstm_stacked" / task / f"sidecar_fold_{fid}.json").read_text())
                test_idx = [i for i in fold["test_indices"] if i >= LOOKBACK - 1]
                y = df.loc[test_idx, target].to_numpy(int)
                p_s = stacked_test_probs(df, ckpt, test_idx)
                f1_check = f1_score(y, (p_s >= 0.5).astype(int), zero_division=0)
                assert abs(f1_check - side["test_f1"]) < 1e-6, f"day fold {fid}: recomputed F1 {f1_check} != sidecar {side['test_f1']}"
                thr_s = threshold_at_fpr(side["val_targets"], side["val_predictions"])
                oof_p, oof_y, p_l = lr_oof_and_full(df, fold["train_indices"], test_idx, feats, target)
                thr_l = threshold_at_fpr(oof_y, oof_p)
                rows.append({"held_out_day": fold["held_out_day"], "n_test": len(y), "positives": int(y.sum()),
                             "stacked": {**ranking(y, p_s), "at_0.5": metrics_at(y, p_s, 0.5), "at_val_5pct_fpr": metrics_at(y, p_s, thr_s)},
                             "lr_baseline": {**ranking(y, p_l), "at_0.5": metrics_at(y, p_l, 0.5), "at_oof_5pct_fpr": metrics_at(y, p_l, thr_l)}})
                for key, val in (("y", y), ("s", p_s), ("s_val", (p_s >= thr_s).astype(float)), ("l", p_l), ("l_val", (p_l >= thr_l).astype(float))):
                    pooled[key].append(val)
                print(f"[B] {task} day {fold['held_out_day']}: stacked AUC {rows[-1]['stacked']['roc_auc']} lr AUC {rows[-1]['lr_baseline']['roc_auc']}", flush=True)
            P = {k: np.concatenate(v) for k, v in pooled.items()}
            res[task] = {"per_day": rows,
                         "pooled": {"stacked": {**ranking(P["y"], P["s"]), "at_0.5": metrics_at(P["y"], P["s"], 0.5), "at_val_5pct_fpr": metrics_at(P["y"], P["s_val"], 0.5)},
                                    "lr_baseline": {**ranking(P["y"], P["l"]), "at_0.5": metrics_at(P["y"], P["l"], 0.5), "at_oof_5pct_fpr": metrics_at(P["y"], P["l_val"], 0.5)}}}
    return res


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    df = pd.read_parquet(DATA).sort_values("window_start_utc").reset_index(drop=True)
    feats = json.loads((REPO / "ml1/artifacts/lr_set_a/set_a_features.json").read_text())["ordered_feature_names"]
    assert len(feats) == 406
    manifest = json.loads((REPO / "ml1/artifacts/loeo/corrected_37fold_manifest.json").read_text())
    bench = json.loads((REPO / "evaluation/benchmark/stacked_benchmark_results.json").read_text())

    a = check_a(df, feats, manifest)
    man = lodo_manifest(df)
    (OUT / "lodo_manifest.json").write_text(json.dumps({**man, "folds": [{k: v for k, v in f.items()} for f in man["folds"]]}))
    b = check_b(df, feats, man)

    out = {"generated_by": "evaluation/leakage_checks/run_leakage_checks.py", "dataset": str(DATA.relative_to(REPO)),
           "reference_test_chosen_threshold": {t: bench[t]["matched_fpr_5pct"] for t, _ in TASKS},
           "check_a_threshold_without_test_labels_37fold": a, "check_b_leave_one_day_out": b}
    (OUT / "leakage_results.json").write_text(json.dumps(out, indent=2))
    write_md(out)
    print("done")


def write_md(o):
    A, B, R = o["check_a_threshold_without_test_labels_37fold"], o["check_b_leave_one_day_out"], o["reference_test_chosen_threshold"]
    L = ["# Robustness checks (leakage controls)", "", "Generated by `evaluation/leakage_checks/run_leakage_checks.py`; all numbers are in `leakage_results.json`.", "",
         "## Check A: threshold chosen without test labels (37-fold LOEO, existing models)", "",
         "| Task | Model | Threshold chosen on | Test F1 | Precision | Recall | Realised test FPR |", "|---|---|---|---|---|---|---|"]
    for t, _ in TASKS:
        r = R[t]["stacked_ensemble"]
        L.append(f"| {t} | Stacked | pooled TEST predictions (reported headline) | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['fpr']:.4f} |")
        a = A[t]["stacked_ensemble"]
        L.append(f"| {t} | Stacked | fold validation split (no test labels) | {a['f1']:.4f} | {a['precision']:.4f} | {a['recall']:.4f} | {a['fpr']:.4f} |")
        r = R[t]["lr_baseline"]
        L.append(f"| {t} | LR (earlier feature version) | pooled TEST predictions (earlier report) | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['fpr']:.4f} |")
        r = A[t]["lr_baseline_refit_test_chosen_threshold"]
        L.append(f"| {t} | LR (refit, same dataset) | pooled TEST predictions (same rule as headline) | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['fpr']:.4f} |")
        a = A[t]["lr_baseline_refit_on_same_dataset"]
        L.append(f"| {t} | LR (refit, same dataset) | out-of-fold training predictions (no test labels) | {a['f1']:.4f} | {a['precision']:.4f} | {a['recall']:.4f} | {a['fpr']:.4f} |")
    L += ["", "ROC-AUC (threshold-free): " + "; ".join(
        f"{t}: stacked {A[t]['stacked_ensemble']['roc_auc']}, LR refit {A[t]['lr_baseline_refit_on_same_dataset']['roc_auc']}" for t, _ in TASKS) + ".", "",
          "**Baseline note:** the earlier LR baseline numbers (`ml1/artifacts/lr_set_a_corrected/`) were computed on an earlier "
          "feature version (max probability difference vs a refit on the final dataset: "
          + ", ".join(f"{t} {A[t]['lr_refit_max_abs_diff_vs_committed_lr_predictions']}" for t, _ in TASKS)
          + "). The 'LR (refit, same dataset)' rows re-run the baseline on the final dataset and are the reference baseline from now on.", "",
          "## Check B: leave-one-DAY-out (6 folds, 36-window purge, models retrained)", "",
          "Each CIC-IDS2018 day holds essentially one attack family, so this is also (mostly) leave-one-family-out: a stress test of generalisation to attack types never seen in training.", ""]
    for t, _ in TASKS:
        L += [f"### {t}", "", "| Held-out day | n (pos) | Stacked ROC-AUC | LR ROC-AUC | Stacked F1 @val-5%FPR (FPR) | LR F1 @oof-5%FPR (FPR) |", "|---|---|---|---|---|---|"]
        for r in B[t]["per_day"]:
            s, l = r["stacked"], r["lr_baseline"]
            fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
            L.append(f"| {r['held_out_day']} | {r['n_test']} ({r['positives']}) | {fmt(s['roc_auc'])} | {fmt(l['roc_auc'])} | "
                     f"{s['at_val_5pct_fpr']['f1']:.4f} ({fmt(s['at_val_5pct_fpr']['fpr'])}) | {l['at_oof_5pct_fpr']['f1']:.4f} ({fmt(l['at_oof_5pct_fpr']['fpr'])}) |")
        ps, pl = B[t]["pooled"]["stacked"], B[t]["pooled"]["lr_baseline"]
        L.append(f"| **pooled** | {ps['at_0.5']['n']} ({ps['at_0.5']['positives']}) | {fmt(ps['roc_auc'])} | {fmt(pl['roc_auc'])} | "
                 f"{ps['at_val_5pct_fpr']['f1']:.4f} ({fmt(ps['at_val_5pct_fpr']['fpr'])}) | {pl['at_oof_5pct_fpr']['f1']:.4f} ({fmt(pl['at_oof_5pct_fpr']['fpr'])}) |")
        L.append("")
    (OUT / "LEAKAGE_CHECKS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
