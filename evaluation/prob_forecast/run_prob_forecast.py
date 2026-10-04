"""Part P (P-t2..P-t4): probabilistic K-step forecasting under 37-fold LOEO.

Question: does the world model's PREDICTIVE DISTRIBUTION over S(t+K) beat fairly-noised persistence
under proper scoring rules, K = 1..5?

Run from the repo root:
    python evaluation/prob_forecast/run_prob_forecast.py

Data      : data-engineering/data/ucs/ucs_windows_models_v1.parquet (2,787 one-minute windows, 6 days)
Features  : the 406 columns of ml1/configs/mdn_v1_feature_schema.txt
CV        : ml1/artifacts/loeo/corrected_37fold_manifest.json (37 folds, 36-window purge), test origins t >= 29
Space     : per-fold z-score + PCA-32, both fitted on the fold's training indices only (ml1/prob_forecast/data.py)
World model: ml1.lstm.model.LSTMGaussianWorldModel (unchanged), trained with ml1.lstm.probabilistic.train_gaussian
            defaults (epochs 30, batch 64, lr 1e-3, wd 1e-4, patience 5) on the chronologically first 80% of the
            fold's training windows; the last 20% is used for early stopping and for the F5 variance factor.
Seed 42 everywhere.

Outputs: evaluation/prob_forecast/prob_results.json, binary_results.json, PROB_FORECAST.md,
         per_window_scores.csv.gz, per_window_binary.csv.gz, pit_k1_k5.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ml1"))

from ml1.prob_forecast.data import load_data, process_fold  # noqa: E402
from ml1.prob_forecast.forecasters import (  # noqa: E402
    Climatology, DiagonalAR1, HeteroscedasticPersistence, NoisedPersistence, inflate, mc_rollout,
)
from lstm.model import LSTMGaussianWorldModel  # noqa: E402
from lstm.probabilistic import train_gaussian  # noqa: E402
from evaluation.prob_forecast import scoring as S  # noqa: E402

SEED = 42
KS = [1, 2, 3, 4, 5]
LOOKBACK = 30
N_SAMPLES = 64
INFLATION_GRID = np.geomspace(0.25, 8.0, 41)
OUT = REPO / "evaluation" / "prob_forecast"
FORECASTERS = ["F0_climatology", "F1_noised_persistence", "F2_noised_persistence_hetero", "F3_ar1",
               "F4_world_model_mc", "F5_world_model_recalibrated"]
BASELINES = FORECASTERS[:4]


def seq_origins(cands, segment, allowed=None, need_next=0):
    """Origins t with a contiguous 30-window lookback (and t+need_next in the same segment); optional membership."""
    n = len(segment)
    out = []
    for t in cands:
        if t < LOOKBACK - 1 or t + need_next >= n:
            continue
        if segment[t - LOOKBACK + 1] != segment[t + need_next]:
            continue
        if allowed is not None and not all((j in allowed) for j in range(t - LOOKBACK + 1, t + need_next + 1)):
            continue
        out.append(t)
    return np.asarray(out, dtype=int)


def main(max_folds=None, no_early_stop=False):
    global OUT
    if no_early_stop:
        OUT = OUT / "sensitivity_no_early_stop"
        OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    df, features = load_data(str(REPO / "data-engineering/data/ucs/ucs_windows_models_v1.parquet"),
                             str(REPO / "ml1/configs/mdn_v1_feature_schema.txt"))
    df = df.sort_values("window_start_utc").reset_index(drop=True)
    assert len(features) == 406, len(features)
    manifest = json.loads((REPO / "ml1/artifacts/loeo/corrected_37fold_manifest.json").read_text())
    ts = pd.to_datetime(df["window_start_utc"], utc=True)
    segment = np.concatenate([[0], np.cumsum(ts.diff().dt.total_seconds().to_numpy()[1:] != 60)])
    labels = df["label_binary"].to_numpy(int)
    future = df["future_attack_label"].to_numpy(int)

    # Current-window detection decision for the TEST origins (committed LOEO predictions, 5%-FPR threshold).
    bench = json.loads((REPO / "evaluation/benchmark/stacked_benchmark_results.json").read_text())
    det_thr = float(bench["detection"]["matched_fpr_5pct"]["stacked_ensemble"]["threshold"])
    det = pd.read_csv(REPO / "ml1/artifacts/lstm/lstm_stacked/lstm_detection_loeo_test_predictions.csv")
    det_prob = dict(zip(pd.to_datetime(det["timestamp"], utc=True), det["y_probability"]))
    onset = pd.read_csv(REPO / "ml1/artifacts/lstm/lstm_stacked/lstm_onset_loeo_test_predictions.csv")
    onset_prob = dict(zip(pd.to_datetime(onset["timestamp"], utc=True), onset["y_probability"]))

    rng_es = np.random.default_rng(SEED)
    rows, brows = [], []
    pit_acc = {(f, k, g): np.zeros(10, int) for f in FORECASTERS for k in KS for g in ("all", "benign_t", "change")}
    fold_meta = []

    for fold in manifest["folds"][:max_folds]:
        fid = fold["fold_id"]
        train_idx = np.asarray(sorted(fold["train_indices"]))
        test_idx = np.asarray([t for t in fold["test_indices"] if t >= LOOKBACK - 1])
        train_set = set(train_idx.tolist())

        _, _, scaler, pca, valid_feats = process_fold(df, features, train_idx, test_idx)
        Z = scaler.transform(df[valid_feats])
        P = pca.transform(Z)

        # ---- world model on the first 80% of training windows; last 20% = validation / calibration
        cut = int(0.8 * len(train_idx))
        inner_tr, inner_val = set(train_idx[:cut].tolist()), set(train_idx[cut:].tolist())
        tr_o = seq_origins(train_idx[:cut], segment, inner_tr, need_next=1)
        va_o = seq_origins(train_idx[cut:], segment, inner_val, need_next=1)
        Xtr = np.stack([P[t - LOOKBACK + 1 : t + 1] for t in tr_o]).astype(np.float32)
        Xva = np.stack([P[t - LOOKBACK + 1 : t + 1] for t in va_o]).astype(np.float32)
        model = LSTMGaussianWorldModel(input_size=P.shape[1], hidden_size=64, state_dim=P.shape[1], dropout=0.2)
        if no_early_stop:
            # Sensitivity: train_gaussian restores the best-VALIDATION epoch; passing the training sequences as the
            # selection set makes it keep the best-training-NLL epoch (~ the full 30 epochs) instead.
            res = train_gaussian(model, Xtr, P[tr_o + 1], Xtr, P[tr_o + 1], seed=SEED, patience=10 ** 6)
        else:
            res = train_gaussian(model, Xtr, P[tr_o + 1], Xva, P[va_o + 1], seed=SEED)
        model = res["model"]

        # ---- F5: one variance-inflation factor per K, fitted on the inner validation windows
        cal_o = seq_origins(train_idx[cut:], segment, None, need_next=0)
        cal_o = np.asarray([t for t in cal_o if t in inner_val])
        cal_samples = mc_rollout(model, P, cal_o, k_max=5, n_samples=N_SAMPLES, seed=SEED)
        factors = {}
        for k in KS:
            ok = np.asarray([(t + k < len(df)) and segment[t + k] == segment[t] for t in cal_o])
            if ok.sum() < 10:
                factors[k] = 1.0
                continue
            ys = P[cal_o[ok] + k]
            smp = cal_samples[ok, :, k - 1, :]
            crps = [S.sample_crps(inflate(smp, c), ys).mean() for c in INFLATION_GRID]
            factors[k] = float(INFLATION_GRID[int(np.argmin(crps))])

        # ---- baselines (all fitted on full training indices)
        f0 = Climatology().fit(P[train_idx])
        f1 = NoisedPersistence().fit(P, train_idx, segment, train_set)
        f2 = HeteroscedasticPersistence().fit(P, train_idx, segment, train_set, labels)
        f3 = DiagonalAR1().fit(P, train_idx, segment, train_set)

        # ---- binary-model components
        prev = labels[train_idx].mean()
        det_clf = LogisticRegression(max_iter=2000).fit(P[train_idx], labels[train_idx])

        test_o = seq_origins(test_idx, segment, None, need_next=0)
        samples = mc_rollout(model, P, test_o, k_max=5, n_samples=N_SAMPLES, seed=SEED)
        dec = np.asarray([int(det_prob.get(ts[t], np.nan) >= det_thr) if ts[t] in det_prob else -1 for t in test_o])
        if (dec < 0).any():
            raise SystemExit(f"fold {fid}: {int((dec < 0).sum())} test origins missing from detection predictions")

        for k in KS:
            ok = np.asarray([(t + k < len(df)) and segment[t + k] == segment[t] for t in test_o])
            o = test_o[ok]
            if len(o) == 0:
                continue
            y = P[o + k]
            P0 = P[o]
            groups = {"all": np.ones(len(o), bool), "benign_t": labels[o] == 0, "change": labels[o + k] != labels[o]}

            gauss = {
                "F0_climatology": f0.predict(P0, k),
                "F1_noised_persistence": f1.predict(P0, k),
                "F2_noised_persistence_hetero": f2.predict(P0, k, dec[ok]),
                "F3_ar1": f3.predict(P0, k),
            }
            per = {}
            for name, (mu, sd) in gauss.items():
                sc = S.gaussian_scores(mu, sd, y)
                sc["es"] = S.energy_score(S.gaussian_samples(mu, sd, N_SAMPLES, rng_es), y)
                per[name] = sc
            smp = samples[ok, :, k - 1, :]
            sc = S.sample_scores(smp, y)
            sc["es"] = S.energy_score(smp, y)
            per["F4_world_model_mc"] = sc
            smp5 = inflate(smp, factors[k])
            sc = S.sample_scores(smp5, y)
            sc["es"] = S.energy_score(smp5, y)
            per["F5_world_model_recalibrated"] = sc

            for name, sc in per.items():
                for g, m in groups.items():
                    if m.any():
                        pit_acc[(name, k, g)] += np.asarray(S.pit_histogram(sc["pit"][m]))
                for i, t in enumerate(o):
                    rows.append({"fold_id": fid, "origin": int(t), "k": k, "forecaster": name,
                                 "label_t": int(labels[t]), "label_tk": int(labels[t + k]),
                                 **{m: float(sc[m][i]) for m in ("crps", "nll", "cov50", "cov90", "width90", "es")}})

            # ---- binary: P(label_binary(t+K) = 1)
            tr_pairs = np.asarray([t for t in train_idx if t + k < len(df) and segment[t + k] == segment[t] and (t + k) in train_set])
            tab = {s: labels[tr_pairs[labels[tr_pairs] == s] + k].mean() for s in (0, 1)}
            m2 = LogisticRegression(max_iter=2000).fit(Z[tr_pairs], labels[tr_pairs + k])
            p_m2 = m2.predict_proba(Z[o])[:, 1]
            smp_flat = samples[ok, :, k - 1, :].reshape(-1, P.shape[1])
            p_m3 = det_clf.predict_proba(smp_flat)[:, 1].reshape(len(o), N_SAMPLES).mean(1)
            for i, t in enumerate(o):
                brows.append({"fold_id": fid, "origin": int(t), "k": k, "y": int(labels[t + k]), "label_t": int(labels[t]),
                              "B0_base_rate": prev, "B1_markov_oracle": tab[int(labels[t])],
                              "B1_markov_realistic": tab[int(dec[ok][i])], "M2_lr_features": float(p_m2[i]),
                              "M3_world_model_mc": float(p_m3[i]),
                              "M1_stacked_onset": float(onset_prob.get(ts[t], np.nan)), "future_attack_label": int(future[t])})

        fold_meta.append({"fold_id": fid, "episode": fold.get("held_out_episode_id"), "n_test_origins": int(len(test_o)),
                          "n_train_seq": int(len(tr_o)), "n_val_seq": int(len(va_o)), "best_epoch": res["best_epoch"],
                          "dropped_zero_std_features": int(406 - len(valid_feats)), "inflation_factors": factors})
        print(f"fold {fid:2d}  test={len(test_o):3d}  epoch={res['best_epoch']:2d}  factors={factors}  {time.time()-t0:6.0f}s", flush=True)

    R = pd.DataFrame(rows)
    B = pd.DataFrame(brows)
    R.to_csv(OUT / "per_window_scores.csv.gz", index=False)
    B.to_csv(OUT / "per_window_binary.csv.gz", index=False)
    summarise(R, B, pit_acc, fold_meta, det_thr)
    print(f"done in {time.time()-t0:.0f}s")


def summarise(R, B, pit_acc, fold_meta, det_thr):
    res = {"generated_by": "evaluation/prob_forecast/run_prob_forecast.py", "space": "per-fold PCA-32 of train-fold z-scores",
           "dataset": "data-engineering/data/ucs/ucs_windows_models_v1.parquet", "n_folds": len(fold_meta),
           "n_samples_mc": N_SAMPLES, "bootstrap": "paired, over the 37 held-out folds (episodes), 2000 resamples",
           "detection_threshold_5pct_fpr": det_thr, "folds": fold_meta, "horizons": {}}
    hstar = 0
    for k in KS:
        hk = {}
        for g in ("all", "benign_t", "change"):
            sub = R[R.k == k]
            if g == "benign_t":
                sub = sub[sub.label_t == 0]
            elif g == "change":
                sub = sub[sub.label_t != sub.label_tk]
            n_win = int(sub[sub.forecaster == FORECASTERS[0]].shape[0])
            gk = {"n_windows": n_win, "forecasters": {}}
            if n_win == 0:
                hk[g] = gk
                continue
            for f in FORECASTERS:
                s = sub[sub.forecaster == f]
                gk["forecasters"][f] = {m: round(float(s[m].mean()), 5) for m in ("crps", "es", "nll", "cov50", "cov90", "width90")}
                gk["forecasters"][f]["pit_hist_10"] = pit_acc[(f, k, g)].tolist()
            best = min(BASELINES, key=lambda f: gk["forecasters"][f]["crps"])
            gk["best_baseline_by_crps"] = best
            piv = {m: sub.pivot_table(index=["fold_id", "origin"], columns="forecaster", values=m) for m in ("crps", "es")}
            grp = piv["crps"].index.get_level_values(0).to_numpy()
            comps = {}
            if n_win >= 20:
                for a, b in [("F4_world_model_mc", "F1_noised_persistence"), ("F5_world_model_recalibrated", "F1_noised_persistence"),
                             ("F4_world_model_mc", best), ("F5_world_model_recalibrated", best)]:
                    for m in ("crps", "es"):
                        pt, lo, hi = S.episode_bootstrap_diff(piv[m][a].to_numpy(), piv[m][b].to_numpy(), grp)
                        comps[f"{a} - {b} [{m}]"] = {"point": round(pt, 5), "ci95": [round(lo, 5), round(hi, 5)],
                                                     "model_better": bool(hi < 0)}
            else:
                comps["note"] = f"too few windows ({n_win} < 20) for a bootstrap CI"
            gk["comparisons"] = comps
            hk[g] = gk
        res["horizons"][str(k)] = hk

    # H*_prob on all windows: F4 or F5 beats the best baseline on CRPS (CI excludes 0) and 90% coverage in [0.85, 0.95]
    for k in KS:
        a = res["horizons"][str(k)]["all"]
        best = a["best_baseline_by_crps"]
        ok = False
        for f in ("F4_world_model_mc", "F5_world_model_recalibrated"):
            c = a["comparisons"].get(f"{f} - {best} [crps]")
            cov = a["forecasters"][f]["cov90"]
            if c and c["model_better"] and 0.85 <= cov <= 0.95:
                ok = True
        if ok and hstar == k - 1:
            hstar = k
    res["h_star_prob"] = hstar
    res["h_star_definition"] = ("largest K (consecutive from 1) at which F4 or F5 beats the best of F0-F3 on mean CRPS "
                                "(pca-space, all windows) with a 95% episode-bootstrap CI excluding 0 AND has 90% coverage in [0.85, 0.95]")
    (OUT / "prob_results.json").write_text(json.dumps(res, indent=2))

    # ---- binary
    bres = {"generated_by": "evaluation/prob_forecast/run_prob_forecast.py", "target": "label_binary(t+K)", "horizons": {}}
    models = ["B0_base_rate", "B1_markov_oracle", "B1_markov_realistic", "M2_lr_features", "M3_world_model_mc"]
    for k in KS:
        hk = {}
        for g in ("all", "benign_t"):
            sub = B[B.k == k] if g == "all" else B[(B.k == k) & (B.label_t == 0)]
            y = sub.y.to_numpy()
            gk = {"n_windows": int(len(sub)), "n_positive": int(y.sum()), "models": {}}
            for m in models:
                p = sub[m].to_numpy(float)
                gk["models"][m] = {"brier": round(float(S.brier(p, y).mean()), 5), "logloss": round(float(S.logloss(p, y).mean()), 5),
                                   "roc_auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.sum() < len(y) and np.unique(p).size > 1 else None,
                                   "reliability": S.reliability(p, y)}
            ref = min(["B0_base_rate", "B1_markov_realistic"], key=lambda m: gk["models"][m]["brier"])
            gk["reference_baseline"] = ref
            grp = sub.fold_id.to_numpy()
            comps = {}
            if 0 < y.sum():
                for m in ("M2_lr_features", "M3_world_model_mc"):
                    pt, lo, hi = S.episode_bootstrap_diff(S.brier(sub[m].to_numpy(float), y), S.brier(sub[ref].to_numpy(float), y), grp)
                    comps[f"{m} - {ref} [brier]"] = {"point": round(pt, 5), "ci95": [round(lo, 5), round(hi, 5)], "model_better": bool(hi < 0)}
            gk["comparisons"] = comps
            hk[g] = gk
        bres["horizons"][str(k)] = hk
    # stacked onset ensemble, its own target (attack window within t+1..t+5), K = 5 origins
    sub = B[(B.k == 5)].dropna(subset=["M1_stacked_onset"])
    for g, s in (("all", sub), ("benign_t", sub[sub.label_t == 0])):
        y = s.future_attack_label.to_numpy()
        p = s.M1_stacked_onset.to_numpy(float)
        bres.setdefault("M1_stacked_onset_target_future_attack_label", {})[g] = {
            "n_windows": int(len(s)), "n_positive": int(y.sum()), "brier": round(float(S.brier(p, y).mean()), 5),
            "brier_base_rate": round(float(S.brier(np.full_like(p, y.mean()), y).mean()), 5),
            "roc_auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.sum() < len(y) else None}
    (OUT / "binary_results.json").write_text(json.dumps(bres, indent=2))
    write_md(res, bres)
    plot_pit(pit_acc)


def write_md(res, bres):
    L = ["# Probabilistic K-step forecast (Part P)", "",
         "Generated by `evaluation/prob_forecast/run_prob_forecast.py`; every number below is in `prob_results.json` / `binary_results.json`.",
         "37-fold LOEO on `ucs_windows_models_v1.parquet`; PCA-32 space fitted per fold on training windows; 64 Monte Carlo samples; "
         "CIs = paired bootstrap over the 37 held-out episodes (2,000 resamples). Lower CRPS / energy score is better.", "",
         f"**H\\*_prob = {res['h_star_prob']}** ({res['h_star_definition']}).", ""]
    for g, title in (("all", "All windows"), ("benign_t", "Benign at t"), ("change", "Change windows (label(t+K) != label(t))")):
        L += [f"## {title}", "", "| K | n | " + " | ".join(f"{f.split('_')[0]} CRPS" for f in FORECASTERS) +
              " | F4 cov90 | F5 cov90 | best baseline | F5 − best CRPS [95% CI] |", "|" + "---|" * (len(FORECASTERS) + 6)]
        for k in KS:
            a = res["horizons"][str(k)][g]
            if a["n_windows"] == 0:
                continue
            fc = a["forecasters"]
            best = a["best_baseline_by_crps"]
            c = a["comparisons"].get(f"F5_world_model_recalibrated - {best} [crps]")
            cs = f"{c['point']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]" if c else a["comparisons"].get("note", "")
            L.append(f"| {k} | {a['n_windows']} | " + " | ".join(f"{fc[f]['crps']:.4f}" for f in FORECASTERS) +
                     f" | {fc['F4_world_model_mc']['cov90']:.3f} | {fc['F5_world_model_recalibrated']['cov90']:.3f} | {best.split('_')[0]} | {cs} |")
        L.append("")
    L += ["## Attack probability P(label_binary(t+K) = 1): Brier score (lower is better)", "",
          "| K | group | n (pos) | B0 base rate | B1 Markov oracle | B1 Markov realistic | M2 LR | M3 world model | M3 − ref [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
    for k in KS:
        for g in ("all", "benign_t"):
            a = bres["horizons"][str(k)][g]
            mm = a["models"]
            c = a["comparisons"].get(f"M3_world_model_mc - {a['reference_baseline']} [brier]")
            cs = f"{c['point']:+.5f} [{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}]" if c else "—"
            L.append(f"| {k} | {g} | {a['n_windows']} ({a['n_positive']}) | " + " | ".join(f"{mm[m]['brier']:.4f}" for m in
                     ["B0_base_rate", "B1_markov_oracle", "B1_markov_realistic", "M2_lr_features", "M3_world_model_mc"]) + f" | {cs} |")
    L += ["", "ref = the better of B0 and B1-realistic (B1-oracle uses the true current label and is shown only as a reference).", ""]
    (OUT / "PROB_FORECAST.md").write_text("\n".join(L) + "\n")


def plot_pit(pit_acc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for r, k in enumerate((1, 5)):
        for c, f in enumerate(("F1_noised_persistence", "F4_world_model_mc", "F5_world_model_recalibrated")):
            h = pit_acc[(f, k, "all")]
            axes[r, c].bar(np.arange(10) / 10 + 0.05, h / h.sum(), width=0.09)
            axes[r, c].axhline(0.1, color="k", lw=0.8, ls="--")
            axes[r, c].set_title(f"{f} K={k}", fontsize=9)
    fig.suptitle("PIT histograms (flat at 0.1 = calibrated), PCA-32 space, all test windows")
    fig.tight_layout()
    fig.savefig(OUT / "pit_k1_k5.png", dpi=110)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-folds", type=int, default=None, help="debug only: run the first N folds")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="sensitivity: train the world model for the full 30 epochs (outputs to sensitivity_no_early_stop/)")
    a = ap.parse_args()
    main(a.max_folds, a.no_early_stop)
