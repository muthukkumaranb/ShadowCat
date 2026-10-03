import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
LOOKBACK = 30
TASKS = [("detection", "label_binary"), ("onset", "future_attack_label")]


class ResidualLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(32, 64, 1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 1)

    def forward_logits(self, x, base_logit):
        out, _ = self.lstm(x)
        return base_logit + self.fc(out[:, -1, :]).squeeze(-1)


def fold_inputs(df, fold, ckpt):
    test_idx = [i for i in fold["test_indices"] if i >= LOOKBACK - 1]
    x = df.loc[test_idx, ckpt["lr_features"]].to_numpy(float)
    lr_logit = ((x - np.array(ckpt["scaler_mean"])) / np.array(ckpt["scaler_scale"])) @ np.array(
        ckpt["lr_coef"]
    ).reshape(-1) + np.array(ckpt["lr_intercept"]).reshape(-1)[0]
    pca = (df[ckpt["pca_columns"]].to_numpy(float) - np.array(ckpt["pca_mean"])) @ np.array(ckpt["pca_components"]).T
    seqs = np.array([pca[i - LOOKBACK + 1 : i + 1] for i in test_idx], dtype=np.float32)
    return test_idx, seqs, lr_logit


def matched_fpr(y, p, target=0.05):
    best = None
    for t in np.unique(p)[::-1]:
        pred = p >= t
        fp = int((pred & (y == 0)).sum())
        tn = int((~pred & (y == 0)).sum())
        if fp / (fp + tn) <= target:
            best = t
        else:
            break
    pred = (p >= best).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "threshold": float(best),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "fpr": fp / (fp + tn),
        "roc_auc": float(roc_auc_score(y, p)),
        "note": "threshold chosen on pooled test predictions for both models (same procedure for both)",
    }


def at_half(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="UCS windows parquet the models were trained on")
    args = ap.parse_args()

    df = pd.read_parquet(REPO / args.data).sort_values("window_start_utc").reset_index(drop=True)
    manifest = json.loads((REPO / "ml1/artifacts/loeo/corrected_37fold_manifest.json").read_text())
    rng = np.random.default_rng(42)
    results = {"data_file": args.data}

    for task, label_col in TASKS:
        rows, mismatches = [], []
        ablation = {"normal": [], "permuted": [], "repeat_last": [], "lr_stage_only": []}
        labels = []
        for fold in manifest["folds"]:
            fid = fold["fold_id"]
            ckpt = torch.load(REPO / f"ml1/artifacts/lstm/lstm_stacked/{task}/model_fold_{fid}.pt", map_location="cpu", weights_only=False)
            sidecar = json.loads((REPO / f"ml1/artifacts/lstm/lstm_stacked/{task}/sidecar_fold_{fid}.json").read_text())
            model = ResidualLSTM()
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            temp = ckpt["temperature"]
            test_idx, seqs, lr_logit = fold_inputs(df, fold, ckpt)
            y = df.loc[test_idx, label_col].to_numpy(int)
            base = torch.tensor(lr_logit, dtype=torch.float32)
            sig = lambda z: 1.0 / (1.0 + np.exp(-z / temp))
            with torch.no_grad():
                p_normal = sig(model.forward_logits(torch.tensor(seqs), base).numpy())
                perm = []
                for _ in range(5):
                    shuffled = seqs.copy()
                    for j in range(len(shuffled)):
                        shuffled[j] = shuffled[j][rng.permutation(LOOKBACK)]
                    perm.append(sig(model.forward_logits(torch.tensor(shuffled), base).numpy()))
                p_repeat = sig(model.forward_logits(torch.tensor(np.repeat(seqs[:, -1:], LOOKBACK, 1)), base).numpy())
            fold_f1 = f1_score(y, (p_normal >= 0.5).astype(int), zero_division=0)
            if abs(fold_f1 - sidecar["test_f1"]) > 1e-6:
                mismatches.append({"fold": fid, "reproduced": fold_f1, "sidecar": sidecar["test_f1"]})
            ablation["normal"].append(p_normal)
            ablation["permuted"].append(np.mean(perm, axis=0))
            ablation["repeat_last"].append(p_repeat)
            ablation["lr_stage_only"].append(sig(lr_logit))
            labels.append(y)
            for idx, yy, pp in zip(test_idx, y, p_normal):
                rows.append({"fold_id": fid, "episode_id": df.loc[idx, "episode_id"],
                             "timestamp": str(df.loc[idx, "window_start_utc"]), "y_true": int(yy), "y_probability": float(pp)})

        if mismatches:
            raise SystemExit(f"{task}: {len(mismatches)} folds do not reproduce sidecar test_f1 -> wrong data file. {mismatches[:5]}")

        pred_path = REPO / f"ml1/artifacts/lstm/lstm_stacked/lstm_{task}_loeo_test_predictions.csv"
        pd.DataFrame(rows).to_csv(pred_path, index=False)

        lstm = pd.DataFrame(rows)
        lr = pd.read_csv(REPO / f"ml1/artifacts/lr_set_a_corrected/lr_{task}_loeo_predictions.csv")
        lstm["ts"] = pd.to_datetime(lstm["timestamp"], utc=True)
        lr["ts"] = pd.to_datetime(lr["timestamp"], utc=True)
        joined = lstm.merge(lr[["fold_id", "ts", "y_true", "y_probability"]], on=["fold_id", "ts"], suffixes=("_lstm", "_lr"))
        assert len(joined) == len(lstm) == len(lr), "LSTM and LR test windows do not align"
        assert (joined["y_true_lstm"] == joined["y_true_lr"]).all(), "label mismatch"
        y_all = joined["y_true_lstm"].to_numpy(int)

        y_cat = np.concatenate(labels)
        normal = np.concatenate(ablation["normal"])
        results[task] = {
            "folds_reproducing_sidecar_f1": len(manifest["folds"]),
            "test_windows": int(len(y_all)),
            "positives": int(y_all.sum()),
            "matched_fpr_5pct": {
                "stacked_ensemble": matched_fpr(y_all, joined["y_probability_lstm"].to_numpy()),
                "lr_baseline": matched_fpr(y_all, joined["y_probability_lr"].to_numpy()),
            },
            "history_ablation_at_0.5": {
                k: {**at_half(y_cat, np.concatenate(v)), "mean_abs_prob_change_vs_normal": float(np.abs(np.concatenate(v) - normal).mean())}
                for k, v in ablation.items()
            },
        }

    out = REPO / "evaluation/benchmark/stacked_benchmark_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
