"""
Train and Evaluate Family / ATT&CK Stage Classifier under 37-Fold LOEO Protocol
Phase B, Task B-t2.

Constraints & Protocol:
- Feature set: ml1/artifacts/lr_set_a/set_a_features.json (406 features)
- Dataset: data-engineering/data/ucs/ucs_windows_models_v1.parquet (2,787 windows, 6 days)
- Manifest: ml1/artifacts/loeo/corrected_37fold_manifest.json (37 folds, seed 42)
- Scaler: StandardScaler fitted on each training fold independently.
- Trained ONLY on windows with label_binary == 1 and non-Benign label_attack_type.
- 1-episode families (DDOS-HOIC, Infiltration-Compromise, Infiltration-Portscan):
  Reported as 'not evaluable under LOEO: 1 episode' plus a separate seen-family
  chronological split (70% train / 30% test with 30-window purge).
- 614 windows with label_binary == 1 and label_attack_type == Benign reported separately.
- Saves 37 fold models to ml1/artifacts/family_classifier/.
- Outputs evaluation/family/family_results.json and evaluation/family/FAMILY.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = REPO_ROOT / "ml1" / "artifacts" / "lr_set_a" / "set_a_features.json"
PARQUET_PATH = REPO_ROOT / "data-engineering" / "data" / "ucs" / "ucs_windows_models_v1.parquet"
MANIFEST_PATH = REPO_ROOT / "ml1" / "artifacts" / "loeo" / "corrected_37fold_manifest.json"
MAPPING_PATH = REPO_ROOT / "data-engineering" / "configs" / "family_to_attck.yaml"

OUTPUT_ARTIFACTS_DIR = REPO_ROOT / "ml1" / "artifacts" / "family_classifier"
OUTPUT_EVAL_DIR = REPO_ROOT / "evaluation" / "family"
RESULTS_JSON_PATH = OUTPUT_EVAL_DIR / "family_results.json"
RESULTS_MD_PATH = OUTPUT_EVAL_DIR / "FAMILY.md"

ALL_ATTACK_FAMILIES = [
    "Botnet",
    "DDOS-HOIC",
    "DDOS-LOIC-UDP",
    "Infiltration-Compromise",
    "Infiltration-Portscan",
    "SSH-Bruteforce",
]
SINGLE_EPISODE_FAMILIES = ["DDOS-HOIC", "Infiltration-Compromise", "Infiltration-Portscan"]
EVALUABLE_LOEO_FAMILIES = ["Botnet", "DDOS-LOIC-UDP", "SSH-Bruteforce"]


def load_inputs() -> Tuple[pd.DataFrame, List[str], Dict[str, Any], Dict[str, Any]]:
    df = pd.read_parquet(PARQUET_PATH)
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feat_meta = json.load(f)
    features = feat_meta["ordered_feature_names"]

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping_cfg = yaml.safe_load(f)

    return df, features, manifest, mapping_cfg


def train_loeo_folds(
    df: pd.DataFrame,
    features: List[str],
    manifest: Dict[str, Any],
    family_to_tactic: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    OUTPUT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    fold_summaries = []
    test_prediction_rows = []

    for fold in manifest["folds"]:
        fold_id = fold["fold_id"]
        held_out_ep = fold["held_out_episode_id"]
        train_idx = fold["train_indices"]
        test_idx = fold["test_indices"]

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Filter: label_binary == 1 and label_attack_type != 'Benign'
        train_att = train_df[(train_df["label_binary"] == 1) & (train_df["label_attack_type"] != "Benign")].copy()
        test_att = test_df[(test_df["label_binary"] == 1) & (test_df["label_attack_type"] != "Benign")].copy()

        X_train = train_att[features].to_numpy(dtype=np.float64)
        y_train = train_att["label_attack_type"].to_numpy()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        )
        clf.fit(X_train_scaled, y_train)

        # Save fold artifact
        fold_model_path = OUTPUT_ARTIFACTS_DIR / f"model_fold_{fold_id}.joblib"
        fold_artifact = {
            "fold_id": fold_id,
            "held_out_episode_id": held_out_ep,
            "model": clf,
            "scaler": scaler,
            "features": features,
            "classes": list(clf.classes_),
            "family_to_tactic": family_to_tactic,
        }
        joblib.dump(fold_artifact, fold_model_path)

        # Evaluate on test attack windows
        if len(test_att) > 0:
            X_test = test_att[features].to_numpy(dtype=np.float64)
            y_test = test_att["label_attack_type"].to_numpy()
            X_test_scaled = scaler.transform(X_test)

            probs = clf.predict_proba(X_test_scaled)
            preds = clf.predict(X_test_scaled)
            classes_list = list(clf.classes_)

            for row_i, (orig_idx, row) in enumerate(test_att.iterrows()):
                pred_fam = preds[row_i]
                true_fam = y_test[row_i]
                prob_vec = probs[row_i]
                conf = float(np.max(prob_vec))

                true_tac = family_to_tactic.get(true_fam, "Unknown")
                pred_tac = family_to_tactic.get(pred_fam, "Unknown")

                test_prediction_rows.append({
                    "window_index": orig_idx,
                    "fold_id": fold_id,
                    "held_out_episode_id": held_out_ep,
                    "window_start_utc": str(row["window_start_utc"]),
                    "true_family": true_fam,
                    "pred_family": pred_fam,
                    "true_tactic": true_tac,
                    "pred_tactic": pred_tac,
                    "confidence": conf,
                    "correct_family": bool(true_fam == pred_fam),
                    "correct_tactic": bool(true_tac == pred_tac),
                })

        fold_summaries.append({
            "fold_id": fold_id,
            "held_out_episode_id": held_out_ep,
            "train_attack_samples": len(train_att),
            "test_attack_samples": len(test_att),
            "classes_in_train": list(clf.classes_),
        })

    preds_df = pd.DataFrame(test_prediction_rows)
    return fold_summaries, preds_df


def evaluate_single_episode_seen(
    df: pd.DataFrame,
    features: List[str],
    family_to_tactic: Dict[str, str],
) -> Dict[str, Any]:
    """
    Evaluates 1-episode families under within-episode chronological split:
    first 70% / last 30% with 30-window purge.
    """
    seen_results = {}

    for fam in SINGLE_EPISODE_FAMILIES:
        fam_df = df[(df["label_binary"] == 1) & (df["label_attack_type"] == fam)].sort_values("window_start_utc")
        n = len(fam_df)
        train_end = int(np.floor(0.70 * n))
        train_end_purged = train_end - 30

        if train_end_purged <= 0:
            seen_results[fam] = {
                "evaluable": False,
                "reason": f"not evaluable: too few windows ({n}) for 30-window purge (train_end_purged={train_end_purged} <= 0)",
                "total_windows": n,
                "train_windows": 0,
                "purged_windows": min(30, train_end),
                "test_windows": n - train_end,
            }
            continue

        train_indices_fam = fam_df.index[:train_end_purged].tolist()
        purge_indices_fam = fam_df.index[train_end_purged:train_end].tolist()
        test_indices_fam = fam_df.index[train_end:].tolist()

        # Build training set: seen train windows of this family + all other attack windows in the dataset
        other_att_indices = df[
            (df["label_binary"] == 1) &
            (df["label_attack_type"] != "Benign") &
            (~df.index.isin(fam_df.index))
        ].index.tolist()

        full_train_indices = sorted(train_indices_fam + other_att_indices)
        train_data = df.loc[full_train_indices]
        test_data = df.loc[test_indices_fam]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_data[features].to_numpy(dtype=np.float64))
        y_tr = train_data["label_attack_type"].to_numpy()

        clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)
        clf.fit(X_tr, y_tr)

        X_te = scaler.transform(test_data[features].to_numpy(dtype=np.float64))
        y_te = test_data["label_attack_type"].to_numpy()

        preds = clf.predict(X_te)
        probs = clf.predict_proba(X_te)
        confs = np.max(probs, axis=1)

        acc = float(np.mean(preds == y_te))
        p, r, f1, _ = precision_recall_fscore_support(
            y_te == fam, preds == fam, average="binary", zero_division=0
        )

        seen_results[fam] = {
            "evaluable": True,
            "total_windows": n,
            "train_windows": len(train_indices_fam),
            "purged_windows": len(purge_indices_fam),
            "test_windows": len(test_indices_fam),
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "mean_confidence": float(np.mean(confs)),
            "predicted_distribution": {k: int(v) for k, v in pd.Series(preds).value_counts().items()},
            "tactic": family_to_tactic.get(fam, "Unknown"),
            "note": "within-episode chronological split, first 70% / last 30%, 30-window purge",
        }

    return seen_results


def evaluate_614_benign_positives(
    df: pd.DataFrame,
    features: List[str],
    manifest: Dict[str, Any],
    family_to_tactic: Dict[str, str],
) -> Dict[str, Any]:
    """
    Evaluates the 614 windows with label_binary == 1 and label_attack_type == 'Benign'.
    These are evaluated across the 37 fold models.
    """
    benign_pos = df[(df["label_binary"] == 1) & (df["label_attack_type"] == "Benign")].copy()
    n_benign_pos = len(benign_pos)
    assert n_benign_pos == 614, f"Expected 614 benign positives, got {n_benign_pos}"

    # Load all 37 fold models and average predicted probabilities
    all_classes = set()
    fold_artifacts = []
    for f in manifest["folds"]:
        path = OUTPUT_ARTIFACTS_DIR / f"model_fold_{f['fold_id']}.joblib"
        art = joblib.load(path)
        fold_artifacts.append(art)
        all_classes.update(art["classes"])

    canonical_classes = sorted(list(all_classes))
    class_to_idx = {c: i for i, c in enumerate(canonical_classes)}

    X_raw = benign_pos[features].to_numpy(dtype=np.float64)
    ensemble_probs = np.zeros((n_benign_pos, len(canonical_classes)), dtype=np.float64)

    for art in fold_artifacts:
        clf = art["model"]
        scaler = art["scaler"]
        classes = art["classes"]
        X_scaled = scaler.transform(X_raw)
        p = clf.predict_proba(X_scaled)
        for local_idx, c in enumerate(classes):
            ensemble_probs[:, class_to_idx[c]] += p[:, local_idx]

    ensemble_probs /= len(fold_artifacts)
    pred_indices = np.argmax(ensemble_probs, axis=1)
    confidences = np.max(ensemble_probs, axis=1)

    predicted_families = [canonical_classes[idx] for idx in pred_indices]
    predicted_tactics = [family_to_tactic.get(f, "Unknown") for f in predicted_families]

    fam_counts = {k: int(v) for k, v in pd.Series(predicted_families).value_counts().items()}
    tac_counts = {k: int(v) for k, v in pd.Series(predicted_tactics).value_counts().items()}

    return {
        "window_count": n_benign_pos,
        "note": "614 windows with label_binary=1 and label_attack_type=Benign (isolated from training)",
        "mean_confidence": float(np.mean(confidences)),
        "min_confidence": float(np.min(confidences)),
        "max_confidence": float(np.max(confidences)),
        "predicted_family_distribution": fam_counts,
        "predicted_tactic_distribution": tac_counts,
    }


def compute_metrics(
    preds_df: pd.DataFrame,
    family_to_tactic: Dict[str, str],
) -> Dict[str, Any]:
    y_true = preds_df["true_family"].tolist()
    y_pred = preds_df["pred_family"].tolist()

    all_families_in_play = sorted(list(set(y_true + y_pred)))
    all_tactics_in_play = sorted(list(set(preds_df["true_tactic"].tolist() + preds_df["pred_tactic"].tolist())))

    # Full confusion matrix across all predicted classes so no window is lost
    cm_fam_df = pd.crosstab(preds_df["true_family"], preds_df["pred_family"])
    # Reindex to all families in play
    cm_fam_full = cm_fam_df.reindex(index=EVALUABLE_LOEO_FAMILIES, columns=all_families_in_play, fill_value=0)

    # Per-family metrics for the 3 evaluable LOEO families
    per_family = {}
    f1_list = []
    for fam in EVALUABLE_LOEO_FAMILIES:
        y_true_binary = (preds_df["true_family"] == fam).to_numpy()
        y_pred_binary = (preds_df["pred_family"] == fam).to_numpy()
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average="binary", zero_division=0
        )
        support = int(np.sum(y_true_binary))
        per_family[fam] = {
            "tactic": family_to_tactic.get(fam, "Unknown"),
            "support": support,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        }
        f1_list.append(f1)

    macro_f1_family = float(np.mean(f1_list))

    # Tactic-level confusion matrix and metrics
    evaluable_tactics = sorted(list(set(family_to_tactic[f] for f in EVALUABLE_LOEO_FAMILIES)))
    cm_tac_df = pd.crosstab(preds_df["true_tactic"], preds_df["pred_tactic"])
    cm_tac_full = cm_tac_df.reindex(index=evaluable_tactics, columns=all_tactics_in_play, fill_value=0)

    per_tactic = {}
    tac_f1_list = []
    for tac in evaluable_tactics:
        y_true_tac_bin = (preds_df["true_tactic"] == tac).to_numpy()
        y_pred_tac_bin = (preds_df["pred_tactic"] == tac).to_numpy()
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_tac_bin, y_pred_tac_bin, average="binary", zero_division=0
        )
        support = int(np.sum(y_true_tac_bin))
        per_tactic[tac] = {
            "support": support,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        }
        tac_f1_list.append(f1)

    macro_f1_tactic = float(np.mean(tac_f1_list))

    return {
        "total_loeo_test_attack_windows": len(preds_df),
        "macro_f1_family": macro_f1_family,
        "macro_f1_tactic": macro_f1_tactic,
        "evaluable_families": EVALUABLE_LOEO_FAMILIES,
        "per_family": per_family,
        "confusion_matrix_family": {
            "row_labels_true": EVALUABLE_LOEO_FAMILIES,
            "col_labels_pred": all_families_in_play,
            "matrix": cm_fam_full.to_numpy().tolist(),
        },
        "evaluable_tactics": evaluable_tactics,
        "per_tactic": per_tactic,
        "confusion_matrix_tactic": {
            "row_labels_true": evaluable_tactics,
            "col_labels_pred": all_tactics_in_play,
            "matrix": cm_tac_full.to_numpy().tolist(),
        },
    }


def write_family_markdown(results: Dict[str, Any], filepath: Path) -> None:
    loeo = results["loeo_evaluation"]
    seen = results["seen_family_evaluation"]
    benign_pos = results["benign_positives_614"]

    lines = [
        "# ATT&CK Stage & Family Classifier Evaluation Report (Phase B)",
        "",
        "## 1. Overview & Protocol",
        "- **Classifier**: Multinomial Logistic Regression (`lbfgs`, `max_iter=2000`, `seed=42`).",
        "- **Feature Set**: `ml1/artifacts/lr_set_a/set_a_features.json` (406 features).",
        "- **Dataset**: `data-engineering/data/ucs/ucs_windows_models_v1.parquet` (2,787 windows, 6 days).",
        "- **Cross-Validation**: 37-fold Leave-One-Episode-Out (LOEO) from `corrected_37fold_manifest.json`.",
        "- **Training Invariant**: Scaler fitted on each training fold independently; trained ONLY on `label_binary == 1` and non-Benign windows.",
        "",
        "---",
        "",
        "## 2. LOEO Evaluation Results (Held-Out Episodes)",
        f"- **Total Evaluable Test Attack Windows**: {loeo['total_loeo_test_attack_windows']}",
        f"- **LOEO Macro-F1 (Family)**: **{loeo['macro_f1_family']:.4f}**",
        f"- **LOEO Macro-F1 (ATT&CK Tactic)**: **{loeo['macro_f1_tactic']:.4f}**",
        "",
        "### Per-Family Performance (LOEO)",
        "| Family | ATT&CK Tactic | Support | Precision | Recall | F1-Score |",
        "|---|---|---|---|---|---|",
    ]

    for fam, m in loeo["per_family"].items():
        lines.append(f"| **{fam}** | {m['tactic']} | {m['support']} | {m['precision']:.4f} | {m['recall']:.4f} | **{m['f1']:.4f}** |")

    lines.extend([
        "",
        "### Confusion Matrix (Family)",
        f"Rows = True (`{loeo['confusion_matrix_family']['row_labels_true']}`), Columns = Predicted (`{loeo['confusion_matrix_family']['col_labels_pred']}`):",
        "```text",
    ])
    header_str = f"{'True \\ Pred':24s} | " + " | ".join(f"{c:18s}" for c in loeo["confusion_matrix_family"]["col_labels_pred"])
    lines.append(header_str)
    lines.append("-" * len(header_str))
    for row, lbl in zip(loeo["confusion_matrix_family"]["matrix"], loeo["confusion_matrix_family"]["row_labels_true"]):
        row_str = f"{lbl:24s} | " + " | ".join(f"{val:18d}" for val in row)
        lines.append(row_str)
    lines.extend([
        "```",
        "",
        "### Per-Tactic Performance (LOEO)",
        "| ATT&CK Tactic | Support | Precision | Recall | F1-Score |",
        "|---|---|---|---|---|",
    ])

    for tac, m in loeo["per_tactic"].items():
        lines.append(f"| **{tac}** | {m['support']} | {m['precision']:.4f} | {m['recall']:.4f} | **{m['f1']:.4f}** |")

    lines.extend([
        "",
        "### Confusion Matrix (ATT&CK Tactic)",
        f"Rows = True (`{loeo['confusion_matrix_tactic']['row_labels_true']}`), Columns = Predicted (`{loeo['confusion_matrix_tactic']['col_labels_pred']}`):",
        "```text",
    ])
    header_tac_str = f"{'True \\ Pred':24s} | " + " | ".join(f"{c:20s}" for c in loeo["confusion_matrix_tactic"]["col_labels_pred"])
    lines.append(header_tac_str)
    lines.append("-" * len(header_tac_str))
    for row, lbl in zip(loeo["confusion_matrix_tactic"]["matrix"], loeo["confusion_matrix_tactic"]["row_labels_true"]):
        row_str = f"{lbl:24s} | " + " | ".join(f"{val:20d}" for val in row)
        lines.append(row_str)
    lines.extend([
        "```",
        "",
        "---",
        "",
        "## 3. Single-Episode Families Evaluation",
        "> [!NOTE]",
        "> Families with only 1 episode (`DDOS-HOIC`, `Infiltration-Compromise`, `Infiltration-Portscan`) cannot be evaluated under LOEO because holding out the episode removes 100% of its training examples.",
        "",
        "### Seen-Family Evaluation (Within-Episode Chronological Split 70% / 30% with 30-Window Purge)",
        "| Family | Status | Windows (Total / Train / Purge / Test) | Precision | Recall | F1-Score | Mean Conf |",
        "|---|---|---|---|---|---|---|",
    ])

    for fam, m in seen.items():
        if m["evaluable"]:
            counts = f"{m['total_windows']} / {m['train_windows']} / {m['purged_windows']} / {m['test_windows']}"
            lines.append(
                f"| **{fam}** | Evaluated | {counts} | {m['precision']:.4f} | {m['recall']:.4f} | **{m['f1']:.4f}** | {m['mean_confidence']:.4f} |"
            )
        else:
            lines.append(
                f"| **{fam}** | Not Evaluable | {m['total_windows']} (too few windows for 30-win purge) | - | - | - | - |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Analysis of 614 Windows (`label_binary = 1` and `label_attack_type = Benign`)",
        "- **Total Isolated Windows**: 614",
        "- **Training Status**: Excluded from family classifier training per rule B-t2.",
        f"- **Mean Classifier Confidence**: {benign_pos['mean_confidence']:.4f} (Min: {benign_pos['min_confidence']:.4f}, Max: {benign_pos['max_confidence']:.4f})",
        "",
        "### Predicted Family Distribution on these 614 Windows",
        "| Predicted Family | Count | Share (%) |",
        "|---|---|---|",
    ])

    for fam, cnt in benign_pos["predicted_family_distribution"].items():
        share = (cnt / benign_pos["window_count"]) * 100
        lines.append(f"| {fam} | {cnt} | {share:.1f}% |")

    lines.extend([
        "",
        "### Predicted ATT&CK Tactic Distribution",
        "| Predicted Tactic | Count | Share (%) |",
        "|---|---|---|",
    ])

    for tac, cnt in benign_pos["predicted_tactic_distribution"].items():
        share = (cnt / benign_pos["window_count"]) * 100
        lines.append(f"| {tac} | {cnt} | {share:.1f}% |")

    lines.append("")
    filepath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[*] Loading inputs...")
    df, features, manifest, mapping_cfg = load_inputs()

    family_to_tactic = {
        fam: meta["tactic_name"]
        for fam, meta in mapping_cfg["families"].items()
        if meta.get("tactic_name")
    }

    print(f"[*] Training 37 LOEO fold models on {len(features)} features...")
    fold_summaries, preds_df = train_loeo_folds(df, features, manifest, family_to_tactic)

    print("[*] Computing LOEO metrics...")
    loeo_metrics = compute_metrics(preds_df, family_to_tactic)

    print("[*] Evaluating single-episode families under chronological split...")
    seen_metrics = evaluate_single_episode_seen(df, features, family_to_tactic)

    print("[*] Evaluating 614 benign-positive windows...")
    benign_pos_metrics = evaluate_614_benign_positives(df, features, manifest, family_to_tactic)

    results = {
        "dataset": "data-engineering/data/ucs/ucs_windows_models_v1.parquet",
        "feature_set": "ml1/artifacts/lr_set_a/set_a_features.json",
        "manifest": "ml1/artifacts/loeo/corrected_37fold_manifest.json",
        "classifier": "LogisticRegression(solver='lbfgs', max_iter=2000, random_state=42)",
        "loeo_evaluation": loeo_metrics,
        "single_episode_loeo_status": {
            fam: "not evaluable under LOEO: 1 episode"
            for fam in SINGLE_EPISODE_FAMILIES
        },
        "seen_family_evaluation": seen_metrics,
        "benign_positives_614": benign_pos_metrics,
        "folds": fold_summaries,
    }

    OUTPUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved results JSON to {RESULTS_JSON_PATH}")

    write_family_markdown(results, RESULTS_MD_PATH)
    print(f"[+] Saved report markdown to {RESULTS_MD_PATH}")

    print("\n========================================================")
    print(f"LOEO Family Macro-F1: {loeo_metrics['macro_f1_family']:.4f}")
    print(f"LOEO Tactic Macro-F1: {loeo_metrics['macro_f1_tactic']:.4f}")
    print("Per-Family F1:")
    for fam, m in loeo_metrics["per_family"].items():
        print(f"  {fam:18s}: F1 = {m['f1']:.4f} (P = {m['precision']:.4f}, R = {m['recall']:.4f}, N = {m['support']})")
    print("Per-Tactic F1:")
    for tac, m in loeo_metrics["per_tactic"].items():
        print(f"  {tac:24s}: F1 = {m['f1']:.4f} (P = {m['precision']:.4f}, R = {m['recall']:.4f}, N = {m['support']})")
    print("========================================================")


if __name__ == "__main__":
    main()
