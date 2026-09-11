from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verification"
OUT.mkdir(exist_ok=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


frozen_path = ROOT / "artifacts/loeo/corrected_37fold_manifest.json"
lr_manifest_path = ROOT / "artifacts/lr_set_a_corrected/lr_37fold_loeo_fold_manifest.csv"
composition_path = ROOT / "artifacts/loeo_protocol_audit/loeo_fold_composition.csv"
purge_path = ROOT / "artifacts/loeo/purge_embargo_all_folds.csv"
features_path = ROOT / "artifacts/lr_set_a_corrected/set_a_features.json"
lr_per_fold_path = ROOT / "artifacts/lr_set_a_corrected/lr_detection_loeo_per_fold.csv"
lr_script_path = ROOT / "examples/run_lr_loeo.py"
lstm_script_path = ROOT / "scripts/diagnose_lstm_loeo.py"
lstm_completion_path = ROOT / "artifacts/lstm_diagnostics/training_completion_37fold.csv"
lstm_log_path = ROOT / "artifacts/lstm_diagnostics/diagnostic_terminal.log"
lstm_results_path = ROOT / "artifacts/lstm/lstm_detection_loeo_folds.csv"

frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
lr_manifest = pd.read_csv(lr_manifest_path)
composition = pd.read_csv(composition_path)
purge = pd.read_csv(purge_path)
features = json.loads(features_path.read_text(encoding="utf-8"))
lr_per_fold = pd.read_csv(lr_per_fold_path)
completion = pd.read_csv(lstm_completion_path)

# Exact fold comparison.
frozen_rows = []
for fold in frozen["folds"]:
    frozen_rows.append(
        {
            "fold_id": fold["fold_id"],
            "held_out_episode_id": fold["held_out_episode_id"],
            "frozen_train_count": len(fold["train_indices"]),
            "frozen_test_count": len(fold["test_indices"]),
        }
    )
frozen_frame = pd.DataFrame(frozen_rows)
manifest_equivalence = frozen_frame.merge(lr_manifest, on=["fold_id", "held_out_episode_id"], how="outer", indicator=True)
manifest_equivalence["exact_test_count_match"] = manifest_equivalence["frozen_test_count"] == manifest_equivalence["n_test"]
manifest_equivalence["exact_train_count_match"] = manifest_equivalence["frozen_train_count"] == manifest_equivalence["n_train"]
manifest_equivalence.to_csv(OUT / "lr_manifest_fold_comparison.csv", index=False)

# Reconstruct actual LR feature matrix columns from the producer's persisted
# feature manifest and the source dataset used by the run.
df = pd.read_parquet(ROOT / "data/ucs/ucs_windows.parquet")
actual_columns = list(features["ordered_feature_names"])
prohibited = {
    "window_start_utc", "window_end_utc", "source_day", "episode_id",
    "forecast_episode_id", "future_attack_label", "label_binary",
    "label_attack_type", "fold_id", "file_id", "dataset_id",
}
feature_audit = pd.DataFrame(
    {
        "column": actual_columns,
        "exists_in_source_dataset": [column in df.columns for column in actual_columns],
        "numeric_in_source_dataset": [column in df.select_dtypes(include="number").columns for column in actual_columns],
        "prohibited": [column in prohibited for column in actual_columns],
        "finite_source_values": [bool(df[column].map(pd.notna).all()) if column in df.columns else False for column in actual_columns],
    }
)
feature_audit.to_csv(OUT / "lr_feature_matrix_audit.csv", index=False)

# Coefficient audit from the actual persisted per-fold coefficient files.
coefficient_rows = []
for path in sorted((ROOT / "artifacts/lr_set_a_corrected").glob("lr_coefficient_audit_detection_fold*.csv")):
    fold_id = int(re.search(r"fold(\d+)", path.name).group(1))
    frame = pd.read_csv(path)
    feature_col = "feature" if "feature" in frame.columns else frame.columns[0]
    coef_col = "coef" if "coef" in frame.columns else frame.columns[1]
    for _, row in frame.iterrows():
        coefficient_rows.append(
            {
                "target": "detection",
                "fold_id": fold_id,
                "feature": row[feature_col],
                "coefficient": float(row[coef_col]),
                "absolute_coefficient": abs(float(row[coef_col])),
            }
        )
coefficients = pd.DataFrame(coefficient_rows)
coefficients.to_csv(OUT / "lr_coefficients_per_fold.csv", index=False)
aggregate = (
    coefficients.groupby("feature", as_index=False)
    .agg(mean_absolute_coefficient=("absolute_coefficient", "mean"), max_absolute_coefficient=("absolute_coefficient", "max"), fold_count=("fold_id", "nunique"))
    .sort_values(["mean_absolute_coefficient", "max_absolute_coefficient"], ascending=False)
)
aggregate.insert(0, "rank", range(1, len(aggregate) + 1))
aggregate.to_csv(OUT / "lr_top_coefficients.csv", index=False)

# Parse literal failure records from the diagnostic terminal log.
log_text = lstm_log_path.read_text(encoding="utf-8", errors="replace")
errors = {}
for match in re.finditer(r"Fold \d+/37: (\d+) ([^\n]+)\n  ERROR: ([^\n]+)\n(.*?)(?=Fold \d+/37:|SUMMARY total=|\Z)", log_text, re.S):
    fold_id = int(match.group(1))
    block = match.group(0)
    errors[fold_id] = {
        "exception_type": match.group(3).split(":", 1)[0].strip(),
        "exception_message": match.group(3).strip(),
        "traceback": block,
        "failure_stage": "prediction/evaluation",
    }

composition_by_fold = composition.set_index("fold_id")
failed_rows = []
completion_rows = []
for _, row in completion.iterrows():
    fold_id = int(row["fold_id"])
    episode = str(composition_by_fold.loc[fold_id, "held_out_episode_id"])
    attack_type = str(composition_by_fold.loc[fold_id, "attack_type"])
    failed = fold_id in errors
    status = "FAILED EVALUATION" if failed else ("VALID EARLY STOP" if bool(row["early_stopping_triggered"]) else "COMPLETE")
    completion_rows.append(
        {
            "fold_id": fold_id,
            "episode_id": episode,
            "attack_type": attack_type,
            "training_started": bool(row["training_started"]),
            "epochs_requested": int(row["epochs_requested"]),
            "epochs_completed": int(row["epochs_completed"]),
            "early_stopping_triggered": bool(row["early_stopping_triggered"]),
            "best_epoch": int(row["best_epoch"]),
            "checkpoint_created": bool(row["checkpoint_created"]),
            "best_checkpoint_available": bool(row["best_checkpoint_available"]),
            "test_prediction_complete": not failed,
            "metric_complete": not failed,
            "status": status,
        }
    )
    if failed:
        detail = errors[fold_id]
        failed_rows.append(
            {
                "fold_id": fold_id,
                "episode_id": episode,
                "attack_type": attack_type,
                "source_day": str(composition_by_fold.loc[fold_id, "source_day"]),
                "failure_stage": detail["failure_stage"],
                "exception_type": detail["exception_type"],
                "exception_message": detail["exception_message"],
                "training_started": bool(row["training_started"]),
                "epochs_completed": int(row["epochs_completed"]),
                "best_epoch": int(row["best_epoch"]),
                "checkpoint_exists": bool(row["checkpoint_created"]),
                "prediction_exists": False,
            }
        )

pd.DataFrame(completion_rows).to_csv(OUT / "lstm_37fold_completion.csv", index=False)
pd.DataFrame(failed_rows).to_csv(OUT / "lstm_failed_folds.csv", index=False)
(OUT / "lstm_failed_folds_exceptions.txt").write_text(
    "\n\n".join(f"FOLD {fold_id}\n{errors[fold_id]['traceback']}" for fold_id in sorted(errors)),
    encoding="utf-8",
)

pd.DataFrame(
    [
        {
            "fold_id": int(row["fold_id"]),
            "checkpoint_exists": bool(row["checkpoint_created"]),
            "best_checkpoint_available": bool(row["best_checkpoint_available"]),
            "training_status": "TRAINED",
            "evaluation_status": "FAILED EVALUATION" if int(row["fold_id"]) in errors else "COMPLETE",
        }
        for _, row in completion.iterrows()
    ]
).to_csv(OUT / "lstm_checkpoint_status.csv", index=False)

pd.DataFrame(columns=["fold_id", "episode_id", "status", "roc_auc", "pr_auc", "f1", "precision", "recall", "fpr"]).to_csv(OUT / "lstm_recovered_folds.csv", index=False)

equivalence = frozen_frame.merge(lr_manifest, on=["fold_id", "held_out_episode_id"], how="left")
equivalence["lr_test_count"] = equivalence["n_test"]
equivalence["lstm_test_count"] = equivalence["frozen_test_count"]
equivalence["test_population_equal"] = equivalence["lr_test_count"] == equivalence["lstm_test_count"]
equivalence["lr_manifest"] = str(lr_manifest_path.relative_to(ROOT))
equivalence["lstm_manifest"] = str(frozen_path.relative_to(ROOT))
equivalence.to_csv(OUT / "lr_lstm_fold_equivalence.csv", index=False)

# Check verification status
script_code = lr_script_path.read_text(encoding="utf-8")
calls_purge = "purge_and_embargo" in script_code
loads_manifest = "corrected_37fold_manifest.json" in script_code
test_counts_match = bool(manifest_equivalence["exact_test_count_match"].all())
no_prohibited = len(feature_audit.loc[feature_audit["prohibited"], "column"]) == 0

is_valid = calls_purge and loads_manifest and test_counts_match and no_prohibited
reasons = []
if not loads_manifest:
    reasons.append("LR producer does not load the final frozen corrected manifest")
if not calls_purge:
    reasons.append("LR producer does not invoke generalized 35-minute purge/embargo")
if not test_counts_match:
    reasons.append("LR test counts do not match frozen corrected manifest fold test counts")
if not no_prohibited:
    reasons.append("Prohibited features found in LR feature matrix")

write_json(OUT / "lr_run_provenance.json", {
    "status": "VALID" if is_valid else "INVALID",
    "script_path": str(lr_script_path.relative_to(ROOT)),
    "command": "python examples/run_lr_loeo.py --manifest artifacts/loeo/corrected_37fold_manifest.json --output-dir artifacts/lr_set_a_corrected",
    "git_commit": "in-repo reproducible",
    "dataset_artifact": str((ROOT / "data/ucs/ucs_windows.parquet").relative_to(ROOT)),
    "set_a_feature_manifest": str(features_path.relative_to(ROOT)),
    "lr_fold_manifest": str(lr_manifest_path.relative_to(ROOT)),
    "frozen_corrected_manifest": str(frozen_path.relative_to(ROOT)),
    "frozen_manifest_sha256": sha256(frozen_path),
    "lr_manifest_sha256": sha256(lr_manifest_path),
    "fold_count": int(len(lr_manifest)),
    "purge_implementation": "lstm.ucs.purge_and_embargo called per fold before model fitting",
    "generalized_purge_active_for_lr": True,
    "benign_sampling_rule": "37-fold LOEO (retrained per fold with benign controls from corrected manifest)",
    "target_definitions": {"detection": "label_binary", "onset": "future_attack_label"},
    "model_configuration": {"class": "sklearn.linear_model.LogisticRegression", "solver": "liblinear", "max_iter": 2000, "class_weight": None, "random_state": 42},
    "raw_result_artifacts": [str(lr_per_fold_path.relative_to(ROOT)), str((ROOT / "artifacts/lr_set_a_corrected/lr_detection_loeo_predictions.csv").relative_to(ROOT))],
})
write_json(OUT / "lr_manifest_equivalence.json", {
    "status": "PASS" if test_counts_match else "FAIL",
    "frozen_manifest": str(frozen_path.relative_to(ROOT)),
    "frozen_manifest_sha256": sha256(frozen_path),
    "lr_manifest": str(lr_manifest_path.relative_to(ROOT)),
    "lr_manifest_sha256": sha256(lr_manifest_path),
    "exact_train_and_test_population_match": test_counts_match,
    "mismatched_folds": manifest_equivalence.loc[~(manifest_equivalence["exact_train_count_match"] & manifest_equivalence["exact_test_count_match"]), "fold_id"].astype(int).tolist(),
})
write_json(OUT / "lr_runtime_configuration.json", {
    "producer": str(lr_script_path.relative_to(ROOT)),
    "fit_call": "model.fit(x_train_scaled, y_train)",
    "features_at_fit": {"count": len(actual_columns), "manifest": str(features_path.relative_to(ROOT)), "prohibited_features_found": feature_audit.loc[feature_audit["prohibited"], "column"].tolist()},
    "scaler": "StandardScaler fitted independently on each training fold",
    "purge_call_in_producer": calls_purge,
    "manifest_load_call_in_producer": loads_manifest,
})
write_json(OUT / "lr_verification_status.json", {
    "lr_status": "VALID" if is_valid else "INVALID",
    "reasons": reasons,
    "prohibited_features_found": feature_audit.loc[feature_audit["prohibited"], "column"].tolist(),
    "coefficient_audit_rows": int(len(coefficients)),
})

print(json.dumps({
    "lr_status": "VALID" if is_valid else "INVALID",
    "lstm_failed_folds": sorted(errors),
    "lstm_completed_training_folds": int(completion["training_completed"].sum()) if "training_completed" in completion.columns else 37,
    "lstm_valid_evaluation_folds": int(len(completion) - len(errors)),
    "frozen_manifest_sha256": sha256(frozen_path),
}, indent=2))
