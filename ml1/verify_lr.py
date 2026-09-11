import pandas as pd
import json
import os
import glob
from pathlib import Path
import traceback

workspace_dir = Path("c:/Users/Vicky/Documents/SIH_2026/LSTM-type-model-CICIDS2018-UCS")
verification_dir = workspace_dir / "verification"
verification_dir.mkdir(parents=True, exist_ok=True)

# 1. LR Provenance & Manifest Verification
manifest_path = workspace_dir / "artifacts/loeo/corrected_37fold_manifest.json"
lr_script = workspace_dir / "scripts/run_lr_frozen.py"
purge_embargo_file = workspace_dir / "artifacts/loeo/purge_embargo_all_folds.csv"
lr_folds_file = workspace_dir / "artifacts/lr/lr_detection_loeo_folds.csv"
lr_set_a = workspace_dir / "artifacts/lr_set_a/set_a_features.json"

with open(manifest_path, "r") as f:
    manifest = json.load(f)

purge_df = pd.read_csv(purge_embargo_file)
lr_folds_df = pd.read_csv(lr_folds_file)

# Assertions
purge_active = purge_df['Violations'].sum() == 0 and purge_df['Min Train->Test Distance'].min() >= 35.0
lr_provenance_status = "VERIFIED" if purge_active and len(manifest['folds']) == 37 and len(lr_folds_df) == 37 else "INVALID"

with open(lr_set_a, "r") as f:
    set_a = json.load(f)
features = set_a['ordered_feature_names']
prohibited = ["window_start_utc", "window_end_utc", "source_day", "episode_id", "forecast_episode_id", "future_attack_label"]
prohibited_found = [p for p in prohibited if p in features]

lr_provenance = {
    "LR script": str(lr_script.relative_to(workspace_dir)),
    "Dataset artifact": "data/ucs/ucs_windows.parquet",
    "Set-A feature manifest": str(lr_set_a.relative_to(workspace_dir)),
    "LOEO manifest": str(manifest_path.relative_to(workspace_dir)),
    "Fold count": len(manifest['folds']),
    "Purge implementation": "generalized 35-minute temporal purge",
    "Generalized purge active": "YES" if purge_active else "NO",
    "LR STATUS": lr_provenance_status,
    "Prohibited Features": prohibited_found
}

with open(verification_dir / "lr_run_provenance.json", "w") as f:
    json.dump(lr_provenance, f, indent=2)

# 5. LR Coefficient Audit
coef_files = glob.glob(str(workspace_dir / "artifacts/lr/lr_coefficient_audit_detection_*.csv"))
all_top_coefs = []
perfect_folds = 0
for cf in coef_files:
    df = pd.read_csv(cf)
    fold_id = int(os.path.basename(cf).split('fold')[1].split('.csv')[0])
    perfect_folds += 1
    top_10 = df.head(10)['feature'].tolist()
    all_top_coefs.extend(top_10)

from collections import Counter
top_coefs_count = Counter(all_top_coefs)

coef_audit = pd.DataFrame(top_coefs_count.most_common(20), columns=["feature", "count"])
coef_audit.to_csv(verification_dir / "lr_top_coefficients.csv", index=False)

# LSTM Evaluation
lstm_smoke4_path = workspace_dir / "artifacts/lstm_diagnostics_smoke4"
lstm_metrics = workspace_dir / "artifacts/ucs_experiment_20260904/lstm_metrics.json" # If applicable?
# I need to see what files exist for LSTM failures. Wait, lstm_smoke4/diagnostic_summary.json probably has it.
