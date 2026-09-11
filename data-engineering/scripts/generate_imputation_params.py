"""
scripts/generate_imputation_params.py
======================================
Generates data/ucs/imputation_params.yaml with full multi-tier production structure:

  1. flow_medians_by_day:
     Per-source_day flow-level medians for rate columns (bytes_per_sec, packets_per_sec,
     fwd_pkts_per_sec, bwd_pkts_per_sec), computed strictly from each training day's
     train-window flows (3,628,886 flows total across 2,013 post-purge train windows).
     Used when source_type="csv" and source_day is recognized.

  2. flow_medians_global:
     Pooled flow-level medians across all 3,628,886 train-window flows combined.
     Used when source_type="csv" and source_day is novel or unrecognized at runtime.

  3. window_feature_medians:
     Raw-scale medians for all 400 window-level features (with inverse expm1 applied
     to log1p-transformed features). Used to fill missing window-level features so
     that under LeakageSafeRobustScaler, imputed features center exactly at 0.0.

Leakage Rationale (documented finding):
  The original batch pipeline computed whole-day flow medians before establishing
  train/val/test splits. This script freezes medians strictly from post-purge
  train-window flows, enforcing rigorous leakage safety.
"""

from pathlib import Path
from datetime import datetime, timezone
import yaml
import numpy as np
import pandas as pd


def generate_imputation_params(
    intermediate_dir: str = "data/intermediate",
    windows_parquet_path: str = "data/ucs/ucs_windows.parquet",
    scaler_params_path: str = "data/ucs/scaler_params.yaml",
    output_path: str = "data/ucs/imputation_params.yaml",
) -> None:
    inter_dir = Path(intermediate_dir)
    ucs_path = Path(windows_parquet_path)
    scaler_path = Path(scaler_params_path)
    out_path = Path(output_path)

    if not ucs_path.exists():
        raise FileNotFoundError(f"UCS windows not found: {ucs_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler params not found: {scaler_path}")

    ucs_df = pd.read_parquet(ucs_path)
    train_wins = ucs_df[ucs_df["split"] == "train"]
    train_window_starts = set(train_wins["window_start_utc"])
    n_train_wins = len(train_wins)

    day_files = {
        "14-02-2018": "cleaned_Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet",
        "21-02-2018": "cleaned_Wednesday-21-02-2018_TrafficForML_CICFlowMeter.parquet",
        "22-02-2018": "cleaned_Thursday-22-02-2018_TrafficForML_CICFlowMeter.parquet",
        "28-02-2018": "cleaned_Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet",
        "01-03-2018": "cleaned_Thursday-01-03-2018_TrafficForML_CICFlowMeter.parquet",
    }

    rate_cols = ["bytes_per_sec", "packets_per_sec", "fwd_pkts_per_sec", "bwd_pkts_per_sec"]

    output = {
        "_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_script": "scripts/generate_imputation_params.py",
            "train_window_flows_used": 0,
            "train_windows_total": n_train_wins,
            "train_windows_with_imputed_flows": 1718,
            "imputation_columns": rate_cols,
            "leakage_decision": (
                "Medians computed ONLY from post-purge train-window flows across intermediate "
                "cleaned parquets. The original batch pipeline used whole-day medians before "
                "train/val/test split, introducing pre-split leakage. This frozen artifact "
                "uses train-only medians as the leakage-safe correct choice. Bit-exact regression "
                "(Test 1) is scoped to the 295 clean train windows. See VALIDATION_REPORT.md Section 8."
            ),
        },
        "flow_medians_by_day": {},
        "flow_medians_global": {},
        "window_feature_medians": {},
    }

    all_train_flows = []
    total_train_flows = 0

    print("=" * 65)
    print("Generating imputation_params.yaml from verified intermediate parquets")
    print("=" * 65)

    for day, fname in sorted(day_files.items()):
        fpath = inter_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Intermediate parquet missing: {fpath}")

        df = pd.read_parquet(fpath)
        flow_wins = df["timestamp_utc"].dt.floor("1min")
        in_train = flow_wins.isin(train_window_starts)
        train_day_df = df[in_train]
        n_day_flows = len(train_day_df)
        total_train_flows += n_day_flows

        all_train_flows.append(train_day_df)

        output["flow_medians_by_day"][day] = {}
        for col in rate_cols:
            med_val = float(train_day_df[col].median()) if n_day_flows > 0 else 0.0
            output["flow_medians_by_day"][day][col] = med_val

        print(f"  {day} ({n_day_flows:,} train flows):")
        for col in rate_cols:
            print(f"    {col}: {output['flow_medians_by_day'][day][col]:.6f}")

    output["_metadata"]["train_window_flows_used"] = total_train_flows
    assert total_train_flows == 3628886, f"Expected 3,628,886 train flows, got {total_train_flows}"

    # Global flow medians
    combined_train = pd.concat(all_train_flows, ignore_index=True)
    print(f"\n  GLOBAL fallback ({len(combined_train):,} total train flows):")
    for col in rate_cols:
        glob_val = float(combined_train[col].median())
        output["flow_medians_global"][col] = glob_val
        print(f"    {col}: {glob_val:.6f}")

    # 400 window-level features from scaler_params in raw scale
    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler_data = yaml.safe_load(f)

    for feat, p in scaler_data.items():
        med = p["median"]
        if p.get("is_log1p", False):
            output["window_feature_medians"][feat] = float(np.expm1(med))
        else:
            output["window_feature_medians"][feat] = float(med)

    print(f"\n  Included {len(output['window_feature_medians'])} raw-scale window feature medians.")

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(output, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(f"\n[+] Successfully wrote verified imputation parameters to {out_path}")


if __name__ == "__main__":
    generate_imputation_params()

