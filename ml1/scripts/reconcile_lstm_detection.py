"""
Reconcile LSTM Detection Result

Reads the existing LSTM LOEO detection metrics to evaluate whether poor F1
is due to a representation/architecture problem (poor ROC-AUC/PR-AUC)
or merely a threshold/calibration problem (strong AUC but poor fixed-threshold F1).
"""
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

def main():
    workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    det_csv = workspace_dir / 'artifacts' / 'lstm' / 'lstm_detection_loeo_folds.csv'
    onset_csv = workspace_dir / 'artifacts' / 'lstm' / 'lstm_onset_loeo_folds.csv'
    out_dir = workspace_dir / 'artifacts' / 'lstm_diagnostics'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'reconciled_lstm_detection_summary.csv'

    print("=== Reconciling LSTM Detection ===")
    if not det_csv.exists() or not onset_csv.exists():
        print(f"ERROR: Missing input files. Need {det_csv.name} and {onset_csv.name}")
        sys.exit(1)

    df_det = pd.read_csv(det_csv)
    df_onset = pd.read_csv(onset_csv)

    print(f"Loaded {len(df_det)} detection folds and {len(df_onset)} onset folds.")

    # Check for failures/NaNs
    det_nans = df_det.isna().sum().sum()
    onset_nans = df_onset.isna().sum().sum()
    print(f"NaNs in detection metrics: {det_nans}")
    print(f"NaNs in onset metrics: {onset_nans}")

    def summarize(df, name):
        f1_mean = df['f1'].mean()
        f1_median = df['f1'].median()
        roc_mean = df['roc_auc'].mean()
        roc_median = df['roc_auc'].median()
        pr_mean = df['pr_auc'].mean()
        pr_median = df['pr_auc'].median()
        
        print(f"\n--- {name} Metrics ---")
        print(f"F1      : Mean={f1_mean:.4f}, Median={f1_median:.4f}")
        print(f"ROC-AUC : Mean={roc_mean:.4f}, Median={roc_median:.4f}")
        print(f"PR-AUC  : Mean={pr_mean:.4f}, Median={pr_median:.4f}")
        
        return {
            'task': name,
            'f1_mean': f1_mean,
            'f1_median': f1_median,
            'roc_auc_mean': roc_mean,
            'roc_auc_median': roc_median,
            'pr_auc_mean': pr_mean,
            'pr_auc_median': pr_median
        }

    res_det = summarize(df_det, "Detection")
    res_onset = summarize(df_onset, "Onset")

    # Conclusion
    print("\n--- Diagnostic Conclusion ---")
    if res_det['roc_auc_mean'] > 0.8 and res_det['f1_mean'] < 0.5:
        msg = "High ROC-AUC but poor F1: The primary issue is THRESHOLD CALIBRATION, not representation/architecture."
    elif res_det['roc_auc_mean'] < 0.6:
        msg = "Poor ROC-AUC: The primary issue is representation/architecture."
    else:
        msg = "Mixed/moderate performance across the board."
    print(f"Conclusion: {msg}")

    # Export
    out_df = pd.DataFrame([res_det, res_onset])
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved reconciled summary to {out_csv}")

if __name__ == '__main__':
    main()
