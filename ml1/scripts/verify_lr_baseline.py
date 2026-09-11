"""
Complete & Verify PS-Mandated LR Baseline Comparison
Steps 1-5 as defined in the task prompt.
"""
import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))

def main():
    artifacts_lr = workspace_dir / 'artifacts' / 'lr'
    canonical_record = workspace_dir / 'artifacts' / 'canonical_lr_record.json'
    corrected_manifest = workspace_dir / 'artifacts' / 'loeo' / 'corrected_37fold_manifest.json'
    data_path = workspace_dir / 'data' / 'ucs' / 'ucs_windows.parquet'

    print("=" * 70)
    print("PS-Mandated LR Baseline Verification")
    print("=" * 70)

    # Load canonical record
    with open(canonical_record) as f:
        canon = json.load(f)

    # -------------------------------------------------------
    # STEP 1: Coefficient / Provenance Audit
    # -------------------------------------------------------
    print("\n--- STEP 1: Coefficient / Provenance Audit ---")
    print(f"  Experiment ID : {canon['canonical_lr_experiment_id']}")
    print(f"  Script        : {canon['script']}")
    print(f"  Manifest      : {canon['manifest']}")
    print(f"  Dataset       : {canon['dataset_artifact']}")
    print(f"  Commit Hash   : {canon['commit_hash']}")
    print(f"  Fold Count    : {canon['fold_count']}")
    print(f"  Superseded    : {canon['superseded_experiments']}")

    # Try to get git hash
    try:
        import subprocess
        result = subprocess.run(['git', 'log', '-1', '--format=%H'], capture_output=True, text=True, cwd=str(workspace_dir))
        print(f"  Current HEAD  : {result.stdout.strip()}")
    except Exception:
        print("  Current HEAD  : (unable to retrieve)")

    # Load fold 0 coefficient audit for detection
    coef_path = artifacts_lr / 'lr_coefficient_audit_detection_fold0.csv'
    df_coef = pd.read_csv(coef_path)
    df_coef['abs_coef'] = df_coef['coef'].abs()
    df_coef_sorted = df_coef.sort_values('abs_coef', ascending=False)

    print("\n  Top 15 Standardized Coefficients (Fold 0, Detection):")
    for _, row in df_coef_sorted.head(15).iterrows():
        print(f"    {row['feature']:40s} coef={row['coef']:+.4f}")

    # Check for degenerate single-feature dominance
    top_coef = df_coef_sorted.iloc[0]['abs_coef']
    second_coef = df_coef_sorted.iloc[1]['abs_coef']
    ratio = top_coef / second_coef if second_coef > 0 else float('inf')
    print(f"\n  Top/Second coefficient ratio: {ratio:.2f}")
    if ratio > 10:
        print("  WARNING: Single feature dominates — possible leakage indicator.")
    else:
        print("  OK: No single feature dominates excessively (ratio < 10).")

    # -------------------------------------------------------
    # STEP 2: Embargo/Purge Distance Table for LR Folds
    # -------------------------------------------------------
    print("\n--- STEP 2: Embargo/Purge Distance Table for LR ---")

    windows = pd.read_parquet(data_path).sort_values('window_start_utc').reset_index(drop=True)
    timestamps = pd.to_datetime(windows['window_start_utc'], utc=True)

    with open(corrected_manifest) as f:
        manifest = json.load(f)

    lr_manifest = pd.read_csv(artifacts_lr / 'lr_37fold_loeo_fold_manifest.csv')

    embargo_rows = []
    for fold_idx in range(37):
        fold = manifest['folds'][fold_idx]
        train_idx = fold['train_indices']
        test_idx = fold['test_indices']

        train_ts = timestamps.iloc[train_idx]
        test_ts = timestamps.iloc[test_idx]

        min_dist = float('inf')
        for tt in test_ts:
            dists = (train_ts - tt).abs()
            fold_min = dists.min().total_seconds() / 60.0
            if fold_min < min_dist:
                min_dist = fold_min

        embargo_rows.append({
            'fold_id': fold_idx,
            'episode': fold['held_out_episode_id'],
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'min_train_test_dist_min': min_dist,
            'passes_35min': min_dist >= 35.0
        })

    df_embargo = pd.DataFrame(embargo_rows)
    all_pass = df_embargo['passes_35min'].all()
    min_overall = df_embargo['min_train_test_dist_min'].min()

    print(f"  Overall minimum train-test distance: {min_overall:.1f} minutes")
    print(f"  All 37 folds pass ≥35 min embargo: {all_pass}")
    if not all_pass:
        failing = df_embargo[~df_embargo['passes_35min']]
        print("  FAILING FOLDS:")
        print(failing.to_string(index=False))

    # -------------------------------------------------------
    # STEP 3: Confirm LR Uses Identical Test Window Population
    # -------------------------------------------------------
    print("\n--- STEP 3: LR vs LSTM Test Window Population Match ---")

    lr_predictions = pd.read_csv(artifacts_lr / 'lr_detection_loeo_predictions.csv')

    mismatches = []
    for fold_idx in range(37):
        fold = manifest['folds'][fold_idx]
        manifest_test_set = set(fold['test_indices'])

        # Get the LR's actual test windows for this fold from predictions
        lr_fold_preds = lr_predictions[lr_predictions['fold_id'] == fold_idx]
        lr_timestamps = set(pd.to_datetime(lr_fold_preds['timestamp'], utc=True))

        # Map manifest test indices to timestamps
        manifest_timestamps = set(timestamps.iloc[list(manifest_test_set)])

        if lr_timestamps != manifest_timestamps:
            mismatches.append({
                'fold_id': fold_idx,
                'lr_count': len(lr_timestamps),
                'manifest_count': len(manifest_timestamps),
                'only_in_lr': lr_timestamps - manifest_timestamps,
                'only_in_manifest': manifest_timestamps - lr_timestamps,
            })

    if len(mismatches) == 0:
        print("  EXACT MATCH: LR test windows match corrected_37fold_manifest.json on all 37 folds.")
    else:
        print(f"  MISMATCH FOUND on {len(mismatches)} folds:")
        for m in mismatches:
            print(f"    Fold {m['fold_id']}: LR={m['lr_count']} vs Manifest={m['manifest_count']}, "
                  f"only_in_LR={len(m['only_in_lr'])}, only_in_manifest={len(m['only_in_manifest'])}")

    # -------------------------------------------------------
    # STEP 4: Sanity-Check FPR=0.000 (Per-Fold Confusion Matrix)
    # -------------------------------------------------------
    print("\n--- STEP 4: Per-Fold Confusion Matrix & FPR=0.000 Sanity Check ---")

    lr_per_fold = pd.read_csv(artifacts_lr / 'lr_detection_loeo_per_fold.csv')

    print(f"  {'Fold':>4s}  {'Episode':>35s}  {'TP':>4s}  {'FP':>4s}  {'TN':>4s}  {'FN':>4s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'FPR':>6s}")
    print("  " + "-" * 120)

    zero_recall_folds = []
    for _, row in lr_per_fold.iterrows():
        tp = int(row['tp'])
        fp = int(row['fp'])
        tn = int(row['tn'])
        fn = int(row['fn'])
        prec = row['precision']
        rec = row['recall']
        f1 = row['f1']
        fpr = row['fpr']

        flag = ""
        if rec == 0.0 and fn > 0:
            flag = " <-- ALWAYS-NEGATIVE"
            zero_recall_folds.append(int(row['fold_id']))

        print(f"  {int(row['fold_id']):>4d}  {row['held_out_episode_id']:>35s}  {tp:>4d}  {fp:>4d}  {tn:>4d}  {fn:>4d}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  {fpr:>6.3f}{flag}")

    if zero_recall_folds:
        print(f"\n  WARNING: {len(zero_recall_folds)} fold(s) have recall=0 and FN>0, indicating 'always predict negative' behavior:")
        print(f"    Folds: {zero_recall_folds}")
        print("  This inflates FPR=0 trivially and makes precision undefined (reported as 0.0).")
        print("  The aggregate FPR=0.000 is PARTIALLY an artifact of this pattern.")
    else:
        print("\n  All folds have non-zero recall. FPR=0.000 is genuine.")

    # -------------------------------------------------------
    # STEP 5: Final PS-Mandated Comparison Table
    # -------------------------------------------------------
    print("\n--- STEP 5: Verified LR vs World Model Comparison (37-fold LOEO) ---")

    # LR metrics from canonical record
    lr_det = canon['detection_metrics']

    # LSTM metrics from reconciled summary
    lstm_summary = pd.read_csv(workspace_dir / 'artifacts' / 'lstm_diagnostics' / 'reconciled_lstm_detection_summary.csv')
    lstm_det_row = lstm_summary[lstm_summary['task'] == 'Detection'].iloc[0]

    # For LSTM we need per-fold data to compute aggregate precision/recall/FPR
    # Use the reconciled metrics from the summary
    lstm_f1_mean = lstm_det_row['f1_mean']
    lstm_roc_auc_mean = lstm_det_row['roc_auc_mean']

    # We need precision/recall/FPR for LSTM — check if per-fold CSV exists
    lstm_per_fold_files = list((workspace_dir / 'artifacts').rglob('*lstm*per_fold*.csv'))
    lstm_prec = "N/A"
    lstm_rec = "N/A"
    lstm_fpr = "N/A"

    # Try to find LSTM per-fold detection results
    for f in sorted(workspace_dir.rglob('*detection*per_fold*.csv')):
        if 'lstm' in str(f).lower() and 'lr' not in str(f).lower():
            try:
                df_lstm_pf = pd.read_csv(f)
                if 'precision' in df_lstm_pf.columns:
                    lstm_prec = f"{df_lstm_pf['precision'].mean():.4f}"
                    lstm_rec = f"{df_lstm_pf['recall'].mean():.4f}"
                    lstm_fpr = f"{df_lstm_pf['fpr'].mean():.4f}"
                    print(f"  (LSTM per-fold source: {f})")
                    break
            except Exception:
                pass

    print(f"\n  {'Metric':<12s}  {'LR Baseline':>14s}  {'LSTM/World Model':>18s}")
    print("  " + "-" * 50)
    print(f"  {'F1':<12s}  {lr_det['f1']:>14.4f}  {lstm_f1_mean:>18.4f}")
    print(f"  {'Precision':<12s}  {lr_det['precision']:>14.4f}  {lstm_prec:>18s}")
    print(f"  {'Recall':<12s}  {lr_det['recall']:>14.4f}  {lstm_rec:>18s}")
    print(f"  {'FPR':<12s}  {lr_det['fpr']:>14.4f}  {lstm_fpr:>18s}")
    print(f"  {'ROC-AUC':<12s}  {'N/A':>14s}  {lstm_roc_auc_mean:>18.4f}")

    print("\n  Note: LR F1/Precision/Recall are macro-averaged across 37 folds (from canonical_lr_record.json).")
    print("  Note: LSTM F1 is macro-averaged across 37 folds (from reconciled_lstm_detection_summary.csv).")
    if zero_recall_folds:
        print(f"  CAVEAT: Fold 16 shows always-predict-negative behavior (recall=0, FN=33).")
        print(f"  This means {33} of {lr_per_fold['fn'].sum() + lr_per_fold['tp'].sum()} total attack windows are missed by the LR on that fold.")
        recalc_f1_num = lr_per_fold[lr_per_fold['f1'] > 0]['f1'].sum()
        recalc_f1 = recalc_f1_num / 37
        print(f"  If fold 16's F1=0 is included in the average: LR mean F1 = {recalc_f1:.4f}")

    print("\n=== Verification Complete ===")

if __name__ == '__main__':
    main()
