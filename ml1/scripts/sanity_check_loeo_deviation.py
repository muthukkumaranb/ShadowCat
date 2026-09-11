"""
FINAL Sanity Check: Fold Distribution on LOEO Deviation Score (0.866)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score

workspace_dir = Path("C:/Users/Vicky/Documents/SIH_2026/LSTM-type-model-CICIDS2018-UCS")
dev_pred_path = workspace_dir / "artifacts/lstm/loeo_world_model/deviation_predictions.csv"
manifest_path = workspace_dir / "artifacts/loeo/corrected_37fold_manifest.json"

df = pd.read_csv(dev_pred_path)
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

# Build fold lookup metadata
fold_meta = {}
for f in manifest['folds']:
    fold_meta[f['fold_id']] = {
        'attack_type': f.get('attack_type', f.get('held_out_attack_type', 'unknown')),
        'held_out_episode_id': f.get('held_out_episode_id', f.get('episode_id', 'unknown'))
    }

# 1. Per-fold metrics
per_fold_rows = []
folds = sorted(df['fold_id'].unique())

for fid in folds:
    sub = df[df['fold_id'] == fid]
    n_pos = (sub['future_attack_label'] == 1).sum()
    n_neg = (sub['future_attack_label'] == 0).sum()
    
    if n_pos > 0 and n_neg > 0:
        score = roc_auc_score(sub['future_attack_label'], sub['deviation_score'])
    else:
        score = np.nan
        
    flag_small_inflated = (n_pos < 5) and (score == 1.0)
    
    meta = fold_meta.get(fid, {'attack_type': 'unknown', 'held_out_episode_id': 'unknown'})
    per_fold_rows.append({
        'fold_id': fid,
        'attack_type': meta['attack_type'],
        'episode_id': meta['held_out_episode_id'],
        'n_pos': n_pos,
        'n_neg': n_neg,
        'total': len(sub),
        'roc_auc': score,
        'flag_small_inflated': flag_small_inflated
    })

fold_df = pd.DataFrame(per_fold_rows)

print("=== PER-FOLD SUMMARY ===")
print(fold_df.to_string(index=False))

# 2. Global ROC-AUC
global_roc = roc_auc_score(df['future_attack_label'], df['deviation_score'])
print(f"\nOverall Global ROC-AUC: {global_roc:.4f}")

# 3. Leave-One-Fold-Out Sensitivity
sensitivity_rows = []
max_shift = 0.0
max_shift_fold = None

for fid in folds:
    sub_loo = df[df['fold_id'] != fid]
    loo_roc = roc_auc_score(sub_loo['future_attack_label'], sub_loo['deviation_score'])
    shift = loo_roc - global_roc
    if abs(shift) > abs(max_shift):
        max_shift = shift
        max_shift_fold = fid
    meta = fold_meta.get(fid, {'attack_type': 'unknown', 'held_out_episode_id': 'unknown'})
    sensitivity_rows.append({
        'removed_fold_id': fid,
        'attack_type': meta['attack_type'],
        'loo_roc_auc': loo_roc,
        'shift_from_global': shift,
        'flag_shift_gt_05': abs(shift) > 0.05
    })

sens_df = pd.DataFrame(sensitivity_rows)
print("\n=== LEAVE-ONE-FOLD-OUT SENSITIVITY (Top 10 largest shifts) ===")
print(sens_df.sort_values(by='shift_from_global', key=abs, ascending=False).head(10).to_string(index=False))

# 4. Fold 16 Breakdown
f16 = df[df['fold_id'] == 16].copy()
f16['future_attack_label_prev'] = f16['future_attack_label'].shift(1).fillna(0)
f16['is_onset'] = (f16['future_attack_label'] == 1) & (f16['future_attack_label_prev'] == 0)
f16['is_sustained'] = (f16['future_attack_label'] == 1) & (f16['future_attack_label_prev'] == 1)
f16['is_benign'] = (f16['future_attack_label'] == 0)

f16_benign_mae = f16[f16['is_benign']]['deviation_score'].mean()
f16_onset_mae = f16[f16['is_onset']]['deviation_score'].mean() if f16['is_onset'].sum() > 0 else np.nan
f16_sustained_mae = f16[f16['is_sustained']]['deviation_score'].mean() if f16['is_sustained'].sum() > 0 else np.nan
f16_roc = roc_auc_score(f16['future_attack_label'], f16['deviation_score']) if (f16['future_attack_label']==1).sum() > 0 and (f16['future_attack_label']==0).sum() > 0 else np.nan

print("\n=== FOLD 16 BREAKDOWN ===")
meta16 = fold_meta.get(16, {'attack_type': 'unknown'})
print(f"Fold 16 Attack Type: {meta16['attack_type']}")
print(f"Fold 16 Total Samples: {len(f16)} (Pos: {(f16['future_attack_label']==1).sum()}, Neg: {(f16['future_attack_label']==0).sum()})")
print(f"Fold 16 ROC-AUC: {f16_roc}")
print(f"Fold 16 Benign MAE: {f16_benign_mae:.4f}")
print(f"Fold 16 Onset MAE: {f16_onset_mae}")
print(f"Fold 16 Sustained MAE: {f16_sustained_mae}")

res_summary = {
    'global_roc': float(global_roc),
    'max_shift': float(max_shift),
    'max_shift_fold': int(max_shift_fold),
    'f16_roc': float(f16_roc) if not np.isnan(f16_roc) else None,
    'f16_benign_mae': float(f16_benign_mae),
    'f16_onset_mae': float(f16_onset_mae) if not np.isnan(f16_onset_mae) else None,
    'f16_sustained_mae': float(f16_sustained_mae) if not np.isnan(f16_sustained_mae) else None,
    'small_inflated_count': int(fold_df['flag_small_inflated'].sum()),
    'shift_gt_05_count': int(sens_df['flag_shift_gt_05'].sum())
}

with open(workspace_dir / "artifacts/lstm/loeo_world_model/sanity_check_results.json", "w") as fp:
    json.dump(res_summary, fp, indent=2)

fold_df.to_csv(workspace_dir / "artifacts/lstm/loeo_world_model/per_fold_roc_table.csv", index=False)
sens_df.to_csv(workspace_dir / "artifacts/lstm/loeo_world_model/leave_one_fold_out_sensitivity.csv", index=False)
