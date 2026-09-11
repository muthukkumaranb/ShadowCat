"""
Audit Deviation Score Robustness
"""
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))

def main():
    print("=" * 70)
    print("Audit Deviation Score Robustness")
    print("=" * 70)

    pred_path = workspace_dir / 'artifacts' / 'lstm' / 'loeo_world_model' / 'deviation_predictions.csv'
    if not pred_path.exists():
        print(f"Error: Could not find {pred_path}")
        # let's try the probabilistic one
        pred_path = workspace_dir / 'artifacts' / 'experiments' / 'world_model_20260904' / 'probabilistic' / 'deviation_predictions.csv'
        if not pred_path.exists():
             print(f"Error: Could not find {pred_path} either.")
             return
             
    print(f"Loading predictions from {pred_path}")
    df = pd.read_csv(pred_path)
    
    label_col = 'future_attack_label' if 'future_attack_label' in df.columns else 'label_binary'
    
    print(f"Total windows: {len(df)}")
    print(f"Positive samples ({label_col}=1): {sum(df[label_col] == 1)}")
    print(f"Negative samples ({label_col}=0): {sum(df[label_col] == 0)}")
    
    y_true = df[label_col].values
    y_score = df['deviation_score'].values
    
    overall_roc = roc_auc_score(y_true, y_score)
    overall_pr = average_precision_score(y_true, y_score)
    
    print(f"Overall ROC-AUC: {overall_roc:.4f}")
    print(f"Overall PR-AUC: {overall_pr:.4f}")
    
    print("\n--- Per-Fold Performance ---")
    fold_rocs = []
    
    for fold_id in sorted(df['fold_id'].unique()):
        fold_df = df[df['fold_id'] == fold_id]
        y_true_fold = fold_df[label_col].values
        y_score_fold = fold_df['deviation_score'].values
        
        pos = sum(y_true_fold == 1)
        neg = sum(y_true_fold == 0)
        
        if pos > 0 and neg > 0:
            fold_roc = roc_auc_score(y_true_fold, y_score_fold)
            fold_rocs.append((fold_id, pos, neg, fold_roc))
            print(f"  Fold {fold_id:2d}: ROC-AUC={fold_roc:.4f} (pos={pos}, neg={neg})")
        else:
            print(f"  Fold {fold_id:2d}: ROC-AUC=N/A (pos={pos}, neg={neg})")
            
    if fold_rocs:
        fold_roc_values = [x[3] for x in fold_rocs]
        print(f"\n  Mean Per-Fold ROC-AUC: {np.mean(fold_roc_values):.4f}")
        print(f"  Median Per-Fold ROC-AUC: {np.median(fold_roc_values):.4f}")
        print(f"  Range: {np.min(fold_roc_values):.4f} - {np.max(fold_roc_values):.4f}")
        
    print("\n--- Checking for Extreme Fold Influence ---")
    # if we exclude the best fold, what happens to overall ROC?
    if fold_rocs:
        best_fold = max(fold_rocs, key=lambda x: x[3])[0]
        df_no_best = df[df['fold_id'] != best_fold]
        roc_no_best = roc_auc_score(df_no_best[label_col].values, df_no_best['deviation_score'].values)
        print(f"  Excluding best fold ({best_fold}): Overall ROC-AUC = {roc_no_best:.4f}")

if __name__ == '__main__':
    main()
