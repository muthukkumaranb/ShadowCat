import pandas as pd
import numpy as np
import json
from pathlib import Path

def main():
    lstm_csv = Path('artifacts/lstm/lstm_detection_loeo_folds.csv')
    df = pd.read_csv(lstm_csv)
    
    print("--- AUTHORITATIVE LSTM RUN IDENTIFIED ---")
    print("Path:", lstm_csv)
    print("Fold count:", len(df))
    
    # Overall metrics
    print("\n--- AGGREGATE METRICS ---")
    print(f"Mean F1: {df['f1'].mean():.4f}")
    print(f"Mean Precision: {df['precision'].mean():.4f}")
    print(f"Mean Recall: {df['recall'].mean():.4f}")
    print(f"Mean FPR: {df['fpr'].mean():.4f}")
    print(f"Mean ROC-AUC: {df['roc_auc'].mean():.4f}")
    print(f"Mean PR-AUC: {df['pr_auc'].mean():.4f}")
    
    # Check threshold vs representation
    # If ROC-AUC is high but F1 is low, threshold is wrong
    low_f1 = df[df['f1'] < 0.1]
    if len(low_f1) > 0:
        print("\n--- FOLDS WITH F1 ≈ 0.0 ---")
        print(low_f1[['fold_id', 'f1', 'roc_auc', 'pr_auc']])
        mean_roc = low_f1['roc_auc'].mean()
        print(f"\nMean ROC-AUC for these failing folds: {mean_roc:.4f}")
        if mean_roc > 0.7:
            print("DIAGNOSIS: Threshold/calibration problem. The model has reasonable separation (ROC-AUC) but fails at the hardcoded threshold.")
        else:
            print("DIAGNOSIS: Genuine representation/separation problem.")
            
if __name__ == '__main__':
    main()
