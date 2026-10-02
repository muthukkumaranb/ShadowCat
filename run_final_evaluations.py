import pandas as pd
import sys
import os
import yaml
import json
import numpy as np
import subprocess
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, os.path.abspath('d:\\sih2026'))
from ml1.examples.train_probabilistic import run


def main():
    print("=== Step 1: Combining datasets ===")
    df_2018 = pd.read_parquet(r'd:\sih2026\data-engineering\data\ucs\ucs_windows.parquet')
    df_2017 = pd.read_parquet(r'd:\sih2026\data-engineering\data\ucs_cic2017\ucs_windows.parquet')

    # We do NOT overwrite window_start_utc!
    # df_combined = pd.concat([df_2018, df_2017], ignore_index=True)
    
    # Wait, the Gate0 LOEO requires dataset merging. But cross-dataset does not.
    # To combine, we will just concatenate and sort.
    # But since 2017 is in 2017 and 2018 is in 2018, there is no timestamp overlap.
    df_combined = pd.concat([df_2018, df_2017], ignore_index=True)
    df_combined['window_start_utc'] = pd.to_datetime(df_combined['window_start_utc'], utc=True)
    df_combined = df_combined.sort_values('window_start_utc').reset_index(drop=True)
    
    # Gate0 expects ucs_windows_path
    combined_path = Path(r'd:\sih2026\data-engineering\data\ucs_combined.parquet')
    
    # Fill NaN for missing packet features ONLY for numeric types
    numeric_cols = df_combined.select_dtypes(include=[np.number]).columns
    df_combined[numeric_cols] = df_combined[numeric_cols].fillna(0.0)
    
    df_combined.to_parquet(combined_path)
    
    sys.path.insert(0, os.path.abspath(r'd:\sih2026\data-engineering'))
    from src.gate0_leakage_test import run_gate0_leakage_test
    
    print("\n=== Step 2: Running Gate 0 Leakage Checks ===")
    report_path = r'd:\sih2026\gate0_combined_leakage_report.md'
    run_gate0_leakage_test(ucs_windows_path=str(combined_path), output_report_path=report_path)
    
    print("\n=== Step 3: Cross-Dataset Generalization Test ===")
    # Train on 2018, test on 2017.
    # Load 2018 train/test logic correctly. 
    # To get 0.738 we need to evaluate using LOEO.
    # Let's run run_lr_frozen.py for the 2018 baseline?
    # No, we will just print the in-distribution chronological split result for transparency, 
    # and compare zero-shot on 2017. Or use the 37-fold LOEO to train an ensemble for zero-shot.
    # Let's train on entire 2018 dataset to get the zero-shot model, and evaluate on 2017.
    
    features_path = r'd:\sih2026\ml1\artifacts\lr_set_a\set_a_features.json'
    with open(features_path) as f:
        features = json.load(f)['ordered_feature_names']
        
    # Make sure we don't scale again if they are already scaled!
    # But wait, run_lr_frozen DOES scale again! So we will do it here to match it.
    
    X_2018 = df_2018[features].values
    y_2018 = df_2018['label_binary'].values
    
    # Impute missing features in 2017 with 0 (since they weren't collected)
    for feat in features:
        if feat not in df_2017.columns:
            df_2017[feat] = 0.0
            
    df_2017[features] = df_2017[features].fillna(0.0)
    X_2017 = df_2017[features].values
    y_2017 = df_2017['label_binary'].values
    
    scaler = StandardScaler()
    X_2018_scaled = scaler.fit_transform(X_2018)
    X_2017_scaled = scaler.transform(X_2017)
    
    clf = LogisticRegression(solver="liblinear", max_iter=2000, random_state=42)
    clf.fit(X_2018_scaled, y_2018)
    
    y_pred_2017 = clf.predict(X_2017_scaled)
    zero_shot_f1 = f1_score(y_2017, y_pred_2017)
    print(f"Zero-Shot F1 on 2017 (Trained on full 2018): {zero_shot_f1:.4f}")
    
    print("\n=== Step 4: Training World Model ===")
    output_dir = Path(r'd:\sih2026\ml1\artifacts\lstm\cic2017_exploratory')
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / 'config.yaml'
    config_data = {
        'model': {
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.2
        },
        'training': {
            'epochs': 50,  # Full training to completion
            'batch_size': 64,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'early_stopping': {
                'patience': 10,
                'min_delta': 0.001
            }
        },
        'world_model': {
            'seed': 42,
            'rollout': {'horizons': [1, 2, 3, 4, 5]},
            'intervals': {'levels': [0.80]}
        }
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
        
    print('Training world model on combined dataset...')
    run(combined_path, output_dir, config_path)
    
    print("\n=== Step 5: Horizon Analysis ===")
    ckpt_path = output_dir / "gaussian_next_state_best.pt"
    eval_script = r"d:\sih2026\evaluation\horizon_analysis\horizon_analysis.py"
    result = subprocess.run(
        ["python", eval_script, "--checkpoint", str(ckpt_path)],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    with open(output_dir / "horizon_eval.txt", "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\nERRORS:\n" + result.stderr)
            
if __name__ == '__main__':
    main()
