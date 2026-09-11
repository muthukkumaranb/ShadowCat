"""Audit pipeline for Logistic Regression Leakage and Fold Composition."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys

# Add parent dir to path to import lstm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lstm.ucs import load_ucs_windows, build_loeo_episode_folds, LOEO_GROUPING_COLUMN

def main():
    # Setup paths
    workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    data_path = workspace_dir / "data" / "ucs" / "ucs_windows.parquet"
    out_dir = workspace_dir / "artifacts" / "lr_leakage_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    lr_repo = Path(r"C:\Users\Vicky\Documents\SIH_2026\SIH2026---UCS-INGESTION-PIPELINE-for-logistic-regression")
    features_path = lr_repo / "artifacts" / "lr_set_a" / "set_a_features.json"
    
    # 1. Load data
    print("Loading UCS data...")
    df = load_ucs_windows(data_path).sort_values("window_start_utc").reset_index(drop=True)
    
    # 2. Load features
    print("Loading features...")
    with open(features_path, 'r') as f:
        meta = json.load(f)
    features = meta["ordered_feature_names"]
    
    # 3. Phase 2: Feature Variance Analysis
    print("Running variance analysis...")
    day_var = []
    ep_var = []
    
    # Group by source_day
    if "source_day" in df.columns:
        day_grouped = df.groupby("source_day")
        day_means = day_grouped[features].mean()
        day_vars = day_grouped[features].var()
        
        between_day_var = day_means.var()
        within_day_var = day_vars.mean()
        
        for feat in features:
            day_var.append({
                "feature": feat,
                "between_day_variance": between_day_var.get(feat, 0),
                "within_day_variance": within_day_var.get(feat, 0)
            })
        pd.DataFrame(day_var).to_csv(out_dir / "day_variance_analysis.csv", index=False)
        
    # Group by episode_id
    if LOEO_GROUPING_COLUMN in df.columns:
        ep_grouped = df.groupby(LOEO_GROUPING_COLUMN)
        ep_means = ep_grouped[features].mean()
        ep_vars = ep_grouped[features].var()
        
        between_ep_var = ep_means.var()
        within_ep_var = ep_vars.mean()
        
        for feat in features:
            ep_var.append({
                "feature": feat,
                "between_episode_variance": between_ep_var.get(feat, 0),
                "within_episode_variance": within_ep_var.get(feat, 0)
            })
        pd.DataFrame(ep_var).to_csv(out_dir / "episode_variance_analysis.csv", index=False)
        
    # 4. Phase 7: Fold Composition Audit
    print("Running fold composition audit...")
    folds = build_loeo_episode_folds(df, episode_col=LOEO_GROUPING_COLUMN)
    
    composition_records = []
    for fold_id, fold in enumerate(folds):
        train_df = df.iloc[fold["train_indices"]]
        test_df = df.iloc[fold["holdout_indices"]]
        
        train_pos = (train_df["label_binary"] == 1).sum()
        train_neg = (train_df["label_binary"] == 0).sum()
        test_pos = (test_df["label_binary"] == 1).sum()
        test_neg = (test_df["label_binary"] == 0).sum()
        
        composition_records.append({
            "fold_id": fold_id,
            "held_out_episode_id": fold["held_out_episode"],
            "attack_type": fold["attack_type"],
            "train_N": len(train_df),
            "test_N": len(test_df),
            "train_positive": train_pos,
            "train_negative": train_neg,
            "test_positive": test_pos,
            "test_negative": test_neg,
            "train_positive_rate": train_pos / len(train_df) if len(train_df) > 0 else 0,
            "test_positive_rate": test_pos / len(test_df) if len(test_df) > 0 else 0,
            "train_benign": train_neg,
            "test_benign": test_neg,
        })
        
    pd.DataFrame(composition_records).to_csv(out_dir / "fold_composition.csv", index=False)

    # 5. Phase 4: Coefficient Audit
    print("Running coefficient audit...")
    coeff_records = []
    
    for fold_id, fold in enumerate(folds):
        train_df = df.iloc[fold["train_indices"]]
        x_train = train_df[features].to_numpy(dtype=float)
        y_train = train_df["label_binary"].to_numpy(dtype=int)
        
        if len(np.unique(y_train)) > 1:
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            
            model = LogisticRegression(solver="liblinear", max_iter=2000, random_state=42)
            model.fit(x_train_scaled, y_train)
            
            for i, feat in enumerate(features):
                coeff_records.append({
                    "fold_id": fold_id,
                    "feature": feat,
                    "coefficient": model.coef_[0, i],
                    "abs_coefficient": abs(model.coef_[0, i])
                })
                
    coeff_df = pd.DataFrame(coeff_records)
    coeff_df.to_csv(out_dir / "coefficient_importance.csv", index=False)
    
    # Aggregate top coefficients
    avg_coeffs = coeff_df.groupby("feature")["abs_coefficient"].mean().reset_index()
    avg_coeffs = avg_coeffs.sort_values("abs_coefficient", ascending=False)
    avg_coeffs.to_csv(out_dir / "top_coefficients.csv", index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
