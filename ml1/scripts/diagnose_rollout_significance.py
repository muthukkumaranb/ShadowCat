"""
Significance and Bias Check for LSTM vs Persistence
"""
import pandas as pd
import numpy as np
import sys
import os
import torch
import json
from pathlib import Path
from scipy.stats import wilcoxon

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))

from lstm.model import LSTMGaussianWorldModel
from lstm.ucs import UCSConfig, validate_ucs_windows, UCSPCA, rollout_targets, persistence_baseline
from lstm.utils import get_device, set_seed
from sklearn.preprocessing import StandardScaler

def lstm_rollout(history, model, device, k=3):
    history = np.asarray(history, dtype=np.float32)
    current = history.copy()
    rollout = np.empty((history.shape[0], k, history.shape[2]), dtype=np.float32)
    
    for step in range(k):
        with torch.no_grad():
            mean, _ = model(torch.as_tensor(current, dtype=torch.float32).to(device))
        predicted = mean.cpu().numpy()
        rollout[:, step, :] = predicted
        
        if step < k - 1:
            current = np.concatenate([current[:, 1:, :], predicted[:, None, :]], axis=1)
            
    return rollout

def main():
    set_seed(42)
    device = get_device()
    
    data_path = workspace_dir / 'data' / 'ucs' / 'ucs_windows.parquet'
    # We use the FIXED model trained in the previous step
    model_path = workspace_dir / 'artifacts' / 'lstm' / 'chunk_4_gaussian_fixed.pt'
    
    if not model_path.exists():
        print(f"Fixed model not found at {model_path}.")
        sys.exit(1)
        
    print("=== Significance & Bias Check (LSTM vs Persistence) ===")
    
    windows = pd.read_parquet(data_path).sort_values('window_start_utc').reset_index(drop=True)
    config = UCSConfig()
    features = validate_ucs_windows(windows, config=config)
    
    total_windows = len(windows)
    ml2_train_end = int(total_windows * 0.70)
    
    chunk_frame = windows.copy()
    chunk_frame['split'] = 'none'
    val_start = int(ml2_train_end * 0.85)
    
    chunk_frame.loc[0:val_start-1, 'split'] = 'train'
    chunk_frame.loc[val_start:ml2_train_end-1, 'split'] = 'val'
    chunk_frame.loc[ml2_train_end:total_windows-1, 'split'] = 'test'

    active_mask = chunk_frame['split'].isin(['train', 'val'])
    active_train_val = chunk_frame[active_mask].copy()

    # Apply PCA directly
    pca = UCSPCA(n_components=32, random_state=42)
    pca.fit(active_train_val.loc[active_train_val['split'] == 'train'], features)
    pca_features = [f"pca_{i}" for i in range(32)]
    full_transformed = pca.transform(chunk_frame)
    full_transformed_np = full_transformed[pca_features].to_numpy(dtype=np.float32)

    # Post-PCA Scaler
    pca_scaler = StandardScaler()
    train_pca_data = full_transformed.loc[chunk_frame['split'] == 'train', pca_features].to_numpy(dtype=np.float32)
    pca_scaler.fit(train_pca_data)
    scaled_pca_np = pca_scaler.transform(full_transformed_np).astype(np.float32)

    # Unscaled representations
    transformed_frame_unscaled = chunk_frame.drop(columns=features)
    for i, p_col in enumerate(pca_features):
        transformed_frame_unscaled[p_col] = full_transformed_np[:, i]
        
    K = 3
    histories_unscaled, targets_unscaled, times = rollout_targets(transformed_frame_unscaled, features=pca_features, config=config, k=K, split='test')
    
    # Baseline Persistence
    pers_pred = np.empty((histories_unscaled.shape[0], K, histories_unscaled.shape[2]), dtype=np.float32)
    last_state = persistence_baseline(histories_unscaled)
    for k_idx in range(K):
        pers_pred[:, k_idx, :] = last_state

    # LSTM
    transformed_frame_scaled = chunk_frame.drop(columns=features)
    for i, p_col in enumerate(pca_features):
        transformed_frame_scaled[p_col] = scaled_pca_np[:, i]
        
    fixed_model = LSTMGaussianWorldModel(input_size=32, hidden_size=64, state_dim=32, num_layers=1, dropout=0.2)
    fixed_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)['model_state_dict'])
    fixed_model.to(device)
    fixed_model.eval()
    
    histories_scaled, _, _ = rollout_targets(transformed_frame_scaled, features=pca_features, config=config, k=K, split='test')
    lstm_pred_scaled = lstm_rollout(histories_scaled, fixed_model, device, k=K)
    
    lstm_pred_unscaled = np.empty_like(lstm_pred_scaled)
    for k_idx in range(K):
        lstm_pred_unscaled[:, k_idx, :] = pca_scaler.inverse_transform(lstm_pred_scaled[:, k_idx, :])

    # ---------------------------------------------------------
    # STEP 1: Paired Significance Test
    # ---------------------------------------------------------
    print("\n--- STEP 1: Paired Significance Test (Wilcoxon Signed-Rank) ---")
    
    def get_errors_per_window(pred, targ, k_idx):
        return np.mean(np.abs(pred[:, k_idx, :] - targ[:, k_idx, :]), axis=1)

    for k_idx in range(K):
        lstm_err = get_errors_per_window(lstm_pred_unscaled, targets_unscaled, k_idx)
        pers_err = get_errors_per_window(pers_pred, targets_unscaled, k_idx)
        
        diff = lstm_err - pers_err
        diff_mean = np.mean(diff)
        diff_median = np.median(diff)
        diff_std = np.std(diff)
        
        # Wilcoxon signed-rank test
        stat, p_val = wilcoxon(lstm_err, pers_err)
        # Cohen's d approximation for effect size
        cohens_d = diff_mean / diff_std if diff_std > 0 else 0
        
        print(f"\nHorizon K={k_idx+1}:")
        print(f"  Diff (LSTM - Pers) : Mean = {diff_mean:.4f}, Median = {diff_median:.4f}, Std = {diff_std:.4f}")
        print(f"  Wilcoxon Statistic : {stat}")
        print(f"  Wilcoxon p-value   : {p_val:.2e}")
        print(f"  Effect Size (Cohen's d): {cohens_d:.4f}")
        
        if p_val < 0.05:
            print("  Conclusion         : The LSTM's higher error is STATISTICALLY SIGNIFICANT (p < 0.05).")
        else:
            print("  Conclusion         : The difference is NOT statistically significant; could plausibly be noise.")

    # ---------------------------------------------------------
    # STEP 2: Systematic Bias Check (PC0)
    # ---------------------------------------------------------
    print("\n--- STEP 2: Systematic Bias Check (PC0) ---")
    
    for k_idx in range(K):
        lstm_pc0_pred = lstm_pred_unscaled[:, k_idx, 0]
        targ_pc0 = targets_unscaled[:, k_idx, 0]
        
        signed_err_pc0 = lstm_pc0_pred - targ_pc0
        bias = np.mean(signed_err_pc0)
        
        print(f"\nHorizon K={k_idx+1}:")
        print(f"  Mean Signed Error (PC0): {bias:.4f}")
        
        if abs(bias) > 50:
            print(f"  Observation: Strong systematic bias detected ({bias:.2f}).")
        
            # Try applying correction on test set just to see upper bound of improvement
            corrected_pc0_pred = lstm_pc0_pred - bias
            
            # Recalculate MAE for PC0 only
            old_pc0_mae = np.mean(np.abs(lstm_pc0_pred - targ_pc0))
            new_pc0_mae = np.mean(np.abs(corrected_pc0_pred - targ_pc0))
            
            # Recalculate Overall MAE
            lstm_pred_corrected = lstm_pred_unscaled.copy()
            lstm_pred_corrected[:, k_idx, 0] = corrected_pc0_pred
            
            old_overall_mae = np.mean(np.mean(np.abs(lstm_pred_unscaled[:, k_idx, :] - targets_unscaled[:, k_idx, :]), axis=1))
            new_overall_mae = np.mean(np.mean(np.abs(lstm_pred_corrected[:, k_idx, :] - targets_unscaled[:, k_idx, :]), axis=1))
            pers_overall_mae = np.mean(np.mean(np.abs(pers_pred[:, k_idx, :] - targets_unscaled[:, k_idx, :]), axis=1))
            
            print(f"  If bias correction applied (post-hoc):")
            print(f"    PC0 MAE would improve: {old_pc0_mae:.4f} -> {new_pc0_mae:.4f}")
            print(f"    Overall MAE would improve: {old_overall_mae:.4f} -> {new_overall_mae:.4f}")
            print(f"    Persistence MAE to beat: {pers_overall_mae:.4f}")

    # ---------------------------------------------------------
    # STEP 3: Autocorrelation Context
    # ---------------------------------------------------------
    print("\n--- STEP 3: Autocorrelation Context ---")
    # Test set targets for PC0 at K=1
    pc0_series = targets_unscaled[:, 0, 0]
    
    if len(pc0_series) > 1:
        # Lag-1 autocorrelation
        acorr = np.corrcoef(pc0_series[:-1], pc0_series[1:])[0, 1]
        print(f"  Lag-1 Autocorrelation of Test PC0: {acorr:.6f}")
        if acorr > 0.99:
            print("  Context: Extreme autocorrelation (>0.99). Persistence is inherently a very strong baseline.")
    
if __name__ == '__main__':
    main()
