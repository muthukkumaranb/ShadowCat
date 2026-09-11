"""
Diagnose LSTM Rollout Failure vs. Persistence Baseline
Executes 6 steps defined by the user to diagnose why Persistence beats LSTM by ~10x on Rollout.
"""
import pandas as pd
import numpy as np
import sys
import os
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))

from lstm.model import LSTMGaussianWorldModel
from lstm.ucs import UCSConfig, build_next_state_sequences, validate_ucs_windows, UCSPCA, rollout_targets, persistence_baseline
from lstm.utils import get_device, set_seed

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
    
    out_dir = workspace_dir / 'artifacts' / 'lstm_diagnostics'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_report = out_dir / 'rollout_diagnosis.txt'
    
    data_path = workspace_dir / 'data' / 'ucs' / 'ucs_windows.parquet'
    model_path = workspace_dir / 'artifacts' / 'lstm' / 'ml2_export_chronological' / 'models' / 'chunk_4_gaussian_best.pt'
    nll_outliers_path = out_dir / 'nll_outliers.csv'
    
    print("=== Diagnosing LSTM Rollout Failure ===")
    
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
    
    explained_variance = pca.pca.explained_variance_ratio_

    transformed_frame = chunk_frame.drop(columns=features)
    for i, p_col in enumerate(pca_features):
        transformed_frame[p_col] = full_transformed_np[:, i]
        
    K = 3
    histories, targets, times = rollout_targets(transformed_frame, features=pca_features, config=config, k=K, split='test')
    
    seqs = build_next_state_sequences(transformed_frame, features=pca_features, config=config)
    train_seq = seqs['train']
    
    lr_model = LinearRegression()
    lr_model.fit(train_seq.X[:, -1, :], train_seq.y)
    
    model = LSTMGaussianWorldModel(input_size=32, hidden_size=64, state_dim=32, num_layers=1, dropout=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Generate Predictions
    pers_pred = np.empty((histories.shape[0], K, histories.shape[2]), dtype=np.float32)
    last_state = persistence_baseline(histories)
    for k_idx in range(K):
        pers_pred[:, k_idx, :] = last_state
        
    lr_pred = np.empty((histories.shape[0], K, histories.shape[2]), dtype=np.float32)
    curr_state = histories[:, -1, :].copy()
    for k_idx in range(K):
        pred_state = lr_model.predict(curr_state)
        lr_pred[:, k_idx, :] = pred_state
        curr_state = pred_state
        
    lstm_pred = lstm_rollout(histories, model, device, k=K)
    
    # ---------------------------------------------------------
    # STEP 1: Harness/Alignment Bug
    # ---------------------------------------------------------
    print("\n--- STEP 1: Harness / Alignment ---")
    print("Confirming identically extracted test windows and lengths:")
    print(f"Histories shape: {histories.shape}")
    print(f"Targets shape  : {targets.shape}")
    print(f"LSTM pred shape: {lstm_pred.shape}")
    
    # We will average the absolute error across the 32 PCA components to get a single MAE scalar per sequence/step
    # Table for first 10 test windows, K=1
    step1_rows = []
    for i in range(10):
        t_target = targets[i, 0, :]
        p_pers = pers_pred[i, 0, :]
        p_lr = lr_pred[i, 0, :]
        p_lstm = lstm_pred[i, 0, :]
        
        # average MAE across components for simplicity in display
        err_pers = np.mean(np.abs(p_pers - t_target))
        err_lr = np.mean(np.abs(p_lr - t_target))
        err_lstm = np.mean(np.abs(p_lstm - t_target))
        
        step1_rows.append({
            'Index': i,
            'Time': str(times[i][0]),
            'Target_PC0': t_target[0],
            'Pers_PC0': p_pers[0],
            'LR_PC0': p_lr[0],
            'LSTM_PC0': p_lstm[0],
            'MAE_Pers': err_pers,
            'MAE_LR': err_lr,
            'MAE_LSTM': err_lstm
        })
    df_step1 = pd.DataFrame(step1_rows)
    print("First 10 Test Windows (K=1, showing PC0 and per-sequence MAE):")
    print(df_step1.to_string(index=False))

    # ---------------------------------------------------------
    # STEP 2: Outlier-driven MAE? Mean vs Median
    # ---------------------------------------------------------
    print("\n--- STEP 2: Mean vs Median Error Distribution ---")
    
    def get_errors(pred, targ, k_idx):
        return np.mean(np.abs(pred[:, k_idx, :] - targ[:, k_idx, :]), axis=1)
        
    for k_idx in range(K):
        err_pers = get_errors(pers_pred, targets, k_idx)
        err_lr = get_errors(lr_pred, targets, k_idx)
        err_lstm = get_errors(lstm_pred, targets, k_idx)
        
        print(f"\nHorizon K={k_idx+1}:")
        print(f"  Persistence : Mean = {np.mean(err_pers):.4f}, Median = {np.median(err_pers):.4f}")
        print(f"  Lagged-LR   : Mean = {np.mean(err_lr):.4f}, Median = {np.median(err_lr):.4f}")
        print(f"  LSTM        : Mean = {np.mean(err_lstm):.4f}, Median = {np.median(err_lstm):.4f}")
        
        if k_idx == 0:
            lstm_k1_err = err_lstm
            print("  LSTM K=1 Percentiles:")
            for p in [5, 25, 50, 75, 95, 99]:
                print(f"    p{p:02d}: {np.percentile(lstm_k1_err, p):.4f}")
            print(f"    max: {np.max(lstm_k1_err):.4f}")

    # Top outlier windows for LSTM K=1
    outlier_idx = np.argsort(lstm_k1_err)[-10:][::-1]
    print("\nTop 10 LSTM Rollout Error Outliers (K=1):")
    for idx in outlier_idx:
        print(f"  Index: {idx}, Time: {times[idx][0]}, LSTM MAE: {lstm_k1_err[idx]:.4f}")

    # ---------------------------------------------------------
    # STEP 3: Per-PCA-Component Error Breakdown
    # ---------------------------------------------------------
    print("\n--- STEP 3: Per-PCA-Component Breakdown (K=1) ---")
    comp_err = np.mean(np.abs(lstm_pred[:, 0, :] - targets[:, 0, :]), axis=0)
    
    comp_rows = []
    for c in range(32):
        comp_rows.append({
            'Component': f"PC{c}",
            'ExplainedVar': explained_variance[c],
            'LSTM_MAE': comp_err[c]
        })
    df_comp = pd.DataFrame(comp_rows).sort_values('LSTM_MAE', ascending=False)
    print("Top 10 Components by LSTM MAE:")
    print(df_comp.head(10).to_string(index=False))

    # ---------------------------------------------------------
    # STEP 4: Predict vs Actual Trajectory
    # ---------------------------------------------------------
    print("\n--- STEP 4: Trajectory Tracking (K=1) ---")
    # Sample 5 evenly spaced indices
    sample_indices = np.linspace(0, len(targets)-1, 5, dtype=int)
    
    plt.figure(figsize=(12, 10))
    for i, idx in enumerate(sample_indices):
        plt.subplot(5, 1, i+1)
        # We will plot PC0 which usually has the highest variance
        history_vals = histories[idx, :, 0]
        targ = targets[idx, 0, 0]
        p_lstm = lstm_pred[idx, 0, 0]
        p_pers = pers_pred[idx, 0, 0]
        
        # Plot history
        x_hist = range(-config.lookback_windows, 0)
        plt.plot(x_hist, history_vals, 'k-', label='History')
        plt.plot(0, targ, 'r*', markersize=10, label='Actual')
        plt.plot(0, p_lstm, 'bo', label='LSTM')
        plt.plot(0, p_pers, 'go', label='Pers')
        
        plt.title(f"Sequence {idx} (PC0)")
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    traj_png = out_dir / 'trajectory_tracking.png'
    plt.savefig(traj_png)
    print(f"Saved trajectory tracking plot to {traj_png}")
    
    # ---------------------------------------------------------
    # STEP 5: Connection to NLL sigma-collapse
    # ---------------------------------------------------------
    print("\n--- STEP 5: Connection to NLL Outliers ---")
    if nll_outliers_path.exists():
        df_nll = pd.read_csv(nll_outliers_path)
        nll_outlier_times = set(df_nll['window_start_utc'].values)
        
        overlap = []
        for idx in outlier_idx:
            ts_str = str(times[idx][0])
            if ts_str in nll_outlier_times:
                overlap.append(ts_str)
                
        print(f"Overlap between top 10 LSTM Rollout Outliers and NLL Outliers: {len(overlap)}")
        if overlap:
            for ts in overlap:
                print(f"  Overlap: {ts}")
    else:
        print("nll_outliers.csv not found.")
        
    print("\n=== Diagnosis Complete ===")

if __name__ == '__main__':
    main()
