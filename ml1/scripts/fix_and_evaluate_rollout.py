"""
Fix and Evaluate Rollout (Path 1b)

Adds a StandardScaler AFTER the PCA step, fitted strictly on the TRAIN split's PCA output.
The LSTM targets and inputs are scaled. Predictions are inverse-transformed
back to raw PCA space to compute MAE identically as before.
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
from sklearn.preprocessing import StandardScaler

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
    
    out_dir = workspace_dir / 'artifacts' / 'lstm'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / 'rollout_error_vs_k_fixed.png'
    
    data_path = workspace_dir / 'data' / 'ucs' / 'ucs_windows.parquet'
    model_path = workspace_dir / 'artifacts' / 'lstm' / 'ml2_export_chronological' / 'models' / 'chunk_4_gaussian_best.pt'
    
    print("=== Step 0 Answer ===")
    print("Yes, standardization was applied before PCA fitting (ucs_windows.parquet features have mean ~0, std ~1).")
    print("However, PC0's output scale is massive (e.g., -2574) because PCA linearly combines 406 features.")
    print("Therefore, Path 1b applies: adding a standardization step AFTER PCA.")
    print("======================\n")

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

    # Apply PCA directly (frozen config)
    pca = UCSPCA(n_components=32, random_state=42)
    pca.fit(active_train_val.loc[active_train_val['split'] == 'train'], features)
    
    pca_features = [f"pca_{i}" for i in range(32)]
    full_transformed = pca.transform(chunk_frame)
    full_transformed_np = full_transformed[pca_features].to_numpy(dtype=np.float32)

    # ---------------------------------------------------------
    # Path 1b Fix: StandardScaler AFTER PCA
    # ---------------------------------------------------------
    pca_scaler = StandardScaler()
    train_pca_data = full_transformed.loc[chunk_frame['split'] == 'train', pca_features].to_numpy(dtype=np.float32)
    pca_scaler.fit(train_pca_data)
    
    # Scale all PCA features for the LSTM
    scaled_pca_np = pca_scaler.transform(full_transformed_np).astype(np.float32)

    # We will build unscaled targets for Persistence/Lagged-LR (which operate in raw PCA space)
    transformed_frame_unscaled = chunk_frame.drop(columns=features)
    for i, p_col in enumerate(pca_features):
        transformed_frame_unscaled[p_col] = full_transformed_np[:, i]
        
    K = 3
    # Unscaled targets and histories for Baseline methods
    histories_unscaled, targets_unscaled, times = rollout_targets(transformed_frame_unscaled, features=pca_features, config=config, k=K, split='test')
    
    # Train sequences for Lagged-LR (unscaled)
    seqs_unscaled = build_next_state_sequences(transformed_frame_unscaled, features=pca_features, config=config)
    train_seq_unscaled = seqs_unscaled['train']
    
    lr_model = LinearRegression()
    lr_model.fit(train_seq_unscaled.X[:, -1, :], train_seq_unscaled.y)
    
    # ---------------------------------------------------------
    # Baseline Predictions
    # ---------------------------------------------------------
    pers_pred = np.empty((histories_unscaled.shape[0], K, histories_unscaled.shape[2]), dtype=np.float32)
    last_state = persistence_baseline(histories_unscaled)
    for k_idx in range(K):
        pers_pred[:, k_idx, :] = last_state
        
    lr_pred = np.empty((histories_unscaled.shape[0], K, histories_unscaled.shape[2]), dtype=np.float32)
    curr_state = histories_unscaled[:, -1, :].copy()
    for k_idx in range(K):
        pred_state = lr_model.predict(curr_state)
        lr_pred[:, k_idx, :] = pred_state
        curr_state = pred_state

    # ---------------------------------------------------------
    # Train LSTM on SCALED data
    # (Since this is a fix script, we need to train the LSTM on the scaled inputs to see if it fixes it)
    # The prompt says: "Do NOT: Retune hyperparameters ... Change the frozen config". 
    # But it says: "Add a standardization step AFTER PCA ... applied before the LSTM sees the data"
    # To evaluate the fixed model, we must train it.
    # ---------------------------------------------------------
    transformed_frame_scaled = chunk_frame.drop(columns=features)
    for i, p_col in enumerate(pca_features):
        transformed_frame_scaled[p_col] = scaled_pca_np[:, i]
        
    seqs_scaled = build_next_state_sequences(transformed_frame_scaled, features=pca_features, config=config)
    train_seq_scaled = seqs_scaled['train']
    val_seq_scaled = seqs_scaled['val']
    
    # We must quickly train the LSTM on this scaled data using the exact frozen config
    print("Training Fixed LSTM on SCALED PCA features...")
    fixed_model = LSTMGaussianWorldModel(input_size=32, hidden_size=64, state_dim=32, num_layers=1, dropout=0.2)
    fixed_checkpoint = out_dir / 'chunk_4_gaussian_fixed.pt'
    
    from lstm.probabilistic import train_gaussian
    train_gaussian(
        fixed_model, train_seq_scaled.X, train_seq_scaled.y, val_seq_scaled.X, val_seq_scaled.y,
        epochs=100, batch_size=64, learning_rate=0.001, weight_decay=0.0001,
        patience=5, min_delta=0.001, seed=42, device=str(device),
        checkpoint_path=fixed_checkpoint
    )
    
    fixed_model.load_state_dict(torch.load(fixed_checkpoint, map_location=device, weights_only=True)['model_state_dict'])
    fixed_model.to(device)
    fixed_model.eval()
    
    # ---------------------------------------------------------
    # LSTM Prediction & Inverse Transform
    # ---------------------------------------------------------
    histories_scaled, _, _ = rollout_targets(transformed_frame_scaled, features=pca_features, config=config, k=K, split='test')
    
    lstm_pred_scaled = lstm_rollout(histories_scaled, fixed_model, device, k=K)
    
    # Inverse transform LSTM predictions back to unscaled PCA space
    lstm_pred_unscaled = np.empty_like(lstm_pred_scaled)
    for k_idx in range(K):
        lstm_pred_unscaled[:, k_idx, :] = pca_scaler.inverse_transform(lstm_pred_scaled[:, k_idx, :])

    # ---------------------------------------------------------
    # Deliverables
    # ---------------------------------------------------------
    print("\n--- Before / After MAE Comparison ---")
    
    def get_errors(pred, targ, k_idx):
        return np.mean(np.abs(pred[:, k_idx, :] - targ[:, k_idx, :]), axis=1)
        
    # We have old numbers hardcoded from the prompt
    old_pers = [ (6.39, 1.58), (5.04, 1.41), (6.25, 1.63) ]
    old_lr = [ (24.64, 15.31), (24.11, 15.84), (30.49, 22.41) ]
    old_lstm = [ (59.89, 57.44), (59.62, 57.54), (59.79, 57.55) ]
    
    results = []
    
    for k_idx in range(K):
        err_pers = get_errors(pers_pred, targets_unscaled, k_idx)
        err_lr = get_errors(lr_pred, targets_unscaled, k_idx)
        err_lstm = get_errors(lstm_pred_unscaled, targets_unscaled, k_idx)
        
        print(f"\nHorizon K={k_idx+1}:")
        print(f"  Persistence (OLD): Mean = {old_pers[k_idx][0]:.2f}, Median = {old_pers[k_idx][1]:.2f}")
        print(f"  Persistence (NEW): Mean = {np.mean(err_pers):.2f}, Median = {np.median(err_pers):.2f}")
        print(f"  Lagged-LR   (OLD): Mean = {old_lr[k_idx][0]:.2f}, Median = {old_lr[k_idx][1]:.2f}")
        print(f"  Lagged-LR   (NEW): Mean = {np.mean(err_lr):.2f}, Median = {np.median(err_lr):.2f}")
        print(f"  LSTM        (OLD): Mean = {old_lstm[k_idx][0]:.2f}, Median = {old_lstm[k_idx][1]:.2f}")
        print(f"  LSTM        (NEW): Mean = {np.mean(err_lstm):.2f}, Median = {np.median(err_lstm):.2f}")
        
        results.append({
            'K': k_idx + 1,
            'Persistence_MAE': np.mean(err_pers),
            'LaggedLR_MAE': np.mean(err_lr),
            'LSTM_MAE': np.mean(err_lstm)
        })

    print("\n--- Before / After Prediction Trajectory (K=1, PC0) ---")
    step4_rows = []
    for i in range(10):
        t_target = targets_unscaled[i, 0, :]
        p_pers = pers_pred[i, 0, :]
        p_lr = lr_pred[i, 0, :]
        p_lstm = lstm_pred_unscaled[i, 0, :]
        
        err_pers = np.mean(np.abs(p_pers - t_target))
        err_lr = np.mean(np.abs(p_lr - t_target))
        err_lstm = np.mean(np.abs(p_lstm - t_target))
        
        step4_rows.append({
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
    df_step4 = pd.DataFrame(step4_rows)
    print(df_step4.to_string(index=False))

    # Plot
    res_df = pd.DataFrame(results)
    plt.figure(figsize=(8, 6))
    k_vals = res_df['K'].values
    plt.plot(k_vals, res_df['Persistence_MAE'].values, 's--', label='Persistence Baseline', color='gray')
    plt.plot(k_vals, res_df['LaggedLR_MAE'].values, 'o-', label='Lagged-LR (PCA)', color='orange')
    plt.plot(k_vals, res_df['LSTM_MAE'].values, '^-', label='LSTM World Model (Fixed)', color='blue')
    
    plt.title("Chronological Rollout Error vs Horizon (K) - FIXED")
    plt.xlabel("Horizon Step (K)")
    plt.ylabel("Mean Absolute Error (PCA space)")
    plt.xticks(k_vals)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"\nUpdated Rollout Error Curve saved to {out_png}")
    
    print("\nConfirmation: seed=42, hidden_size=64, state_dim=32, num_layers=1, dropout=0.2 were UNCHANGED.")
    print("Confirmation: PCA scaler was fit strictly on the 'train' split of the chronological chunks.")

if __name__ == '__main__':
    main()
