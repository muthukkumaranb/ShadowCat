"""
Rollout Evaluation & Error Curve

Evaluates K=1, 2, 3 step rollout for the LSTM world model, Persistence baseline,
and Lagged-LR baseline. The lagged-LR baseline must operate on the PCA-reduced representation.
Generates an explicit PNG plot and raw CSV.
"""
import pandas as pd
import numpy as np
import sys
import os
import torch
from pathlib import Path
import matplotlib.pyplot as plt

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))

from lstm.model import LSTMGaussianWorldModel
from lstm.ucs import UCSConfig, build_next_state_sequences, validate_ucs_windows, UCSPCA, rollout_targets, persistence_baseline, fit_pca_lagged_linear_baseline, pca_rollout_predictions
from lstm.utils import get_device, set_seed
from sklearn.linear_model import LinearRegression

def lstm_rollout(history, model, device, k=3):
    # history shape: (samples, lookback, features)
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
    out_csv = out_dir / 'rollout_errors.csv'
    out_png = out_dir / 'rollout_error_vs_k.png'
    
    data_path = workspace_dir / 'data' / 'ucs' / 'ucs_windows.parquet'
    model_path = workspace_dir / 'artifacts' / 'lstm' / 'ml2_export_chronological' / 'models' / 'chunk_4_gaussian_best.pt'
    
    if not model_path.exists():
        print(f"Model not found at {model_path}. Please run chronological export first.")
        sys.exit(1)
        
    print("=== Rollout Evaluation (K=1, 2, 3) ===")
    
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

    # We need contiguous rollout targets.
    # The rollout_targets function in ucs.py uses the 'split' column.
    # But it takes the whole dataframe. We need to pass the PCA transformed one.
    transformed_frame = chunk_frame.drop(columns=features)
    for i, p_col in enumerate(pca_features):
        transformed_frame[p_col] = full_transformed_np[:, i]
        
    # Get histories and targets for TEST split
    # rollout_targets returns: (samples, lookback, features), (samples, k, features), ...
    K = 3
    histories, targets, times = rollout_targets(transformed_frame, features=pca_features, config=config, k=K, split='test')
    
    print(f"Test rollout samples: {len(histories)}")
    
    # Get training sequences for lagged LR
    seqs = build_next_state_sequences(transformed_frame, features=pca_features, config=config)
    train_seq = seqs['train']
    
    # Train Lagged-LR on PCA
    print("Training Lagged-LR baseline on PCA...")
    lr_model = LinearRegression()
    lr_model.fit(train_seq.X[:, -1, :], train_seq.y)
    
    # Load LSTM model
    print("Loading LSTM model...")
    model = LSTMGaussianWorldModel(input_size=32, hidden_size=64, state_dim=32, num_layers=1, dropout=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 1. Persistence Baseline
    print("Evaluating Persistence...")
    pers_pred = np.empty((histories.shape[0], K, histories.shape[2]), dtype=np.float32)
    # persistence means S(t+k) = S(t)
    last_state = persistence_baseline(histories)
    for k_idx in range(K):
        pers_pred[:, k_idx, :] = last_state
        
    # 2. Lagged LR Baseline
    print("Evaluating Lagged-LR...")
    # we need to write a simple recursive function for LR on PCA
    lr_pred = np.empty((histories.shape[0], K, histories.shape[2]), dtype=np.float32)
    curr_state = histories[:, -1, :].copy()
    for k_idx in range(K):
        pred_state = lr_model.predict(curr_state)
        lr_pred[:, k_idx, :] = pred_state
        curr_state = pred_state
        
    # 3. LSTM World Model
    print("Evaluating LSTM World Model...")
    lstm_pred = lstm_rollout(histories, model, device, k=K)
    
    # Calculate MAE (Mean Absolute Error) for each K
    def compute_mae(predictions, targets):
        # average over samples and features
        return np.mean(np.abs(predictions - targets), axis=(0, 2))
        
    pers_mae = compute_mae(pers_pred, targets)
    lr_mae = compute_mae(lr_pred, targets)
    lstm_mae = compute_mae(lstm_pred, targets)
    
    results = []
    for k_idx in range(K):
        results.append({
            'K': k_idx + 1,
            'Persistence_MAE': pers_mae[k_idx],
            'LaggedLR_MAE': lr_mae[k_idx],
            'LSTM_MAE': lstm_mae[k_idx]
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_csv, index=False)
    print(f"\nRollout Errors saved to {out_csv}")
    print(res_df.to_string(index=False))
    
    # Plotting
    plt.figure(figsize=(8, 6))
    k_vals = res_df['K'].values
    plt.plot(k_vals, res_df['Persistence_MAE'].values, 's--', label='Persistence Baseline', color='gray')
    plt.plot(k_vals, res_df['LaggedLR_MAE'].values, 'o-', label='Lagged-LR (PCA)', color='orange')
    plt.plot(k_vals, res_df['LSTM_MAE'].values, '^-', label='LSTM World Model', color='blue')
    
    plt.title("Chronological Rollout Error vs Horizon (K)")
    plt.xlabel("Horizon Step (K)")
    plt.ylabel("Mean Absolute Error (PCA space)")
    plt.xticks(k_vals)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Rollout Error Curve saved to {out_png}")

if __name__ == '__main__':
    main()
