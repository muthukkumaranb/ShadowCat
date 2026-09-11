"""
Diagnose Attack vs Benign Rollout Split
"""
import pandas as pd
import numpy as np
import sys
import os
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))

from lstm.model import LSTMGaussianWorldModel
from lstm.ucs import UCSConfig, validate_ucs_windows, UCSPCA, rollout_targets, persistence_baseline, build_next_state_sequences
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
    
    data_path = workspace_dir / 'data' / 'ucs' / 'ucs_windows.parquet'
    model_path = workspace_dir / 'artifacts' / 'lstm' / 'chunk_4_gaussian_fixed.pt'
    
    if not model_path.exists():
        print(f"Fixed model not found at {model_path}.")
        sys.exit(1)
        
    print("=== Attack vs. Benign Rollout Performance Split ===\n")
    
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

    # PCA
    pca = UCSPCA(n_components=32, random_state=42)
    pca.fit(active_train_val.loc[active_train_val['split'] == 'train'], features)
    pca_features = [f"pca_{i}" for i in range(32)]
    full_transformed = pca.transform(chunk_frame)
    full_transformed_np = full_transformed[pca_features].to_numpy(dtype=np.float32)
    
    # Identify top 4 components by variance
    explained_variance = pca.pca.explained_variance_ratio_
    top_components_idx = np.argsort(explained_variance)[::-1][:4]

    # Post-PCA Scaler
    pca_scaler = StandardScaler()
    train_pca_data = full_transformed.loc[chunk_frame['split'] == 'train', pca_features].to_numpy(dtype=np.float32)
    pca_scaler.fit(train_pca_data)
    scaled_pca_np = pca_scaler.transform(full_transformed_np).astype(np.float32)

    # Unscaled representations
    transformed_frame_unscaled = chunk_frame.drop(columns=features)
    for i, p_col in enumerate(pca_features):
        transformed_frame_unscaled[p_col] = full_transformed_np[:, i]
        
    # We also need the label_binary
    transformed_frame_unscaled['label_binary'] = chunk_frame['label_binary']
        
    K = 3
    # Extract Unscaled histories and targets
    histories_unscaled, targets_unscaled, times = rollout_targets(transformed_frame_unscaled, features=pca_features, config=config, k=K, split='test')
    
    # Map timestamps to label_binary manually to bypass feature validation guards
    test_subset = chunk_frame.loc[chunk_frame["split"] == 'test']
    time_to_label = dict(zip(pd.to_datetime(test_subset["window_start_utc"], utc=True), test_subset["label_binary"]))
    
    labels_target = np.empty((len(times), K, 1))
    for i, t_idx in enumerate(times):
        for k_idx in range(K):
            labels_target[i, k_idx, 0] = time_to_label[t_idx[k_idx]]

    # Baseline Persistence
    pers_pred = np.empty((histories_unscaled.shape[0], K, histories_unscaled.shape[2]), dtype=np.float32)
    last_state = persistence_baseline(histories_unscaled)
    for k_idx in range(K):
        pers_pred[:, k_idx, :] = last_state

    # Baseline LR
    seqs_unscaled = build_next_state_sequences(transformed_frame_unscaled, features=pca_features, config=config)
    train_seq_unscaled = seqs_unscaled['train']
    lr_model = LinearRegression()
    lr_model.fit(train_seq_unscaled.X[:, -1, :], train_seq_unscaled.y)
    
    lr_pred = np.empty((histories_unscaled.shape[0], K, histories_unscaled.shape[2]), dtype=np.float32)
    curr_state = histories_unscaled[:, -1, :].copy()
    for k_idx in range(K):
        pred_state = lr_model.predict(curr_state)
        lr_pred[:, k_idx, :] = pred_state
        curr_state = pred_state

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
    # Execution
    # ---------------------------------------------------------
    
    def get_errors(pred, targ, k_idx):
        # returns array of shape (samples,) representing overall MAE per sample
        return np.mean(np.abs(pred[:, k_idx, :] - targ[:, k_idx, :]), axis=1)

    print("--- STEP 1 & 2: Benign vs. Attack Group x Model x K MAE Table ---")
    
    # Store rows for dataframe display
    results_rows = []
    
    for k_idx in range(K):
        labels_k = labels_target[:, k_idx, 0]
        benign_mask = (labels_k == 0)
        attack_mask = (labels_k == 1)
        
        n_benign = np.sum(benign_mask)
        n_attack = np.sum(attack_mask)
        
        err_pers = get_errors(pers_pred, targets_unscaled, k_idx)
        err_lr = get_errors(lr_pred, targets_unscaled, k_idx)
        err_lstm = get_errors(lstm_pred_unscaled, targets_unscaled, k_idx)
        
        for group, mask in [('Benign', benign_mask), ('Attack', attack_mask)]:
            if np.sum(mask) == 0:
                continue
            
            results_rows.append({
                'Group': group,
                'Count': np.sum(mask),
                'K': k_idx + 1,
                'Pers_Mean': np.mean(err_pers[mask]), 'Pers_Med': np.median(err_pers[mask]),
                'LR_Mean': np.mean(err_lr[mask]), 'LR_Med': np.median(err_lr[mask]),
                'LSTM_Mean': np.mean(err_lstm[mask]), 'LSTM_Med': np.median(err_lstm[mask]),
            })

    df_res = pd.DataFrame(results_rows)
    print(df_res.to_string(index=False))

    print("\n--- STEP 3 & 4: Per-Component MAE & Variance Split (Top 4 Components) ---")
    
    # Output structure
    comp_rows = []
    
    for c_idx in top_components_idx:
        c_name = f"PC{c_idx}"
        for k_idx in range(K):
            labels_k = labels_target[:, k_idx, 0]
            for group, mask in [('Benign', labels_k == 0), ('Attack', labels_k == 1)]:
                if np.sum(mask) == 0:
                    continue
                
                # Targets for this component
                targ_c = targets_unscaled[mask, k_idx, c_idx]
                pred_lstm_c = lstm_pred_unscaled[mask, k_idx, c_idx]
                pred_pers_c = pers_pred[mask, k_idx, c_idx]
                
                mae_lstm = np.mean(np.abs(pred_lstm_c - targ_c))
                mae_pers = np.mean(np.abs(pred_pers_c - targ_c))
                var_targ = np.var(targ_c)
                
                comp_rows.append({
                    'Component': c_name,
                    'K': k_idx + 1,
                    'Group': group,
                    'Target_Var': var_targ,
                    'Pers_MAE': mae_pers,
                    'LSTM_MAE': mae_lstm,
                    'LSTM_vs_Pers': mae_lstm - mae_pers
                })
                
    df_comp = pd.DataFrame(comp_rows)
    print(df_comp.to_string(index=False))
    
    print("\n--- STEP 5: Interpretation ---")
    
    # Auto-generate a plain statement based on the extracted values
    # Check if LSTM ever beats Pers in 'Attack' group for K=1,2,3 overall
    beats_overall_attack = df_res[df_res['Group'] == 'Attack']['LSTM_Mean'].values < df_res[df_res['Group'] == 'Attack']['Pers_Mean'].values
    if np.any(beats_overall_attack):
        print(f"Overall MAE: LSTM BEATS Persistence on Attack windows for at least one K horizon.")
    else:
        print("Overall MAE: LSTM STILL LOSES to Persistence on Attack windows across all K horizons.")
        
    # Check components
    attack_comp_df = df_comp[df_comp['Group'] == 'Attack']
    lstm_wins = attack_comp_df[attack_comp_df['LSTM_vs_Pers'] < 0]
    if len(lstm_wins) > 0:
        print("\nPer-Component Advantage: The LSTM DOES show a genuine advantage over Persistence during attack windows on the following components:")
        for _, row in lstm_wins.iterrows():
            print(f"  - {row['Component']} (K={row['K']}): LSTM MAE {row['LSTM_MAE']:.4f} vs Pers {row['Pers_MAE']:.4f} (Delta: {row['LSTM_vs_Pers']:.4f})")
    else:
        print("\nPer-Component Advantage: No. The LSTM does NOT show a genuine advantage over Persistence during attack windows on ANY of the top components.")

if __name__ == '__main__':
    main()
