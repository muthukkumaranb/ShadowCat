"""
B2 Within-Episode Progression Forecast
"""
import pandas as pd
import numpy as np
import torch
import sys
import os
from pathlib import Path
import json

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))

from lstm.model import LSTMGaussianWorldModel
from lstm.ucs import UCSConfig, validate_ucs_windows, apply_training_only_pca
from lstm.utils import get_device, set_seed

def get_persistence(history):
    return history[-1]

def main():
    set_seed(42)
    device = get_device()
    
    data_path = workspace_dir / 'data' / 'ucs' / 'ucs_windows.parquet'
    manifest_path = workspace_dir / 'artifacts' / 'loeo' / 'corrected_37fold_manifest.json'
    out_dir = workspace_dir / 'artifacts' / 'lstm' / 'b2_progression'
    model_dir = workspace_dir / 'artifacts' / 'lstm' / 'loeo_world_model'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    windows = pd.read_parquet(data_path).sort_values('window_start_utc').reset_index(drop=True)
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    config = UCSConfig()
    features = validate_ucs_windows(windows, config=config)
    
    k_context = 5
    K_horizons = [1, 2, 5]
    
    results = []
    
    for fold in manifest['folds']:
        fold_id = fold['fold_id']
        test_indices = np.array(fold['test_indices'])
        held_out_ep = fold['held_out_episode_id']
        
        fold_frame = windows.copy()
        
        # Get the actual test set windows for this fold
        test_df = fold_frame.loc[test_indices]
        
        # We only care about the attack episode
        attack_windows = test_df[test_df['episode_id'] == held_out_ep].sort_values('window_start_utc')
        if len(attack_windows) < k_context + max(K_horizons):
            # Not enough windows in this episode to do B2 progression
            continue
            
        train_indices = np.array(fold['train_indices'])
        train_df = fold_frame.loc[train_indices].sort_values('window_start_utc')
        val_start = int(len(train_df) * 0.8)
        
        fold_frame['split'] = 'none'
        fold_frame.loc[test_indices, 'split'] = 'test'
        fold_frame.loc[train_df.index[:val_start], 'split'] = 'train'
        fold_frame.loc[train_df.index[val_start:], 'split'] = 'val'
        
        purged = fold_frame[fold_frame['split'].isin(['train', 'val', 'test'])].copy()
        transformed, pca = apply_training_only_pca(purged, features, 32, random_state=42)
        pca_features = [f"pca_{i}" for i in range(32)]
        
        full_transformed = pca.transform(fold_frame)
        full_transformed_np = full_transformed[pca_features].to_numpy(dtype=np.float32)
        
        model = LSTMGaussianWorldModel(input_size=32, hidden_size=64, state_dim=32, num_layers=1, dropout=0.2)
        model_path = model_dir / str(fold_id) / 'gaussian_best.pt'
        
        if not model_path.exists():
            continue
            
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        model.to(device)
        model.eval()
        
        # Identify the first k_context windows of the attack
        attack_idx = attack_windows.index.to_numpy()
        start_idx = attack_idx[k_context - 1]
        
        # build context
        idx_map = {idx: pos for pos, idx in enumerate(fold_frame.index)}
        t_pos = idx_map[start_idx]
        
        hist_start = t_pos - config.lookback_windows + 1
        if hist_start < 0:
            continue
            
        history = full_transformed_np[hist_start:t_pos+1]
        if len(history) != config.lookback_windows:
            continue
            
        # Rollout
        current = np.expand_dims(history, axis=0) # (1, L, D)
        predictions = []
        for step in range(max(K_horizons)):
            with torch.no_grad():
                mean, _ = model(torch.as_tensor(current, dtype=torch.float32).to(device))
            pred = mean.cpu().numpy() # (1, D)
            predictions.append(pred[0])
            current = np.concatenate([current[:, 1:, :], np.expand_dims(pred, axis=1)], axis=1)
            
        pers_pred = get_persistence(history)
        
        for K in K_horizons:
            target_pos = t_pos + K
            if target_pos >= len(full_transformed_np):
                continue
            target_state = full_transformed_np[target_pos]
            pred_state = predictions[K-1]
            
            lstm_mae = np.mean(np.abs(pred_state - target_state))
            pers_mae = np.mean(np.abs(pers_pred - target_state))
            
            # also get PC2 error specifically
            lstm_pc2 = np.abs(pred_state[2] - target_state[2])
            pers_pc2 = np.abs(pers_pred[2] - target_state[2])
            
            results.append({
                'episode_id': held_out_ep,
                'fold_id': fold_id,
                'horizon_K': K,
                'lstm_mae_all': lstm_mae,
                'pers_mae_all': pers_mae,
                'lstm_mae_pc2': lstm_pc2,
                'pers_mae_pc2': pers_pc2
            })
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df.to_csv(out_dir / 'b2_progression_results.csv', index=False)
        print("B2 Progression Results (Mean MAE):")
        
        summary = res_df.groupby('horizon_K')[['lstm_mae_all', 'pers_mae_all', 'lstm_mae_pc2', 'pers_mae_pc2']].mean().reset_index()
        print(summary)
        print(f"\nUsable Episodes: {len(res_df['episode_id'].unique())}")
    else:
        print("No usable episodes found.")

if __name__ == '__main__':
    main()
