"""
Feature-Group Attribution via Ablation

Computes attribution for the LSTM world model by masking/zeroing specific feature groups
over the 30-window history, transforming through PCA, and measuring the change in prediction error.
"""
import pandas as pd
import numpy as np
import sys
import os
import torch
from pathlib import Path
import json

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))

from lstm.model import LSTMGaussianWorldModel
from lstm.ucs import UCSConfig, validate_ucs_windows, UCSPCA
from lstm.utils import get_device, set_seed
from lstm.probabilistic import deviation_scores

def main():
    set_seed(42)
    device = get_device()
    
    out_dir = workspace_dir / 'artifacts' / 'lstm'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'feature_attribution.csv'
    
    data_path = workspace_dir / 'data' / 'ucs' / 'ucs_windows.parquet'
    model_path = workspace_dir / 'artifacts' / 'lstm' / 'ml2_export_chronological' / 'models' / 'chunk_4_gaussian_best.pt'
    
    if not model_path.exists():
        print(f"Model not found at {model_path}. Please run chronological export first.")
        sys.exit(1)
        
    print("=== Feature-Group Attribution (Ablation) ===")
    
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
    
    # Load model
    model = LSTMGaussianWorldModel(input_size=32, hidden_size=64, state_dim=32, num_layers=1, dropout=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)['model_state_dict'])
    model.to(device)
    model.eval()
    
    def evaluate_with_mask(mask_cols):
        # Create a masked dataframe
        masked_frame = chunk_frame.copy()
        if mask_cols:
            masked_frame[mask_cols] = 0.0
            
        full_transformed = pca.transform(masked_frame)
        full_transformed_np = full_transformed[pca_features].to_numpy(dtype=np.float32)
        
        # Test sequences
        X, y = [], []
        test_indices = list(range(ml2_train_end, total_windows))
        for t_pos in test_indices:
            start = t_pos - config.lookback_windows
            if start < 0 or t_pos >= len(chunk_frame):
                continue
            hist = full_transformed_np[start:t_pos]
            if len(hist) == config.lookback_windows:
                X.append(hist)
                # target is always unmasked!
                # so we need the unmasked target from the original transformed space.
                pass
        
        # We need the original unmasked y!
        unmasked_transformed = pca.transform(chunk_frame)
        unmasked_np = unmasked_transformed[pca_features].to_numpy(dtype=np.float32)
        
        X, y = [], []
        for t_pos in test_indices:
            start = t_pos - config.lookback_windows
            if start < 0 or t_pos >= len(chunk_frame):
                continue
            hist = full_transformed_np[start:t_pos]
            if len(hist) == config.lookback_windows:
                X.append(hist)
                y.append(unmasked_np[t_pos])
                
        test_X = np.asarray(X, dtype=np.float32)
        test_y = np.asarray(y, dtype=np.float32)
        
        with torch.no_grad():
            mean, _ = model(torch.as_tensor(test_X, dtype=torch.float32).to(device))
        
        dev = deviation_scores(mean.cpu().numpy(), test_y)
        return float(np.mean(dev))
        
    print("Evaluating baseline (no mask)...")
    baseline_mae = evaluate_with_mask([])
    print(f"Baseline MAE: {baseline_mae:.4f}")
    
    groups = {
        'byte_count': [c for c in features if 'byte' in c],
        'packet_count': [c for c in features if 'packet' in c],
        'duration': [c for c in features if 'duration' in c],
        'fwd': [c for c in features if 'fwd' in c],
        'bwd': [c for c in features if 'bwd' in c],
    }
    
    results = []
    
    for group_name, cols in groups.items():
        if not cols:
            continue
        print(f"Evaluating mask group: {group_name} ({len(cols)} columns)...")
        mae = evaluate_with_mask(cols)
        # Attribution = Error with mask - Error without mask
        # If masking it INCREASES error, it was a USEFUL feature (positive attribution).
        # If masking it DECREASES error, it was a HARMFUL feature (negative attribution).
        attribution = mae - baseline_mae
        
        results.append({
            'feature_group': group_name,
            'masked_mae': mae,
            'attribution': attribution
        })
        
    res_df = pd.DataFrame(results).sort_values('attribution', ascending=False)
    res_df.to_csv(out_csv, index=False)
    
    print("\n--- Feature Attribution Results ---")
    print(res_df.to_string(index=False))
    print(f"Saved attribution to {out_csv}")

if __name__ == '__main__':
    main()
