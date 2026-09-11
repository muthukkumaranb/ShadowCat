import json
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))

from lstm.model import LSTMGaussianWorldModel
from lstm.probabilistic import deviation_scores, train_gaussian
from lstm.ucs import UCSConfig, apply_training_only_pca, build_next_state_sequences, validate_ucs_windows
from lstm.utils import get_device, set_seed

def main():
    set_seed(42)
    device = get_device()
    data_path = workspace_dir / 'data' / 'ucs' / 'ucs_windows.parquet'
    manifest_path = workspace_dir / 'artifacts' / 'loeo' / 'corrected_37fold_manifest.json'
    out_dir = workspace_dir / 'artifacts' / 'lstm' / 'loeo_world_model'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    windows = pd.read_parquet(data_path).sort_values('window_start_utc').reset_index(drop=True)
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    config = UCSConfig()
    features = validate_ucs_windows(windows, config=config)
    
    all_deviation_rows = []
    
    for fold in manifest['folds']:
        fold_id = fold['fold_id']
        print(f"Evaluating World Model on Fold {fold_id}...")
        
        train_indices = np.array(fold['train_indices'])
        test_indices = np.array(fold['test_indices'])
        
        fold_frame = windows.copy()
        fold_frame['split'] = 'none'
        fold_frame.loc[test_indices, 'split'] = 'test'
        
        train_df_sorted = fold_frame.loc[train_indices].sort_values('window_start_utc')
        n_train = len(train_df_sorted)
        val_start = int(n_train * 0.8)
        actual_train_indices = train_df_sorted.index[:val_start]
        actual_val_indices = train_df_sorted.index[val_start:]
        
        fold_frame.loc[actual_train_indices, 'split'] = 'train'
        fold_frame.loc[actual_val_indices, 'split'] = 'val'
        
        active_mask = fold_frame['split'].isin(['train', 'val', 'test'])
        purged = fold_frame[active_mask].copy()
        
        transformed, pca = apply_training_only_pca(purged, features, 32, random_state=42)
        pca_features = [f"pca_{i}" for i in range(32)]
        
        sequences = build_next_state_sequences(transformed, features=pca_features, config=config)
        train_set, val_set = sequences['train'], sequences['val']
        
        full_transformed = pca.transform(fold_frame)
        full_transformed_np = full_transformed[pca_features].to_numpy(dtype=np.float32)
        
        histories = []
        labels = []
        test_timestamps = []
        
        idx_to_pos = {idx: pos for pos, idx in enumerate(fold_frame.index)}
        test_pos_indices = [idx_to_pos[idx] for idx in test_indices]
        times_array = pd.DatetimeIndex(pd.to_datetime(fold_frame['window_start_utc'], utc=True))
        
        for t_pos in test_pos_indices:
            start = t_pos - config.lookback_windows
            if start < 0 or t_pos >= len(fold_frame):
                continue
            hist = full_transformed_np[start:t_pos]
            if len(hist) == config.lookback_windows:
                histories.append(hist)
                labels.append(full_transformed_np[t_pos])
                test_timestamps.append(times_array[t_pos])
                
        if len(histories) == 0:
            print(f"Skipping fold {fold_id}: no valid test sequences")
            continue
            
        test_X = np.asarray(histories, dtype=np.float32)
        test_y = np.asarray(labels, dtype=np.float32)
        
        model = LSTMGaussianWorldModel(input_size=32, hidden_size=64, state_dim=32, num_layers=1, dropout=0.2)
        checkpoint_path = out_dir / str(fold_id) / 'gaussian_best.pt'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        train_gaussian(
            model,
            train_set.X,
            train_set.y,
            val_set.X,
            val_set.y,
            epochs=2,
            batch_size=64,
            learning_rate=0.001,
            weight_decay=0.0001,
            patience=5,
            min_delta=0.001,
            seed=42,
            device=str(device),
            checkpoint_path=checkpoint_path,
        )
        
        model.eval()
        with torch.no_grad():
            mean, std = model(torch.as_tensor(test_X, dtype=torch.float32).to(device))
        mean = mean.cpu().numpy()
        std = std.cpu().numpy()
        
        deviation = deviation_scores(mean, test_y)
        test_labels = fold_frame.loc[test_indices, 'future_attack_label'].to_numpy(dtype=np.float32)
        
        fold_df = pd.DataFrame({
            'timestamp': [str(t) for t in test_timestamps],
            'deviation_score': deviation,
            'future_attack_label': test_labels[:len(deviation)].astype(int),
            'protocol': 'LOEO',
            'fold_id': fold_id,
            'actual_state': [list(y) for y in test_y],
            'predicted_mean': [list(m) for m in mean],
            'predicted_std': [list(s) for s in std]
        })
        all_deviation_rows.append(fold_df)
        
    final_df = pd.concat(all_deviation_rows, ignore_index=True)
    output_csv = out_dir / 'deviation_predictions.csv'
    final_df.to_csv(output_csv, index=False)
    print(f"\nSaved LOEO deviation predictions to {output_csv}")

if __name__ == '__main__':
    main()
