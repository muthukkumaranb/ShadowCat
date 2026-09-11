"""
NLL Diagnostic and Uncertainty Coverage

Diagnoses the NLL behavior on the chronological split to identify sigma collapse.
Evaluates interval coverage.
"""
import pandas as pd
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import json
from pathlib import Path
from scipy.stats import norm

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))

from lstm.model import LSTMGaussianWorldModel
from lstm.probabilistic import train_gaussian
from lstm.ucs import UCSConfig, apply_training_only_pca, build_next_state_sequences, validate_ucs_windows, UCSPCA, UCSRobustScaler
from lstm.utils import get_device, set_seed

def gaussian_nll_elementwise(mean, std, target):
    # NLL = 0.5 * ((target - mean)/std)^2 + log(std) + 0.5 * log(2*pi)
    variance = std ** 2
    nll = 0.5 * ((target - mean) ** 2) / variance + np.log(std) + 0.5 * np.log(2 * np.pi)
    return nll

def main():
    set_seed(42)
    device = get_device()
    
    out_dir = workspace_dir / 'artifacts' / 'lstm_diagnostics'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'nll_outliers.csv'
    metrics_path = out_dir / 'uncertainty_metrics.json'

    data_path = workspace_dir / 'data' / 'ucs' / 'ucs_windows.parquet'
    
    # We will use Chunk 4 from the chronological export because it evaluates 
    # the entire ML2 train split (0-1949). Or rather, Chunk 4 evaluates ML2 Val/Test 
    # using a model trained on ML2 Train. We should look at this model.
    model_path = workspace_dir / 'artifacts' / 'lstm' / 'ml2_export_chronological' / 'models' / 'chunk_4_gaussian_best.pt'
    
    if not model_path.exists():
        print(f"Model not found at {model_path}. Please run chronological export first.")
        sys.exit(1)

    print("=== NLL Diagnostic & Uncertainty Evaluation ===")
    
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

    # Apply PCA directly (as in the fixed export script)
    pca = UCSPCA(n_components=32, random_state=42)
    pca.fit(active_train_val.loc[active_train_val['split'] == 'train'], features)
    
    pca_features = [f"pca_{i}" for i in range(32)]
    full_transformed = pca.transform(chunk_frame)
    full_transformed_np = full_transformed[pca_features].to_numpy(dtype=np.float32)

    # Build test sequences manually
    def get_sequences(start_idx, end_idx):
        X, y, t = [], [], []
        times_array = pd.DatetimeIndex(pd.to_datetime(chunk_frame['window_start_utc'], utc=True))
        for t_pos in range(start_idx, end_idx):
            start = t_pos - config.lookback_windows
            if start < 0 or t_pos >= len(chunk_frame):
                continue
            hist = full_transformed_np[start:t_pos]
            if len(hist) == config.lookback_windows:
                X.append(hist)
                y.append(full_transformed_np[t_pos])
                t.append(times_array[t_pos])
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), t

    train_X, train_y, train_t = get_sequences(0, ml2_train_end)
    test_X, test_y, test_t = get_sequences(ml2_train_end, total_windows)
    
    model = LSTMGaussianWorldModel(input_size=32, hidden_size=64, state_dim=32, num_layers=1, dropout=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)['model_state_dict'])
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        train_mean_t, train_std_t = model(torch.as_tensor(train_X, dtype=torch.float32).to(device))
        test_mean_t, test_std_t = model(torch.as_tensor(test_X, dtype=torch.float32).to(device))
        
    train_mean, train_std = train_mean_t.cpu().numpy(), train_std_t.cpu().numpy()
    test_mean, test_std = test_mean_t.cpu().numpy(), test_std_t.cpu().numpy()

    # NLL Diagnosis on Train Set
    train_nll = gaussian_nll_elementwise(train_mean, train_std, train_y)
    train_nll_per_window = train_nll.mean(axis=1) # average over 32 features
    
    mean_nll = float(np.mean(train_nll_per_window))
    median_nll = float(np.median(train_nll_per_window))
    
    print(f"\n--- Train NLL Diagnostics ---")
    print(f"Mean Train NLL   : {mean_nll:.4f}")
    print(f"Median Train NLL : {median_nll:.4f}")
    
    # Outliers
    nll_threshold = np.percentile(train_nll_per_window, 99)
    outlier_indices = np.where(train_nll_per_window > nll_threshold)[0]
    outliers = []
    
    for idx in outlier_indices:
        outliers.append({
            'window_start_utc': str(train_t[idx]),
            'nll': float(train_nll_per_window[idx]),
            'mean_std': float(np.mean(train_std[idx]))
        })
    
    outliers_df = pd.DataFrame(outliers).sort_values('nll', ascending=False)
    outliers_df.to_csv(out_csv, index=False)
    print(f"Found {len(outliers_df)} top 1% NLL outliers.")
    print(f"Saved NLL outliers to {out_csv}")
    
    if mean_nll > median_nll + 10:
        print("Diagnosis: Large gap between mean and median NLL. Suggests a few collapsed-sigma or high-error outliers are dominating.")

    # Uncertainty / Interval Coverage on Test Set
    print(f"\n--- Uncertainty Interval Coverage (Test Set) ---")
    # 95% interval: mean +/- 1.96 * std
    z_score = 1.96
    lower_bound = test_mean - z_score * test_std
    upper_bound = test_mean + z_score * test_std
    
    within_interval = (test_y >= lower_bound) & (test_y <= upper_bound)
    coverage = float(within_interval.mean())
    mean_interval_width = float((upper_bound - lower_bound).mean())
    
    print(f"95% Interval Empirical Coverage : {coverage*100:.2f}%")
    print(f"Mean Interval Width             : {mean_interval_width:.4f}")

    metrics = {
        'train_nll_mean': mean_nll,
        'train_nll_median': median_nll,
        'test_empirical_coverage_95': coverage,
        'test_mean_interval_width': mean_interval_width,
        'protocol': 'chronological',
        'eval_split': 'test'
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved uncertainty metrics to {metrics_path}")

if __name__ == '__main__':
    main()
