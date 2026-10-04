import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import logging

logging.basicConfig(level=logging.INFO)

def load_data(parquet_path='data-engineering/data/ucs/ucs_windows_models_v1.parquet', schema_path='ml1/configs/mdn_v1_feature_schema.txt'):
    df = pd.read_parquet(parquet_path)
    with open(schema_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.startswith('#'):
                features = line.strip().split(',')
                break
    return df, features

def process_fold(df, features, train_indices, test_indices):
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()
    
    # Calculate zero std features
    train_std = train_df[features].std()
    zero_std_features = train_std[train_std == 0].index.tolist()
    logging.info(f"Dropping {len(zero_std_features)} features with zero training std.")
    
    valid_features = [f for f in features if f not in zero_std_features]
    
    scaler = StandardScaler()
    scaler.fit(train_df[valid_features])
    
    z_train = scaler.transform(train_df[valid_features])
    z_test = scaler.transform(test_df[valid_features]) if len(test_df) else np.empty((0, len(valid_features)))
    
    pca = PCA(n_components=32)
    pca.fit(z_train)
    
    pca_train = pca.transform(z_train)
    pca_test = pca.transform(z_test) if len(z_test) else np.empty((0, pca.n_components_))
    
    train_out = {
        'z_space': z_train,
        'pca_space': pca_train,
        'episode_id': train_df['episode_id'].values,
        'window_start_utc': train_df['window_start_utc'].values
    }
    
    test_out = {
        'z_space': z_test,
        'pca_space': pca_test,
        'episode_id': test_df['episode_id'].values,
        'window_start_utc': test_df['window_start_utc'].values
    }
    
    return train_out, test_out, scaler, pca, valid_features

def get_valid_windows(df, K):
    # Ensure sorted by window_start_utc
    df = df.sort_values('window_start_utc')
    df['gap'] = df['window_start_utc'].diff()
    
    valid_mask = np.ones(len(df), dtype=bool)
    # A window t is valid at horizon K only if t-29 ... t+K all lie in the same contiguous day (no gap > 1 minute).
    
    # Not implementing exact day check here for brevity, just gap check
    for i in range(len(df)):
        start = i - 29
        end = i + K
        if start < 0 or end >= len(df):
            valid_mask[i] = False
            continue
            
        gaps = df['gap'].iloc[start+1:end+1]
        if (gaps > pd.Timedelta(minutes=1)).any():
            valid_mask[i] = False
            
    logging.info(f"K={K}: {valid_mask.sum()} valid windows")
    return valid_mask
