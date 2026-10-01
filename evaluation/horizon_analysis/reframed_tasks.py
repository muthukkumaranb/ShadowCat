import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, mean_absolute_error

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, str(workspace_dir))

from ml1.lstm.model import LSTMGaussianWorldModel
from ml1.lstm.ucs import UCSConfig, validate_ucs_windows

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    device = get_device()
    
    # 1. Load Dataset
    data_path = workspace_dir / "data-engineering/data/ucs/ucs_windows.parquet"
    if not data_path.exists():
        data_path = workspace_dir / "data/ucs/ucs_windows.parquet"
        
    windows = pd.read_parquet(data_path)
    config = UCSConfig()
    features = validate_ucs_windows(windows, config=config)
    
    # Define Binary Labels: 0 for Benign, 1 for Attack
    windows['binary_label'] = windows['label_binary']
    
    # Define Time-to-next-transition
    # For windows inside an attack episode, how many windows until stage change?
    # Stage is defined by attack_type here
    windows['attack_stage'] = windows['label_attack_type'].fillna('Benign')
    
    # Calculate time to next transition
    ttnt = np.zeros(len(windows))
    last_stage = None
    last_change_idx = len(windows)
    
    # Backward pass to calculate time to NEXT transition
    for i in range(len(windows)-1, -1, -1):
        curr_stage = windows['attack_stage'].iloc[i]
        if i == len(windows)-1 or curr_stage != windows['attack_stage'].iloc[i+1]:
            last_change_idx = i + 1
        ttnt[i] = last_change_idx - i
        
    windows['ttnt'] = ttnt
    
    # 2. Load World Model
    world_model_path = workspace_dir / "ml1/artifacts/lstm/probabilistic_world_model_v3/gaussian_next_state_best.pt"
    if not world_model_path.exists():
        world_model_path = workspace_dir / "ml1/artifacts/lstm/gaussian_next_state_best_v3.pt"
    if not world_model_path.exists():
        world_model_path = workspace_dir / "ml1/artifacts/lstm/gaussian_next_state_best_v2.pt"
        
    world_model = LSTMGaussianWorldModel(
        input_size=len(features),
        hidden_size=64,
        state_dim=len(features),
        num_layers=1,
        dropout=0.2
    )
    ckpt = torch.load(world_model_path, map_location=device, weights_only=False)
    world_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    world_model.to(device)
    world_model.eval()
    
    # 3. Rolling-Origin Setup
    origins = [0.9] # using different training end points
    purge_windows = config.purge_embargo_width
    lookback = config.lookback_windows
    
    all_f1_bin_model = {k: [] for k in range(1, 6)}
    all_f1_bin_pers = {k: [] for k in range(1, 6)}
    
    all_mae_ttnt_model = {k: [] for k in range(1, 6)}
    all_mae_ttnt_pers = {k: [] for k in range(1, 6)}
    
    total_len = len(windows)
    
    for split_ratio in origins:
        train_end_idx = int(total_len * split_ratio)
        train_df = windows.iloc[:train_end_idx].copy()
        
        test_start_idx = train_end_idx + purge_windows
        test_end_idx = test_start_idx + int(total_len * 0.1)
        if test_end_idx > total_len:
            test_end_idx = total_len
            
        if test_start_idx >= total_len:
            continue
            
        test_df = windows.iloc[test_start_idx:test_end_idx].copy()
        
        # Prepare Train sequences for linear heads
        train_values = train_df[features].values
        train_bin = train_df['binary_label'].values
        train_ttnt = train_df['ttnt'].values
        
        valid_starts_train = list(range(len(train_df) - lookback - 5))
        if not valid_starts_train:
            continue
            
        train_histories = np.array([train_values[s:s+lookback] for s in valid_starts_train], dtype=np.float32)
        
        with torch.no_grad():
            curr_seq_train = torch.tensor(train_histories).to(device)
            # We will just train on representations at K=1 for simplicity, or we can train a separate head for each K
            # Actually, to be fair to the model, let's train a head for each K using rolled-out representations.
            train_reps = {k: [] for k in range(1, 6)}
            for k in range(5):
                mean_next_train, _ = world_model(curr_seq_train)
                train_reps[k+1] = mean_next_train.cpu().numpy()
                curr_seq_train = torch.cat([curr_seq_train[:, 1:, :], mean_next_train.unsqueeze(1)], dim=1)
                
        print(f"Split {split_ratio} - Train shape: {train_df.shape}")
        # Train linear models for each K
        bin_models = {}
        ttnt_models = {}
        
        from sklearn.preprocessing import StandardScaler
        scalers = {}
        
        for k in range(1, 6):
            print(f"Training K={k} models")
            y_bin = np.array([train_bin[s+lookback+k-1] for s in valid_starts_train])
            y_ttnt = np.array([train_ttnt[s+lookback+k-1] for s in valid_starts_train])
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(train_reps[k])
            scalers[k] = scaler
            
            clf = LogisticRegression(solver='liblinear', max_iter=100, class_weight='balanced')
            clf.fit(X_scaled, y_bin)
            bin_models[k] = clf
            
            reg = LinearRegression()
            reg.fit(X_scaled, y_ttnt)
            ttnt_models[k] = reg
            
        # Test targets & histories
        test_values = test_df[features].values
        test_bin = test_df['binary_label'].values
        test_ttnt = test_df['ttnt'].values
        
        valid_starts = list(range(len(test_df) - lookback - 5))
        if not valid_starts:
            continue
            
        histories = np.array([test_values[start:start+lookback] for start in valid_starts], dtype=np.float32)
        
        pred_bin_model = np.zeros((len(valid_starts), 5))
        pred_ttnt_model = np.zeros((len(valid_starts), 5))
        
        pred_bin_pers = np.zeros((len(valid_starts), 5))
        pred_ttnt_pers = np.zeros((len(valid_starts), 5))
        
        base_bin = np.array([test_bin[start+lookback-1] for start in valid_starts])
        base_ttnt = np.array([test_ttnt[start+lookback-1] for start in valid_starts])
        
        with torch.no_grad():
            curr_seq = torch.tensor(histories).to(device)
            for k in range(5):
                mean_next, _ = world_model(curr_seq)
                rep = mean_next.cpu().numpy()
                rep_scaled = scalers[k+1].transform(rep)
                
                pred_bin_model[:, k] = bin_models[k+1].predict(rep_scaled)
                pred_ttnt_model[:, k] = ttnt_models[k+1].predict(rep_scaled)
                
                curr_seq = torch.cat([curr_seq[:, 1:, :], mean_next.unsqueeze(1)], dim=1)
                
                # Persistence baseline
                pred_bin_pers[:, k] = base_bin
                # For TTNT persistence: if TTNT was T at t, it should be max(1, T-k) at t+k
                # Actually, simple persistence is just T-k (clamped to 1 minimum for within-attack)
                pred_ttnt_pers[:, k] = np.maximum(1, base_ttnt - (k+1))
                
        for k in range(5):
            y_true_bin = np.array([test_bin[start+lookback+k] for start in valid_starts])
            y_true_ttnt = np.array([test_ttnt[start+lookback+k] for start in valid_starts])
            
            f1_m = f1_score(y_true_bin, pred_bin_model[:, k], average='macro', zero_division=0)
            f1_p = f1_score(y_true_bin, pred_bin_pers[:, k], average='macro', zero_division=0)
            
            mae_m = mean_absolute_error(y_true_ttnt, pred_ttnt_model[:, k])
            mae_p = mean_absolute_error(y_true_ttnt, pred_ttnt_pers[:, k])
            
            all_f1_bin_model[k+1].append(f1_m)
            all_f1_bin_pers[k+1].append(f1_p)
            
            all_mae_ttnt_model[k+1].append(mae_m)
            all_mae_ttnt_pers[k+1].append(mae_p)

    # 4. Report
    print("=== Reframed Task 3a: Binary Escalation (Benign vs Attack) ===")
    print("K\tModel_F1\tPers_F1\tBeat_Baseline?")
    for k in range(1, 6):
        mean_m = np.mean(all_f1_bin_model[k])
        mean_p = np.mean(all_f1_bin_pers[k])
        print(f"{k}\t{mean_m:.4f}\t\t{mean_p:.4f}\t{mean_m > mean_p}")
        
    print("\n=== Reframed Task 3b: Time-to-next-transition (MAE) ===")
    print("K\tModel_MAE\tPers_MAE\tBeat_Baseline?")
    for k in range(1, 6):
        mean_m = np.mean(all_mae_ttnt_model[k])
        mean_p = np.mean(all_mae_ttnt_pers[k])
        # Lower MAE is better
        print(f"{k}\t{mean_m:.4f}\t\t{mean_p:.4f}\t{mean_m < mean_p}")

if __name__ == "__main__":
    main()
