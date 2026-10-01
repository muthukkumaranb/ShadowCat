import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, str(workspace_dir))

from ml1.lstm.model import LSTMGaussianWorldModel
from ml1.lstm.ucs import UCSConfig, validate_ucs_windows, build_next_state_sequences
from ml1.scripts.train_stage_head import StageClassificationHead

def load_attack_stage_mapping(yaml_path: Path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    mappings = {}
    for attack_name, info in data.get('mappings', {}).items():
        tactic = info.get('tactic', 'Unknown/Other')
        if '/' in tactic and 'Credential Access' in tactic:
            tactic = 'Credential Access'
        mappings[attack_name] = tactic
    return mappings

def assign_stage(row, tactic_map):
    if row['label_binary'] == 0:
        return "Unknown/Other"
    attack = row.get('label_attack_type', 'Benign')
    return tactic_map.get(attack, "Unknown/Other")

import argparse

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to custom world model checkpoint")
    parser.add_argument("--stage-head-checkpoint", type=str, default=None, help="Path to custom stage head checkpoint")
    args = parser.parse_args()

    device = get_device()
    
    # 1. Load Dataset & Mapping
    data_path = workspace_dir / "data-engineering/data/ucs/ucs_windows.parquet"
    if not data_path.exists():
        data_path = workspace_dir / "data/ucs/ucs_windows.parquet"
        
    windows = pd.read_parquet(data_path)
    config = UCSConfig()
    features = validate_ucs_windows(windows, config=config)
    
    mapping_path = workspace_dir / "data-engineering/data/ucs/attack_tactics_mapping.yaml"
    if not mapping_path.exists():
        mapping_path = workspace_dir / "data/ucs/attack_tactics_mapping.yaml"
    tactic_map = load_attack_stage_mapping(mapping_path)
    
    windows['stage_label'] = windows.apply(lambda r: assign_stage(r, tactic_map), axis=1)
    stage_classes = sorted(list(set(windows['stage_label'].unique())))
    if "Unknown/Other" not in stage_classes:
        stage_classes.append("Unknown/Other")
    stage_classes = sorted(stage_classes)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(stage_classes)}
    windows['stage_idx'] = windows['stage_label'].map(class_to_idx)
    
    num_classes = len(stage_classes)
    
    # 2. Load Models
    if args.checkpoint:
        world_model_path = Path(args.checkpoint)
        # Infer architecture from sweep configs
        # Default is hidden=64, layers=1, but if we pass checkpoint, we might need to know architecture!
        # Oh, `LSTMGaussianWorldModel` needs input_size, hidden_size, state_dim, num_layers.
        # We can try to load the state dict to guess or just pass via args.
        ckpt = torch.load(world_model_path, map_location=device, weights_only=False)
        model_state = ckpt.get('model_state_dict', ckpt)
        # Infer hidden size and num layers from state dict
        lstm_weight = model_state.get('lstm.weight_ih_l0')
        if lstm_weight is not None:
            hidden_size = lstm_weight.shape[0] // 4
            num_layers = max([int(k.split('_l')[1]) for k in model_state.keys() if k.startswith('lstm.weight_ih_l')]) + 1
        else:
            hidden_size = 64
            num_layers = 1
    else:
        world_model_path = workspace_dir / "ml1/artifacts/lstm/probabilistic_world_model_v3/gaussian_next_state_best.pt"
        if not world_model_path.exists():
            world_model_path = workspace_dir / "ml1/artifacts/lstm/gaussian_next_state_best_v3.pt"
        if not world_model_path.exists():
            world_model_path = workspace_dir / "ml1/artifacts/lstm/gaussian_next_state_best_v2.pt"
        hidden_size = 64
        num_layers = 1
        ckpt = torch.load(world_model_path, map_location=device, weights_only=False)
        model_state = ckpt.get('model_state_dict', ckpt)
        
    world_model = LSTMGaussianWorldModel(
        input_size=len(features),
        hidden_size=hidden_size,
        state_dim=len(features),
        num_layers=num_layers,
        dropout=0.2
    )
    world_model.load_state_dict(model_state)
    world_model.to(device)
    world_model.eval()
    
    if args.stage_head_checkpoint:
        stage_head_path = Path(args.stage_head_checkpoint)
    else:
        stage_head_path = workspace_dir / "ml1/artifacts/lstm/stage_head_v3/stage_head_best.pt"
        if not stage_head_path.exists():
            stage_head_path = workspace_dir / "ml1/artifacts/lstm/stage_head/stage_head_best.pt"
        
    stage_head = StageClassificationHead(
        state_dim=len(features),
        num_classes=num_classes,
        hidden_dim=64,
        dropout=0.2
    )
    st_ckpt = torch.load(stage_head_path, map_location=device, weights_only=False)
    stage_head.load_state_dict(st_ckpt)
    stage_head.to(device)
    stage_head.eval()
    
    # 3. Rolling-Origin Setup
    origins = [0.5, 0.6, 0.7, 0.8, 0.9] # using different training end points
    purge_windows = config.purge_embargo_width
    lookback = config.lookback_windows
    
    all_f1_model = {k: [] for k in range(1, 6)}
    all_f1_pers = {k: [] for k in range(1, 6)}
    all_f1_markov = {k: [] for k in range(1, 6)}
    
    times = pd.to_datetime(windows['window_start_utc'], utc=True)
    total_len = len(windows)
    
    for split_ratio in origins:
        train_end_idx = int(total_len * split_ratio)
        train_df = windows.iloc[:train_end_idx].copy()
        
        test_start_idx = train_end_idx + purge_windows
        test_end_idx = test_start_idx + int(total_len * 0.1) # test on 10%
        if test_end_idx > total_len:
            test_end_idx = total_len
            
        if test_start_idx >= total_len:
            continue
            
        test_df = windows.iloc[test_start_idx:test_end_idx].copy()
        
        # Markov baseline on train
        stage_counts = np.zeros((num_classes, num_classes))
        train_stages = train_df['stage_idx'].values
        for i in range(len(train_stages) - 1):
            stage_counts[train_stages[i], train_stages[i+1]] += 1
        
        # Add slight smoothing
        stage_counts += 0.01 
        markov_trans = stage_counts / stage_counts.sum(axis=1, keepdims=True)
        
        # Test targets & histories
        test_values = test_df[features].values
        test_stages = test_df['stage_idx'].values
        
        # We need histories of length 'lookback' to forecast K steps
        valid_starts = []
        for i in range(len(test_df) - lookback - 5):
            valid_starts.append(i)
            
        if not valid_starts:
            continue
            
        histories = np.array([test_values[start:start+lookback] for start in valid_starts], dtype=np.float32)
        true_stages = np.array([[test_stages[start+lookback+k-1] for k in range(1, 6)] for start in valid_starts])
        base_stages = np.array([test_stages[start+lookback-1] for start in valid_starts])
        
        # Model predictions
        pred_stages_model = np.zeros((len(valid_starts), 5))
        pred_stages_pers = np.zeros((len(valid_starts), 5))
        pred_stages_markov = np.zeros((len(valid_starts), 5))
        
        with torch.no_grad():
            curr_seq = torch.tensor(histories).to(device)
            for k in range(5):
                mean_next, _ = world_model(curr_seq)
                logits = stage_head(mean_next)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                pred_stages_model[:, k] = preds
                
                # Update seq for next step
                curr_seq = torch.cat([curr_seq[:, 1:, :], mean_next.unsqueeze(1)], dim=1)
                
                # Persistence baseline
                pred_stages_pers[:, k] = base_stages
                
                # Markov baseline
                if k == 0:
                    curr_probs = np.zeros((len(valid_starts), num_classes))
                    curr_probs[np.arange(len(valid_starts)), base_stages] = 1.0
                curr_probs = curr_probs @ markov_trans
                pred_stages_markov[:, k] = np.argmax(curr_probs, axis=1)
                
        for k in range(5):
            y_true = true_stages[:, k]
            y_pred_m = pred_stages_model[:, k]
            y_pred_p = pred_stages_pers[:, k]
            y_pred_mk = pred_stages_markov[:, k]
            
            f1_m = f1_score(y_true, y_pred_m, average='macro', zero_division=0)
            f1_p = f1_score(y_true, y_pred_p, average='macro', zero_division=0)
            f1_mk = f1_score(y_true, y_pred_mk, average='macro', zero_division=0)
            
            all_f1_model[k+1].append(f1_m)
            all_f1_pers[k+1].append(f1_p)
            all_f1_markov[k+1].append(f1_mk)

    # 4. Report
    print("=== Horizon Predictability Analysis ===")
    print("K\tModel_F1\tSpread\tPers_F1\tMarkov_F1\tBeat_Baselines?")
    
    h_star = 0
    for k in range(1, 6):
        mean_m = np.mean(all_f1_model[k])
        std_m = np.std(all_f1_model[k])
        mean_p = np.mean(all_f1_pers[k])
        mean_mk = np.mean(all_f1_markov[k])
        
        beats = (mean_m > mean_p) and (mean_m > mean_mk)
        if beats:
            h_star = k
            
        print(f"{k}\t{mean_m:.4f}\t\t{std_m:.4f}\t{mean_p:.4f}\t{mean_mk:.4f}\t\t{beats}")
        
    print(f"\nPredictability Horizon H* = {h_star}")

if __name__ == "__main__":
    main()
