import os
import sys
import pandas as pd
import numpy as np
import torch
import argparse
from pathlib import Path

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, str(workspace_dir))

from ml1.lstm.model import LSTMMixtureDensityWorldModel
from ml1.lstm.ucs import UCSConfig, validate_ucs_windows

def main():
    parser = argparse.ArgumentParser(description="Analyze MDN mixture weights for known-benign vs known-attack windows.")
    parser.add_argument("--checkpoint", type=Path, default=Path("ml1/artifacts/lstm/mdn_v1/mdn_next_state_best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data-engineering/data/ucs/ucs_windows.parquet"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_path = workspace_dir / args.data
    if not data_path.exists():
        data_path = workspace_dir / "data/ucs/ucs_windows.parquet"
        
    print(f"Loading data from {data_path}...")
    windows = pd.read_parquet(data_path)
    config = UCSConfig()
    features = validate_ucs_windows(windows, config=config)

    ckpt_path = workspace_dir / args.checkpoint
    print(f"Loading MDN checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = ckpt.get('model_state_dict', ckpt)

    num_components = model_state['pi_head.2.bias'].shape[0]
    lstm_weight = model_state.get('lstm.weight_ih_l0')
    hidden_size = lstm_weight.shape[0] // 4
    num_layers = max([int(k.split('_l')[1]) for k in model_state.keys() if k.startswith('lstm.weight_ih_l')]) + 1

    model = LSTMMixtureDensityWorldModel(
        input_size=len(features),
        hidden_size=hidden_size,
        state_dim=len(features),
        num_layers=num_layers,
        dropout=0.2,
        num_components=num_components
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # Sample 100 benign and 100 attack windows
    np.random.seed(42)
    benign = windows[windows['label_binary'] == 0].sample(100, random_state=42)
    attack = windows[windows['label_binary'] == 1].sample(100, random_state=42)

    def extract_pi(df):
        lookback = config.lookback_windows
        idx_map = {idx: pos for pos, idx in enumerate(windows.index)}
        full_features_np = windows[features].to_numpy(dtype=np.float32)
        
        X_list = []
        for idx in df.index:
            t_pos = idx_map[idx]
            start = t_pos - lookback + 1
            if start < 0 or t_pos >= len(windows):
                continue
            hist = full_features_np[start:t_pos+1]
            if len(hist) == lookback:
                X_list.append(hist)
        
        if not X_list:
            return np.zeros((0, num_components))
            
        X_arr = np.array(X_list, dtype=np.float32)
        with torch.no_grad():
            pi, _, _ = model(torch.tensor(X_arr).to(device))
        return pi.cpu().numpy()

    pi_benign = extract_pi(benign)
    pi_attack = extract_pi(attack)

    avg_benign = np.mean(pi_benign, axis=0)
    avg_attack = np.mean(pi_attack, axis=0)

    print("\n--- MDN Mixture Weight Diagnostic ---")
    print(f"Condition | " + " | ".join([f"pi_{i+1}" for i in range(num_components)]))
    print("-" * 50)
    print(f"Benign    | " + " | ".join([f"{p*100:.2f}%" for p in avg_benign]))
    print(f"Attack    | " + " | ".join([f"{p*100:.2f}%" for p in avg_attack]))

if __name__ == "__main__":
    main()
