import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path

workspace_dir = Path(r"d:\sih2026")
sys.path.insert(0, str(workspace_dir))

from ml1.lstm.model import LSTMMixtureDensityWorldModel
from ml1.lstm.ucs import UCSConfig, validate_ucs_windows, build_next_state_sequences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

windows = pd.read_parquet(workspace_dir / "data-engineering/data/ucs/ucs_windows.parquet")
config = UCSConfig()
features = validate_ucs_windows(windows, config=config)

model_state = torch.load(workspace_dir / "ml1/artifacts/lstm/mdn_v1/mdn_next_state_best.pt", map_location=device)['model_state_dict']

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
        if start < 0:
            continue
        hist = full_features_np[start:t_pos+1]
        if len(hist) == lookback:
            X_list.append(hist)
    
    X_arr = np.array(X_list, dtype=np.float32)
    with torch.no_grad():
        pi, _, _ = model(torch.tensor(X_arr).to(device))
    return pi.cpu().numpy()

pi_benign = extract_pi(benign)
pi_attack = extract_pi(attack)

print("Average pi for Benign: ", np.mean(pi_benign, axis=0))
print("Average pi for Attack: ", np.mean(pi_attack, axis=0))
