import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

workspace_dir = Path("d:/sih2026").resolve()
sys.path.insert(0, str(workspace_dir))

from ml1.lstm.model import LSTMGaussianWorldModel
from ml1.scripts.train_stage_head import StageClassificationHead
from evaluation.horizon_analysis.horizon_analysis import load_attack_stage_mapping, assign_stage
from ml1.lstm.ucs import UCSConfig, validate_ucs_windows

data_path = workspace_dir / "data-engineering/data/ucs/ucs_windows.parquet"
windows = pd.read_parquet(data_path)
features = validate_ucs_windows(windows, config=UCSConfig())

world_model = LSTMGaussianWorldModel(input_size=len(features), hidden_size=128, state_dim=len(features), num_layers=2, dropout=0.2)
ckpt = torch.load(workspace_dir / "ml1/artifacts/lstm/sweep_v1/hidden128_L2_low_lr/gaussian_next_state_best.pt", map_location='cpu', weights_only=False)
world_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
world_model.eval()

stage_head = StageClassificationHead(state_dim=len(features), num_classes=6, hidden_dim=64, dropout=0.2)
st_ckpt = torch.load(workspace_dir / "ml1/artifacts/lstm/stage_head_v3/stage_head_best.pt", map_location='cpu', weights_only=False)
stage_head.load_state_dict(st_ckpt)
stage_head.eval()

test_data = torch.tensor(windows[features].values[:1000], dtype=torch.float32).unsqueeze(1)
with torch.no_grad():
    mu, _ = world_model(test_data)
    preds = stage_head(mu).argmax(dim=-1)
    
print("Predictions distribution:", np.bincount(preds.numpy()))
