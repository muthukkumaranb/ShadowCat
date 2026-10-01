import os
import sys
import json
import argparse
from pathlib import Path

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))
sys.path.insert(0, str(workspace_dir / "backend"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support

from ml1.lstm.model import LSTMGaussianWorldModel
from ml1.lstm.ucs import UCSConfig, validate_ucs_windows, build_next_state_sequences, load_ucs_windows, purge_and_embargo
from ml1.lstm.utils import get_device, set_seed
from backend.models import MIMOStageClassificationHead, StageClassificationHead

def load_attack_stage_mapping(yaml_path: Path):
    import yaml
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    mappings = {}
    for attack_name, info in data.get('mappings', {}).items():
        tactic = info.get('tactic', 'Unknown/Other')
        if '/' in tactic and 'Credential Access' in tactic:
            tactic = 'Credential Access'
        mappings[attack_name] = tactic
    return mappings

def main():
    set_seed(42)
    device = get_device()
    
    data_path = workspace_dir / "data-engineering/data/ucs/ucs_windows.parquet"
    mapping_path = workspace_dir / "data-engineering/data/ucs/attack_tactics_mapping.yaml"
    wm_ckpt = workspace_dir / "ml1/artifacts/lstm/probabilistic_world_model_v2/gaussian_next_state_best.pt"
    if not wm_ckpt.exists():
        wm_ckpt = workspace_dir / "ml1/artifacts/lstm/gaussian_next_state_best_v3.pt"
    if not wm_ckpt.exists():
        wm_ckpt = workspace_dir / "ml1/artifacts/lstm/gaussian_next_state_best_v2.pt"

    print("Loading data...")
    windows = load_ucs_windows(data_path)
    config = UCSConfig()
    features = validate_ucs_windows(windows, config=config)
    tactic_map = load_attack_stage_mapping(mapping_path)

    def assign_stage(row):
        if row['label_binary'] == 0:
            return "Unknown/Other"
        return tactic_map.get(row.get('label_attack_type', 'Benign'), "Unknown/Other")

    windows['stage_label'] = windows.apply(assign_stage, axis=1)
    stage_classes = sorted(list(set(windows['stage_label'].unique())))
    if "Unknown/Other" not in stage_classes:
        stage_classes.append("Unknown/Other")
    stage_classes = sorted(stage_classes)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(stage_classes)}
    windows['stage_idx'] = windows['stage_label'].map(class_to_idx)

    purged = purge_and_embargo(windows, config=config)
    sequences = build_next_state_sequences(purged, features=features, config=config)
    
    world_model = LSTMGaussianWorldModel(
        input_size=len(features),
        hidden_size=64,
        state_dim=len(features),
        num_layers=1,
        dropout=0.2
    )
    ckpt = torch.load(wm_ckpt, map_location=device, weights_only=False)
    world_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    world_model.to(device)
    world_model.eval()

    time_to_stage = dict(zip(pd.to_datetime(purged['window_start_utc'], utc=True), purged['stage_idx']))

    def get_mimo_data(seq_set):
        X_tensor = torch.as_tensor(seq_set.X, dtype=torch.float32).to(device)
        with torch.no_grad():
            output, _ = world_model.lstm(X_tensor)
            z_t = world_model.dropout(output[:, -1, :]).cpu().numpy()
        
        y_stages = []
        for ts in seq_set.timestamps:
            stages_k = []
            for k in range(5):
                ts_k = ts + pd.Timedelta(minutes=k)
                stages_k.append(time_to_stage.get(ts_k, class_to_idx["Unknown/Other"]))
            y_stages.append(stages_k)
        return z_t, np.array(y_stages, dtype=int), seq_set.X

    z_train, y_train, X_train = get_mimo_data(sequences['train'])
    z_val, y_val, X_val = get_mimo_data(sequences['val'])
    z_test, y_test, X_test = get_mimo_data(sequences['test'])

    print("Training MIMO Stage Head...")
    mimo_head = MIMOStageClassificationHead(latent_dim=64, num_classes=len(stage_classes), horizons=5)
    mimo_head.to(device)
    optimizer = torch.optim.Adam(mimo_head.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    tr_ds = TensorDataset(torch.as_tensor(z_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.long))
    tr_ld = DataLoader(tr_ds, batch_size=64, shuffle=True)
    
    for epoch in range(15):
        mimo_head.train()
        for zb, yb in tr_ld:
            zb, yb = zb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = mimo_head(zb) # (B, 5, num_classes)
            loss = criterion(logits.view(-1, len(stage_classes)), yb.view(-1))
            loss.backward()
            optimizer.step()
    
    # Load original stage head
    st_path = workspace_dir / "ml1/artifacts/lstm/stage_head_v3/reeval_packetcov/stage_head_best.pt"
    if not st_path.exists():
        st_path = workspace_dir / "ml1/artifacts/lstm/stage_head_v3/stage_head_best.pt"
    if not st_path.exists():
        st_path = workspace_dir / "ml1/artifacts/lstm/stage_head/stage_head_best.pt"
    
    orig_head = StageClassificationHead(state_dim=len(features), num_classes=len(stage_classes))
    if st_path.exists():
        orig_head.load_state_dict(torch.load(st_path, map_location=device, weights_only=False))
    orig_head.to(device)
    orig_head.eval()

    print("Evaluating K=4 and K=5...")
    mimo_head.eval()
    
    # Recursive rollout for orig head
    X_t = torch.as_tensor(X_test, dtype=torch.float32).to(device)
    orig_preds = {4: [], 5: []}
    
    with torch.no_grad():
        for i in range(len(X_t)):
            seq = X_t[i:i+1]
            for k in range(5):
                mean, _ = world_model(seq)
                if k == 3: # K=4
                    logits = orig_head(mean)
                    orig_preds[4].append(logits.argmax(-1).item())
                if k == 4: # K=5
                    logits = orig_head(mean)
                    orig_preds[5].append(logits.argmax(-1).item())
                seq = torch.cat([seq[:, 1:, :], mean.unsqueeze(1)], dim=1)
                
        # MIMO predictions
        mimo_logits = mimo_head(torch.as_tensor(z_test, dtype=torch.float32).to(device))
        mimo_preds = mimo_logits.argmax(-1).cpu().numpy() # (N, 5)

    for k_idx, k_val in [(3, 4), (4, 5)]:
        y_true_k = y_test[:, k_idx]
        orig_p = np.array(orig_preds[k_val])
        mimo_p = mimo_preds[:, k_idx]
        
        _, _, f1_orig, _ = precision_recall_fscore_support(y_true_k, orig_p, average='weighted', zero_division=0)
        _, _, f1_mimo, _ = precision_recall_fscore_support(y_true_k, mimo_p, average='weighted', zero_division=0)
        _, _, f1_orig_macro, _ = precision_recall_fscore_support(y_true_k, orig_p, average='macro', zero_division=0)
        _, _, f1_mimo_macro, _ = precision_recall_fscore_support(y_true_k, mimo_p, average='macro', zero_division=0)
        
        _, _, f1_orig_cls, _ = precision_recall_fscore_support(y_true_k, orig_p, average=None, labels=range(len(stage_classes)), zero_division=0)
        _, _, f1_mimo_cls, _ = precision_recall_fscore_support(y_true_k, mimo_p, average=None, labels=range(len(stage_classes)), zero_division=0)
        
        print(f"\n=== Horizon K={k_val} ===")
        print(f"{'Class Name':<25} | {'Recursive F1':<12} | {'MIMO F1':<10} | {'Delta':<8}")
        print("-" * 65)
        for i, cls_name in enumerate(stage_classes):
            orig_val = f1_orig_cls[i]
            mimo_val = f1_mimo_cls[i]
            delta = mimo_val - orig_val
            print(f"{cls_name:<25} | {orig_val:<12.4f} | {mimo_val:<10.4f} | {delta:<8.4f}")
        print("-" * 65)
        print(f"{'Weighted Average':<25} | {f1_orig:<12.4f} | {f1_mimo:<10.4f} | {f1_mimo - f1_orig:<8.4f}")
        print(f"{'Macro Average':<25} | {f1_orig_macro:<12.4f} | {f1_mimo_macro:<10.4f} | {f1_mimo_macro - f1_orig_macro:<8.4f}\n")

if __name__ == '__main__':
    main()

