import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))
sys.path.insert(0, str(workspace_dir / "backend"))

from ml1.lstm.model import LSTMGaussianWorldModel
from ml1.lstm.ucs import UCSConfig, validate_ucs_windows, build_next_state_sequences, load_ucs_windows, purge_and_embargo
from ml1.lstm.utils import get_device, set_seed
from backend.models import MIMOStageClassificationHead, StageClassificationHead

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

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

def train_mimo_head(z_train, y_train, loss_fn, stage_classes, device, epochs=15):
    head = MIMOStageClassificationHead(latent_dim=64, num_classes=len(stage_classes), horizons=5).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
    ds = TensorDataset(torch.as_tensor(z_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.long))
    ld = DataLoader(ds, batch_size=64, shuffle=True)
    for epoch in range(epochs):
        head.train()
        for zb, yb in ld:
            zb, yb = zb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = head(zb)
            loss = loss_fn(logits.view(-1, len(stage_classes)), yb.view(-1))
            loss.backward()
            optimizer.step()
    head.eval()
    return head

def train_recursive_head(S_hat_train, y_train_k1, loss_fn, stage_classes, state_dim, device, epochs=15):
    head = StageClassificationHead(state_dim=state_dim, num_classes=len(stage_classes)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
    ds = TensorDataset(torch.as_tensor(S_hat_train, dtype=torch.float32), torch.as_tensor(y_train_k1, dtype=torch.long))
    ld = DataLoader(ds, batch_size=64, shuffle=True)
    for epoch in range(epochs):
        head.train()
        for sb, yb in ld:
            sb, yb = sb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = head(sb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
    head.eval()
    return head

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
        if row['label_binary'] == 0: return "Unknown/Other"
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
        input_size=len(features), hidden_size=64, state_dim=len(features),
        num_layers=1, dropout=0.2
    ).to(device)
    ckpt = torch.load(wm_ckpt, map_location=device, weights_only=False)
    world_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    world_model.eval()

    time_to_stage = dict(zip(pd.to_datetime(purged['window_start_utc'], utc=True), purged['stage_idx']))

    def get_data(seq_set):
        X_tensor = torch.as_tensor(seq_set.X, dtype=torch.float32).to(device)
        with torch.no_grad():
            mean, _ = world_model(X_tensor)
            output, _ = world_model.lstm(X_tensor)
            z_t = world_model.dropout(output[:, -1, :]).cpu().numpy()
            s_hat = mean.cpu().numpy()
        
        y_stages = []
        for ts in seq_set.timestamps:
            stages_k = []
            for k in range(5):
                ts_k = ts + pd.Timedelta(minutes=k)
                stages_k.append(time_to_stage.get(ts_k, class_to_idx["Unknown/Other"]))
            y_stages.append(stages_k)
        return z_t, s_hat, np.array(y_stages, dtype=int), seq_set.X

    z_train, s_hat_train, y_train, X_train = get_data(sequences['train'])
    z_test, s_hat_test, y_test, X_test = get_data(sequences['test'])

    print("\n--- TASK 4: Training Set Raw Class Counts BEFORE Augmentation ---")
    for k_idx, k_val in [(3, 4), (4, 5)]:
        counts = np.bincount(y_train[:, k_idx], minlength=len(stage_classes))
        print(f"K={k_val} Counts:")
        for i, c in enumerate(stage_classes):
            print(f"  {c}: {counts[i]}")

    # TASK 1: Data augmentation for rare attack classes at K=4/K=5
    # Approach: Time-warping / jittering existing short sequences.
    # We find sequences where a rare class attack occurs in K=1-3, and we synthesize continuations into K=4/K=5 by 
    # injecting small Gaussian noise to the latent representation z_t and carrying the attack class forward,
    # compensating for the chronological purge that truncated these episodes.
    rare_classes = ["Command and Control", "Discovery", "Impact", "Credential Access"]
    rare_class_indices = set(class_to_idx[c] for c in rare_classes if c in class_to_idx)
    unknown_idx = class_to_idx["Unknown/Other"]

    z_aug = []
    s_hat_aug = []
    y_aug = []

    for i in range(len(y_train)):
        # Check if sequence contains a rare class
        seq_classes = set(y_train[i])
        if seq_classes.intersection(rare_class_indices):
            # Create jittered copies (e.g., 5 copies per sequence to boost counts)
            for _ in range(5):
                # Realistic bounds: std of z_train is ~0.1, we inject 0.05 noise
                z_synth = z_train[i] + np.random.normal(0, 0.05, size=z_train[i].shape)
                s_hat_synth = s_hat_train[i] + np.random.normal(0, 0.05, size=s_hat_train[i].shape)
                y_synth = y_train[i].copy()
                
                # Carry forward the last known attack stage into K=4/5 if it became Unknown/Other
                last_known = None
                for k in range(5):
                    if y_synth[k] in rare_class_indices:
                        last_known = y_synth[k]
                    elif y_synth[k] == unknown_idx and last_known is not None:
                        y_synth[k] = last_known
                
                z_aug.append(z_synth)
                s_hat_aug.append(s_hat_synth)
                y_aug.append(y_synth)

    if len(z_aug) > 0:
        print(f"\nAugmented with {len(z_aug)} synthetic sequences (time-warped continuations + jitter).")
        z_train = np.vstack([z_train, np.array(z_aug)])
        s_hat_train = np.vstack([s_hat_train, np.array(s_hat_aug)])
        y_train = np.vstack([y_train, np.array(y_aug)])

    print("\n--- TASK 4: Training Set Raw Class Counts AFTER Augmentation ---")
    for k_idx, k_val in [(3, 4), (4, 5)]:
        counts = np.bincount(y_train[:, k_idx], minlength=len(stage_classes))
        print(f"K={k_val} Counts:")
        for i, c in enumerate(stage_classes):
            print(f"  {c}: {counts[i]}")

    # Compute class weights from Y(t+1) (which is y_train[:, 0]) for recursive head
    counts_k1 = np.bincount(y_train[:, 0], minlength=len(stage_classes))
    total_samples = len(y_train)
    weights_np = total_samples / (len(stage_classes) * np.maximum(counts_k1, 1))
    class_weights_t = torch.as_tensor(weights_np, dtype=torch.float32).to(device)

    print("\nTraining Models...")
    # Loss variants
    loss_unweighted = nn.CrossEntropyLoss()
    loss_weighted = nn.CrossEntropyLoss(weight=class_weights_t)
    loss_focal = FocalLoss(alpha=class_weights_t, gamma=2.0)

    # Train MIMO heads
    mimo_unw = train_mimo_head(z_train, y_train, loss_unweighted, stage_classes, device)
    mimo_wht = train_mimo_head(z_train, y_train, loss_weighted, stage_classes, device)
    mimo_foc = train_mimo_head(z_train, y_train, loss_focal, stage_classes, device)

    # Train Recursive heads (on t+1)
    rec_unw = train_recursive_head(s_hat_train, y_train[:, 0], loss_unweighted, stage_classes, len(features), device)
    rec_wht = train_recursive_head(s_hat_train, y_train[:, 0], loss_weighted, stage_classes, len(features), device)
    rec_foc = train_recursive_head(s_hat_train, y_train[:, 0], loss_focal, stage_classes, len(features), device)

    # Evaluate
    print("\nEvaluating K=4 and K=5...")
    X_t = torch.as_tensor(X_test, dtype=torch.float32).to(device)
    
    def rollout_eval(head):
        preds = {4: [], 5: []}
        with torch.no_grad():
            for i in range(len(X_t)):
                seq = X_t[i:i+1]
                for k in range(5):
                    mean, _ = world_model(seq)
                    if k == 3: preds[4].append(head(mean).argmax(-1).item())
                    if k == 4: preds[5].append(head(mean).argmax(-1).item())
                    seq = torch.cat([seq[:, 1:, :], mean.unsqueeze(1)], dim=1)
        return np.array(preds[4]), np.array(preds[5])

    r_unw_p4, r_unw_p5 = rollout_eval(rec_unw)
    r_wht_p4, r_wht_p5 = rollout_eval(rec_wht)
    r_foc_p4, r_foc_p5 = rollout_eval(rec_foc)
    
    with torch.no_grad():
        m_z = torch.as_tensor(z_test, dtype=torch.float32).to(device)
        m_unw_p = mimo_unw(m_z).argmax(-1).cpu().numpy()
        m_wht_p = mimo_wht(m_z).argmax(-1).cpu().numpy()
        m_foc_p = mimo_foc(m_z).argmax(-1).cpu().numpy()

    def get_f1s(y_true, y_pred):
        _, _, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(len(stage_classes)), zero_division=0)
        return f1_cls

    for k_idx, k_val in [(3, 4), (4, 5)]:
        yt = y_test[:, k_idx]
        
        f1_r_unw = get_f1s(yt, r_unw_p4 if k_val==4 else r_unw_p5)
        f1_r_wht = get_f1s(yt, r_wht_p4 if k_val==4 else r_wht_p5)
        f1_r_foc = get_f1s(yt, r_foc_p4 if k_val==4 else r_foc_p5)
        
        f1_m_unw = get_f1s(yt, m_unw_p[:, k_idx])
        f1_m_wht = get_f1s(yt, m_wht_p[:, k_idx])
        f1_m_foc = get_f1s(yt, m_foc_p[:, k_idx])

        print(f"\n=== Horizon K={k_val} ===")
        print(f"{'Class Name':<20} | {'Rec Base':<8} | {'Rec Wght':<8} | {'Rec Focl':<8} | {'MIM Base':<8} | {'MIM Wght':<8} | {'MIM Focl':<8}")
        print("-" * 95)
        for i, cls_name in enumerate(stage_classes):
            print(f"{cls_name[:20]:<20} | {f1_r_unw[i]:<8.4f} | {f1_r_wht[i]:<8.4f} | {f1_r_foc[i]:<8.4f} | {f1_m_unw[i]:<8.4f} | {f1_m_wht[i]:<8.4f} | {f1_m_foc[i]:<8.4f}")
        print("-" * 95)
        
        # Macro averages
        m_r_u = np.mean(f1_r_unw)
        m_r_w = np.mean(f1_r_wht)
        m_r_f = np.mean(f1_r_foc)
        m_m_u = np.mean(f1_m_unw)
        m_m_w = np.mean(f1_m_wht)
        m_m_f = np.mean(f1_m_foc)
        print(f"{'Macro Average':<20} | {m_r_u:<8.4f} | {m_r_w:<8.4f} | {m_r_f:<8.4f} | {m_m_u:<8.4f} | {m_m_w:<8.4f} | {m_m_f:<8.4f}")

if __name__ == '__main__':
    main()

