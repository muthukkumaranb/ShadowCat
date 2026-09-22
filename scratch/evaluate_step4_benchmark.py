"""
Step 4: Re-train Detection and Evaluate LSTM vs LR under Standardized PCA
"""
import os
import sys
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, average_precision_score
from tqdm import tqdm

repo_root = Path(__file__).resolve().parent.parent
ml1_dir = repo_root / "ml1"
sys.path.insert(0, str(ml1_dir))

from lstm.model import LSTMClassifier
from lstm.ucs import UCSConfig, validate_ucs_windows
from lstm.utils import get_device, set_seed

def train_classifier(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64,
                     lr=0.001, patience=5, device='cpu', checkpoint_path=None):
    train_dataset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), 
                                  torch.as_tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.as_tensor(X_val, dtype=torch.float32), 
                                torch.as_tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model.forward_logits(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_b.size(0)
            
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model.forward_logits(X_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * X_b.size(0)
                
        val_loss /= len(val_dataset)
        
        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            early_stop_counter = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                break
                
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model, best_val_loss, train_loss

def main():
    set_seed(42)
    device = 'cpu'
    
    data_path = ml1_dir / 'data' / 'ucs' / 'ucs_windows.parquet'
    manifest_path = ml1_dir / 'artifacts' / 'loeo' / 'corrected_37fold_manifest.json'
    pca_path = repo_root / 'models' / 'pca_32.pkl'
    
    print("Loading data, manifest, and PCA...")
    df = pd.read_parquet(data_path).sort_values('window_start_utc').reset_index(drop=True)
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    with open(pca_path, 'rb') as f:
        pca_artifact = pickle.load(f)
        
    pca = pca_artifact['pca']
    pca_scaler = pca_artifact['scaler']
    features = [f"pca_{i}" for i in range(32)]
    
    config = UCSConfig()
    base_features = validate_ucs_windows(df, config=config)
    
    # Pre-transform all features
    raw_vals = df[base_features].to_numpy(dtype=np.float64)
    scaled_vals = pca_scaler.transform(raw_vals)
    scaled_frame = pd.DataFrame(scaled_vals, columns=base_features, index=df.index)
    trans_frame = pca.transform(scaled_frame)
    full_transformed_np = trans_frame[features].to_numpy(dtype=np.float32)
    
    det_out_dir = ml1_dir / 'artifacts' / 'lstm' / 'detection_v5'
    det_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Train & evaluate Detection (label_binary) across 37 folds
    print("\n--- Training Detection Models on Standardized PCA (37 folds) ---")
    det_metrics_050 = []
    det_metrics_038 = []
    
    for fold in tqdm(manifest['folds'], desc="Detection Folds"):
        fold_id = fold['fold_id']
        held_out_episode = fold['held_out_episode_id']
        train_indices = np.array(fold['train_indices'])
        test_indices = fold['test_indices']
        attack_type = df.loc[df['episode_id'] == held_out_episode, 'label_attack_type'].iloc[0]
        
        # 80/20 train/val split
        train_df_sorted = df.loc[train_indices].sort_values('window_start_utc')
        val_start = int(len(train_df_sorted) * 0.8)
        actual_train_idx = train_df_sorted.index[:val_start]
        actual_val_idx = train_df_sorted.index[val_start:]
        
        def extract_seqs(indices, target_col):
            X, y = [], []
            targets_array = df[target_col].to_numpy(dtype=np.float32)
            idx_map = {idx: pos for pos, idx in enumerate(df.index)}
            for idx in indices:
                t_pos = idx_map[idx]
                start = t_pos - config.lookback_windows + 1
                if start < 0 or t_pos >= len(df):
                    continue
                hist = full_transformed_np[start:t_pos+1]
                if len(hist) == config.lookback_windows:
                    X.append(hist)
                    y.append(targets_array[t_pos])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
            
        X_train, y_train = extract_seqs(actual_train_idx, 'label_binary')
        X_val, y_val = extract_seqs(actual_val_idx, 'label_binary')
        X_test, y_test = extract_seqs(test_indices, 'label_binary')
        
        ckpt = det_out_dir / f"model_fold_{fold_id}.pt"
        model = LSTMClassifier(input_size=32, hidden_size=64, num_layers=1, dropout=0.20)
        
        if not ckpt.exists():
            model, _, _ = train_classifier(model, X_train, y_train, X_val, y_val,
                                           epochs=100, batch_size=64, lr=0.001, patience=5,
                                           device=device, checkpoint_path=ckpt)
        else:
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            
        model.eval()
        with torch.no_grad():
            probs = model.predict_proba(torch.as_tensor(X_test, dtype=torch.float32)).numpy()
            
        y_test_int = y_test.astype(int)
        roc = float(roc_auc_score(y_test_int, probs)) if len(np.unique(y_test_int)) > 1 else None
        pr = float(average_precision_score(y_test_int, probs)) if len(np.unique(y_test_int)) > 1 else None
        
        for thresh, m_list in [(0.50, det_metrics_050), (0.38, det_metrics_038)]:
            preds = (probs >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test_int, preds, labels=[0, 1]).ravel()
            f1 = float(f1_score(y_test_int, preds, zero_division=0))
            prec = float(precision_score(y_test_int, preds, zero_division=0))
            rec = float(recall_score(y_test_int, preds, zero_division=0))
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            
            m_list.append({
                "fold_id": fold_id,
                "held_out_episode_id": held_out_episode,
                "attack_type": attack_type,
                "target": "label_binary",
                "train_sequences": len(X_train),
                "validation_sequences": len(X_val),
                "test_sequences": len(X_test),
                "f1": f1,
                "pr_auc": pr,
                "precision": prec,
                "recall": rec,
                "fpr": fpr,
                "roc_auc": roc,
            })
            
    # 2. Evaluate Onset from hazard_head_v5 H5 across 37 folds
    print("\n--- Evaluating Onset Models (hazard_head_v5 H5) across 37 folds ---")
    onset_metrics_050 = []
    onset_metrics_038 = []
    
    for fold in tqdm(manifest['folds'], desc="Onset Folds"):
        fold_id = fold['fold_id']
        held_out_episode = fold['held_out_episode_id']
        train_indices = np.array(fold['train_indices'])
        test_indices = fold['test_indices']
        attack_type = df.loc[df['episode_id'] == held_out_episode, 'label_attack_type'].iloc[0]
        
        X_test, y_test = extract_seqs(test_indices, 'future_attack_label')
        ckpt = ml1_dir / 'artifacts' / 'lstm' / 'hazard_head_v5' / 'H5' / f"model_fold_{fold_id}.pt"
        
        model = LSTMClassifier(input_size=32, hidden_size=64, num_layers=1, dropout=0.20)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        
        with torch.no_grad():
            probs = model.predict_proba(torch.as_tensor(X_test, dtype=torch.float32)).numpy()
            
        y_test_int = y_test.astype(int)
        roc = float(roc_auc_score(y_test_int, probs)) if len(np.unique(y_test_int)) > 1 else None
        pr = float(average_precision_score(y_test_int, probs)) if len(np.unique(y_test_int)) > 1 else None
        
        for thresh, m_list in [(0.50, onset_metrics_050), (0.38, onset_metrics_038)]:
            preds = (probs >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test_int, preds, labels=[0, 1]).ravel()
            f1 = float(f1_score(y_test_int, preds, zero_division=0))
            prec = float(precision_score(y_test_int, preds, zero_division=0))
            rec = float(recall_score(y_test_int, preds, zero_division=0))
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            
            m_list.append({
                "fold_id": fold_id,
                "held_out_episode_id": held_out_episode,
                "attack_type": attack_type,
                "target": "future_attack_label",
                "train_sequences": len(X_train),
                "validation_sequences": len(X_val),
                "test_sequences": len(X_test),
                "f1": f1,
                "pr_auc": pr,
                "precision": prec,
                "recall": rec,
                "fpr": fpr,
                "roc_auc": roc,
            })
            
    # Save the fresh evaluation files
    # We will write both thresh 0.50 and thresh 0.38 summaries
    lstm_dir = ml1_dir / 'artifacts' / 'lstm'
    
    # Save default (thresh 0.38 to match the original evaluation protocol, or thresh 0.50)
    # Let's save both versions
    pd.DataFrame(det_metrics_038).to_csv(lstm_dir / 'lstm_detection_loeo_folds_t38.csv', index=False)
    pd.DataFrame(onset_metrics_038).to_csv(lstm_dir / 'lstm_onset_loeo_folds_t38.csv', index=False)
    
    pd.DataFrame(det_metrics_050).to_csv(lstm_dir / 'lstm_detection_loeo_folds_t50.csv', index=False)
    pd.DataFrame(onset_metrics_050).to_csv(lstm_dir / 'lstm_onset_loeo_folds_t50.csv', index=False)
    
    # Write the canonical target files expected by compare_results.py
    # Using t38 (which is what run_lstm_frozen.py originally used) or t50
    # Let's check both in compare_results!
    print("\n--- RESULTS AT THRESHOLD 0.38 (Original LSTM Protocol Threshold) ---")
    df_det_38 = pd.DataFrame(det_metrics_038)
    df_onset_38 = pd.DataFrame(onset_metrics_038)
    print("Detection (All): F1=", df_det_38['f1'].mean(), "Prec=", df_det_38['precision'].mean(), "Rec=", df_det_38['recall'].mean(), "FPR=", df_det_38['fpr'].mean())
    print("Onset (All):     F1=", df_onset_38['f1'].mean(), "Prec=", df_onset_38['precision'].mean(), "Rec=", df_onset_38['recall'].mean(), "FPR=", df_onset_38['fpr'].mean())
    
    print("\n--- RESULTS AT THRESHOLD 0.50 (Standard Calibration Threshold) ---")
    df_det_50 = pd.DataFrame(det_metrics_050)
    df_onset_50 = pd.DataFrame(onset_metrics_050)
    print("Detection (All): F1=", df_det_50['f1'].mean(), "Prec=", df_det_50['precision'].mean(), "Rec=", df_det_50['recall'].mean(), "FPR=", df_det_50['fpr'].mean())
    print("Onset (All):     F1=", df_onset_50['f1'].mean(), "Prec=", df_onset_50['precision'].mean(), "Rec=", df_onset_50['recall'].mean(), "FPR=", df_onset_50['fpr'].mean())
    
    # Update canonical artifacts/lstm/lstm_detection_loeo_folds.csv and lstm_onset_loeo_folds.csv
    # with the retrained models at threshold 0.38 (or 0.50)
    pd.DataFrame(det_metrics_038).to_csv(lstm_dir / 'lstm_detection_loeo_folds.csv', index=False)
    pd.DataFrame(onset_metrics_038).to_csv(lstm_dir / 'lstm_onset_loeo_folds.csv', index=False)
    print("\nCanonical files updated at artifacts/lstm/lstm_detection_loeo_folds.csv and lstm_onset_loeo_folds.csv")

if __name__ == '__main__':
    main()
