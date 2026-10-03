import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import sys
from pathlib import Path

WORKSPACE_DIR = Path("d:/sih2026")
sys.path.insert(0, str(WORKSPACE_DIR))

import torch.nn as nn
from ml1.lstm.ucs import UCSConfig, apply_training_only_pca, build_lstm_sequences, purge_and_embargo, validate_ucs_windows
from ml1.lstm.utils import get_device

class ResidualLSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.20):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward_logits(self, x: torch.Tensor, base_logit: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_step = self.dropout(output[:, -1, :])
        residual = self.fc(last_step).squeeze(-1)
        return base_logit + residual

    def forward(self, x: torch.Tensor, base_logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x, base_logit))

SEED = 42
PCA_COMPONENTS = 32
THRESHOLD = 0.7844065427780151

def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(f1), float(precision), float(recall), float(fpr)

def main():
    data_path = WORKSPACE_DIR / "ml1/data/ucs/ucs_windows.parquet"
    manifest_path = WORKSPACE_DIR / "ml1/artifacts/loeo/corrected_37fold_manifest.json"
    feature_manifest_path = WORKSPACE_DIR / "ml1/artifacts/lr_set_a/set_a_features.json"
    
    df = pd.read_parquet(data_path)
    df["window_start_utc"] = pd.to_datetime(df["window_start_utc"], utc=True)
    df = df.sort_values("window_start_utc").reset_index(drop=True)
    
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(feature_manifest_path, encoding="utf-8") as f:
        feature_manifest = json.load(f)
        
    base_features = list(feature_manifest["ordered_feature_names"])
    config = UCSConfig()
    
    device = get_device()
    
    all_y_true = []
    all_normal_prob = []
    all_perm_prob_sum = []
    all_repeat_prob = []
    
    folds = manifest["folds"]
    
    np.random.seed(SEED)
    
    for position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        checkpoint_path = WORKSPACE_DIR / f"ml1/artifacts/lstm/lstm_stacked/onset/model_fold_{fold_id}.pt"
        if not checkpoint_path.is_file():
            print(f"Skipping fold {fold_id}, no checkpoint.")
            continue
            
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        test_indices = np.asarray(fold["test_indices"], dtype=int)
        
        fold_frame = df.copy()
        fold_frame["split"] = "none"
        fold_frame.loc[train_indices, "split"] = "train"
        fold_frame.loc[test_indices, "split"] = "test"
        
        train_rows = fold_frame.loc[train_indices].sort_values("window_start_utc")
        val_start = int(len(train_rows) * 0.8)
        fold_frame.loc[train_rows.index[val_start:], "split"] = "val"
        
        active_mask = fold_frame["split"].isin(["train", "val", "test"])
        purged = purge_and_embargo(fold_frame[active_mask].copy(), config=config)
        transformed, pca = apply_training_only_pca(purged, base_features, PCA_COMPONENTS, random_state=SEED)
        pca_features = [f"pca_{index}" for index in range(PCA_COMPONENTS)]
        
        full_transformed = pca.transform(fold_frame)
        full_values = full_transformed[pca_features].to_numpy(dtype=np.float32)
        
        positions = {index: pos for pos, index in enumerate(fold_frame.index)}
        
        test_histories, test_labels = [], []
        for index in test_indices:
            pos = positions[index]
            start = pos - config.lookback_windows + 1
            if start < 0: continue
            history = full_values[start:pos + 1]
            if len(history) == config.lookback_windows:
                test_histories.append(history)
                test_labels.append(int(fold_frame.loc[index, "future_attack_label"]))
                
        if not test_histories:
            continue
            
        X = np.asarray(test_histories, dtype=np.float32)
        y = np.asarray(test_labels, dtype=int)
        all_y_true.extend(y)
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        T = checkpoint["temperature"]
        lr_coef = np.array(checkpoint["lr_coef"])
        lr_intercept = np.array(checkpoint["lr_intercept"])
        scaler_mean = np.array(checkpoint["scaler_mean"])
        scaler_scale = np.array(checkpoint["scaler_scale"])
        lr_features = checkpoint["lr_features"]
        
        test_df = fold_frame.loc[test_indices]
        x_lr_test = (test_df[lr_features].to_numpy(dtype=float) - scaler_mean) / scaler_scale
        base_logits = (x_lr_test @ lr_coef.T + lr_intercept).reshape(-1)
        
        model = ResidualLSTMClassifier(input_size=PCA_COMPONENTS, hidden_size=64, dropout=0.20)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        base_logits_t = torch.as_tensor(base_logits, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            # a) Normal
            normal_logits = model.forward_logits(torch.as_tensor(X, device=device), base_logits_t)
            normal_prob = torch.sigmoid(normal_logits / T).cpu().numpy().reshape(-1)
            all_normal_prob.extend(normal_prob)
            
            # b) Permuted 5 times
            perm_prob_sum = np.zeros(len(y))
            for _ in range(5):
                X_perm = X.copy()
                for i in range(len(X_perm)):
                    np.random.shuffle(X_perm[i])
                logits = model.forward_logits(torch.as_tensor(X_perm, device=device), base_logits_t)
                perm_prob = torch.sigmoid(logits / T).cpu().numpy().reshape(-1)
                perm_prob_sum += perm_prob
            all_perm_prob_sum.extend(perm_prob_sum / 5.0)
            
            # c) Repeat last window
            X_rep = X.copy()
            for i in range(len(X_rep)):
                X_rep[i] = np.repeat(X[i, -1:], config.lookback_windows, axis=0)
            logits = model.forward_logits(torch.as_tensor(X_rep, device=device), base_logits_t)
            rep_prob = torch.sigmoid(logits / T).cpu().numpy().reshape(-1)
            all_repeat_prob.extend(rep_prob)

    all_y_true = np.array(all_y_true)
    print(f"Total samples: {len(all_y_true)}, Positives: {all_y_true.sum()}")
    print(f"Normal Prob Max: {np.max(all_normal_prob)}")
    
    f1_a, p_a, r_a, fpr_a = compute_metrics(all_y_true, np.array(all_normal_prob), THRESHOLD)
    f1_b, p_b, r_b, fpr_b = compute_metrics(all_y_true, np.array(all_perm_prob_sum), THRESHOLD)
    f1_c, p_c, r_c, fpr_c = compute_metrics(all_y_true, np.array(all_repeat_prob), THRESHOLD)
    
    res = {
        "normal": {"F1": f1_a, "Precision": p_a, "Recall": r_a, "FPR": fpr_a},
        "permuted": {"F1": f1_b, "Precision": p_b, "Recall": r_b, "FPR": fpr_b},
        "repeated_last": {"F1": f1_c, "Precision": p_c, "Recall": r_c, "FPR": fpr_c}
    }
    
    out = Path("d:/sih2026/evaluation/ablation/history_shuffle_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=4)
        
    print("Done")

if __name__ == "__main__":
    main()
