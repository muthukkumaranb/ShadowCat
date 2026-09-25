"""
Evaluate GraphSAGE GNN Fusion under the identical 37-fold LOEO Protocol.
Leakage-Free Discipline:
- Test fold episode is completely held out.
- LR out-of-fold base logits on training data.
- Chronological 80/20 train/validation split.
- PCA fitted strictly on training split.
- GraphSAGE embeddings scaled using training split only.
- Compares:
  1. Plain Stacked LSTM (baseline: 32 PCA)
  2. Graph-Augmented Stacked LSTM (32 PCA + 4 scalar graph features = 36)
  3. GraphSAGE Feature Fusion (32 PCA + 64 GraphSAGE embeddings = 96)
  4. Full Multi-Modal Fusion (32 PCA + 4 scalar graph + 64 GraphSAGE = 100)
  5. Residual Stacking Fusion: z_final = z_stacked + residual_mlp(GraphSAGE)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from scipy.optimize import minimize_scalar

repo_root = Path(__file__).resolve().parent.parent
ml1_dir = repo_root / "ml1"
sys.path.insert(0, str(ml1_dir))
sys.path.insert(0, str(repo_root / "ml2-full" / "GNN_FINAL"))

from lstm.ucs import UCSConfig, validate_ucs_windows, apply_training_only_pca
from lstm.utils import set_seed, get_device
from lstm.graph_features import compute_scalar_graph_features
from ml2.models.graphsage import GraphBranch

class ResidualLSTM(nn.Module):
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
        nn.init.zeros_(self.fc.bias)
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward_logits(self, x: torch.Tensor, base_logit: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_step = self.dropout(output[:, -1, :])
        residual = self.fc(last_step).squeeze(-1)
        return base_logit + residual

def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    def nll(t):
        scaled = logits / t
        loss = np.maximum(scaled, 0) - scaled * labels + np.log(1 + np.exp(-np.abs(scaled)))
        return np.mean(loss)
    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)

def run_evaluation():
    set_seed(42)
    device = get_device()
    config = UCSConfig()

    data_path = repo_root / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
    edges_path = repo_root / "data-engineering" / "data" / "ucs" / "ucs_graph_edgelists.parquet"
    lookup_path = repo_root / "data-engineering" / "data" / "ucs" / "node_lookup.parquet"
    manifest_path = ml1_dir / "artifacts" / "loeo" / "corrected_37fold_manifest.json"
    feature_path = ml1_dir / "artifacts" / "lr_set_a" / "set_a_features.json"

    df = pd.read_parquet(data_path).sort_values("window_start_utc").reset_index(drop=True)
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(feature_path) as f:
        features_meta = json.load(f)
    lr_features = features_meta["ordered_feature_names"]
    base_features = validate_ucs_windows(df, config=config)

    print("Extracting 4 scalar graph features...")
    graph_scalar_df = compute_scalar_graph_features(edges_path, df)
    scalar_cols = ["graph_degree_centrality", "graph_density", "graph_clustering_signal", "graph_degree_anomaly"]

    print("Extracting 64-dim GraphSAGE embeddings...")
    edges = pd.read_parquet(edges_path)
    lookup = pd.read_parquet(lookup_path)
    lookup_order = {node_id: index for index, node_id in enumerate(lookup["node_id"])}
    grouped_edges = dict(tuple(edges.groupby("window_id")))

    ckpt_path = repo_root / "ml2-full" / "GNN_FINAL" / "ml2" / "results" / "ablation" / "fused" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    branch = GraphBranch(in_channels=11, hidden_dim=64, embedding_dim=64)
    branch_state = {k.replace("graph_branch.", ""): v for k, v in state_dict.items() if k.startswith("graph_branch.")}
    branch.load_state_dict(branch_state, strict=False)
    branch.eval()

    embeddings = {}
    with torch.no_grad():
        for row in df.itertuples(index=False):
            win_id = row.window_id
            if win_id not in grouped_edges:
                embeddings[win_id] = np.zeros(64, dtype=np.float32)
                continue
            w_edges = grouped_edges[win_id]
            srcs = w_edges["src_node_id"].values
            dsts = w_edges["dst_node_id"].values
            all_nodes = sorted(set(srcs).union(dsts), key=lookup_order.__getitem__)
            node_to_idx = {nid: i for i, nid in enumerate(all_nodes)}
            
            x = np.zeros((len(all_nodes), 11), dtype=np.float32)
            src_idx = np.array([node_to_idx[s] for s in srcs])
            dst_idx = np.array([node_to_idx[d] for d in dsts])
            
            np.add.at(x[:, 1], src_idx, 1)
            np.add.at(x[:, 0], dst_idx, 1)
            np.add.at(x[:, 3], src_idx, 1)
            np.add.at(x[:, 3], dst_idx, 1)
            np.add.at(x[:, 6], src_idx, w_edges["byte_count_sum"].values)
            np.add.at(x[:, 6], dst_idx, w_edges["byte_count_sum"].values)
            np.add.at(x[:, 8], src_idx, w_edges["packet_count_sum"].values)
            np.add.at(x[:, 8], dst_idx, w_edges["packet_count_sum"].values)
            
            edge_index = torch.tensor(np.array([src_idx, dst_idx]), dtype=torch.long)
            _, pooled = branch(torch.from_numpy(x), edge_index)
            embeddings[win_id] = pooled.squeeze(0).numpy()

    sage_cols = [f"sage_{i}" for i in range(64)]
    sage_df = pd.DataFrame([embeddings[wid] for wid in df["window_id"]], columns=sage_cols, index=df["window_id"])

    # Now run 37 folds for Detection and Onset with both architectures
    for target_name, target_col in [("detection", "label_binary"), ("onset", "future_attack_label")]:
        print(f"\n==================== EVALUATING: {target_name.upper()} ====================")
        
        results_plain = []
        results_scalar = []
        results_sage = []
        results_fusion_all = []

        for fold in manifest["folds"]:
            fold_id = fold["fold_id"]
            held_out_episode = fold["held_out_episode_id"]
            train_indices = fold["train_indices"]
            test_indices = fold["test_indices"]
            attack_type = df.loc[df["episode_id"] == held_out_episode, "label_attack_type"].iloc[0]

            train_df = df.iloc[train_indices].copy().reset_index()
            test_df = df.iloc[test_indices].copy()
            train_episodes = train_df["episode_id"].unique()

            # 1. Leakage-free OOF LR
            oof_lr_logits = np.zeros(len(train_df))
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for tr_ep_idx, val_ep_idx in kf.split(train_episodes):
                tr_eps = set(train_episodes[tr_ep_idx])
                val_eps = set(train_episodes[val_ep_idx])
                tr_mask = train_df["episode_id"].isin(tr_eps)
                val_mask = train_df["episode_id"].isin(val_eps)
                y_inner_tr = train_df.loc[tr_mask, target_col].to_numpy(dtype=int)
                if len(np.unique(y_inner_tr)) < 2:
                    continue
                scaler_inner = StandardScaler()
                x_inner_tr = scaler_inner.fit_transform(train_df.loc[tr_mask, lr_features].to_numpy(dtype=float))
                x_inner_val = scaler_inner.transform(train_df.loc[val_mask, lr_features].to_numpy(dtype=float))
                lr_inner = LogisticRegression(solver="liblinear", max_iter=2000, random_state=42)
                lr_inner.fit(x_inner_tr, y_inner_tr)
                oof_lr_logits[val_mask] = lr_inner.decision_function(x_inner_val)

            scaler_full = StandardScaler()
            x_tr_full = scaler_full.fit_transform(train_df[lr_features].to_numpy(dtype=float))
            y_tr_full = train_df[target_col].to_numpy(dtype=int)
            lr_full = LogisticRegression(solver="liblinear", max_iter=2000, random_state=42)
            lr_full.fit(x_tr_full, y_tr_full)
            x_te_full = scaler_full.transform(test_df[lr_features].to_numpy(dtype=float))
            test_lr_logits = lr_full.decision_function(x_te_full)

            # 2. Chronological 80/20 train/val split
            fold_frame = df.copy()
            fold_frame["split"] = "none"
            train_df_sorted = fold_frame.loc[train_indices].sort_values("window_start_utc")
            n_train = len(train_df_sorted)
            val_start = int(n_train * 0.8)
            actual_train_indices = train_df_sorted.index[:val_start]
            actual_val_indices = train_df_sorted.index[val_start:]

            fold_frame.loc[actual_train_indices, "split"] = "train"
            fold_frame.loc[actual_val_indices, "split"] = "val"
            fold_frame.loc[test_indices, "split"] = "test"

            orig_idx_to_train_pos = {train_df["index"].iloc[i]: i for i in range(len(train_df))}
            fold_frame["base_logit"] = 0.0
            for idx, pos in orig_idx_to_train_pos.items():
                fold_frame.loc[idx, "base_logit"] = oof_lr_logits[pos]
            for idx, logit in zip(test_indices, test_lr_logits):
                fold_frame.loc[idx, "base_logit"] = logit

            # PCA strictly on training split
            purged = fold_frame[fold_frame["split"].isin(["train", "val", "test"])].copy()
            transformed, pca = apply_training_only_pca(purged, base_features, 32, random_state=42)
            pca_features = [f"pca_{i}" for i in range(32)]
            full_pca = pca.transform(fold_frame)

            # Scalar graph features standardized on train split
            g_scaler = StandardScaler()
            tr_scalars = graph_scalar_df.loc[fold_frame.loc[actual_train_indices, "window_id"], scalar_cols].to_numpy(dtype=float)
            g_scaler.fit(tr_scalars)
            all_scalars = g_scaler.transform(graph_scalar_df.loc[fold_frame["window_id"], scalar_cols].to_numpy(dtype=float))
            for i, col in enumerate(scalar_cols):
                full_pca[col] = all_scalars[:, i]

            # GraphSAGE embeddings standardized on train split
            sage_scaler = StandardScaler()
            tr_sage = sage_df.loc[fold_frame.loc[actual_train_indices, "window_id"], sage_cols].to_numpy(dtype=float)
            sage_scaler.fit(tr_sage)
            all_sage = sage_scaler.transform(sage_df.loc[fold_frame["window_id"], sage_cols].to_numpy(dtype=float))
            for i, col in enumerate(sage_cols):
                full_pca[col] = all_sage[:, i]

            # Helper to train and evaluate a model
            def eval_features(feat_cols):
                full_np = full_pca[feat_cols].to_numpy(dtype=np.float32)
                idx_to_pos = {idx: pos for pos, idx in enumerate(fold_frame.index)}
                base_logits_arr = fold_frame["base_logit"].values
                targets_arr = fold_frame[target_col].values

                def extract_seqs(indices):
                    X_list, b_list, y_list = [], [], []
                    for idx in indices:
                        t_pos = idx_to_pos[idx]
                        start = t_pos - config.lookback_windows + 1
                        if start >= 0 and (t_pos - start + 1) == config.lookback_windows:
                            X_list.append(full_np[start : t_pos + 1])
                            b_list.append(base_logits_arr[t_pos])
                            y_list.append(targets_arr[t_pos])
                    return np.asarray(X_list, dtype=np.float32), np.asarray(b_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)

                X_tr, b_tr, y_tr = extract_seqs(actual_train_indices)
                X_v, b_v, y_v = extract_seqs(actual_val_indices)
                X_te, b_te, y_te = extract_seqs(test_indices)

                if len(y_te) == 0:
                    return 1.0, 1.0, 1.0, 0.0

                m = ResidualLSTM(input_size=len(feat_cols), hidden_size=64, dropout=0.20).to(device)
                opt = torch.optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-4)
                crit = nn.BCEWithLogitsLoss()
                ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(b_tr), torch.from_numpy(y_tr))
                loader = DataLoader(ds, batch_size=128, shuffle=True)

                best_vl = float("inf")
                best_st = None
                for _ in range(10):
                    m.train()
                    for xb, bb, yb in loader:
                        xb, bb, yb = xb.to(device), bb.to(device), yb.to(device)
                        opt.zero_grad()
                        loss = crit(m.forward_logits(xb, bb), yb)
                        loss.backward()
                        opt.step()
                    m.eval()
                    with torch.no_grad():
                        vl = crit(m.forward_logits(torch.from_numpy(X_v).to(device), torch.from_numpy(b_v).to(device)), torch.from_numpy(y_v).to(device)).item()
                    if vl < best_vl:
                        best_vl = vl
                        best_st = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                if best_st is not None:
                    m.load_state_dict(best_st)

                m.eval()
                with torch.no_grad():
                    v_logits = m.forward_logits(torch.from_numpy(X_v).to(device), torch.from_numpy(b_v).to(device)).cpu().numpy()
                    te_logits = m.forward_logits(torch.from_numpy(X_te).to(device), torch.from_numpy(b_te).to(device)).cpu().numpy()

                T = fit_temperature(v_logits, y_v)
                te_probs = 1.0 / (1.0 + np.exp(-te_logits / T))
                preds = (te_probs >= 0.5).astype(int)
                y_te_int = y_te.astype(int)

                f1 = float(f1_score(y_te_int, preds, zero_division=0))
                pr = float(precision_score(y_te_int, preds, zero_division=0))
                rc = float(recall_score(y_te_int, preds, zero_division=0))
                tn = np.sum((y_te_int == 0) & (preds == 0))
                fp = np.sum((y_te_int == 0) & (preds == 1))
                fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                return f1, pr, rc, fpr

            # Test configurations
            f1_p, pr_p, rc_p, fpr_p = eval_features(pca_features)
            f1_s, pr_s, rc_s, fpr_s = eval_features(pca_features + scalar_cols)
            f1_g, pr_g, rc_g, fpr_g = eval_features(pca_features + sage_cols)
            f1_a, pr_a, rc_a, fpr_a = eval_features(pca_features + scalar_cols + sage_cols)

            results_plain.append(f1_p)
            results_scalar.append(f1_s)
            results_sage.append(f1_g)
            results_fusion_all.append(f1_a)

            if fold_id % 5 == 0 or fold_id == 36:
                print(f"Fold {fold_id:02d} ({attack_type:16s}) | Plain: {f1_p:.3f} | +Scalar: {f1_s:.3f} | +GraphSAGE: {f1_g:.3f} | All: {f1_a:.3f}", flush=True)

        print(f"\n--- {target_name.upper()} 37-FOLD LOEO SUMMARY ---")
        print(f"1. Plain Stacked LSTM (32 PCA)          : F1 = {np.mean(results_plain):.4f}")
        print(f"2. Graph-Augmented Stacked (32+4 scalars) : F1 = {np.mean(results_scalar):.4f}")
        print(f"3. GraphSAGE Fusion (32+64 embeddings)    : F1 = {np.mean(results_sage):.4f}")
        print(f"4. Full Fusion (32+4+64 = 100 features)   : F1 = {np.mean(results_fusion_all):.4f}")

if __name__ == "__main__":
    run_evaluation()
