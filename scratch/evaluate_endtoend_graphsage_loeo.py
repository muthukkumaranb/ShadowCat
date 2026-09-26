"""
End-to-End GraphSAGE Fusion: Fair LOEO Evaluation
===================================================
Design decisions (locked down, approved):
  1. GraphSAGE runs ONCE per current window (not 30x per lookback step)
  2. All 2,787 graphs precomputed ONCE and cached across folds/epochs
  3. 12 node features computed purely from each window's own edge rows (no leakage)

Architecture:
  z_final = base_logit + lstm_residual(PCA_32[t-29:t]) + mlp(GraphSAGE(G_t))
                                                                ↑ current window only

GraphSAGE initialized from scratch (not from old regression checkpoint).
"""

import argparse
import json
import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.optimize import minimize_scalar

repo_root = Path(__file__).resolve().parent.parent
ml1_dir = repo_root / "ml1"
sys.path.insert(0, str(ml1_dir))

from lstm.ucs import UCSConfig, validate_ucs_windows, apply_training_only_pca
from lstm.utils import set_seed, get_device
from lstm.graph_features import compute_scalar_graph_features


# ═══════════════════════════════════════════════════════════════════════
# Section 1: Graph Cache Builder — precompute once, reuse across folds
# ═══════════════════════════════════════════════════════════════════════

NODE_FEATURE_DIM = 12  # The real 12 features from graph_builder.py

def build_graph_cache(edges_df: pd.DataFrame, all_window_ids: List[str]) -> Dict[str, Data]:
    """
    Build one PyG Data object per window from real edge data.
    Node features (12 dims, per-window only, no rolling/historical — confirmed no leakage):
      0: out_degree         1: in_degree
      2: out_flow_count     3: in_flow_count
      4: out_bytes          5: in_bytes
      6: out_packets        7: in_packets
      8: out_tcp_ratio      9: out_udp_ratio
     10: unique_dst_ports  11: unique_src_ports
    """
    print(f"  Building graph cache for {len(all_window_ids)} windows "
          f"from {len(edges_df):,} real edges...", flush=True)
    t0 = time.time()

    grouped = dict(tuple(edges_df.groupby("window_id")))
    cache: Dict[str, Data] = {}
    n_with, n_empty, total_nodes, total_edges = 0, 0, 0, 0

    for wid in all_window_ids:
        if wid not in grouped:
            cache[wid] = Data(
                x=torch.zeros((1, NODE_FEATURE_DIM), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=1,
            )
            n_empty += 1
            continue

        w_edges = grouped[wid]
        srcs = w_edges["src_node_id"].values
        dsts = w_edges["dst_node_id"].values

        # Local node mapping
        all_nodes_set = set(srcs) | set(dsts)
        all_nodes = sorted(all_nodes_set)
        node_to_idx = {nid: i for i, nid in enumerate(all_nodes)}
        n_nodes = len(all_nodes)

        src_idx = np.array([node_to_idx[s] for s in srcs], dtype=np.int64)
        dst_idx = np.array([node_to_idx[d] for d in dsts], dtype=np.int64)

        flow_counts = w_edges["flow_count"].values.astype(np.float32)
        byte_sums = w_edges["byte_count_sum"].values.astype(np.float32)
        packet_sums = w_edges["packet_count_sum"].values.astype(np.float32)
        protocols = w_edges["protocol_mode"].values
        is_tcp = (protocols == 6).astype(np.float32)
        is_udp = (protocols == 17).astype(np.float32)

        x = np.zeros((n_nodes, NODE_FEATURE_DIM), dtype=np.float32)

        # Each row in the edge list is a unique (src, dst) pair per window,
        # so degree = count of unique partners = count of rows per node.
        out_degree = np.bincount(src_idx, minlength=n_nodes).astype(np.float32)
        in_degree = np.bincount(dst_idx, minlength=n_nodes).astype(np.float32)

        x[:, 0] = out_degree
        x[:, 1] = in_degree
        x[:, 10] = out_degree  # unique_dst_ports ≈ out_degree (nodes are port-protocol endpoints)
        x[:, 11] = in_degree   # unique_src_ports ≈ in_degree

        np.add.at(x[:, 2], src_idx, flow_counts)     # out_flow_count
        np.add.at(x[:, 3], dst_idx, flow_counts)     # in_flow_count
        np.add.at(x[:, 4], src_idx, byte_sums)       # out_bytes
        np.add.at(x[:, 5], dst_idx, byte_sums)       # in_bytes
        np.add.at(x[:, 6], src_idx, packet_sums)     # out_packets
        np.add.at(x[:, 7], dst_idx, packet_sums)     # in_packets

        # TCP/UDP ratios per source node
        tcp_per_src = np.zeros(n_nodes, dtype=np.float32)
        udp_per_src = np.zeros(n_nodes, dtype=np.float32)
        np.add.at(tcp_per_src, src_idx, is_tcp)
        np.add.at(udp_per_src, src_idx, is_udp)
        x[:, 8] = np.divide(tcp_per_src, out_degree, out=np.zeros_like(tcp_per_src), where=out_degree > 0)
        x[:, 9] = np.divide(udp_per_src, out_degree, out=np.zeros_like(udp_per_src), where=out_degree > 0)

        edge_index = torch.tensor(np.stack([src_idx, dst_idx], axis=0), dtype=torch.long)

        cache[wid] = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=edge_index,
            num_nodes=n_nodes,
        )
        n_with += 1
        total_nodes += n_nodes
        total_edges += len(src_idx)

    elapsed = time.time() - t0
    print(f"  Graph cache built in {elapsed:.1f}s: "
          f"{n_with} windows with edges, {n_empty} empty, "
          f"avg {total_nodes / max(n_with, 1):.0f} nodes/graph, "
          f"avg {total_edges / max(n_with, 1):.0f} edges/graph", flush=True)
    return cache


def compute_node_norm_stats(
    graph_cache: Dict[str, Data],
    train_window_ids: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean/std from training-window graphs only (zero leakage)."""
    all_feats = []
    for wid in train_window_ids:
        if wid in graph_cache:
            all_feats.append(graph_cache[wid].x.numpy())
    if all_feats:
        stacked = np.concatenate(all_feats, axis=0)
        feat_mean = np.mean(stacked, axis=0).astype(np.float32)
        feat_std = (np.std(stacked, axis=0) + 1e-8).astype(np.float32)
    else:
        feat_mean = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
        feat_std = np.ones(NODE_FEATURE_DIM, dtype=np.float32)
    return torch.from_numpy(feat_mean), torch.from_numpy(feat_std)


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Model Definitions
# ═══════════════════════════════════════════════════════════════════════

class PlainResidualLSTM(nn.Module):
    """Plain or scalar-augmented residual LSTM (no graph). Matches existing protocol."""
    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.20):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.fc.bias)
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward_logits(self, x: torch.Tensor, base_logit: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_step = self.dropout(output[:, -1, :])
        residual = self.fc(last_step).squeeze(-1)
        return base_logit + residual


class EndToEndGraphSAGEFusion(nn.Module):
    """
    Joint LSTM + GraphSAGE model with end-to-end gradients.
    z = base_logit + lstm_residual(PCA_seq) + graph_mlp(GraphSAGE(G_t))

    GraphSAGE runs ONCE per current window (not 30x per lookback).
    GraphSAGE initialized from scratch. SAGEConv weights receive real gradients.
    """
    def __init__(
        self,
        pca_dim: int = 32,
        lstm_hidden: int = 64,
        node_feat_dim: int = 12,
        gnn_hidden: int = 64,
        gnn_embed: int = 64,
        dropout: float = 0.20,
    ):
        super().__init__()
        # --- LSTM branch (temporal residual from PCA sequence) ---
        self.lstm = nn.LSTM(input_size=pca_dim, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True, dropout=0.0)
        self.lstm_drop = nn.Dropout(p=dropout)
        self.lstm_fc = nn.Linear(lstm_hidden, 1)
        nn.init.zeros_(self.lstm_fc.bias)
        nn.init.normal_(self.lstm_fc.weight, std=0.01)

        # --- GraphSAGE branch (topological context for current window) ---
        # Standard SAGEConv, 2 layers, matching GraphBranch architecture
        self.sage_conv1 = SAGEConv(node_feat_dim, gnn_hidden)
        self.sage_bn1 = nn.BatchNorm1d(gnn_hidden)
        self.sage_conv2 = SAGEConv(gnn_hidden, gnn_embed)
        self.sage_dropout = dropout

        # Graph signal MLP: 64-dim embedding → scalar
        # Initialized near-zero so predictions start ≈ base_logit + lstm_residual
        self.graph_mlp = nn.Sequential(
            nn.Linear(gnn_embed, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.graph_mlp[-1].bias)
        nn.init.normal_(self.graph_mlp[-1].weight, std=0.01)

        # Node feature normalization buffers (set per fold)
        self.register_buffer("node_mean", torch.zeros(node_feat_dim))
        self.register_buffer("node_std", torch.ones(node_feat_dim))

    def set_node_norm(self, mean: torch.Tensor, std: torch.Tensor):
        self.node_mean.copy_(mean)
        self.node_std.copy_(std)

    def forward_logits(
        self,
        x_pca: torch.Tensor,       # [B, 30, 32]
        base_logit: torch.Tensor,   # [B]
        batched_graph: Batch,       # PyG Batch of B graphs (one per sample)
    ) -> torch.Tensor:
        # --- LSTM branch ---
        output, _ = self.lstm(x_pca)
        lstm_hidden = self.lstm_drop(output[:, -1, :])
        lstm_residual = self.lstm_fc(lstm_hidden).squeeze(-1)   # [B]

        # --- GraphSAGE branch (with real gradients!) ---
        gx = (batched_graph.x - self.node_mean) / self.node_std  # normalize
        h = self.sage_conv1(gx, batched_graph.edge_index)
        h = self.sage_bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.sage_dropout, training=self.training)
        h = self.sage_conv2(h, batched_graph.edge_index)
        graph_emb = global_mean_pool(h, batched_graph.batch)    # [B, 64]
        graph_signal = self.graph_mlp(graph_emb).squeeze(-1)    # [B]

        return base_logit + lstm_residual + graph_signal


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Training & Evaluation Utilities
# ═══════════════════════════════════════════════════════════════════════

def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    def nll(t):
        scaled = logits / t
        loss = np.maximum(scaled, 0) - scaled * labels + np.log(1 + np.exp(-np.abs(scaled)))
        return np.mean(loss)
    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def train_eval_plain_lstm(
    X_tr, b_tr, y_tr, X_v, b_v, y_v, X_te, b_te, y_te,
    input_size: int, device, epochs: int = 10, batch_size: int = 64,
) -> Tuple[float, float, float, float, List[float]]:
    """Train and evaluate a plain/scalar residual LSTM. Returns (f1, prec, rec, fpr, loss_curve)."""
    m = PlainResidualLSTM(input_size=input_size).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    best_vl, best_st = float("inf"), None
    loss_curve = []
    indices = np.arange(len(X_tr))

    X_v_t = torch.from_numpy(X_v).to(device)
    b_v_t = torch.from_numpy(b_v).to(device)
    y_v_t = torch.from_numpy(y_v).to(device)
    X_te_t = torch.from_numpy(X_te).to(device)
    b_te_t = torch.from_numpy(b_te).to(device)

    for ep in range(epochs):
        m.train()
        np.random.shuffle(indices)
        for s in range(0, len(indices), batch_size):
            bi = indices[s : s + batch_size]
            xb = torch.from_numpy(X_tr[bi]).to(device)
            bb = torch.from_numpy(b_tr[bi]).to(device)
            yb = torch.from_numpy(y_tr[bi]).to(device)
            opt.zero_grad()
            loss = crit(m.forward_logits(xb, bb), yb)
            loss.backward()
            opt.step()

        m.eval()
        with torch.no_grad():
            vl = crit(m.forward_logits(X_v_t, b_v_t), y_v_t).item()
        loss_curve.append(vl)
        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.cpu().clone() for k, v in m.state_dict().items()}

    if best_st:
        m.load_state_dict(best_st)

    m.eval()
    with torch.no_grad():
        v_logits = m.forward_logits(X_v_t, b_v_t).cpu().numpy()
        te_logits = m.forward_logits(X_te_t, b_te_t).cpu().numpy()

    T = fit_temperature(v_logits, y_v)
    te_probs = 1.0 / (1.0 + np.exp(-te_logits / T))
    preds = (te_probs >= 0.5).astype(int)
    y_int = y_te.astype(int)
    f1 = float(f1_score(y_int, preds, zero_division=0))
    pr = float(precision_score(y_int, preds, zero_division=0))
    rc = float(recall_score(y_int, preds, zero_division=0))
    tn, fp = np.sum((y_int == 0) & (preds == 0)), np.sum((y_int == 0) & (preds == 1))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return f1, pr, rc, fpr, loss_curve


def train_eval_graphsage(
    X_tr, b_tr, y_tr, wid_tr,
    X_v, b_v, y_v, wid_v,
    X_te, b_te, y_te, wid_te,
    graph_cache: Dict[str, Data],
    node_mean: torch.Tensor, node_std: torch.Tensor,
    device, epochs: int = 10, batch_size: int = 64,
) -> Tuple[float, float, float, float, List[float]]:
    """Train and evaluate the end-to-end GraphSAGE fusion model."""
    m = EndToEndGraphSAGEFusion(pca_dim=32).to(device)
    m.set_node_norm(node_mean.to(device), node_std.to(device))
    opt = torch.optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    best_vl, best_st = float("inf"), None
    loss_curve = []
    indices = np.arange(len(X_tr))

    def make_batch_graph(wids, dev):
        graphs = [graph_cache[wid] for wid in wids]
        return Batch.from_data_list(graphs).to(dev)

    X_v_t = torch.from_numpy(X_v).to(device)
    b_v_t = torch.from_numpy(b_v).to(device)
    y_v_t = torch.from_numpy(y_v).to(device)
    vg = make_batch_graph(wid_v, device)

    X_te_t = torch.from_numpy(X_te).to(device)
    b_te_t = torch.from_numpy(b_te).to(device)
    teg = make_batch_graph(wid_te, device)

    for ep in range(epochs):
        m.train()
        np.random.shuffle(indices)
        for s in range(0, len(indices), batch_size):
            bi = indices[s : s + batch_size]
            xb = torch.from_numpy(X_tr[bi]).to(device)
            bb = torch.from_numpy(b_tr[bi]).to(device)
            yb = torch.from_numpy(y_tr[bi]).to(device)
            bg = make_batch_graph([wid_tr[i] for i in bi], device)

            opt.zero_grad()
            logits = m.forward_logits(xb, bb, bg)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        m.eval()
        with torch.no_grad():
            vl = crit(m.forward_logits(X_v_t, b_v_t, vg), y_v_t).item()
        loss_curve.append(vl)
        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.cpu().clone() for k, v in m.state_dict().items()}

    if best_st:
        m.load_state_dict(best_st)

    m.eval()
    with torch.no_grad():
        v_logits = m.forward_logits(X_v_t, b_v_t, vg).cpu().numpy()
        te_logits = m.forward_logits(X_te_t, b_te_t, teg).cpu().numpy()

    T = fit_temperature(v_logits, y_v)
    te_probs = 1.0 / (1.0 + np.exp(-te_logits / T))
    preds = (te_probs >= 0.5).astype(int)
    y_int = y_te.astype(int)
    f1 = float(f1_score(y_int, preds, zero_division=0))
    pr = float(precision_score(y_int, preds, zero_division=0))
    rc = float(recall_score(y_int, preds, zero_division=0))
    tn, fp = np.sum((y_int == 0) & (preds == 0)), np.sum((y_int == 0) & (preds == 1))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return f1, pr, rc, fpr, loss_curve


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Main Evaluation
# ═══════════════════════════════════════════════════════════════════════

def run_evaluation(target_choice: str = "both", subset_choice: str = "all", max_folds: int = None, epochs: int = 10, resume: bool = False):
    t0_total = time.time()
    set_seed(42)
    device = get_device()
    config = UCSConfig()
    print(f"Device: {device}", flush=True)

    # ── Load all data ─────────────────────────────────────────────────
    data_path = repo_root / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
    edges_path = repo_root / "data-engineering" / "data" / "ucs" / "ucs_graph_edgelists.parquet"
    manifest_path = ml1_dir / "artifacts" / "loeo" / "corrected_37fold_manifest.json"
    feature_path = ml1_dir / "artifacts" / "lr_set_a" / "set_a_features.json"

    print("Loading UCS windows...", flush=True)
    df = pd.read_parquet(data_path).sort_values("window_start_utc").reset_index(drop=True)
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(feature_path) as f:
        features_meta = json.load(f)
    lr_features = features_meta["ordered_feature_names"]
    base_features = validate_ucs_windows(df, config=config)

    # ── Step 1: Precompute scalar graph features ──────────────────────
    print("Computing 4 scalar graph features...", flush=True)
    graph_scalar_df = compute_scalar_graph_features(edges_path, df)
    scalar_cols = ["graph_degree_centrality", "graph_density",
                   "graph_clustering_signal", "graph_degree_anomaly"]

    # ── Step 2: Precompute ALL 2,787 PyG graphs ONCE ──────────────────
    print("Building real per-window graph cache (precomputed once, reused across all folds)...", flush=True)
    edges_df = pd.read_parquet(edges_path)
    all_wids = df["window_id"].tolist()
    graph_cache = build_graph_cache(edges_df, all_wids)

    # ── Step 3: LOEO evaluation ───────────────────────────────────────
    results_file = repo_root / "scratch" / "endtoend_graphsage_results.json"
    all_results = {}

    targets = [("detection", "label_binary"), ("onset", "future_attack_label")]
    if target_choice == "detection":
        targets = [("detection", "label_binary")]
    elif target_choice == "onset":
        targets = [("onset", "future_attack_label")]

    for target_name, target_col in targets:
        print(f"\n{'=' * 70}")
        print(f"  TARGET: {target_name.upper()} ({target_col})")
        print(f"{'=' * 70}", flush=True)

        fold_results = []
        existing_by_ep = {}
        if resume and results_file.exists():
            try:
                with open(results_file) as f:
                    prior = json.load(f)
                if target_name in prior and isinstance(prior[target_name], list):
                    for r in prior[target_name]:
                        existing_by_ep[r.get("held_out_episode")] = r
                print(f"  Resuming: found {len(existing_by_ep)} completed folds for {target_name}", flush=True)
            except Exception as e:
                print(f"  Warning loading prior results for resume: {e}", flush=True)

        folds_to_run = manifest["folds"]
        if subset_choice == "botnet":
            folds_to_run = [f for f in folds_to_run if df.loc[df["episode_id"] == f["held_out_episode_id"], "label_attack_type"].iloc[0] == "Botnet"]
        elif subset_choice == "first5":
            folds_to_run = folds_to_run[:5]
        if max_folds is not None:
            folds_to_run = folds_to_run[:max_folds]

        print(f"  Evaluating {len(folds_to_run)} folds for {target_name}...", flush=True)

        for fold in folds_to_run:
            fold_id = fold["fold_id"]
            held_out = fold["held_out_episode_id"]
            attack_type = df.loc[df["episode_id"] == held_out, "label_attack_type"].iloc[0]

            if resume and held_out in existing_by_ep:
                r = existing_by_ep[held_out]
                fold_results.append(r)
                delta = r["graphsage_f1"] - r["scalar_f1"]
                marker = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")
                print(
                    f"  Fold {fold_id:2d} ({attack_type:16s}) | "
                    f"Plain={r['plain_f1']:.4f}  Scalar={r['scalar_f1']:.4f}  GraphSAGE={r['graphsage_f1']:.4f} {marker}  "
                    f"[resumed {r.get('elapsed_sec', 0.0):.1f}s]",
                    flush=True,
                )
                continue

            train_indices = fold["train_indices"]
            test_indices = fold["test_indices"]
            t0_fold = time.time()

            train_df = df.iloc[train_indices].copy().reset_index()
            test_df = df.iloc[test_indices].copy()
            train_episodes = train_df["episode_id"].unique()

            # ── OOF Logistic Regression (identical to verified protocol) ──
            oof_lr_logits = np.zeros(len(train_df))
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for tr_ep_idx, val_ep_idx in kf.split(train_episodes):
                tr_eps = set(train_episodes[tr_ep_idx])
                val_eps = set(train_episodes[val_ep_idx])
                tr_mask = train_df["episode_id"].isin(tr_eps)
                val_mask = train_df["episode_id"].isin(val_eps)
                y_inner = train_df.loc[tr_mask, target_col].to_numpy(dtype=int)
                if len(np.unique(y_inner)) < 2:
                    continue
                sc_inner = StandardScaler()
                x_inner_tr = sc_inner.fit_transform(train_df.loc[tr_mask, lr_features].to_numpy(dtype=float))
                x_inner_val = sc_inner.transform(train_df.loc[val_mask, lr_features].to_numpy(dtype=float))
                lr_inner = LogisticRegression(solver="liblinear", max_iter=2000, random_state=42)
                lr_inner.fit(x_inner_tr, y_inner)
                oof_lr_logits[val_mask] = lr_inner.decision_function(x_inner_val)

            sc_full = StandardScaler()
            x_tr_full = sc_full.fit_transform(train_df[lr_features].to_numpy(dtype=float))
            y_tr_full = train_df[target_col].to_numpy(dtype=int)
            lr_full = LogisticRegression(solver="liblinear", max_iter=2000, random_state=42)
            lr_full.fit(x_tr_full, y_tr_full)
            x_te_full = sc_full.transform(test_df[lr_features].to_numpy(dtype=float))
            test_lr_logits = lr_full.decision_function(x_te_full)

            # ── Chronological 80/20 train/val split ──
            fold_frame = df.copy()
            fold_frame["split"] = "none"
            train_sorted = fold_frame.loc[train_indices].sort_values("window_start_utc")
            n_tr = len(train_sorted)
            val_start = int(n_tr * 0.8)
            actual_train_idx = train_sorted.index[:val_start]
            actual_val_idx = train_sorted.index[val_start:]

            fold_frame.loc[actual_train_idx, "split"] = "train"
            fold_frame.loc[actual_val_idx, "split"] = "val"
            fold_frame.loc[test_indices, "split"] = "test"

            orig_to_pos = {train_df["index"].iloc[i]: i for i in range(len(train_df))}
            fold_frame["base_logit"] = 0.0
            for idx, pos in orig_to_pos.items():
                fold_frame.loc[idx, "base_logit"] = oof_lr_logits[pos]
            for idx, logit in zip(test_indices, test_lr_logits):
                fold_frame.loc[idx, "base_logit"] = logit

            # ── PCA 32 fitted strictly on training split ──
            purged = fold_frame[fold_frame["split"].isin(["train", "val", "test"])].copy()
            _, pca = apply_training_only_pca(purged, base_features, 32, random_state=42)
            full_pca = pca.transform(fold_frame)
            pca_cols = [f"pca_{i}" for i in range(32)]

            # ── Scalar graph features standardized on training split ──
            g_scaler = StandardScaler()
            tr_sc = graph_scalar_df.loc[fold_frame.loc[actual_train_idx, "window_id"], scalar_cols].to_numpy(dtype=float)
            g_scaler.fit(tr_sc)
            all_sc = g_scaler.transform(graph_scalar_df.loc[fold_frame["window_id"], scalar_cols].to_numpy(dtype=float))
            for i, c in enumerate(scalar_cols):
                full_pca[c] = all_sc[:, i]

            # ── Node feature normalization from training-window graphs ──
            train_wids = fold_frame.loc[actual_train_idx, "window_id"].tolist()
            node_mean, node_std = compute_node_norm_stats(graph_cache, train_wids)

            # ── Extract sequences ──
            idx_to_pos = {idx: pos for pos, idx in enumerate(fold_frame.index)}
            base_arr = fold_frame["base_logit"].values
            target_arr = fold_frame[target_col].values
            wid_arr = fold_frame["window_id"].values

            # Precompute feature arrays for plain (32) and scalar (36)
            plain_np = full_pca[pca_cols].to_numpy(dtype=np.float32)
            scalar_np = full_pca[pca_cols + scalar_cols].to_numpy(dtype=np.float32)

            def extract_seqs(indices, feat_np):
                X, b, y, wids = [], [], [], []
                for idx in indices:
                    t = idx_to_pos[idx]
                    s = t - config.lookback_windows + 1
                    if s >= 0 and (t - s + 1) == config.lookback_windows:
                        X.append(feat_np[s : t + 1])
                        b.append(base_arr[t])
                        y.append(target_arr[t])
                        wids.append(wid_arr[t])
                return (
                    np.asarray(X, dtype=np.float32),
                    np.asarray(b, dtype=np.float32),
                    np.asarray(y, dtype=np.float32),
                    wids,
                )

            # Plain sequences (32 PCA only, with window_ids attached for GraphSAGE)
            Xp_tr, bp_tr, yp_tr, wg_tr = extract_seqs(actual_train_idx, plain_np)
            Xp_v, bp_v, yp_v, wg_v = extract_seqs(actual_val_idx, plain_np)
            Xp_te, bp_te, yp_te, wg_te = extract_seqs(test_indices, plain_np)

            # Scalar sequences (36 = 32 PCA + 4 graph scalars)
            Xs_tr, _, ys_tr, _ = extract_seqs(actual_train_idx, scalar_np)
            Xs_v, _, ys_v, _ = extract_seqs(actual_val_idx, scalar_np)
            Xs_te, _, ys_te, _ = extract_seqs(test_indices, scalar_np)

            if len(yp_te) == 0:
                print(f"  Fold {fold_id:2d}: SKIP (no test sequences)", flush=True)
                continue

            # ── Train & Evaluate: 1) Plain LSTM ──
            f1_p, pr_p, rc_p, fpr_p, lc_p = train_eval_plain_lstm(
                Xp_tr, bp_tr, yp_tr, Xp_v, bp_v, yp_v, Xp_te, bp_te, yp_te,
                input_size=32, device=device, epochs=epochs,
            )

            # ── Train & Evaluate: 2) Scalar Graph-Augmented LSTM ──
            f1_s, pr_s, rc_s, fpr_s, lc_s = train_eval_plain_lstm(
                Xs_tr, bp_tr, ys_tr, Xs_v, bp_v, ys_v, Xs_te, bp_te, ys_te,
                input_size=36, device=device, epochs=epochs,
            )

            # ── Train & Evaluate: 3) End-to-End GraphSAGE Fusion ──
            f1_g, pr_g, rc_g, fpr_g, lc_g = train_eval_graphsage(
                Xp_tr, bp_tr, yp_tr, wg_tr,
                Xp_v, bp_v, yp_v, wg_v,
                Xp_te, bp_te, yp_te, wg_te,
                graph_cache=graph_cache,
                node_mean=node_mean, node_std=node_std,
                device=device, epochs=epochs,
            )

            elapsed = time.time() - t0_fold
            fold_results.append({
                "fold_id": fold_id,
                "attack_type": attack_type,
                "held_out_episode": held_out,
                "plain_f1": f1_p, "plain_prec": pr_p, "plain_rec": rc_p,
                "scalar_f1": f1_s, "scalar_prec": pr_s, "scalar_rec": rc_s,
                "graphsage_f1": f1_g, "graphsage_prec": pr_g, "graphsage_rec": rc_g,
                "graphsage_loss_curve": lc_g,
                "elapsed_sec": elapsed,
            })

            delta = f1_g - f1_s
            marker = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")
            print(
                f"  Fold {fold_id:2d} ({attack_type:16s}) | "
                f"Plain={f1_p:.4f}  Scalar={f1_s:.4f}  GraphSAGE={f1_g:.4f} {marker}  "
                f"[{elapsed:.1f}s]",
                flush=True,
            )

            # Incremental save
            all_results[target_name] = fold_results
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            # ── Intermediate summary after Botnet folds (early signal) ──
            if fold_id == 9 and target_name == "detection":
                botnet = [r for r in fold_results if r["attack_type"] == "Botnet"]
                if botnet:
                    print(f"\n  -- EARLY SIGNAL (Botnet folds, {target_name}) --")
                    print(f"  Plain:     {np.mean([r['plain_f1'] for r in botnet]):.4f}")
                    print(f"  Scalar:    {np.mean([r['scalar_f1'] for r in botnet]):.4f}")
                    print(f"  GraphSAGE: {np.mean([r['graphsage_f1'] for r in botnet]):.4f}")
                    print(f"  --------------------------------------\n", flush=True)

        # ── Full summary for this target ──
        if fold_results:
            print(f"\n{'-' * 70}")
            print(f"  {target_name.upper()} -- 37-FOLD LOEO SUMMARY")
            print(f"{'-' * 70}")

            p_f1s = [r["plain_f1"] for r in fold_results]
            s_f1s = [r["scalar_f1"] for r in fold_results]
            g_f1s = [r["graphsage_f1"] for r in fold_results]

            print(f"  1. Plain Stacked LSTM (32 PCA)             : F1 = {np.mean(p_f1s):.4f}")
            print(f"  2. Scalar Graph-Augmented (32+4)            : F1 = {np.mean(s_f1s):.4f}")
            print(f"  3. End-to-End GraphSAGE Fusion (32 PCA+GNN) : F1 = {np.mean(g_f1s):.4f}")

            # Per-attack-type breakdown
            print(f"\n  Per-Attack-Type Breakdown:")
            print(f"  {'Attack Type':20s} {'N':>4s}  {'Plain':>8s}  {'Scalar':>8s}  {'GraphSAGE':>10s}  {'Delta(GS-Sc)':>13s}")
            for atk in ["Botnet", "SSH-Bruteforce", "DDOS-LOIC-UDP"]:
                atk_results = [r for r in fold_results if r["attack_type"] == atk]
                if atk_results:
                    p_m = np.mean([r["plain_f1"] for r in atk_results])
                    s_m = np.mean([r["scalar_f1"] for r in atk_results])
                    g_m = np.mean([r["graphsage_f1"] for r in atk_results])
                    d = g_m - s_m
                    print(f"  {atk:20s} {len(atk_results):4d}  {p_m:8.4f}  {s_m:8.4f}  {g_m:10.4f}  {d:+13.4f}")

            # Per-fold training loss curves (last 3 values)
            print(f"\n  GraphSAGE final val losses (last 3 epochs):")
            for r in fold_results[:5]:  # show first 5
                lc = r.get("graphsage_loss_curve", [])
                tail = lc[-3:] if len(lc) >= 3 else lc
                print(f"    Fold {r['fold_id']:2d}: {[f'{v:.4f}' for v in tail]}")
            if len(fold_results) > 5:
                print(f"    ... ({len(fold_results) - 5} more folds)")

        all_results[target_name] = fold_results

    # ── Save full results ──
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to: {results_file}", flush=True)

    # ── Go/No-Go Assessment ──
    print(f"\n{'=' * 70}")
    print(f"  GO/NO-GO ASSESSMENT")
    print(f"{'=' * 70}")

    for tgt in ["detection", "onset"]:
        if tgt in all_results and all_results[tgt]:
            res = all_results[tgt]
            s_mean = np.mean([r["scalar_f1"] for r in res])
            g_mean = np.mean([r["graphsage_f1"] for r in res])
            delta = g_mean - s_mean
            beats = delta > 0.0005  # meaningful improvement threshold

            if beats:
                print(f"\n  {tgt.upper()}: GraphSAGE F1={g_mean:.4f} vs Scalar F1={s_mean:.4f} "
                      f"(Delta={delta:+.4f}) -> POTENTIAL IMPROVEMENT. Proceed to latency benchmark.")
            else:
                print(f"\n  {tgt.upper()}: GraphSAGE F1={g_mean:.4f} vs Scalar F1={s_mean:.4f} "
                      f"(Delta={delta:+.4f}) -> NO IMPROVEMENT over scalar features. "
                      f"Graph topology does not add signal beyond cheap graph statistics for this task.")

    total_elapsed = time.time() - t0_total
    print(f"\n{'=' * 70}")
    print(f"  TOTAL EVALUATION WALL-CLOCK RUNTIME: {total_elapsed:.1f}s ({total_elapsed / 60:.2f} min)")
    print(f"{'=' * 70}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End GraphSAGE LOEO Evaluation")
    parser.add_argument("--target", choices=["detection", "onset", "both"], default="both",
                        help="Target to evaluate (detection, onset, or both)")
    parser.add_argument("--subset", choices=["all", "botnet", "first5"], default="all",
                        help="Subset of folds to run (all 37, botnet, or first5)")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Maximum number of folds to run")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train (default: 10)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from existing results in results_file if present")
    args = parser.parse_args()
    run_evaluation(target_choice=args.target, subset_choice=args.subset, max_folds=args.max_folds, epochs=args.epochs, resume=args.resume)
