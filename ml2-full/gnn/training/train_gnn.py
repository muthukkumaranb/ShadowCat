"""
Training pipeline for GNN graph representation encoder.

Trains the GraphEncoder + DiagnosticClassifier (SnapshotGNN) and optionally
the TemporalAttackGNN for sequence-based forecasting experiments.

Key design decisions:
    - Edge features are passed through to EdgeAwareSAGEConv layers
    - Classifier is diagnostic only; the primary output is the 64-D graph embedding
    - All hyperparameters come from gnn/configs/gnn.yaml
    - Checkpoints include full metadata per spec §34
"""
import os
import sys
import random
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional, Any

from gnn.models.graphsage import SnapshotGNN, TemporalAttackGNN, GraphEncoder
from gnn.baselines import compute_metrics
from gnn.graph_dataset import create_chronological_splits, TemporalGraphDataset
from gnn.graph_builder import NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES


def set_seed(seed: int = 42):
    """Set reproducible random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def detect_device() -> torch.device:
    """Detect and log compute device information."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Device] Using CUDA Device: {device_name} ({vram_gb:.2f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available; using CPU.")
    return device


def train_single_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train model for one epoch, passing edge_attr through to the model."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Pass edge_attr to the model — this is the critical fix
        logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        y = batch.y.view(-1, 1).to(device)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model on a DataLoader, passing edge_attr through."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        y = batch.y.view(-1, 1).to(device)

        loss = criterion(logits, y)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

        all_preds.extend(probs)
        all_targets.extend(y.cpu().numpy().flatten())

    avg_loss = total_loss / max(1, total_samples)
    y_true = np.array(all_targets, dtype=np.int64)
    y_prob = np.array(all_preds, dtype=np.float32)

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    return avg_loss, metrics, y_true, y_prob


def train_gnn(
    graphs: List[Data],
    splits: Dict[str, Tuple[int, int]],
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    feature_mode: str = "all",  # 'all', 'structure_only', 'node_only', 'edge_only'
) -> Dict[str, Any]:
    """
    Train and evaluate SnapshotGNN (GraphEncoder + DiagnosticClassifier)
    with early stopping and checkpointing.

    The primary output of the trained model is the GraphEncoder's 64-D embedding.
    The DiagnosticClassifier is used only for training signal and evaluation.
    """
    set_seed(config.get("random_seed", 42))
    if device is None:
        device = detect_device()

    train_start, train_end = splits["train"]
    val_start, val_end = splits["val"]
    test_start, test_end = splits["test"]

    # Clone graphs to apply feature ablation if requested
    processed_graphs = []
    for g in graphs:
        g_copy = g.clone()
        if feature_mode == "structure_only":
            # Node features set to constant 1.0 (topology only)
            g_copy.x = torch.ones((g_copy.num_nodes, 1), dtype=torch.float32)
            g_copy.edge_attr = None
        elif feature_mode == "node_only":
            # Keep node features, remove edge features
            g_copy.edge_attr = None
        # 'all' = keep everything (node + edge features)
        processed_graphs.append(g_copy)

    train_graphs = processed_graphs[train_start : train_end + 1]
    val_graphs = processed_graphs[val_start : val_end + 1]
    test_graphs = processed_graphs[test_start : test_end + 1]

    # Calculate class imbalance pos_weight from training set only
    train_labels = [int(g.y.item()) for g in train_graphs]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight_val = float(n_neg / max(1, n_pos)) * config.get("training", {}).get("pos_weight_multiplier", 1.0)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    batch_size = config.get("training", {}).get("batch_size", 16)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # Determine feature dimensions from processed data
    node_feature_dim = processed_graphs[0].x.shape[1]
    edge_feature_dim = 0
    if processed_graphs[0].edge_attr is not None and processed_graphs[0].edge_attr.shape[1] > 0:
        edge_feature_dim = processed_graphs[0].edge_attr.shape[1]

    model_cfg = config.get("model", {})
    hidden_dim = model_cfg.get("hidden_channels", 64)
    embedding_dim = model_cfg.get("embedding_dim", hidden_dim)  # default: same as hidden

    model = SnapshotGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=hidden_dim,
        embedding_dim=model_cfg.get("embedding_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.3),
        pooling=model_cfg.get("pooling", "mean_max"),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    lr = float(config.get("training", {}).get("learning_rate", 0.001))
    weight_decay = float(config.get("training", {}).get("weight_decay", 0.0001))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = config.get("training", {}).get("epochs", 100)
    patience = config.get("training", {}).get("patience", 15)

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_single_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pr_auc": val_metrics["pr_auc"],
            "val_f1": val_metrics["f1"],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    # Final evaluation on all splits
    train_loss, train_metrics, _, _ = evaluate_model(model, train_loader, criterion, device)
    val_loss, val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
    test_loss, test_metrics, y_true_test, y_prob_test = evaluate_model(model, test_loader, criterion, device)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "history": history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "y_prob_test": y_prob_test,
        "y_true_test": y_true_test,
        "feature_mode": feature_mode,
        "node_feature_dim": node_feature_dim,
        "edge_feature_dim": edge_feature_dim,
        "in_channels": node_feature_dim,  # backward compat
        "pos_weight": pos_weight_val,
    }


def train_temporal_forecaster(
    graphs: List[Data],
    splits: Dict[str, Tuple[int, int]],
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    pretrained_gnn: Optional[SnapshotGNN] = None,
) -> Dict[str, Any]:
    """
    Train a sequence-based temporal GNN for future attack forecasting.
    Input: [G(t-W+1), ..., G(t)] → Output: P(future_attack in [t, t+H])
    Precomputes graph snapshot embeddings for instant high-speed training.

    NOTE: This is an optional experiment — it does NOT replace the LSTM/Fusion system.
    """
    set_seed(config.get("random_seed", 42))
    if device is None:
        device = detect_device()

    window_size = config.get("data", {}).get("sequence_window", 30)
    node_feature_dim = graphs[0].x.shape[1]
    edge_feature_dim = 0
    if graphs[0].edge_attr is not None and graphs[0].edge_attr.shape[1] > 0:
        edge_feature_dim = graphs[0].edge_attr.shape[1]

    model_cfg = config.get("model", {})

    # Use pretrained GNN encoder or create one
    if pretrained_gnn is not None:
        gnn_encoder = pretrained_gnn
    else:
        gnn_encoder = SnapshotGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=model_cfg.get("hidden_channels", 64),
            embedding_dim=model_cfg.get("embedding_dim", 64),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.3),
            pooling=model_cfg.get("pooling", "mean_max"),
        ).to(device)

    # 1. Precompute graph embeddings for all graphs
    gnn_encoder.eval()
    print("  Precomputing graph embeddings for temporal sequence forecasting...", flush=True)
    embeddings = []
    labels = []
    with torch.no_grad():
        for g in graphs:
            g_dev = g.to(device)
            batch_vec = torch.zeros(g_dev.num_nodes, dtype=torch.long, device=device)
            emb = gnn_encoder.get_graph_embedding(
                g_dev.x, g_dev.edge_index, batch_vec, edge_attr=g_dev.edge_attr
            )
            embeddings.append(emb.cpu().squeeze(0))
            labels.append(int(g.y.item()))

    stacked_embeddings = torch.stack(embeddings, dim=0)  # (N_graphs, D)
    embed_dim = stacked_embeddings.shape[1]

    # 2. Build temporal dataset of sequences
    valid_indices = list(range(window_size - 1, len(graphs)))
    X_seq = []
    y_seq = []
    for vi in valid_indices:
        start_idx = vi - window_size + 1
        seq = stacked_embeddings[start_idx : vi + 1]
        X_seq.append(seq)
        y_seq.append(labels[vi])

    X_seq = torch.stack(X_seq, dim=0)
    y_seq = torch.tensor(y_seq, dtype=torch.float32)

    # Get split index masks
    train_start, train_end = splits["train"]
    val_start, val_end = splits["val"]
    test_start, test_end = splits["test"]

    train_mask = [i for i, vi in enumerate(valid_indices) if train_start <= vi <= train_end]
    val_mask = [i for i, vi in enumerate(valid_indices) if val_start <= vi <= val_end]
    test_mask = [i for i, vi in enumerate(valid_indices) if test_start <= vi <= test_end]

    X_train, y_train = X_seq[train_mask].to(device), y_seq[train_mask].to(device)
    X_val, y_val = X_seq[val_mask].to(device), y_seq[val_mask].to(device)
    X_test, y_test = X_seq[test_mask].to(device), y_seq[test_mask].to(device)

    # 3. Model
    temporal_model = TemporalAttackGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=model_cfg.get("hidden_channels", 64),
        embedding_dim=model_cfg.get("embedding_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.3),
        pooling=model_cfg.get("pooling", "mean_max"),
        temporal_method=model_cfg.get("temporal_method", "mean"),
    ).to(device)
    temporal_model.gnn.load_state_dict(gnn_encoder.state_dict())

    # Class weighting
    n_pos = int(y_train.sum().item())
    n_neg = len(y_train) - n_pos
    pos_weight_val = float(n_neg / max(1, n_pos)) * config.get("training", {}).get("pos_weight_multiplier", 1.0)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    lr = float(config.get("training", {}).get("learning_rate", 0.001))
    weight_decay = float(config.get("training", {}).get("weight_decay", 0.0001))
    optimizer = torch.optim.Adam(temporal_model.temporal_classifier.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = config.get("training", {}).get("epochs", 80)
    patience = config.get("training", {}).get("patience", 15)

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 0
    patience_counter = 0

    temporal_method = model_cfg.get("temporal_method", "mean")

    def forward_seq(X_batch):
        if temporal_method == "mean":
            t_emb = X_batch.mean(dim=1)
        elif temporal_method == "last":
            t_emb = X_batch[:, -1, :]
        else:
            t_emb = X_batch.mean(dim=1)
        return temporal_model.temporal_classifier(t_emb)

    for epoch in range(1, epochs + 1):
        temporal_model.train()
        optimizer.zero_grad()
        logits = forward_seq(X_train).squeeze(-1)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        temporal_model.eval()
        with torch.no_grad():
            val_logits = forward_seq(X_val).squeeze(-1)
            val_loss = criterion(val_logits, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in temporal_model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state_dict is not None:
        temporal_model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    # Test Evaluation
    temporal_model.eval()
    with torch.no_grad():
        test_logits = forward_seq(X_test).squeeze(-1)
        test_preds = torch.sigmoid(test_logits).cpu().numpy()
        test_targets = y_test.cpu().numpy().astype(np.int64)

    test_metrics = compute_metrics(test_targets, test_preds)

    lead_time_min = 0.0
    for pred, true_val in zip(test_preds, test_targets):
        if true_val == 1 and pred >= 0.5:
            lead_time_min = config.get("data", {}).get("forecast_horizon_sec", 600) / 60.0
            break

    return {
        "model": temporal_model,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "lead_time_min": lead_time_min,
        "y_prob_test": test_preds,
        "y_true_test": test_targets,
    }
