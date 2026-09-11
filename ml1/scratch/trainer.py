"""
ML2 Trainer: training loop for graph-based next-state prediction models.

Handles:
    - MSE loss for next-state embedding prediction
    - Gradient clipping
    - Early stopping with patience
    - Checkpoint saving
    - Per-epoch train/val loss logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from ml2.data.dataset import build_paired_sequences, SnapshotDataset

logger = logging.getLogger(__name__)


class ML2Trainer:
    """
    Trains a dynamics-prediction model (FusedModel or TemporalOnlyBaseline)
    on (graph_t, graph_{t+1}) pairs using MSE loss.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: str = None,
        checkpoint_dir: str = None,
    ):
        """
        Args:
            model: Model with .forward(x, edge_index, z_t, batch) → pred embedding.
            config: Training config dict (from ml2_config.yaml).
            device: 'cuda' or 'cpu'. Auto-detects if None.
            checkpoint_dir: Directory for saving checkpoints.
        """
        self.model = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        training_cfg = config.get("training", {})
        self.lr = training_cfg.get("lr", 0.001)
        self.epochs = training_cfg.get("epochs", 100)
        self.patience = training_cfg.get("patience", 15)
        self.max_grad_norm = training_cfg.get("max_grad_norm", 1.0)

        ml1_cfg = config.get("ml1_interface", {})
        self.z_dim = ml1_cfg.get("z_dim", 64)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        self.criterion = nn.MSELoss()

        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def _get_target_embedding(self, graph: Data) -> torch.Tensor:
        """
        Computes the target embedding for graph_{t+1}.

        Always uses mean-pooled raw node features as the target.
        This ensures:
          1. No self-referential target (graph_branch is never used for targets).
          2. Both FusedModel and TemporalOnlyBaseline predict into the
             same fixed 11-dim target space, making ablation comparison valid.
        """
        graph = graph.to(self.device)
        batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
        target = global_mean_pool(graph.x, batch)
        return target

    def _train_epoch(self, pairs: List[Tuple[Data, Data]]) -> float:
        """Run one training epoch over (graph_t, graph_{t+1}) pairs."""
        self.model.train()
        total_loss = 0.0

        for graph_t, graph_tp1 in pairs:
            graph_t = graph_t.to(self.device)
            graph_tp1 = graph_tp1.to(self.device)

            batch_t = torch.zeros(graph_t.x.size(0), dtype=torch.long, device=self.device)
            z_t = torch.zeros(1, self.z_dim, device=self.device)

            # Forward
            pred = self.model(graph_t.x, graph_t.edge_index, z_t, batch_t)

            # Target: embedding of next graph (detached from computation graph)
            target = self._get_target_embedding(graph_tp1).detach()

            # Match dimensions if needed
            if pred.shape[-1] != target.shape[-1]:
                min_dim = min(pred.shape[-1], target.shape[-1])
                loss = self.criterion(pred[:, :min_dim], target[:, :min_dim])
            else:
                loss = self.criterion(pred, target)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(pairs), 1)

    @torch.no_grad()
    def _eval_epoch(self, pairs: List[Tuple[Data, Data]]) -> float:
        """Evaluate on validation pairs."""
        self.model.eval()
        total_loss = 0.0

        for graph_t, graph_tp1 in pairs:
            graph_t = graph_t.to(self.device)
            graph_tp1 = graph_tp1.to(self.device)

            batch_t = torch.zeros(graph_t.x.size(0), dtype=torch.long, device=self.device)
            z_t = torch.zeros(1, self.z_dim, device=self.device)

            pred = self.model(graph_t.x, graph_t.edge_index, z_t, batch_t)
            target = self._get_target_embedding(graph_tp1)

            if pred.shape[-1] != target.shape[-1]:
                min_dim = min(pred.shape[-1], target.shape[-1])
                loss = self.criterion(pred[:, :min_dim], target[:, :min_dim])
            else:
                loss = self.criterion(pred, target)

            total_loss += loss.item()

        return total_loss / max(len(pairs), 1)

    def train(
        self,
        train_dataset: SnapshotDataset,
        val_dataset: SnapshotDataset,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.

        Args:
            train_dataset: Training SnapshotDataset.
            val_dataset: Validation SnapshotDataset.
            verbose: Print progress.

        Returns:
            Training history dict with 'train_loss' and 'val_loss' lists.
        """
        train_pairs = build_paired_sequences(train_dataset)
        val_pairs = build_paired_sequences(val_dataset)

        if len(train_pairs) == 0:
            logger.warning("No training pairs (need ≥2 snapshots). Skipping training.")
            return self.history

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_pairs)
            val_loss = self._eval_epoch(val_pairs) if len(val_pairs) > 0 else float("nan")

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            self.scheduler.step(val_loss if not np.isnan(val_loss) else train_loss)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch:3d}/{self.epochs} | "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e}"
                )

            # Early stopping
            if not np.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if self.checkpoint_dir:
                    self._save_checkpoint(epoch, val_loss, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch} (patience={self.patience})")
                    break

        # Save final history
        if self.checkpoint_dir:
            self._save_history()

        return self.history

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }

        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(ckpt, path)

    def _save_history(self) -> None:
        """Save training history to JSON."""
        if self.checkpoint_dir is None:
            return

        path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, path: str) -> None:
        """Load a model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
