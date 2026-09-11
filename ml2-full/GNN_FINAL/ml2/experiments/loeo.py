"""
Leave-One-Episode-Out (LOEO) Cross-Validation Harness.

Jointly owned by ML1 and ML2. Designed to be import-agnostic so either module
can use it directly.

An "episode" is a contiguous sub-sequence of temporal snapshots (graphs or
observation windows), typically corresponding to a scenario/attack campaign/
normal-baseline segment.

Usage:
    harness = LOEOHarness()
    harness.register_model(my_model, "fused_v1")
    results = harness.run(episodes, metrics_fn, config)
    print(harness.summary())
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import logging
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    NamedTuple,
)
from dataclasses import dataclass, field
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """
    Represents a single episode — a contiguous time segment of data.

    Attributes:
        episode_id: Unique identifier for this episode.
        data: List of PyG Data objects (graphs) or any sequential data.
        metadata: Optional dict of episode-level metadata (e.g., label, scenario name).
    """
    episode_id: str
    data: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class FoldResult:
    """Results from a single LOEO fold."""
    fold_idx: int
    held_out_episode_id: str
    metrics: Dict[str, float]
    train_episode_ids: List[str]
    num_train_samples: int
    num_test_samples: int


class LOEOHarness:
    """
    Leave-One-Episode-Out cross-validation harness.

    Interface contract (designed for ML1/ML2 joint use):
        1. Register one or more models via .register_model()
        2. Call .run() with a list of Episodes and a metrics function
        3. Call .summary() to get a DataFrame of per-fold and aggregate results
    """

    def __init__(self):
        self._models: Dict[str, nn.Module] = {}
        self._results: Dict[str, List[FoldResult]] = {}

    def register_model(self, model: nn.Module, name: str) -> None:
        """
        Register a model for evaluation.

        Args:
            model: PyTorch model. Will be deep-copied per fold to avoid weight leakage.
            name: Human-readable name for this model variant.
        """
        self._models[name] = model
        self._results[name] = []
        logger.info(f"Registered model '{name}' ({type(model).__name__})")

    def run(
        self,
        episodes: List[Episode],
        metrics_fn: Callable,
        train_fn: Optional[Callable] = None,
        config: Optional[dict] = None,
        device: str = None,
        verbose: bool = True,
    ) -> Dict[str, List[FoldResult]]:
        """
        Runs LOEO cross-validation for all registered models.

        For each fold:
            - Hold out one episode for testing
            - Train on all other episodes
            - Evaluate on the held-out episode

        Args:
            episodes: List of Episode objects.
            metrics_fn: Callable(model, test_data, device) → Dict[str, float].
                Must return a dict of metric_name → metric_value.
            train_fn: Optional callable(model, train_data, config, device) → trained_model.
                If None, model is used as-is (for pre-trained or frozen models).
            config: Config dict passed to train_fn.
            device: 'cuda' or 'cpu'. Auto-detects if None.
            verbose: Print progress.

        Returns:
            Dict mapping model_name → list of FoldResult.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if len(episodes) < 2:
            raise ValueError(f"LOEO requires ≥2 episodes, got {len(episodes)}")

        if not self._models:
            raise ValueError("No models registered. Call .register_model() first.")

        n_folds = len(episodes)

        for model_name, base_model in self._models.items():
            if verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"LOEO for model: {model_name} ({n_folds} folds)")
                logger.info(f"{'='*60}")

            fold_results = []

            for fold_idx in range(n_folds):
                held_out = episodes[fold_idx]
                train_episodes = [ep for i, ep in enumerate(episodes) if i != fold_idx]

                if verbose:
                    logger.info(
                        f"  Fold {fold_idx + 1}/{n_folds}: "
                        f"held-out={held_out.episode_id} "
                        f"({len(held_out)} samples)"
                    )

                # Deep-copy model to avoid weight contamination across folds
                fold_model = copy.deepcopy(base_model).to(device)

                # Concatenate training data from all non-held-out episodes
                train_data = []
                train_ep_ids = []
                for ep in train_episodes:
                    train_data.extend(ep.data)
                    train_ep_ids.append(ep.episode_id)

                # Train (if train_fn provided)
                if train_fn is not None:
                    fold_model = train_fn(fold_model, train_data, config, device)

                # Evaluate on held-out episode
                test_data = held_out.data
                try:
                    metrics = metrics_fn(fold_model, test_data, device)
                except Exception as e:
                    logger.error(f"  Metrics computation failed for fold {fold_idx}: {e}")
                    metrics = {"error": float("nan")}

                fold_result = FoldResult(
                    fold_idx=fold_idx,
                    held_out_episode_id=held_out.episode_id,
                    metrics=metrics,
                    train_episode_ids=train_ep_ids,
                    num_train_samples=len(train_data),
                    num_test_samples=len(test_data),
                )
                fold_results.append(fold_result)

                if verbose:
                    metrics_str = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
                    logger.info(f"    → {metrics_str}")

            self._results[model_name] = fold_results

        return self._results

    def run_manifest(
        self,
        graphs: list,
        manifest_path: str,
        metrics_fn: Callable,
        train_fn: Optional[Callable] = None,
        config: Optional[dict] = None,
        device: str = None,
        verbose: bool = True,
    ) -> Dict[str, List[FoldResult]]:
        """
        Runs LOEO cross-validation using a pre-computed manifest.
        
        Note: This LOEO harness code (manifest parsing, fold-local scaler logic) is specifically 
        for ML1's detection/onset/B2 work and/or a potential future second ablation variant — 
        it is NOT to be used for the chronological-split dynamics ablation required for Sep 12.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        import json
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if not self._models:
            raise ValueError("No models registered. Call .register_model() first.")

        n_folds = manifest.get("fold_count", len(manifest["folds"]))
        
        # Determine purge and embargo
        loeo_cfg = config.get("loeo", {}) if config else {}
        purge_window = loeo_cfg.get("purge_window", 0)
        embargo_window = loeo_cfg.get("embargo_window", 0)

        from ml2.data.dataset import get_loeo_splits

        for model_name, base_model in self._models.items():
            if verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"LOEO Manifest for model: {model_name} ({n_folds} folds)")
                logger.info(f"{'='*60}")

            fold_results = []

            for fold_entry in manifest["folds"]:
                fold_idx = fold_entry["fold_id"]
                held_out_id = fold_entry["held_out_episode_id"]
                train_indices = fold_entry["train_indices"]
                test_indices = fold_entry["test_indices"]

                if verbose:
                    logger.info(
                        f"  Fold {fold_idx + 1}/{n_folds}: "
                        f"held-out={held_out_id} "
                        f"({len(test_indices)} test samples, {len(train_indices)} raw train samples)"
                    )

                train_dataset, test_dataset = get_loeo_splits(
                    graphs, train_indices, test_indices, purge_window, embargo_window
                )
                
                # Unwrap to lists of graphs for the training/metrics functions
                train_data = [train_dataset.get(i) for i in range(len(train_dataset))]
                test_data = [test_dataset.get(i) for i in range(len(test_dataset))]

                fold_model = copy.deepcopy(base_model).to(device)

                if train_fn is not None:
                    # Note: train_fn signature is expected to take (model, SnapshotDataset, SnapshotDataset) 
                    # but we are passing train_data as a list if we adhere to old interface.
                    # We will pass SnapshotDataset directly to be compatible with ML2Trainer if train_fn wraps it.
                    fold_model = train_fn(fold_model, train_dataset, test_dataset, config, device)

                try:
                    metrics = metrics_fn(fold_model, test_data, device)
                except Exception as e:
                    logger.error(f"  Metrics computation failed for fold {fold_idx}: {e}")
                    metrics = {"error": float("nan")}

                fold_result = FoldResult(
                    fold_idx=fold_idx,
                    held_out_episode_id=held_out_id,
                    metrics=metrics,
                    train_episode_ids=[], # Manifest doesn't explicitly list train episode names
                    num_train_samples=len(train_data),
                    num_test_samples=len(test_data),
                )
                fold_results.append(fold_result)

                if verbose:
                    metrics_str = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
                    logger.info(f"    → {metrics_str}")

            self._results[model_name] = fold_results

        return self._results

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary DataFrame with per-fold and aggregate (mean ± std) results.

        Columns: model, fold, held_out_episode, metric_1, metric_2, ...
        Last row per model: "AGGREGATE" with mean ± std.
        """
        rows = []

        for model_name, fold_results in self._results.items():
            if not fold_results:
                continue

            # Collect all metric keys
            all_metrics = set()
            for fr in fold_results:
                all_metrics.update(fr.metrics.keys())

            # Per-fold rows
            for fr in fold_results:
                row = {
                    "model": model_name,
                    "fold": fr.fold_idx,
                    "held_out_episode": fr.held_out_episode_id,
                    "n_train": fr.num_train_samples,
                    "n_test": fr.num_test_samples,
                }
                for metric in all_metrics:
                    row[metric] = fr.metrics.get(metric, float("nan"))
                rows.append(row)

            # Aggregate row (mean ± std)
            agg_row = {
                "model": model_name,
                "fold": "AGGREGATE",
                "held_out_episode": "mean ± std",
                "n_train": "",
                "n_test": "",
            }
            for metric in all_metrics:
                values = [
                    fr.metrics.get(metric, float("nan"))
                    for fr in fold_results
                    if not np.isnan(fr.metrics.get(metric, float("nan")))
                ]
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    agg_row[metric] = f"{mean:.6f} ± {std:.6f}"
                else:
                    agg_row[metric] = "N/A"
            rows.append(agg_row)

        return pd.DataFrame(rows)

    def get_results(self, model_name: str = None) -> Dict[str, List[FoldResult]]:
        """
        Returns raw results.

        Args:
            model_name: Specific model to retrieve. Returns all if None.
        """
        if model_name:
            return {model_name: self._results.get(model_name, [])}
        return self._results

    def reset(self) -> None:
        """Clear all results (keeps registered models)."""
        self._results = {name: [] for name in self._models}


# --- Convenience functions for ML1/ML2 coordination ---

def create_episodes_from_graphs(
    graphs: List[Data],
    episode_boundaries: List[int] = None,
    episode_ids: List[str] = None,
) -> List[Episode]:
    """
    Segments a flat list of graphs into Episodes.

    Args:
        graphs: Chronologically ordered PyG Data objects.
        episode_boundaries: List of indices where episodes start.
            E.g., [0, 50, 120] → 3 episodes: [0:50), [50:120), [120:end).
            If None, treats entire sequence as one episode.
        episode_ids: Optional names for each episode.

    Returns:
        List of Episode objects.
    """
    if episode_boundaries is None:
        return [Episode(episode_id="full_sequence", data=graphs)]

    boundaries = sorted(episode_boundaries)
    if boundaries[0] != 0:
        boundaries = [0] + boundaries
    boundaries.append(len(graphs))

    episodes = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        ep_id = episode_ids[i] if episode_ids and i < len(episode_ids) else f"episode_{i}"
        episodes.append(Episode(episode_id=ep_id, data=graphs[start:end]))

    return episodes


def default_metrics_fn(
    model: nn.Module,
    test_data: List[Data],
    device: str,
    z_dim: int = 64,
) -> Dict[str, float]:
    """
    Default metrics function for LOEO evaluation.
    Computes next-state error and rollout error at K=3.

    This is the shared contract between ML1 and ML2.
    """
    from ml2.data.dataset import build_paired_sequences, SnapshotDataset
    from ml2.training.metrics import next_state_error, rollout_error

    ds = SnapshotDataset(test_data)
    pairs = build_paired_sequences(ds)

    nse = next_state_error(model, pairs, z_dim=z_dim, device=device) if pairs else float("nan")
    re = rollout_error(model, test_data, K=3, z_dim=z_dim, device=device)

    return {
        "next_state_error": nse,
        "rollout_K3_mean_mse": re["mean_mse"],
        "rollout_K3_cumulative_mse": re["cumulative_mse"],
    }
