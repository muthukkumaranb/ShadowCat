"""
Fused-vs-Temporal-Only Ablation Harness.

Trains both FusedModel and TemporalOnlyBaseline on identical temporal splits,
then evaluates on:
    1. Next-state error (1-step MSE on test set)
    2. Rollout error vs K (K = 1, 2, 3, 5, 10)

Outputs structured results and optional comparison plots.
"""

import torch
import numpy as np
import pandas as pd
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from torch_geometric.data import Data

from ml2.models.fusion import FusedModel, TemporalOnlyBaseline, GraphOnlyBaseline
from ml2.training.trainer import ML2Trainer
from ml2.training.metrics import next_state_error, compute_all_rollout_errors
from ml2.data.dataset import (
    SnapshotDataset,
    get_temporal_splits,
    build_paired_sequences,
)
from ml2.configs import load_config

logger = logging.getLogger(__name__)


def _build_model(model_type: str, config: dict) -> torch.nn.Module:
    """Factory for instantiating ablation models from config."""
    model_cfg = config.get("model", {})
    ml1_cfg = config.get("ml1_interface", {})

    kwargs = {
        "z_dim": ml1_cfg.get("z_dim", 64),
        "fusion_hidden_dim": model_cfg.get("fusion_hidden_dim", 128),
        "output_dim": model_cfg.get("node_feature_dim", 11),
    }

    if model_type == "fused":
        return FusedModel(
            node_feature_dim=model_cfg.get("node_feature_dim", 11),
            graph_hidden_dim=model_cfg.get("hidden_dim", 64),
            graph_embedding_dim=model_cfg.get("embedding_dim", 64),
            graph_num_layers=model_cfg.get("num_layers", 2),
            graph_dropout=model_cfg.get("dropout", 0.3),
            **kwargs,
        )
    elif model_type == "temporal_only":
        return TemporalOnlyBaseline(**kwargs)
    elif model_type == "graph_only":
        return GraphOnlyBaseline(
            node_feature_dim=model_cfg.get("node_feature_dim", 11),
            graph_hidden_dim=model_cfg.get("hidden_dim", 64),
            graph_embedding_dim=model_cfg.get("embedding_dim", 64),
            graph_num_layers=model_cfg.get("num_layers", 2),
            graph_dropout=model_cfg.get("dropout", 0.3),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_ablation(
    graphs: List[Data],
    config: dict = None,
    model_types: List[str] = None,
    K_values: List[int] = None,
    output_dir: str = None,
    device: str = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Runs the fused-vs-temporal-only ablation experiment.

    Args:
        graphs: Chronologically ordered list of PyG Data objects.
        config: ML2 config dict. Loads default if None.
        model_types: List of model types to compare.
            Defaults to ['fused', 'temporal_only'].
        K_values: Rollout horizons for evaluation.
            Defaults to [1, 2, 3, 5, 10].
        output_dir: Directory for saving results. Creates if doesn't exist.
        device: 'cuda' or 'cpu'. Auto-detects if None.
        verbose: Print progress.

    Returns:
        Dict mapping model_type → {
            'next_state_error': float,
            'rollout_errors': {K: {...}},
            'training_history': {'train_loss': [...], 'val_loss': [...]},
        }
    """
    if config is None:
        config = load_config()

    if model_types is None:
        model_types = ["fused", "temporal_only"]

    if K_values is None:
        K_values = [1, 2, 3, 5, 10]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Split data (same split for all models — fair comparison)
    splits_cfg = config.get("splits", {})
    train_ds, val_ds, test_ds = get_temporal_splits(
        graphs,
        train_frac=splits_cfg.get("train_frac", 0.70),
        val_frac=splits_cfg.get("val_frac", 0.15),
        test_frac=splits_cfg.get("test_frac", 0.15),
    )

    if verbose:
        logger.info(
            f"Ablation splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )

    ml1_cfg = config.get("ml1_interface", {})
    z_dim = ml1_cfg.get("z_dim", 64)

    results = {}

    for model_type in model_types:
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training model: {model_type}")
            logger.info(f"{'='*60}")

        model = _build_model(model_type, config)

        ckpt_dir = os.path.join(output_dir, model_type) if output_dir else None
        trainer = ML2Trainer(model, config, device=device, checkpoint_dir=ckpt_dir)

        # Train
        history = trainer.train(train_ds, val_ds, verbose=verbose)

        # Evaluate: next-state error
        test_pairs = build_paired_sequences(test_ds)
        nse = next_state_error(model, test_pairs, z_dim=z_dim, device=device)

        # Evaluate: rollout error vs K
        test_graphs = [test_ds[i] for i in range(len(test_ds))]
        rollout_results = compute_all_rollout_errors(
            model, test_graphs, K_values=K_values, z_dim=z_dim, device=device
        )

        results[model_type] = {
            "next_state_error": nse,
            "rollout_errors": rollout_results,
            "training_history": history,
        }

        if verbose:
            logger.info(f"  Next-state error: {nse:.6f}")
            for K, re in rollout_results.items():
                logger.info(f"  Rollout K={K}: mean_mse={re['mean_mse']:.6f}")

    # Save results
    if output_dir:
        _save_ablation_results(results, output_dir)

    return results


def _save_ablation_results(results: Dict, output_dir: str) -> None:
    """Save ablation results to CSV and JSON."""

    # JSON dump (full results)
    json_path = os.path.join(output_dir, "ablation_results.json")

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(json_path, "w") as f:
        json.dump(_convert(results), f, indent=2)

    # CSV summary table
    rows = []
    for model_type, res in results.items():
        row = {"model": model_type, "next_state_error": res["next_state_error"]}
        for K, re in res["rollout_errors"].items():
            row[f"rollout_K{K}_mean_mse"] = re["mean_mse"]
            row[f"rollout_K{K}_cumulative_mse"] = re["cumulative_mse"]
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "ablation_summary.csv")
    df.to_csv(csv_path, index=False)

    logger.info(f"Ablation results saved to {output_dir}")


def generate_ablation_report(
    results: Dict[str, Dict],
    output_path: str = None,
) -> str:
    """
    Generates a markdown report comparing ablation results.

    Args:
        results: Output from run_ablation().
        output_path: Optional path to write the report.

    Returns:
        Markdown string.
    """
    lines = ["# Fused vs Temporal-Only Ablation Report\n"]

    # Next-state error comparison
    lines.append("## Next-State Error (1-step MSE)\n")
    lines.append("| Model | MSE |")
    lines.append("|-------|-----|")
    for model_type, res in results.items():
        lines.append(f"| {model_type} | {res['next_state_error']:.6f} |")
    lines.append("")

    # Rollout error table
    lines.append("## Rollout Error vs K\n")

    # Collect all K values
    all_Ks = set()
    for res in results.values():
        all_Ks.update(res["rollout_errors"].keys())
    all_Ks = sorted(all_Ks)

    header = "| K | " + " | ".join(results.keys()) + " |"
    separator = "|---|" + "|".join(["---"] * len(results)) + "|"
    lines.append(header)
    lines.append(separator)

    for K in all_Ks:
        row = f"| {K} |"
        for model_type, res in results.items():
            mse = res["rollout_errors"].get(K, {}).get("mean_mse", float("nan"))
            row += f" {mse:.6f} |"
        lines.append(row)
    lines.append("")

    # Delta analysis
    model_types = list(results.keys())
    if "fused" in results and "temporal_only" in results:
        fused_nse = results["fused"]["next_state_error"]
        temp_nse = results["temporal_only"]["next_state_error"]
        delta_pct = ((temp_nse - fused_nse) / temp_nse) * 100 if temp_nse > 0 else 0

        lines.append("## Key Finding\n")
        if delta_pct > 0:
            lines.append(
                f"Fused model reduces next-state error by **{delta_pct:.1f}%** "
                f"compared to temporal-only baseline."
            )
        else:
            lines.append(
                f"Temporal-only baseline outperforms fused model by "
                f"**{abs(delta_pct):.1f}%** on next-state error."
            )

    report = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report
