import sys
import os
import logging
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml2.data.dataset import SnapshotDataset, get_temporal_splits, build_paired_sequences
from ml2.models.fusion import FusedModel, TemporalOnlyBaseline
from ml2.training.trainer import ML2Trainer
from ml2.training.metrics import next_state_error, compute_all_rollout_errors
from ml2.configs import load_config
from ml2.data.canonical_ucs import build_canonical_graphs, load_canonical_ucs
from ml2.data.ml1_interface import ML1Adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multiseed")

UCS_DIR = os.path.join(os.path.dirname(__file__), "data", "ucs")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ablation_multiseed")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

def main():
    config = load_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info("Loading graphs...")
    canonical_windows, full_edges_df, node_lookup = load_canonical_ucs(Path(UCS_DIR))
    graphs = build_canonical_graphs(canonical_windows, full_edges_df, node_lookup)
    window_starts = [g.timestamp for g in graphs]
    
    ml1_cfg = config.get("ml1_interface", {})
    adapter = ML1Adapter(ml1_output_file=ml1_cfg.get("output_dir"), mock_mode=False)
    graphs = adapter.inject_novelty_into_graphs(graphs, window_starts)

    # Discard 29 warmup graphs so z_store has causal history
    graphs = graphs[29:]

    splits_cfg = config.get("splits", {})
    train_ds, val_ds, test_ds = get_temporal_splits(
        graphs,
        train_frac=splits_cfg.get("train_frac", 0.70),
        val_frac=splits_cfg.get("val_frac", 0.15),
        test_frac=splits_cfg.get("test_frac", 0.15),
    )

    seeds = [42, 100, 2026, 777, 999]
    results = {"fused": [], "temporal_only": []}
    
    z_dim = ml1_cfg.get("z_dim", 64)
    model_cfg = config.get("model", {})
    
    for seed in seeds:
        logger.info(f"=== RUNNING SEED {seed} ===")
        
        for model_type in ["fused", "temporal_only"]:
            set_seed(seed)
            if model_type == "fused":
                model = FusedModel(
                    node_feature_dim=model_cfg.get("node_feature_dim", 11),
                    graph_hidden_dim=model_cfg.get("hidden_dim", 64),
                    graph_embedding_dim=model_cfg.get("embedding_dim", 64),
                    graph_num_layers=model_cfg.get("num_layers", 2),
                    graph_dropout=model_cfg.get("dropout", 0.3),
                    z_dim=z_dim,
                    fusion_hidden_dim=model_cfg.get("fusion_hidden_dim", 128),
                    output_dim=model_cfg.get("node_feature_dim", 11),
                )
            else:
                model = TemporalOnlyBaseline(
                    z_dim=z_dim,
                    fusion_hidden_dim=model_cfg.get("fusion_hidden_dim", 128),
                    output_dim=model_cfg.get("node_feature_dim", 11),
                )
            
            trainer = ML2Trainer(model, config, device=DEVICE, checkpoint_dir=None)
            history = trainer.train(train_ds, val_ds, verbose=False)
            
            # --- CRITICAL FIX: SCALE THE TEST DATA ---
            test_graphs_unscaled = [test_ds[i] for i in range(len(test_ds))]
            test_graphs_scaled = trainer.scaler.transform(test_graphs_unscaled)
            scaled_test_ds = SnapshotDataset(test_graphs_scaled)
            test_pairs = build_paired_sequences(scaled_test_ds)
            
            nse = next_state_error(model, test_pairs, z_dim=z_dim, device=DEVICE)
            
            best_val_loss = min(history["val_loss"])
            best_epoch = history["val_loss"].index(best_val_loss) + 1
            stopped_epoch = len(history["val_loss"])
            
            res = {
                "seed": seed,
                "nse": nse,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch
            }
            results[model_type].append(res)
            logger.info(f"{model_type} -> NSE: {nse:.6f}, Val Loss: {best_val_loss:.6f} @ Ep {best_epoch} (Stop {stopped_epoch})")

    with open(os.path.join(OUTPUT_DIR, "multiseed_results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
