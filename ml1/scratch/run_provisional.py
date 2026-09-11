"""
ML2 Runner: Provisional Ablation + Lateral-Movement Spot-Check

This script performs two tasks:
1. PROVISIONAL ABLATION: Runs fused-vs-temporal-only comparison with novelty
   slot zeroed (ML1 outputs pending). Results labeled "PROVISIONAL".
2. LATERAL-MOVEMENT SPOT-CHECK: Verifies anomaly-velocity threshold fires
   at the infiltration phase transition in CICIDS2018 day 01-03-2018.

Data source: UCS Precomputed Edgelists
    E:/SIH 2026 - .../data/ucs/ucs_graph_edgelists.parquet

NOTE: Novelty slot (feature 10) is zeroed throughout. This is a known
placeholder — do NOT treat ablation numbers as final. Labeled as
"PROVISIONAL — novelty slot zeroed pending ML1 export" per project protocol.
"""

import sys
import os
import logging
import json

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from pathlib import Path

# PyG imports
from torch_geometric.data import Data

from ml2.data.dataset import SnapshotDataset, get_temporal_splits, build_paired_sequences
from ml2.models.fusion import FusedModel, TemporalOnlyBaseline
from ml2.training.trainer import ML2Trainer
from ml2.training.metrics import next_state_error, compute_all_rollout_errors
from ml2.experiments.lateral_movement_demo import (
    compute_embedding_trajectory,
    detect_anomalous_transitions,
    compute_graph_statistics,
)
from ml2.configs import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ml2_runner")

# ─── Constants ───────────────────────────────────────────────────────────────
EDGELISTS_PATH_ACTUAL = r"E:\SIH 2026 - UCS Ingestion Pipeline (Main)\data\ucs\ucs_graph_edgelists.parquet"
WINDOWS_PATH_ACTUAL = r"E:\SIH 2026 - UCS Ingestion Pipeline (Main)\data\ucs\ucs_windows.parquet"

INFILTRATION_DAY = "01-03-2018"
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_edgelists(data_path: str, day: str = None) -> pd.DataFrame:
    """Load and optionally filter precomputed edgelists."""
    logger.info(f"Loading edgelists from {data_path}")
    df = pd.read_parquet(data_path)
    if day:
        df = df[df["source_day"] == day].copy()
        logger.info(f"Filtered to {day}: {len(df)} edges")
    return df


def build_graphs_from_edgelists(edges_df: pd.DataFrame) -> tuple:
    """
    Build PyG Data objects from precomputed edgelists.
    Returns (graphs, window_starts).
    """
    # Sort by window_start_utc to ensure temporal ordering
    edges_df = edges_df.sort_values("window_start_utc")
    
    windows = edges_df["window_start_utc"].unique()
    logger.info(f"Building {len(windows)} graph snapshots from precomputed edges...")

    graphs = []
    window_starts = []

    for t in windows:
        w_df = edges_df[edges_df["window_start_utc"] == t]
        
        src = w_df["src_node_id"].values
        dst = w_df["dst_node_id"].values
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        # Edge attributes (optional, but good to have)
        # flow_count, byte_count_sum, packet_count_sum, duration_mean_sec, protocol_mode
        edge_attr = torch.tensor(
            w_df[["flow_count", "byte_count_sum", "packet_count_sum", "duration_mean_sec", "protocol_mode"]].values,
            dtype=torch.float
        )

        # Node features (11 dims)
        # 0: in_degree, 1: out_degree, 2: flows_in, 3: flows_out, 4: bytes_in, 5: bytes_out
        # 6: pkts_in, 7: pkts_out, 8: dur_in, 9: dur_out, 10: novelty_tminus1 (zeroed)
        num_nodes = max(src.max(), dst.max()) + 1 if len(src) > 0 else 0
        if num_nodes == 0:
            continue
            
        x = torch.zeros((num_nodes, 11), dtype=torch.float)
        
        # Aggregate out features (from src)
        for i, s in enumerate(src):
            x[s, 1] += 1  # out_degree
            x[s, 3] += w_df.iloc[i]["flow_count"]
            x[s, 5] += w_df.iloc[i]["byte_count_sum"]
            x[s, 7] += w_df.iloc[i]["packet_count_sum"]
            x[s, 9] += w_df.iloc[i]["duration_mean_sec"]
            
        # Aggregate in features (from dst)
        for i, d in enumerate(dst):
            x[d, 0] += 1  # in_degree
            x[d, 2] += w_df.iloc[i]["flow_count"]
            x[d, 4] += w_df.iloc[i]["byte_count_sum"]
            x[d, 6] += w_df.iloc[i]["packet_count_sum"]
            x[d, 8] += w_df.iloc[i]["duration_mean_sec"]
            
        # Normalize duration by degree (simple average)
        out_deg = x[:, 1].clone()
        out_deg[out_deg == 0] = 1
        x[:, 9] /= out_deg
        
        in_deg = x[:, 0].clone()
        in_deg[in_deg == 0] = 1
        x[:, 8] /= in_deg

        # Note: x[:, 10] (novelty) remains 0.0

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, timestamp=t)
        graphs.append(graph)
        window_starts.append(t)

    logger.info(f"Built {len(graphs)} graphs")
    return graphs, window_starts


# ═══════════════════════════════════════════════════════════════════════════
# TASK 1: PROVISIONAL ABLATION (novelty-zeroed)
# ═══════════════════════════════════════════════════════════════════════════


def run_provisional_ablation(graphs, config):
    logger.info("=" * 70)
    logger.info("PROVISIONAL ABLATION — novelty slot zeroed pending ML1 export")
    logger.info("=" * 70)

    # Temporal split
    splits_cfg = config.get("splits", {})
    train_ds, val_ds, test_ds = get_temporal_splits(
        graphs,
        train_frac=splits_cfg.get("train_frac", 0.70),
        val_frac=splits_cfg.get("val_frac", 0.15),
        test_frac=splits_cfg.get("test_frac", 0.15),
    )
    logger.info(
        f"Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    ml1_cfg = config.get("ml1_interface", {})
    z_dim = ml1_cfg.get("z_dim", 64)
    model_cfg = config.get("model", {})

    K_values = config.get("ablation", {}).get("K_values", [1, 2, 3, 5, 10])
    results = {}

    for model_type in ["fused", "temporal_only"]:
        logger.info(f"\n--- Training: {model_type} ---")

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

        ckpt_dir = os.path.join(OUTPUT_DIR, "ablation", model_type)
        trainer = ML2Trainer(model, config, device=DEVICE, checkpoint_dir=ckpt_dir)
        history = trainer.train(train_ds, val_ds, verbose=True)

        # Evaluate
        test_pairs = build_paired_sequences(test_ds)
        test_graphs = [test_ds[i] for i in range(len(test_ds))]

        nse = next_state_error(model, test_pairs, z_dim=z_dim, device=DEVICE)
        rollout_res = compute_all_rollout_errors(
            model, test_graphs, K_values=K_values, z_dim=z_dim, device=DEVICE
        )

        results[model_type] = {
            "next_state_error": nse,
            "rollout_errors": rollout_res,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "epochs_trained": len(history["train_loss"]),
        }

        logger.info(f"  Next-state error: {nse:.6f}")
        for K, re in rollout_res.items():
            logger.info(f"  Rollout K={K}: mean_mse={re['mean_mse']:.6f}")

    # Print comparison table
    logger.info("\n" + "=" * 70)
    logger.info("PROVISIONAL ABLATION RESULTS (novelty slot zeroed pending ML1 export)")
    logger.info("=" * 70)

    header = f"{'Metric':<30} | {'Fused':>15} | {'Temporal-Only':>15} | {'Delta':>10}"
    logger.info(header)
    logger.info("-" * len(header))

    fused_nse = results["fused"]["next_state_error"]
    temp_nse = results["temporal_only"]["next_state_error"]
    delta = ((temp_nse - fused_nse) / temp_nse * 100) if temp_nse > 0 else 0
    logger.info(
        f"{'Next-state MSE':<30} | {fused_nse:>15.6f} | {temp_nse:>15.6f} | {delta:>+9.1f}%"
    )

    for K in K_values:
        if K in results["fused"]["rollout_errors"] and K in results["temporal_only"]["rollout_errors"]:
            f_mse = results["fused"]["rollout_errors"][K]["mean_mse"]
            t_mse = results["temporal_only"]["rollout_errors"][K]["mean_mse"]
            d = ((t_mse - f_mse) / t_mse * 100) if t_mse > 0 and not np.isnan(t_mse) else 0
            logger.info(
                f"{'Rollout K=' + str(K) + ' mean MSE':<30} | {f_mse:>15.6f} | {t_mse:>15.6f} | {d:>+9.1f}%"
            )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# TASK 2: LATERAL-MOVEMENT SPOT-CHECK
# ═══════════════════════════════════════════════════════════════════════════


def run_lateral_movement_spotcheck(edges_df, config):
    """
    Spot-check: verify anomaly-velocity threshold fires at the infiltration
    phase transition in the CICIDS2018 infiltration episodes.
    """
    logger.info("=" * 70)
    logger.info("LATERAL-MOVEMENT SPOT-CHECK — CICIDS2018 Infiltration Episode")
    logger.info("=" * 70)

    # Load ucs_windows to find exact episode times
    try:
        windows_df = pd.read_parquet(WINDOWS_PATH_ACTUAL)
        ep1 = windows_df[windows_df['episode_id'] == '01-03-2018_Infiltration-Compromise_0']
        ep2 = windows_df[windows_df['episode_id'] == '01-03-2018_Infiltration-Portscan_0']
        
        comp_start = ep1['window_start_utc'].min()
        comp_end = ep1['window_start_utc'].max()
        scan_start = ep2['window_start_utc'].min()
        scan_end = ep2['window_start_utc'].max()
        
        logger.info(f"Compromise phase: {comp_start} to {comp_end}")
        logger.info(f"Portscan phase: {scan_start} to {scan_end}")
        
    except Exception as e:
        logger.error(f"Failed to load ucs_windows.parquet for exact boundaries: {e}")
        return None

    # Context window: 30 mins before compromise to 15 mins after portscan
    def _to_utc_ts(t):
        ts = pd.Timestamp(t)
        return ts.tz_localize('UTC') if ts.tzinfo is None else ts

    context_start = _to_utc_ts(comp_start) - pd.Timedelta(minutes=30)
    context_end = _to_utc_ts(scan_end) + pd.Timedelta(minutes=15)

    # Convert edges window_start_utc to tz-aware for safe comparison
    edges_df = edges_df.copy()
    
    # Check if tz-aware
    sample_time = edges_df["window_start_utc"].iloc[0]
    if pd.Timestamp(sample_time).tzinfo is None:
        edges_df["window_start_utc"] = edges_df["window_start_utc"].dt.tz_localize("UTC")
        
    context_edges = edges_df[
        (edges_df["window_start_utc"] >= context_start) &
        (edges_df["window_start_utc"] <= context_end)
    ]
    
    logger.info(f"Context window: {context_start} → {context_end} ({len(context_edges)} edges)")

    # Build graphs
    graphs, window_starts = build_graphs_from_edgelists(context_edges)
    logger.info(f"Built {len(graphs)} context graphs")

    if len(graphs) < 3:
        logger.error("Too few graphs for spot-check")
        return None

    # Run embedding trajectory with untrained GraphBranch
    from ml2.models.graphsage import GraphBranch
    model = GraphBranch(in_channels=11, hidden_dim=64, embedding_dim=64)
    trajectory = compute_embedding_trajectory(model, graphs, device=DEVICE)

    embeddings = trajectory["embeddings"]
    velocities = trajectory["velocities"]

    logger.info(f"Embedding trajectory: {embeddings.shape}")
    logger.info(f"Velocity stats: mean={velocities.mean():.6f}, std={velocities.std():.6f}, "
                f"min={velocities.min():.6f}, max={velocities.max():.6f}")

    # Detect anomalous transitions
    anomalous_2sigma = detect_anomalous_transitions(velocities, threshold_sigma=2.0)
    for sigma in [1.0, 1.5, 2.0, 2.5, 3.0]:
        anom = detect_anomalous_transitions(velocities, threshold_sigma=sigma)
        logger.info(f"  Threshold σ={sigma:.1f}: {len(anom)} anomalous transitions at indices {anom}")

    # The phase transition gap is between comp_end and scan_start
    gap_start_time = _to_utc_ts(comp_end)
    gap_end_time = _to_utc_ts(scan_start)

    logger.info(f"\n--- Phase Transition Analysis ---")
    logger.info(f"Gap: {gap_start_time} → {gap_end_time}")

    transition_indices = []
    for i, ws in enumerate(window_starts):
        ws_ts = _to_utc_ts(ws)
        if (ws_ts >= gap_start_time - pd.Timedelta(minutes=15) and
            ws_ts <= gap_end_time + pd.Timedelta(minutes=15)):
            transition_indices.append(i)

    logger.info(f"Graph indices near phase transition: {transition_indices}")

    threshold_fires = False
    transition_hits = [idx for idx in anomalous_2sigma if idx in transition_indices or (idx + 1) in transition_indices]

    if transition_hits:
        logger.info(f"✅ THRESHOLD FIRES at phase transition! Anomalous indices near gap: {transition_hits}")
        threshold_fires = True
    else:
        logger.warning(f"⚠ Threshold did NOT fire at phase transition with σ=2.0")
        anomalous_1sigma = detect_anomalous_transitions(velocities, threshold_sigma=1.0)
        hits_1sigma = [idx for idx in anomalous_1sigma if idx in transition_indices or (idx + 1) in transition_indices]
        if hits_1sigma:
            logger.info(f"  → Fires at σ=1.0 threshold: {hits_1sigma}")
            threshold_fires = True

    # Save results
    os.makedirs(os.path.join(OUTPUT_DIR, "lateral_movement"), exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "lateral_movement", "embeddings.npy"), embeddings)
    np.save(os.path.join(OUTPUT_DIR, "lateral_movement", "velocities.npy"), velocities)

    result = {
        "n_graphs": len(graphs),
        "phase_boundary": str(gap_start_time),
        "threshold_fires_at_transition": threshold_fires,
        "anomalous_2sigma": anomalous_2sigma,
        "velocity_mean": float(velocities.mean()),
        "velocity_std": float(velocities.std()),
    }

    with open(os.path.join(OUTPUT_DIR, "lateral_movement", "spotcheck_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    config = load_config()

    if not os.path.exists(EDGELISTS_PATH_ACTUAL):
        logger.error(f"Edgelists file not found at {EDGELISTS_PATH_ACTUAL}")
        return

    # Load full day's edgelists for the infiltration day
    edges_df = load_edgelists(EDGELISTS_PATH_ACTUAL, day=INFILTRATION_DAY)
    
    if len(edges_df) == 0:
        logger.error("No edges found for infiltration day.")
        return

    # Task 2: Lateral-movement spot-check
    spotcheck_result = run_lateral_movement_spotcheck(edges_df, config)

    # Task 1: Provisional ablation
    # Build graphs from the full day's data
    graphs, window_starts = build_graphs_from_edgelists(edges_df)
    ablation_result = run_provisional_ablation(graphs, config)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("RUN COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("⚠ REMINDER: All ablation numbers are PROVISIONAL —")
    logger.info("  novelty slot zeroed pending ML1 export (deviation_predictions.csv)")
    if spotcheck_result:
        fires = spotcheck_result["threshold_fires_at_transition"]
        logger.info(f"Lateral-movement threshold fires at transition: {'YES ✅' if fires else 'NO ⚠'}")


if __name__ == "__main__":
    main()
