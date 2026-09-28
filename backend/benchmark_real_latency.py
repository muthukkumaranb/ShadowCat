"""
Real Latency Benchmark for Production Deployed ShadowCat Pipeline
==================================================================
Measures genuine wall-clock inference latency on the actual deployed pipeline:
  1. Sequence ingestion & 406-dim feature extraction / tensor creation
  2. 32-dim PCA feature projection
  3. Stacked & calibrated Residual LSTM 37-fold ensemble inference (detection & onset)
  4. World Model continuous dynamics & multi-step trajectory rollout
  5. Real Dynamic / Canonical Graph Topology Resolution & Propagation Traversal
  6. 4-Stage Cryptographic Hash Generation for Lineage Integrity
  7. On-Chain Hyperledger Fabric Consensus Transaction Commit (measured on real test network)

Uses real UCS benchmark windows from `data-engineering/data/ucs/ucs_windows.parquet`,
evaluating both Benign and Malicious attack episodes across repeated trials.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from backend.predict import ShadowcatPipeline
import backend.predict as predict_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_benchmark(n_trials: int = 30, n_warmup: int = 5, n_fabric_trials: int = 5):
    print("=" * 80)
    print("  SHADOWCAT REAL LATENCY BENCHMARK — PRODUCTION DEPLOYED PIPELINE")
    print("=" * 80)
    print(f"Configuration: {n_warmup} warmup trials, {n_trials} ML/graph trials, {n_fabric_trials} live Fabric trials")

    # 1. Initialize Deployed Pipeline
    print("\n[1/5] Initializing deployed ShadowcatPipeline (loading checkpoints, PCA, topology)...")
    t_init_start = time.perf_counter()
    pipeline = ShadowcatPipeline()
    t_init_end = time.perf_counter()
    print(f" -> Pipeline initialized in {(t_init_end - t_init_start)*1000:.2f} ms")

    # 2. Load Real UCS Parquet Windows
    print("\n[2/5] Loading real UCS benchmark windows from parquet...")
    parquet_path = repo_root / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet dataset not found at {parquet_path}")

    df_windows = pd.read_parquet(parquet_path).sort_values("window_start_utc")
    benign_pool = df_windows[df_windows["has_malicious_flows"] == 0]
    attack_pool = df_windows[df_windows["has_malicious_flows"] == 1]

    print(f" -> Dataset loaded: {len(df_windows)} total windows ({len(benign_pool)} benign, {len(attack_pool)} attack)")

    # Construct test sample sequences (30 consecutive windows per sequence)
    test_sequences = []
    for i in range(0, min(150, len(benign_pool) - 30), 30):
        test_sequences.append(("Benign", benign_pool.iloc[i : i + 30].copy()))
    for i in range(0, min(150, len(attack_pool) - 30), 30):
        test_sequences.append(("Malicious", attack_pool.iloc[i : i + 30].copy()))

    print(f" -> Prepared {len(test_sequences)} distinct real 30-window evaluation episodes")

    # Temporarily stub fabric notarization in predict_module for pure ML/graph latency measurement
    orig_notarize_alert = predict_module.notarize_alert
    orig_record_lineage = predict_module.record_prediction_lineage
    predict_module.notarize_alert = lambda *args, **kwargs: True
    predict_module.record_prediction_lineage = lambda *args, **kwargs: True
    logging.disable(logging.INFO)

    # 3. Warmup Phase
    print(f"\n[3/5] Warming up PyTorch inference engines & caches ({n_warmup} iterations)...")
    for i in range(n_warmup):
        seq_label, seq_df = test_sequences[i % len(test_sequences)]
        _ = pipeline.predict(seq_df, source_type="flows")
    print(" -> Warmup completed successfully.")

    # 4. Timed Benchmark Runs (ML + Graph Traversal)
    print(f"\n[4/5] Executing {n_trials} timed inference passes across real sequences...")
    
    total_ml_latencies = []
    extraction_latencies = []
    pca_latencies = []
    lstm_latencies = []
    world_model_latencies = []
    graph_latencies = []
    hashing_latencies = []

    for trial_idx in range(n_trials):
        seq_label, seq_df = test_sequences[trial_idx % len(test_sequences)]

        # --- Component-level fine-grained profiling ---
        # A. Feature extraction from raw window dataframe
        t0 = time.perf_counter()
        window_df = pipeline.extractor.extract(seq_df, source_type="flows")
        model_tensor = pipeline.extractor.extract_model_tensor(seq_df, source_type="flows")
        seq_30x406 = np.ascontiguousarray(model_tensor[-30:], dtype=np.float32)
        seq_tensor = torch.as_tensor(seq_30x406, dtype=torch.float32).unsqueeze(0).to(pipeline.device)
        t_ext = time.perf_counter()
        extraction_latencies.append((t_ext - t0) * 1000.0)

        # B. 32-dim PCA projection
        t1 = time.perf_counter()
        cols = pipeline.pca_features if (pipeline.pca_features is not None and len(pipeline.pca_features) == 406) else [f"f_{i}" for i in range(406)]
        df_for_pca = pd.DataFrame(seq_30x406, columns=cols)
        if getattr(pipeline, "pca_scaler", None) is not None:
            scaled_vals = pipeline.pca_scaler.transform(df_for_pca[cols].to_numpy(dtype=np.float64))
            df_for_pca = pd.DataFrame(scaled_vals, columns=cols)
        transformed = pipeline.pca.transform(df_for_pca)
        pca_cols = [f"pca_{i}" for i in range(32)]
        seq_32 = transformed[pca_cols].to_numpy(dtype=np.float32)
        seq_tensor_32 = torch.as_tensor(seq_32, dtype=torch.float32).unsqueeze(0).to(pipeline.device)
        t_pca = time.perf_counter()
        pca_latencies.append((t_pca - t1) * 1000.0)

        # C. Stacked Residual LSTM 37-fold ensemble inference
        t2 = time.perf_counter()
        curr_window_406 = seq_30x406[-1]
        det_probs = [m.predict_proba(seq_tensor_32, curr_window_406) for m in pipeline.stacked_detection_models]
        onset_hazards = pipeline._predict_hazard_ensemble(seq_30x406)
        t_lstm = time.perf_counter()
        lstm_latencies.append((t_lstm - t2) * 1000.0)

        # D. World Model continuous dynamics & latent rollout
        t3 = time.perf_counter()
        with torch.no_grad():
            pred_mean_t1, pred_std_t1 = pipeline.world_model(seq_tensor)
            z_t = pipeline.world_model.extract_latent_z(seq_tensor)
        t_wm = time.perf_counter()
        world_model_latencies.append((t_wm - t3) * 1000.0)

        # E. Real dynamic / canonical graph resolution & graph traversal
        t4 = time.perf_counter()
        from backend.graph_traversal import compute_graph_traversal, classify_endpoint_role
        w_df = seq_df.tail(1)
        w_id = w_df["window_id"].iloc[0]
        window_edges = pipeline.ucs_edges[pipeline.ucs_edges["window_id"] == w_id]
        id_to_ep = dict(zip(pipeline.node_lookup["node_id"], pipeline.node_lookup["endpoint_identifier"]))
        edge_subset = window_edges.head(45)
        active_nids = sorted(set(edge_subset["src_node_id"]).union(set(edge_subset["dst_node_id"])))
        nodes_info = [{"id": str(id_to_ep.get(n, n)), "role": classify_endpoint_role(str(id_to_ep.get(n, n)))} for n in active_nids]
        edges_info = [{"source": str(id_to_ep.get(r.src_node_id, r.src_node_id)), "target": str(id_to_ep.get(r.dst_node_id, r.dst_node_id)), "flow_count": int(getattr(r, "flow_count", 1))} for r in edge_subset.itertuples(index=False)]
        _ = compute_graph_traversal(graph_nodes=nodes_info, graph_edges=edges_info, flagged_flows=[], max_k=5)
        t_graph = time.perf_counter()
        graph_latencies.append((t_graph - t4) * 1000.0)

        # F. Real Lineage Hashing
        t5 = time.perf_counter()
        import hashlib
        raw_hash = hashlib.sha256(str(seq_df.values).encode("utf-8")).hexdigest()
        feat_hash = hashlib.sha256(seq_30x406.tobytes()).hexdigest()
        pred_hash = hashlib.sha256(str(det_probs).encode("utf-8")).hexdigest()
        t_hash = time.perf_counter()
        hashing_latencies.append((t_hash - t5) * 1000.0)

        # G. Full end-to-end ML + Graph pipeline call
        t_e2e_0 = time.perf_counter()
        res = pipeline.predict(seq_df, source_type="flows")
        t_e2e_1 = time.perf_counter()
        total_ml_latencies.append((t_e2e_1 - t_e2e_0) * 1000.0)

    # Restore original fabric functions and logging
    logging.disable(logging.NOTSET)
    predict_module.notarize_alert = orig_notarize_alert
    predict_module.record_prediction_lineage = orig_record_lineage

    # 5. Measure On-Chain Fabric Ledger Commit Latency
    print(f"\n[5/5] Measuring live Hyperledger Fabric transaction commit latency ({n_fabric_trials} trials)...")
    fabric_latencies = []
    fabric_success_count = 0
    fabric_fallback_count = 0
    from backend.fabric_bridge import record_prediction_lineage
    for trial_idx in range(n_fabric_trials):
        t_fab_0 = time.perf_counter()
        commit_ok = record_prediction_lineage(
            lineage_id=f"bench_lin_{trial_idx}_{int(time.time() * 1000)}",
            raw_data_hash="a1b2c3d4e5f67890",
            feature_hash="f1e2d3c4b5a67890",
            model_id="lstm-stacked-v1",
            prediction_hash="9988776655443322",
            severity="HIGH" if trial_idx % 2 == 0 else "LOW",
            timestamp="2026-09-26T10:00:00Z",
            target_node="172.31.69.21",
        )
        t_fab_1 = time.perf_counter()
        dur_ms = (t_fab_1 - t_fab_0) * 1000.0
        if commit_ok:
            fabric_success_count += 1
            fabric_latencies.append(dur_ms)
            print(f" -> Trial {trial_idx + 1}/{n_fabric_trials}: ON-CHAIN COMMIT CONFIRMED ({dur_ms:.2f} ms)")
        else:
            fabric_fallback_count += 1
            print(f" -> Trial {trial_idx + 1}/{n_fabric_trials}: FELL BACK TO LOCAL/SHA256 (excluded from on-chain commit latency, {dur_ms:.2f} ms)")

    # 6. Compute Statistics
    def stats(arr):
        return {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "median_ms": float(np.median(arr)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
        }

    if fabric_success_count > 0:
        fabric_commit_stats = stats(fabric_latencies)
        fabric_commit_stats["trials_attempted"] = n_fabric_trials
        fabric_commit_stats["trials_committed"] = fabric_success_count
        fabric_commit_stats["trials_fell_back"] = fabric_fallback_count
        fabric_commit_stats["status"] = "COMMITTED_ON_CHAIN"
    else:
        fabric_commit_stats = f"SKIPPED - Fabric unreachable during this run, {fabric_fallback_count}/{n_fabric_trials} trials fell back to sha256_fallback"

    results = {
        "pipeline_name": "ShadowCat Stacked Residual LSTM + Graph Traversal",
        "benchmark_environment": {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "n_ml_trials": n_trials,
            "n_fabric_trials": n_fabric_trials,
            "window_size": 30,
            "raw_feature_dim": 406,
            "pca_dim": 32,
            "stacked_ensemble_folds": len(pipeline.stacked_detection_models),
        },
        "ml_inference_pipeline_e2e": stats(total_ml_latencies),
        "stages": {
            "1_feature_extraction_ms": stats(extraction_latencies),
            "2_pca_projection_ms": stats(pca_latencies),
            "3_stacked_lstm_37folds_ms": stats(lstm_latencies),
            "4_world_model_dynamics_ms": stats(world_model_latencies),
            "5_real_graph_traversal_ms": stats(graph_latencies),
            "6_lineage_hashing_ms": stats(hashing_latencies),
        },
        "onchain_fabric_commit": fabric_commit_stats,
        "ml_throughput_windows_per_sec": float(1000.0 / np.median(total_ml_latencies)),
    }

    # Print Summary Table
    e2e = results["ml_inference_pipeline_e2e"]
    fab = results["onchain_fabric_commit"]
    print("\n" + "=" * 80)
    print(" REAL LATENCY BENCHMARK RESULTS (WALL-CLOCK PER 30-WINDOW EVALUATION)")
    print("=" * 80)
    print(f"{'Pipeline Stage':<38} | {'Median (P50)':<12} | {'P95':<10} | {'Mean ± Std':<16}")
    print("-" * 78)
    for stg_name, s in results["stages"].items():
        clean_name = stg_name.split("_", 1)[1].replace("_ms", "").replace("_", " ").title()
        print(f"{clean_name:<38} | {s['median_ms']:>8.2f} ms | {s['p95_ms']:>6.2f} ms | {s['mean_ms']:>6.2f} ± {s['std_ms']:<5.2f} ms")
    print("-" * 80)
    print(f"{'TOTAL ML + GRAPH PIPELINE (P50)':<38} | {e2e['median_ms']:>8.2f} ms | {e2e['p95_ms']:>6.2f} ms | {e2e['mean_ms']:>6.2f} ± {e2e['std_ms']:<5.2f} ms")
    print(f"Min / Max: {e2e['min_ms']:.2f} ms / {e2e['max_ms']:.2f} ms")
    print(f"Core Inference Throughput: {results['ml_throughput_windows_per_sec']:.2f} windows / second")
    print("-" * 80)
    if isinstance(fab, dict) and "median_ms" in fab:
        header_fab = f"FABRIC COMMIT ({fab.get('trials_committed', len(fabric_latencies))}/{n_fabric_trials} ON-CHAIN)"
        print(f"{header_fab:<38} | {fab['median_ms']:>8.2f} ms | {fab['p95_ms']:>6.2f} ms | {fab['mean_ms']:>6.2f} ± {fab['std_ms']:<5.2f} ms")
    else:
        print(f"{'FABRIC ON-CHAIN COMMIT':<38} | {str(fab)}")
    print("=" * 80)

    # Save to JSON
    out_path = repo_root / "docs" / "real_pipeline_latency_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed latency telemetry written to: {out_path}")

    return results


if __name__ == "__main__":
    run_benchmark(n_trials=20, n_warmup=3, n_fabric_trials=3)
