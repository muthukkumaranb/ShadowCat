"""
ML1 → ML2 Deviation Exporter

Generates the real ML1 world-model deviation predictions in exactly the format
required by ML2's ML1Adapter (per-window JSON files).

Contract (from ML1Adapter._preload):
  - Directory of per-window JSON files: {output_dir}/{window_id}.json
  - Each file maps host_id → novelty_score (float)

Temporal alignment:
  - ML2's inject_novelty_into_graphs performs t-1 shift itself
    (line 172: prev_window_id = str(window_ids[i - 1]))
  - Therefore ML1 stores deviation AT window t, NOT shifted

Deviation definition:
  d(t) = mean(|S_actual(t) - S_predicted(t)|)  (mean absolute error in PCA space)
  This is the established deviation_scores() from lstm.probabilistic.

Since ML1 produces a global window-level deviation (not per-host), every host
in a given window receives the same deviation score.
"""
import json
import sys
import os
import platform
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))

from lstm.model import LSTMGaussianWorldModel
from lstm.probabilistic import deviation_scores, train_gaussian
from lstm.ucs import UCSConfig, apply_training_only_pca, build_next_state_sequences, validate_ucs_windows
from lstm.utils import get_device, set_seed


def get_git_hash():
    """Get current git hash if available."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=str(workspace_dir)
        )
        return result.stdout.strip() if result.returncode == 0 else 'unavailable'
    except Exception:
        return 'unavailable'


def main():
    set_seed(42)
    device = get_device()

    # --- Paths ---
    data_path = workspace_dir / 'data' / 'ucs' / 'ucs_windows.parquet'
    edges_path = workspace_dir / 'data' / 'ucs' / 'ucs_graph_edgelists.parquet'
    manifest_path = workspace_dir / 'artifacts' / 'loeo' / 'corrected_37fold_manifest.json'
    model_dir = workspace_dir / 'artifacts' / 'lstm' / 'loeo_world_model'
    out_dir = workspace_dir / 'artifacts' / 'lstm' / 'ml2_export'
    json_dir = out_dir / 'per_window_json'
    json_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("Loading UCS windows...")
    windows = pd.read_parquet(data_path).sort_values('window_start_utc').reset_index(drop=True)

    print("Loading edge lists for node-to-window mapping...")
    edges = pd.read_parquet(edges_path)

    # Build mapping: window_start_utc → (window_id, set of active node_ids)
    # Use vectorized groupby instead of row-by-row iteration
    edges['window_start_str'] = edges['window_start_utc'].astype(str)

    # window_id_map: window_start_utc string → window_id string
    wid_df = edges[['window_start_str', 'window_id']].drop_duplicates('window_start_str')
    window_id_map = dict(zip(wid_df['window_start_str'], wid_df['window_id'].astype(str)))

    # window_nodes: window_start_utc string → set of active node_ids
    src_groups = edges.groupby('window_start_str')['src_node_id'].apply(set)
    dst_groups = edges.groupby('window_start_str')['dst_node_id'].apply(set)
    window_nodes = {}
    for ts in src_groups.index:
        window_nodes[ts] = src_groups[ts] | dst_groups.get(ts, set())
    for ts in dst_groups.index:
        if ts not in window_nodes:
            window_nodes[ts] = dst_groups[ts]

    print(f"Found {len(window_nodes)} windows with active nodes in edge lists")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    config = UCSConfig()
    features = validate_ucs_windows(windows, config=config)

    # --- Generate deviations per fold ---
    all_deviation_rows = []

    for fold in manifest['folds']:
        fold_id = fold['fold_id']
        print(f"Evaluating World Model on Fold {fold_id}...")

        train_indices = np.array(fold['train_indices'])
        test_indices = np.array(fold['test_indices'])

        fold_frame = windows.copy()
        fold_frame['split'] = 'none'
        fold_frame.loc[test_indices, 'split'] = 'test'

        train_df_sorted = fold_frame.loc[train_indices].sort_values('window_start_utc')
        n_train = len(train_df_sorted)
        val_start = int(n_train * 0.8)
        actual_train_indices = train_df_sorted.index[:val_start]
        actual_val_indices = train_df_sorted.index[val_start:]

        fold_frame.loc[actual_train_indices, 'split'] = 'train'
        fold_frame.loc[actual_val_indices, 'split'] = 'val'

        active_mask = fold_frame['split'].isin(['train', 'val', 'test'])
        purged = fold_frame[active_mask].copy()

        transformed, pca = apply_training_only_pca(purged, features, 32, random_state=42)
        pca_features = [f"pca_{i}" for i in range(32)]

        sequences = build_next_state_sequences(transformed, features=pca_features, config=config)
        train_set, val_set = sequences['train'], sequences['val']

        full_transformed = pca.transform(fold_frame)
        full_transformed_np = full_transformed[pca_features].to_numpy(dtype=np.float32)

        histories = []
        labels = []
        test_window_timestamps = []

        idx_to_pos = {idx: pos for pos, idx in enumerate(fold_frame.index)}
        test_pos_indices = [idx_to_pos[idx] for idx in test_indices]
        times_array = pd.DatetimeIndex(pd.to_datetime(fold_frame['window_start_utc'], utc=True))

        for t_pos in test_pos_indices:
            start = t_pos - config.lookback_windows
            if start < 0 or t_pos >= len(fold_frame):
                continue
            hist = full_transformed_np[start:t_pos]
            if len(hist) == config.lookback_windows:
                histories.append(hist)
                labels.append(full_transformed_np[t_pos])
                test_window_timestamps.append(times_array[t_pos])

        if len(histories) == 0:
            print(f"  Skipping fold {fold_id}: no valid test sequences")
            continue

        test_X = np.asarray(histories, dtype=np.float32)
        test_y = np.asarray(labels, dtype=np.float32)

        # --- Load trained model ---
        model = LSTMGaussianWorldModel(
            input_size=32, hidden_size=64, state_dim=32, num_layers=1, dropout=0.2
        )
        checkpoint_path = model_dir / str(fold_id) / 'gaussian_best.pt'

        if not checkpoint_path.exists():
            print(f"  Training fold {fold_id} (checkpoint not found at {checkpoint_path})")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            train_gaussian(
                model, train_set.X, train_set.y, val_set.X, val_set.y,
                epochs=2, batch_size=64, learning_rate=0.001, weight_decay=0.0001,
                patience=5, min_delta=0.001, seed=42, device=str(device),
                checkpoint_path=checkpoint_path,
            )
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])

        model.to(device)
        model.eval()
        with torch.no_grad():
            mean, std = model(torch.as_tensor(test_X, dtype=torch.float32).to(device))
        mean_np = mean.cpu().numpy()

        # --- Compute deviation ---
        deviation = deviation_scores(mean_np, test_y)

        for i, ts in enumerate(test_window_timestamps):
            all_deviation_rows.append({
                'window_start_utc': str(ts),
                'deviation_score': float(deviation[i]),
                'fold_id': int(fold_id),
            })

    # --- Aggregate: if a window appears in multiple folds, average ---
    print(f"\nTotal raw deviation entries: {len(all_deviation_rows)}")
    dev_df = pd.DataFrame(all_deviation_rows)
    dev_agg = dev_df.groupby('window_start_utc').agg(
        deviation_score=('deviation_score', 'mean'),
        fold_count=('fold_id', 'count'),
        folds=('fold_id', lambda x: list(x)),
    ).reset_index()
    dev_agg = dev_agg.sort_values('window_start_utc').reset_index(drop=True)
    print(f"Unique windows with deviation: {len(dev_agg)}")

    # --- Write per-window JSON files ---
    json_count = 0
    windows_written = []
    for _, row in dev_agg.iterrows():
        ts_str = row['window_start_utc']
        score = float(row['deviation_score'])

        # Find the window_id from edges
        if ts_str in window_id_map:
            wid = window_id_map[ts_str]
        else:
            # Fallback: construct from timestamp
            ts = pd.Timestamp(ts_str)
            wid = f"W_{ts.strftime('%d-%m-%Y')}_{ts.strftime('%Y%m%d_%H%M%S')}"

        # Get active nodes for this window
        if ts_str in window_nodes:
            active = window_nodes[ts_str]
        else:
            active = set()

        # Write JSON: {str(node_id): deviation_score} for ALL active nodes
        # ML1's deviation is global per window, so every host gets the same value
        host_scores = {str(nid): score for nid in sorted(active)}

        json_path = json_dir / f"{wid}.json"
        with open(json_path, 'w') as f:
            json.dump(host_scores, f)
        json_count += 1
        windows_written.append(wid)

    print(f"Wrote {json_count} per-window JSON files to {json_dir}")

    # --- Audit CSV ---
    audit_path = out_dir / 'deviation_audit.csv'
    dev_agg.to_csv(audit_path, index=False)
    print(f"Saved audit CSV: {audit_path}")

    # --- Validation ---
    scores = dev_agg['deviation_score'].values
    print(f"\n--- Deviation Statistics ---")
    print(f"  Count:  {len(scores)}")
    print(f"  Min:    {scores.min():.6f}")
    print(f"  Max:    {scores.max():.6f}")
    print(f"  Mean:   {scores.mean():.6f}")
    print(f"  Std:    {scores.std():.6f}")
    print(f"  NaN:    {np.isnan(scores).sum()}")
    print(f"  Inf:    {np.isinf(scores).sum()}")

    assert np.isnan(scores).sum() == 0, "ERROR: NaN values in deviation!"
    assert np.isinf(scores).sum() == 0, "ERROR: Inf values in deviation!"
    assert len(scores) > 0, "ERROR: No deviations generated!"

    # --- Metadata ---
    metadata = {
        'model': 'LSTMGaussianWorldModel',
        'model_config': {
            'input_size': 32,
            'hidden_size': 64,
            'state_dim': 32,
            'num_layers': 1,
            'dropout': 0.2,
        },
        'sequence_length': config.lookback_windows,
        'feature_dim': 32,
        'pca_components': 32,
        'deviation_metric': 'mean_absolute_error_in_pca_space',
        'deviation_formula': 'd(t) = mean(|S_actual(t) - S_predicted(t)|)',
        'temporal_alignment': 'stored_at_window_t_adapter_performs_t-1_shift',
        'temporal_alignment_detail': (
            'ML1 stores deviation at window t. '
            'ML2 inject_novelty_into_graphs uses window_ids[i-1] to fetch t-1 novelty. '
            'No pre-shift applied by ML1.'
        ),
        'output_format': 'per_window_json',
        'output_dir': str(json_dir),
        'num_windows': json_count,
        'num_predictions': len(scores),
        'deviation_stats': {
            'min': float(scores.min()),
            'max': float(scores.max()),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'nan_count': 0,
            'inf_count': 0,
        },
        'protocol': 'LOEO 37-fold (corrected manifest)',
        'manifest': str(manifest_path),
        'checkpoint_dir': str(model_dir),
        'seed': 42,
        'generation_timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'git_hash': get_git_hash(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
    }

    metadata_path = out_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata: {metadata_path}")

    print("\n=== ML1 Export Complete ===")
    print(f"  Output dir:     {out_dir}")
    print(f"  JSON dir:       {json_dir}")
    print(f"  Windows:        {json_count}")
    print(f"  Audit CSV:      {audit_path}")
    print(f"  Metadata:       {metadata_path}")


if __name__ == '__main__':
    main()
