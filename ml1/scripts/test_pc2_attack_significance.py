"""
Significance Test: LSTM v2 World Model vs Persistence on PC2, Attack Windows Only
Evaluates the canonical v2 checkpoint: artifacts/lstm/gaussian_next_state_best_v2.pt
"""
import pandas as pd
import numpy as np
import sys
import os
import torch
from pathlib import Path
from scipy.stats import wilcoxon

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))

from lstm.model import LSTMGaussianWorldModel
from lstm.ucs import UCSConfig, validate_ucs_windows, UCSPCA, build_next_state_sequences, purge_and_embargo, rollout_targets, persistence_baseline
from lstm.utils import get_device, set_seed

def lstm_rollout(history, model, device, k=3):
    history = np.asarray(history, dtype=np.float32)
    current = history.copy()
    rollout = np.empty((history.shape[0], k, history.shape[2]), dtype=np.float32)
    for step in range(k):
        with torch.no_grad():
            mean, _ = model(torch.as_tensor(current, dtype=torch.float32).to(device))
        predicted = mean.cpu().numpy()
        rollout[:, step, :] = predicted
        if step < k - 1:
            current = np.concatenate([current[:, 1:, :], predicted[:, None, :]], axis=1)
    return rollout

def main():
    set_seed(42)
    device = get_device()

    data_path = workspace_dir / 'data' / 'ucs' / 'ucs_windows.parquet'
    model_path = workspace_dir / 'artifacts' / 'lstm' / 'gaussian_next_state_best_v2.pt'

    print("=" * 70)
    print("Significance Test: LSTM v2 World Model vs Persistence on PC2, Attack Windows")
    print(f"Model Checkpoint: {model_path.relative_to(workspace_dir)}")
    print("=" * 70)

    windows = pd.read_parquet(data_path).sort_values('window_start_utc').reset_index(drop=True)
    config = UCSConfig()
    features = validate_ucs_windows(windows, config=config)
    purged = purge_and_embargo(windows, config=config)

    # 1. Fit PCA on purged train partition (32 components)
    train_frame = purged[purged['split'] == 'train']
    pca = UCSPCA(n_components=32, random_state=42)
    pca.fit(train_frame, features)

    # 2. Build 406-dimensional sequence inputs for test partition
    sequences = build_next_state_sequences(purged, features=features, config=config)
    test_seq = sequences['test']

    # 3. Load v2 Model Checkpoint (input_size=406, state_dim=406)
    model = LSTMGaussianWorldModel(input_size=406, hidden_size=64, state_dim=406, num_layers=1, dropout=0.2)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Multi-step Rollout in 406-dimensional feature space
    K = 3
    histories_406, targets_406, times = rollout_targets(
        purged, features=features, config=config, k=K, split='test')

    lstm_pred_406 = lstm_rollout(histories_406, model, device, k=K)

    # Persistence baseline in 406-dim space
    pers_pred_406 = np.empty_like(targets_406)
    last_state = persistence_baseline(histories_406)
    for k_idx in range(K):
        pers_pred_406[:, k_idx, :] = last_state

    # Map timestamps to labels and episode_id
    purged_test = purged[purged['split'] == 'test']
    time_to_label = dict(zip(pd.to_datetime(purged_test["window_start_utc"], utc=True), purged_test["label_binary"]))
    time_to_episode = dict(zip(pd.to_datetime(purged_test["window_start_utc"], utc=True), purged_test.get("episode_id", purged_test["source_day"])))

    labels_target = np.empty((len(times), K, 1))
    episodes_target = []
    for i, t_idx in enumerate(times):
        ep_row = []
        for k_idx in range(K):
            labels_target[i, k_idx, 0] = time_to_label[t_idx[k_idx]]
            ep_row.append(time_to_episode[t_idx[k_idx]])
        episodes_target.append(ep_row)

    # 5. Project all 406-dim predictions & targets onto PCA space
    PC2 = 2
    for k_idx in range(K):
        attack_mask = (labels_target[:, k_idx, 0] == 1)
        N = int(np.sum(attack_mask))

        # Transform 406-dim features to 32 PCA components
        targ_pca = pca.transform(pd.DataFrame(targets_406[:, k_idx, :], columns=features))
        lstm_pca = pca.transform(pd.DataFrame(lstm_pred_406[:, k_idx, :], columns=features))
        pers_pca = pca.transform(pd.DataFrame(pers_pred_406[:, k_idx, :], columns=features))

        # Extract PC2 column (pca_2) as numpy array filtered by attack mask
        targ_pc2 = targ_pca["pca_2"].to_numpy()[attack_mask]
        lstm_pc2 = lstm_pca["pca_2"].to_numpy()[attack_mask]
        pers_pc2 = pers_pca["pca_2"].to_numpy()[attack_mask]

        lstm_mae = np.abs(lstm_pc2 - targ_pc2)
        pers_mae = np.abs(pers_pc2 - targ_pc2)
        diff = lstm_mae - pers_mae

        # --- STEP 1: Wilcoxon + Cohen's d ---
        stat, p_val = wilcoxon(lstm_mae, pers_mae)
        d_mean = np.mean(diff)
        d_std = np.std(diff, ddof=1)
        cohens_d = d_mean / d_std if d_std > 0 else 0

        print(f"\n--- K={k_idx+1} (N={N} attack windows) ---")
        print(f"  STEP 1: Paired Significance")
        print(f"    Mean(LSTM-Pers) = {d_mean:.4f}")
        print(f"    Median(LSTM-Pers) = {np.median(diff):.4f}")
        print(f"    Std(LSTM-Pers) = {d_std:.4f}")
        print(f"    Wilcoxon stat  = {stat}")
        print(f"    p-value        = {p_val:.4e}")
        print(f"    Cohen's d      = {cohens_d:.4f}")
        if p_val < 0.05:
            if d_mean < 0:
                print(f"    ==> LSTM advantage is STATISTICALLY SIGNIFICANT (p < 0.05), LSTM wins.")
            else:
                print(f"    ==> Persistence advantage is STATISTICALLY SIGNIFICANT (p < 0.05).")
        else:
            print(f"    ==> Difference is NOT statistically significant.")

        # --- STEP 2: Percentile Breakdown ---
        print(f"\n  STEP 2: Percentile Breakdown (PC2 MAE, Attack Windows)")
        pcts = [5, 25, 50, 75, 95]
        header = f"    {'':>6s}"
        for p in pcts:
            header += f"  {'p'+str(p):>8s}"
        header += f"  {'max':>8s}"
        print(header)

        lstm_line = f"    {'LSTM':>6s}"
        pers_line = f"    {'Pers':>6s}"
        for p in pcts:
            lstm_line += f"  {np.percentile(lstm_mae, p):>8.2f}"
            pers_line += f"  {np.percentile(pers_mae, p):>8.2f}"
        lstm_line += f"  {np.max(lstm_mae):>8.2f}"
        pers_line += f"  {np.max(pers_mae):>8.2f}"
        print(lstm_line)
        print(pers_line)

        lstm_wins = np.sum(diff < 0)
        ties = np.sum(diff == 0)
        pers_wins = np.sum(diff > 0)
        print(f"    LSTM wins: {lstm_wins}/{N} ({100*lstm_wins/N:.1f}%), "
              f"Pers wins: {pers_wins}/{N} ({100*pers_wins/N:.1f}%), "
              f"Ties: {ties}/{N}")

    print(f"\n--- STEP 3: Conclusion ---")
    print("PC2 significance evaluation against v2 checkpoint completed.")

if __name__ == '__main__':
    main()
