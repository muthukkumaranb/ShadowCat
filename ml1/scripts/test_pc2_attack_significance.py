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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-report", type=str, default=None)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    data_path = Path(args.data_path) if args.data_path else (workspace_dir / 'data' / 'ucs' / 'ucs_windows.parquet')
    if not data_path.exists():
        data_path = workspace_dir.parent / 'data-engineering' / 'data' / 'ucs' / 'ucs_windows.parquet'
    
    model_path = Path(args.model_path) if args.model_path else (workspace_dir / 'artifacts' / 'lstm' / 'gaussian_next_state_best_v3.pt')
    if not model_path.exists():
        model_path = workspace_dir / 'artifacts' / 'lstm' / 'gaussian_next_state_best_v2.pt'

    print("=" * 70)
    print("Significance Test: LSTM World Model vs Persistence on PC2, Attack Windows")
    print(f"Model Checkpoint: {model_path.name}")
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

    # 3. Load Model Checkpoint (input_size=406, state_dim=406)
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
    report_k_data = []
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

        report_k_data.append({
            'k': k_idx + 1,
            'N': N,
            'lstm_mae': lstm_mae,
            'pers_mae': pers_mae,
            'd_mean': d_mean,
            'd_median': float(np.median(diff)),
            'd_std': d_std,
            'stat': stat,
            'p_val': p_val,
            'cohens_d': cohens_d,
            'lstm_wins': int(lstm_wins),
            'pers_wins': int(pers_wins),
            'ties': int(ties),
        })

    print(f"\n--- STEP 3: Conclusion ---")
    print("PC2 significance evaluation completed.")

    # Write Markdown Report
    report_file = Path(args.output_report) if args.output_report else (workspace_dir / 'artifacts' / 'lstm' / 'pc2_significance_report_v3.md')
    report_lines = []
    report_lines.append(f"# PC2 Attack Window Significance Test Report (v3 Retrain Cascade)")
    report_lines.append("")
    report_lines.append("## Summary & Key Findings")
    report_lines.append("")
    report_lines.append(f"- **Test Goal**: Evaluate whether the retrained LSTM Gaussian World Model (`v3`) outperforms the Persistence baseline on Principal Component 2 (PC2) during attack windows in the test set.")
    report_lines.append(f"- **Model Checkpoint**: `{model_path.name}`")
    report_lines.append(f"- **Dataset**: `{data_path.name}` (Test split, contiguous attack windows).")
    report_lines.append("- **Statistical Test**: Wilcoxon signed-rank test (two-sided paired non-parametric test) + Cohen's d effect size.")
    verdict = "PERSISTENCE BASELINE IS STATISTICALLY SUPERIOR ON PC2" if all(item['d_mean'] > 0 and item['p_val'] < 0.05 for item in report_k_data) else "LSTM WORLD MODEL MEETS OR EXCEEDS BASELINE"
    report_lines.append(f"- **Overall Verdict**: **{verdict}**.")
    for item in report_k_data:
        k = item['k']
        N = item['N']
        dm = item['d_mean']
        pv = item['p_val']
        cd = item['cohens_d']
        pw = item['pers_wins']
        report_lines.append(f"  - At **K={k}** ($N={N}$ attack windows): Mean MAE difference $\\Delta = {dm:+.4f}$ points (Persistence advantage), $p = {pv:.2e}$, Cohen's $d = {cd:.4f}$. Persistence wins {100*pw/N:.1f}\\% ({pw}/{N}) of attack windows.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Detailed Results by Rollout Horizon $K$")
    report_lines.append("")

    for item in report_k_data:
        k = item['k']
        N = item['N']
        lm_mean = float(np.mean(item['lstm_mae']))
        pm_mean = float(np.mean(item['pers_mae']))
        report_lines.append(f"### Rollout Horizon $K={k}$ ($N = {N}$ Attack Windows)")
        report_lines.append(f"- **LSTM Mean MAE**: ${lm_mean:.2f}$ | **Persistence Mean MAE**: ${pm_mean:.2f}$ | **Mean Delta (LSTM - Pers)**: **${item['d_mean']:+.4f}$**")
        report_lines.append(f"- **Median Delta (LSTM - Pers)**: ${item['d_median']:+.4f}$")
        report_lines.append(f"- **Wilcoxon Signed-Rank Statistic**: ${item['stat']:.1f}$ ($p = {item['p_val']:.4e}$)")
        report_lines.append(f"- **Cohen's d**: ${item['cohens_d']:.4f}$")
        report_lines.append(f"- **Win Breakdown**: LSTM wins on {item['lstm_wins']} / {N} windows ({100*item['lstm_wins']/N:.1f}\\%), Persistence wins on {item['pers_wins']} / {N} windows ({100*item['pers_wins']/N:.1f}\\%).")
        report_lines.append("")
        report_lines.append(f"#### PC2 MAE Percentile Distribution ($K={k}$)")
        report_lines.append("| Model | p5 | p25 | p50 (Median) | p75 | p95 | Max |")
        report_lines.append("|---|---|---|---|---|---|---|")
        pcts = [5, 25, 50, 75, 95]
        l_vals = [f"{np.percentile(item['lstm_mae'], p):.2f}" for p in pcts] + [f"{np.max(item['lstm_mae']):.2f}"]
        p_vals = [f"{np.percentile(item['pers_mae'], p):.2f}" for p in pcts] + [f"{np.max(item['pers_mae']):.2f}"]
        report_lines.append(f"| **LSTM** | " + " | ".join(l_vals) + " |")
        report_lines.append(f"| **Persistence** | " + " | ".join(p_vals) + " |")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"Saved PC2 significance report to: {report_file}")

if __name__ == '__main__':
    main()
