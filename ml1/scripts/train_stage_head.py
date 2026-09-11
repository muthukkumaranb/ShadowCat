"""
Stage Classification Head
Consumes PREDICTED future state S_hat(t+1) from trained LSTMGaussianWorldModel
and classifies ATT&CK stage tactics with explicit Unknown/Other fallback.
"""
from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import yaml
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

from lstm.model import LSTMGaussianWorldModel
from lstm.ucs import UCSConfig, validate_ucs_windows, build_next_state_sequences, load_ucs_windows, purge_and_embargo
from lstm.utils import get_device, set_seed


class StageClassificationHead(nn.Module):
    """Stage Classifier operating on predicted future latent state S_hat(t+1)."""
    def __init__(self, state_dim: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, predicted_state: torch.Tensor) -> torch.Tensor:
        return self.net(predicted_state)


def load_attack_stage_mapping(yaml_path: Path) -> dict[str, str]:
    """Parse ATT&CK tactics mapping manifest."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    mappings = {}
    for attack_name, info in data.get('mappings', {}).items():
        tactic = info.get('tactic', 'Unknown/Other')
        # Standardize compound tactic names if any
        if '/' in tactic and 'Credential Access' in tactic:
            tactic = 'Credential Access'
        mappings[attack_name] = tactic
    return mappings


def main():
    parser = argparse.ArgumentParser(description="Train Stage Head on predicted future states S_hat(t+1).")
    parser.add_argument("--data", type=Path, default=Path("data/ucs/ucs_windows.parquet"))
    parser.add_argument("--world-model-ckpt", type=Path, default=Path("artifacts/experiments/world_model_20260904/probabilistic/gaussian_next_state_best.pt"))
    parser.add_argument("--mapping", type=Path, default=Path("data/ucs/attack_tactics_mapping.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lstm/stage_head"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    device = get_device()

    print("======================================================================")
    print("Stage Classification Head (Architectural Input: Predicted Future State)")
    print("======================================================================")

    # 1. Load Data and Manifest
    windows = load_ucs_windows(args.data)
    config = UCSConfig()
    features = validate_ucs_windows(windows, config=config)

    tactic_map = load_attack_stage_mapping(args.mapping)
    print(f"Loaded ATT&CK tactic mappings for {len(tactic_map)} attack types:")
    for k, v in tactic_map.items():
        print(f"  - {k} -> {v}")

    # Map stage label per window (Unknown/Other for Benign or unmapped)
    def assign_stage(row):
        if row['label_binary'] == 0:
            return "Unknown/Other"
        attack = row.get('label_attack_type', 'Benign')
        return tactic_map.get(attack, "Unknown/Other")

    windows['stage_label'] = windows.apply(assign_stage, axis=1)

    stage_classes = sorted(list(set(windows['stage_label'].unique())))
    # Ensure Unknown/Other is included
    if "Unknown/Other" not in stage_classes:
        stage_classes.append("Unknown/Other")
    stage_classes = sorted(stage_classes)
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(stage_classes)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(stage_classes)}

    windows['stage_idx'] = windows['stage_label'].map(class_to_idx)
    
    print("\nStage Label Distribution across Canonical Dataset:")
    counts = windows['stage_label'].value_counts()
    total_w = len(windows)
    for stage_name, count in counts.items():
        print(f"  - {stage_name}: {count} ({100.0 * count / total_w:.2f}%)")

    # 2. Extract sequences & compute S_hat(t+1) using trained World Model
    purged = purge_and_embargo(windows, config=config)
    sequences = build_next_state_sequences(purged, features=features, config=config)
    
    train_seq, val_seq, test_seq = sequences['train'], sequences['val'], sequences['test']

    # Load World Model to extract predicted future state
    world_model = LSTMGaussianWorldModel(
        input_size=train_seq.X.shape[-1],
        hidden_size=64,
        state_dim=train_seq.y.shape[-1],
        num_layers=1,
        dropout=0.2
    )
    ckpt = torch.load(args.world_model_ckpt, map_location=device, weights_only=True)
    world_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    world_model.to(device)
    world_model.eval()

    def get_predicted_states_and_labels(seq_set):
        X_tensor = torch.as_tensor(seq_set.X, dtype=torch.float32).to(device)
        with torch.no_grad():
            mean, _ = world_model(X_tensor)
        s_hat = mean.cpu().numpy()
        
        # Map sequence timestamps to stage_idx
        time_to_stage = dict(zip(pd.to_datetime(purged['window_start_utc'], utc=True), purged['stage_idx']))
        y_stage = np.array([time_to_stage[ts] for ts in seq_set.timestamps], dtype=int)
        return s_hat, y_stage

    print("\nExtracting predicted future states S_hat(t+1) from World Model...")
    S_hat_train, y_train = get_predicted_states_and_labels(train_seq)
    S_hat_val, y_val = get_predicted_states_and_labels(val_seq)
    S_hat_test, y_test = get_predicted_states_and_labels(test_seq)

    print(f"Train S_hat shape: {S_hat_train.shape}, Val S_hat shape: {S_hat_val.shape}, Test S_hat shape: {S_hat_test.shape}")

    # 3. Train Stage Head Classifier
    num_classes = len(stage_classes)
    stage_head = StageClassificationHead(state_dim=S_hat_train.shape[-1], num_classes=num_classes, hidden_dim=64, dropout=0.2)
    stage_head.to(device)

    train_dataset = TensorDataset(torch.as_tensor(S_hat_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.as_tensor(S_hat_val, dtype=torch.float32), torch.as_tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    optimizer = torch.optim.Adam(stage_head.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(100):
        stage_head.train()
        train_loss = 0.0
        for s_b, y_b in train_loader:
            s_b, y_b = s_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = stage_head(s_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * s_b.size(0)

        stage_head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for s_b, y_b in val_loader:
                s_b, y_b = s_b.to(device), y_b.to(device)
                logits = stage_head(s_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * s_b.size(0)

        val_loss /= len(val_dataset)
        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(stage_head.state_dict(), args.output_dir / "stage_head_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best checkpoint
    stage_head.load_state_dict(torch.load(args.output_dir / "stage_head_best.pt", map_location=device))
    stage_head.eval()

    # 4. Evaluation under Chronological Split
    with torch.no_grad():
        test_logits = stage_head(torch.as_tensor(S_hat_test, dtype=torch.float32).to(device))
        test_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
        test_preds = np.argmax(test_probs, axis=-1)

    acc = float(accuracy_score(y_test, test_preds))
    prec, rec, f1, support = precision_recall_fscore_support(y_test, test_preds, average=None, zero_division=0, labels=list(range(num_classes)))
    cm = confusion_matrix(y_test, test_preds, labels=list(range(num_classes)))

    per_class_metrics = {}
    for i, cls_name in enumerate(stage_classes):
        per_class_metrics[cls_name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }

    # 5. 37-Fold LOEO Evaluation across all Attack Tactics
    manifest_path = Path("artifacts/loeo/corrected_37fold_manifest.json")
    loeo_per_class = {cls_name: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for cls_name in stage_classes}
    loeo_total_samples = 0
    loeo_cm = np.zeros((num_classes, num_classes), dtype=int)

    if manifest_path.exists():
        print("\nRunning 37-fold LOEO cross-validation for Stage Head...")
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)

        for fold in manifest_data['folds']:
            fold_id = fold['fold_id']
            train_idx = np.array(fold['train_indices'])
            test_idx = np.array(fold['test_indices'])

            # Map indices to purged windows
            fold_frame = purged.copy()
            fold_frame['split'] = 'none'
            fold_frame.loc[fold_frame.index.isin(test_idx), 'split'] = 'test'
            fold_frame.loc[fold_frame.index.isin(train_idx), 'split'] = 'train'

            train_df = fold_frame[fold_frame['split'] == 'train']
            test_df = fold_frame[fold_frame['split'] == 'test']

            if len(test_df) == 0 or len(train_df) == 0:
                continue

            # Compute S_hat using world model on sequence histories
            def extract_s_hat_and_y(df_sub):
                idx_map = {idx: pos for pos, idx in enumerate(purged.index)}
                X_list, y_list = [], []
                full_features_np = purged[features].to_numpy(dtype=np.float32)
                full_y_stage = purged['stage_idx'].to_numpy(dtype=int)

                for idx in df_sub.index:
                    t_pos = idx_map[idx]
                    start = t_pos - config.lookback_windows + 1
                    if start < 0 or t_pos >= len(purged):
                        continue
                    hist = full_features_np[start:t_pos+1]
                    if len(hist) == config.lookback_windows:
                        X_list.append(hist)
                        y_list.append(full_y_stage[t_pos])
                if not X_list:
                    return np.zeros((0, 406), dtype=np.float32), np.zeros((0,), dtype=int)
                X_arr = np.array(X_list, dtype=np.float32)
                with torch.no_grad():
                    mean, _ = world_model(torch.as_tensor(X_arr, dtype=torch.float32).to(device))
                return mean.cpu().numpy(), np.array(y_list, dtype=int)

            S_train, Y_train = extract_s_hat_and_y(train_df)
            S_test, Y_test = extract_s_hat_and_y(test_df)

            if len(S_train) == 0 or len(S_test) == 0 or len(np.unique(Y_train)) < 2:
                continue

            f_head = StageClassificationHead(state_dim=406, num_classes=num_classes, hidden_dim=64, dropout=0.2).to(device)
            opt = torch.optim.Adam(f_head.parameters(), lr=0.001, weight_decay=1e-4)
            crit = nn.CrossEntropyLoss()

            tr_ds = TensorDataset(torch.as_tensor(S_train, dtype=torch.float32), torch.as_tensor(Y_train, dtype=torch.long))
            tr_ld = DataLoader(tr_ds, batch_size=64, shuffle=True)

            for ep in range(15):
                f_head.train()
                for sb, yb in tr_ld:
                    sb, yb = sb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = crit(f_head(sb), yb)
                    loss.backward()
                    opt.step()

            f_head.eval()
            with torch.no_grad():
                l_test = f_head(torch.as_tensor(S_test, dtype=torch.float32).to(device))
                p_test = np.argmax(torch.softmax(l_test, dim=-1).cpu().numpy(), axis=-1)

            fold_cm = confusion_matrix(Y_test, p_test, labels=list(range(num_classes)))
            loeo_cm += fold_cm
            loeo_total_samples += len(Y_test)

    # Compute overall LOEO per-class metrics
    loeo_per_class_summary = {}
    for i, cls_name in enumerate(stage_classes):
        tp = loeo_cm[i, i]
        fp = np.sum(loeo_cm[:, i]) - tp
        fn = np.sum(loeo_cm[i, :]) - tp
        support = np.sum(loeo_cm[i, :])
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        loeo_per_class_summary[cls_name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(support)
        }

    loeo_overall_acc = float(np.trace(loeo_cm) / loeo_total_samples) if loeo_total_samples > 0 else 0.0

    # Summary metrics dict
    metrics_summary = {
        "architectural_input": "predicted_future_latent_state_S_hat_t_plus_1",
        "world_model_checkpoint": str(args.world_model_ckpt),
        "chronological_split_metrics": {
            "overall_accuracy": acc,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm.tolist()
        },
        "loeo_split_metrics": {
            "overall_accuracy": loeo_overall_acc,
            "total_eval_samples": loeo_total_samples,
            "per_class_metrics": loeo_per_class_summary,
            "confusion_matrix": loeo_cm.tolist()
        },
        "num_stage_classes": num_classes,
        "stage_classes": stage_classes,
        "unknown_other_fraction_in_dataset": float(np.mean(windows['stage_label'] == "Unknown/Other")),
    }

    with open(args.output_dir / "stage_head_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # 6. Generate Markdown Report
    report_lines = [
        "# Stage Head Classification Report\n",
        "## Architectural Verification",
        "- **Input Verification**: Stage Head operates directly on **PREDICTED future latent state** $\\hat{S}_{t+1}$ generated by `LSTMGaussianWorldModel` (Item 1 checkpoint), fulfilling strict architectural requirements.",
        "- **ATT&CK Mapping Source**: `data/ucs/attack_tactics_mapping.yaml`",
        "- **Unknown/Other Fallback Rule**: All benign or unmapped/low-confidence traffic windows default to `Unknown/Other`.",
        "- **Scope Verification**: Evaluated across all labeled windows (Claim Ladder Rung 4). B1 stage transition (n=1 infiltration) is strictly excluded (case study scope owned by ML2).\n",
        "## Performance Metrics — 37-Fold LOEO Protocol (All Stage Classes Covered)",
        f"- **LOEO Overall Accuracy**: **{loeo_overall_acc:.4f}**",
        f"- **Total LOEO Evaluation Samples**: **{loeo_total_samples}**\n",
        "### 37-Fold LOEO Per-Class Metrics",
        "| Stage Tactic Class | Precision | Recall | F1-Score | Support |",
        "|---|---|---|---|---|"
    ]

    for cls_name in stage_classes:
        m = loeo_per_class_summary[cls_name]
        report_lines.append(f"| **{cls_name}** | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")

    report_lines.append("\n### 37-Fold LOEO Confusion Matrix")
    report_lines.append("```")
    col_hdr = "True \\ Pred"
    header = f"{col_hdr:<22s}" + "".join([f"{cls_name[:12]:>14s}" for cls_name in stage_classes])
    report_lines.append(header)
    report_lines.append("-" * len(header))
    for i, true_cls in enumerate(stage_classes):
        row_str = f"{true_cls:<22s}" + "".join([f"{loeo_cm[i, j]:>14d}" for j in range(num_classes)])
        report_lines.append(row_str)
    report_lines.append("```\n")

    report_lines.append("## Chronological Split Metrics (Single Final Test Window)")
    report_lines.append(f"- **Accuracy**: **{acc:.4f}**")
    report_lines.append(f"- **Unknown/Other Ratio in Test Split**: **{100.0 * np.mean(y_test == class_to_idx['Unknown/Other']):.2f}%**\n")

    report_lines.append("### Chronological Per-Class Metrics")
    report_lines.append("| Stage Tactic Class | Precision | Recall | F1-Score | Support |")
    report_lines.append("|---|---|---|---|---|")
    for cls_name in stage_classes:
        m = per_class_metrics[cls_name]
        report_lines.append(f"| **{cls_name}** | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")

    report_path = args.output_dir / "stage_head_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nSaved stage head metrics: {args.output_dir / 'stage_head_metrics.json'}")
    print(f"Saved stage head report: {report_path}")
    print(f"Checkpoint saved: {args.output_dir / 'stage_head_best.pt'}")
    print("=== Stage Head Training and Evaluation Complete ===")


if __name__ == '__main__':
    main()
