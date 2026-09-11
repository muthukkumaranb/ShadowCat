import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import sys
import os
workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))
from lstm.ucs import UCSConfig, validate_ucs_windows, build_lstm_sequences, apply_training_only_pca
from lstm.model import LSTMClassifier
from lstm.trainer import train
from lstm.evaluator import evaluate
from lstm.utils import set_seed, get_device

def run_target(df: pd.DataFrame, manifest_folds: list, target_col: str, output_dir: Path):
    set_seed(42)
    device = get_device()
    config = UCSConfig()
    base_features = validate_ucs_windows(df, config=config)
    
    fold_metrics = []
    
    for fold in manifest_folds:
        fold_id = fold["fold_id"]
        held_out_episode = fold["held_out_episode_id"]
        train_indices = fold["train_indices"]
        test_indices = fold["test_indices"]
        
        # Attack type lookup
        attack_type = df.loc[df['episode_id'] == held_out_episode, 'label_attack_type'].iloc[0]

        fold_frame = df.copy()
        fold_frame["split"] = "none"
        fold_frame.loc[train_indices, "split"] = "train"
        fold_frame.loc[test_indices, "split"] = "test"
        
        # Note: evaluate_loeo previously split val from train using chronological split inside purge_and_embargo
        # Let's manually do an 80/20 train/val chronological split on train_indices
        train_indices = np.array(train_indices)
        train_df_sorted = fold_frame.loc[train_indices].sort_values('window_start_utc')
        n_train = len(train_df_sorted)
        if n_train == 0:
            print(f"Skipping fold {fold_id}, no train indices")
            continue
            
        val_start = int(n_train * 0.8)
        actual_train_indices = train_df_sorted.index[:val_start]
        actual_val_indices = train_df_sorted.index[val_start:]
        
        fold_frame.loc[actual_train_indices, "split"] = "train"
        fold_frame.loc[actual_val_indices, "split"] = "val"
        fold_frame.loc[test_indices, "split"] = "test"
        
        # Only keep train, val, test
        active_mask = fold_frame["split"].isin(["train", "val", "test"])
        purged = fold_frame[active_mask].copy()

        transformed, pca = apply_training_only_pca(
            purged,
            base_features,
            32,
            random_state=42,
        )
        features = [f"pca_{index}" for index in range(32)]
        
        sequences = build_lstm_sequences(
            transformed,
            features=features,
            config=config,
            target_col=target_col,
        )
        
        # Holdout sequences
        # In build_lstm_sequences, we only built train and val. We need test.
        full_transformed = pca.transform(fold_frame)
        full_transformed_np = full_transformed[features].to_numpy(dtype=np.float32)
        test_df = fold_frame.loc[test_indices].copy()
        
        histories = []
        labels = []
        times_array = fold_frame['window_start_utc'].values
        targets_array = fold_frame[target_col].values
        
        idx_to_pos = {idx: pos for pos, idx in enumerate(fold_frame.index)}
        
        test_pos_indices = [idx_to_pos[idx] for idx in test_indices]
        for t_pos in test_pos_indices:
            start = t_pos - config.lookback_windows + 1
            if start < 0:
                continue
            hist = full_transformed_np[start:t_pos+1]
            if len(hist) == config.lookback_windows:
                histories.append(hist)
                labels.append(targets_array[t_pos])
        
        if len(histories) == 0:
            print(f"Skipping fold {fold_id}, no valid test sequences.")
            continue
            
        test_X = np.asarray(histories, dtype=np.float32)
        test_y = np.asarray(labels, dtype=np.float32)

        model = LSTMClassifier(input_size=32, hidden_size=64, num_layers=1, dropout=0.20)
        checkpoint = output_dir / f"{target_col}_folds" / str(fold_id) / "classifier_best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        
        result = train(
            model,
            sequences["train"].X,
            sequences["train"].y,
            sequences["val"].X,
            sequences["val"].y,
            epochs=2,
            batch_size=64,
            learning_rate=0.001,
            seed=42,
            device=str(device),
            checkpoint_path=checkpoint,
            weight_decay=0.0001,
            early_stopping_patience=5,
            early_stopping_min_delta=0.001,
        )
        
        evaluation = evaluate(
            model,
            test_X,
            test_y,
            threshold=0.38,
            device=device,
        )
        
        tn, fp, _, _ = evaluation["confusion_matrix"].ravel() if len(evaluation["confusion_matrix"].ravel()) == 4 else (0,0,0,0)
        fpr = float(fp / (tn + fp)) if tn + fp else 0.0
        
        fold_metrics.append({
            "fold_id": fold_id,
            "held_out_episode_id": held_out_episode,
            "attack_type": attack_type,
            "target": target_col,
            "train_sequences": len(sequences["train"].y),
            "validation_sequences": len(sequences["val"].y),
            "test_sequences": len(test_y),
            "f1": float(evaluation["f1"]),
            "pr_auc": None if evaluation["pr_auc"] is None else float(evaluation["pr_auc"]),
            "precision": float(evaluation["precision"]),
            "recall": float(evaluation["recall"]),
            "fpr": fpr,
            "roc_auc": None if evaluation["roc_auc"] is None else float(evaluation["roc_auc"]),
        })
        target_name = "detection" if target_col == "label_binary" else "onset"
        print(f"Fold {fold_id} ({held_out_episode}) completed. F1: {fold_metrics[-1]['f1']:.4f}")
        pd.DataFrame(fold_metrics).to_csv(output_dir / f"lstm_{target_name}_loeo_folds.csv", index=False)
        
    print(f"Finished {target_col}")
    
def main():
    data_path = Path("data/ucs/ucs_windows.parquet")
    manifest_path = Path("artifacts/loeo/corrected_37fold_manifest.json")
    output_dir = Path("artifacts/lstm")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_parquet(data_path).sort_values("window_start_utc").reset_index(drop=True)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    print("Running LSTM Detection...")
    run_target(df, manifest["folds"], "label_binary", output_dir)
    print("Running LSTM Onset...")
    run_target(df, manifest["folds"], "future_attack_label", output_dir)
    print("LSTM Eval Done")

if __name__ == "__main__":
    main()
