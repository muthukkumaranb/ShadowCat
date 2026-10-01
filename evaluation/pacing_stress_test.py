import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data-engineering"))

from src.ucs_extractor import UCSExtractor
from backend.predict import ShadowcatPipeline

def evaluate_pacing():
    print("Loading CSE-CIC-IDS2018 UCS windows...")
    data_path = REPO_ROOT / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
    df = pd.read_parquet(data_path).sort_values("window_start_utc").reset_index(drop=True)
    
    # Extract model tensor directly since it's already windowed
    extractor = UCSExtractor()
    # The DF is already windowed, so we just extract the model columns directly
    # Wait, the df has standard names. We can just use the extractor to do it by passing source_type="flows"? No, source_type="flows" still expects unwindowed canonical flows.
    # The dataframe is ALREADY `ucs_windows.parquet`. The columns are exactly `extractor.OUTPUT_COLUMNS` or a superset.
    model_tensor = df[extractor.MODEL_INPUT_COLUMNS].to_numpy(dtype=np.float32)
    
    # Pick a benign window for interleaving (first window is usually benign)
    benign_idx = df.index[df['label_binary'] == 0][0]
    benign_window_tensor = model_tensor[benign_idx]
    
    # Pick a specific attack episode to evaluate
    attack_episodes = df[df['label_binary'] == 1]['episode_id'].unique()
    target_episode = attack_episodes[0]
    
    episode_mask = df['episode_id'] == target_episode
    ep_indices = df.index[episode_mask]
    
    # The episode has a sequence of windows. 
    # Let's say the attack starts at index `attack_start_idx` within this episode.
    ep_df = df.iloc[ep_indices]
    
    print(f"Evaluating Adversarial Pacing on Episode: {target_episode} ({len(ep_indices)} windows)")
    
    pipeline = ShadowcatPipeline()
    device = pipeline.device
    lookback = 30
    
    def run_inference_on_sequence(sequence_tensor):
        predictions_det = []
        hazards_list = []
        for i in range(len(sequence_tensor)):
            start = max(0, i - lookback + 1)
            seq = sequence_tensor[start:i+1]
            if len(seq) < lookback:
                pad_count = lookback - len(seq)
                seq = np.pad(seq, ((pad_count, 0), (0, 0)), mode='edge')
            
            seq_t = torch.as_tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            det_prob = pipeline._predict_detection_ensemble(seq)
            hazards = pipeline._predict_hazard_ensemble(seq)
            
            pred_det = 1 if (det_prob is not None and det_prob >= 0.5) else 0
            predictions_det.append(pred_det)
            hazards_list.append(hazards)
        return predictions_det, hazards_list

    # 1x Baseline
    baseline_tensor = model_tensor[ep_indices]
    baseline_labels = ep_df['label_binary'].values
    
    preds_1x, haz_1x = run_inference_on_sequence(baseline_tensor)
    f1_1x = f1_score(baseline_labels, preds_1x, zero_division=0)
    
    # Helper to create dilated sequence
    def create_dilated_sequence(dilation_factor):
        dilated_tensor = []
        dilated_labels = []
        
        attack_started = False
        for idx in range(len(baseline_tensor)):
            w_tensor = baseline_tensor[idx]
            label = baseline_labels[idx]
            
            if label == 1:
                attack_started = True
                
            dilated_tensor.append(w_tensor)
            dilated_labels.append(label)
            
            if attack_started:
                # Interleave (dilation_factor - 1) benign windows
                for _ in range(dilation_factor - 1):
                    dilated_tensor.append(benign_window_tensor)
                    dilated_labels.append(0) # These inserted windows are technically benign
                    
        return np.array(dilated_tensor), np.array(dilated_labels)

    # 2x Dilation
    tensor_2x, labels_2x = create_dilated_sequence(2)
    preds_2x, haz_2x = run_inference_on_sequence(tensor_2x)
    f1_2x = f1_score(labels_2x, preds_2x, zero_division=0)

    # 4x Dilation
    tensor_4x, labels_4x = create_dilated_sequence(4)
    preds_4x, haz_4x = run_inference_on_sequence(tensor_4x)
    f1_4x = f1_score(labels_4x, preds_4x, zero_division=0)
    
    # Calculate effective lead-time (how many steps before the final attack window is it detected)
    # We define lead time as: from the first hazard alert (hazard >= 0.5) to the end of the sequence.
    def get_lead_time(hazards_list, true_labels):
        # find the index where true attack starts and finishes
        # find the earliest hazard alert (e.g. h1 >= 0.5)
        alert_idx = -1
        for i, h in enumerate(hazards_list):
            if h.get(1, 0) >= 0.5:
                alert_idx = i
                break
        
        if alert_idx == -1:
            return 0
            
        last_attack_idx = len(true_labels) - 1
        for i in range(len(true_labels)-1, -1, -1):
            if true_labels[i] == 1:
                last_attack_idx = i
                break
                
        return max(0, last_attack_idx - alert_idx)
        
    lead_1x = get_lead_time(haz_1x, baseline_labels)
    lead_2x = get_lead_time(haz_2x, labels_2x)
    lead_4x = get_lead_time(haz_4x, labels_4x)

    print("\n--- Adversarial Pacing Stress Test Results ---")
    print("Dilation | F1 Score | Effective Lead Time (windows)")
    print(f"   1x    |  {f1_1x:.4f}  |  {lead_1x}")
    print(f"   2x    |  {f1_2x:.4f}  |  {lead_2x}")
    print(f"   4x    |  {f1_4x:.4f}  |  {lead_4x}")

if __name__ == '__main__':
    evaluate_pacing()
