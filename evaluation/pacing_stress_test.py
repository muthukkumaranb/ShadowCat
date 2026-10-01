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
    
    extractor = UCSExtractor()
    model_tensor = df[extractor.MODEL_INPUT_COLUMNS].to_numpy(dtype=np.float32)
    labels = df['label_binary'].values
    
    # Find a transition from Benign (0) to Attack (1)
    diffs = np.diff(labels)
    transition_indices = np.where(diffs == 1)[0]
    
    if len(transition_indices) == 0:
        print("No Benign -> Attack transition found in the dataset.")
        return
        
    trans_idx = transition_indices[0] # The last benign window before attack
    
    # Let's extract a realistic sequence: 30 benign windows, followed by the attack, and some benign after.
    # Find the end of this attack segment
    attack_end = trans_idx + 1
    while attack_end < len(labels) and labels[attack_end] == 1:
        attack_end += 1
        
    start_idx = max(0, trans_idx - 29) # Up to 30 benign lead-in
    end_idx = min(len(labels), attack_end + 30) # Up to 30 benign lead-out
    
    seq_indices = list(range(start_idx, end_idx))
    
    # Ensure we use an actual benign window for padding (the one right before the attack)
    pad_benign_idx = trans_idx
    pad_benign_tensor = model_tensor[pad_benign_idx]
    
    print(f"Evaluating Adversarial Pacing on Transition Boundary.")
    print(f"Original Sequence Length: {len(seq_indices)} windows.")
    print(f"Contains {trans_idx - start_idx + 1} Benign lead-in, {attack_end - trans_idx - 1} Attack windows, {end_idx - attack_end} Benign lead-out.")
    
    pipeline = ShadowcatPipeline()
    device = pipeline.device
    lookback = 30
    
    def run_inference(sequence_tensor):
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

    def evaluate_dilated_sequence(dilation_factor):
        dilated_tensor = []
        dilated_labels = []
        
        # We dilate the transition: inserting benign windows right AT the boundary.
        for idx in seq_indices:
            w_tensor = model_tensor[idx]
            label = labels[idx]
            
            if idx == trans_idx + 1:
                # We are at the first attack window. Insert (dilation - 1) * 30 benign windows to simulate a long pause
                # Wait, if we want to dilate the gap between benign and attack, we can insert them here.
                # Let's insert (dilation_factor - 1) * 5 benign windows to stretch the transition.
                num_inserts = (dilation_factor - 1) * 5 
                for _ in range(num_inserts):
                    dilated_tensor.append(pad_benign_tensor)
                    dilated_labels.append(0)
                    
            dilated_tensor.append(w_tensor)
            dilated_labels.append(label)
            
        dilated_tensor = np.array(dilated_tensor)
        dilated_labels = np.array(dilated_labels)
        
        preds, haz = run_inference(dilated_tensor)
        f1 = f1_score(dilated_labels, preds, zero_division=0)
        
        # Calculate lead time: from first alert in the ATTACK phase to the end of the attack phase
        # Actually, let's measure from the first alert anywhere after the benign lead-in.
        # But wait, to be comparable, we can measure how many windows the alert happens BEFORE the final attack window.
        alert_idx = -1
        for i, h in enumerate(haz):
            if h.get(1, 0) >= 0.5:
                alert_idx = i
                break
                
        last_attack_idx = len(dilated_labels) - 1
        for i in range(len(dilated_labels)-1, -1, -1):
            if dilated_labels[i] == 1:
                last_attack_idx = i
                break
                
        lead_time = max(0, last_attack_idx - alert_idx) if alert_idx != -1 else 0
        return f1, lead_time, len(dilated_tensor), dilated_labels

    f1_1x, lead_1x, len_1x, lbl_1x = evaluate_dilated_sequence(1)
    f1_2x, lead_2x, len_2x, lbl_2x = evaluate_dilated_sequence(2)
    f1_4x, lead_4x, len_4x, lbl_4x = evaluate_dilated_sequence(4)

    print("\n--- Adversarial Pacing Stress Test Results ---")
    print("Dilation | F1 Score | Effective Lead Time | Sequence Length")
    print(f"   1x    |  {f1_1x:.4f}  |        {lead_1x:2d}         |       {len_1x}")
    print(f"   2x    |  {f1_2x:.4f}  |        {lead_2x:2d}         |       {len_2x}")
    print(f"   4x    |  {f1_4x:.4f}  |        {lead_4x:2d}         |       {len_4x}")
    
    if f1_1x == f1_2x and lead_1x == lead_2x:
        print("\nNote: The metrics are identical across dilations.")
        print(f"Evidence of different inputs:")
        print(f"1x Labels (length {len(lbl_1x)}): {lbl_1x.tolist()}")
        print(f"2x Labels (length {len(lbl_2x)}): {lbl_2x.tolist()}")

if __name__ == '__main__':
    evaluate_pacing()
