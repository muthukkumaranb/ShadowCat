import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data-engineering"))

from src.ctu13_translator import CTU13Translator
from src.ucs_extractor import UCSExtractor
from backend.predict import ShadowcatPipeline

def evaluate_ctu13():
    print("Loading CTU-13 Dataset...")
    ctu13_path = REPO_ROOT / "scratch" / "CTU13_Attack_Traffic.csv"
    if not ctu13_path.exists():
        print("CTU-13 data not found. Please ensure scratch/CTU13_Attack_Traffic.csv is available.")
        return
    
    df_ctu13 = pd.read_csv(ctu13_path)
    # Ensure there's a Label column, default to 1 if it's attack traffic
    if 'Label' not in df_ctu13.columns:
        df_ctu13['Label'] = 1
        
    print(f"Translating {len(df_ctu13)} CTU-13 flows...")
    translator = CTU13Translator()
    translated_df, fidelity = translator.translate(df_ctu13)
    print(f"Translation Fidelity: {fidelity.fidelity_score_pct}%")
    
    print("Extracting UCS Windows...")
    extractor = UCSExtractor()
    window_df = extractor.extract(translated_df, source_type="csv")
    model_tensor = extractor.extract_model_tensor(translated_df, source_type="csv")
    
    print(f"Extracted {len(window_df)} windows.")
    
    pipeline = ShadowcatPipeline()
    
    # We will slide a 30-window over the tensor and predict
    lookback = 30
    predictions_det = []
    predictions_onset = []
    targets_det = []
    
    device = pipeline.device
    
    # For ground truth, we need to know if the window has attacks.
    # Assuming window_df has 'mask_has_malicious_flows' or we can just assume 1 if all are attacks.
    # We'll just assume all windows in CTU13_Attack_Traffic have Label=1.
    for i in range(len(model_tensor)):
        start = max(0, i - lookback + 1)
        seq = model_tensor[start:i+1]
        
        # pad if needed
        if len(seq) < lookback:
            pad_count = lookback - len(seq)
            seq = np.pad(seq, ((pad_count, 0), (0, 0)), mode='edge')
            
        seq_tensor = torch.as_tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        det_prob = pipeline._predict_detection_ensemble(seq)
        hazards = pipeline._predict_hazard_ensemble(seq)
        
        pred_det = 1 if (det_prob is not None and det_prob >= 0.5) else 0
        pred_onset = 1 if (hazards.get(1, 0) >= 0.5) else 0
        
        predictions_det.append(pred_det)
        predictions_onset.append(pred_onset)
        targets_det.append(1) # Ground truth is 1 for attack traffic
        
    y_true = np.array(targets_det)
    y_pred = np.array(predictions_det)
    
    print("\n--- CTU-13 Cross-Dataset Evaluation (Detection) ---")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    
    # We can't compute full MCC/Precision properly if there are no benign samples.
    # Let's check if there are 0s in targets_det
    if len(np.unique(y_true)) > 1:
        print(f"F1: {f1_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
        print(f"FPR: {fpr:.4f}")
        print(f"MCC: {matthews_corrcoef(y_true, y_pred):.4f}")
    else:
        print("Dataset only contains attack windows. Precision, FPR, and MCC are undefined.")

if __name__ == '__main__':
    evaluate_ctu13()
