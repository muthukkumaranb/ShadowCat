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
    print("Loading CTU-13 Attack Dataset...")
    ctu13_path = REPO_ROOT / "scratch" / "CTU13_Attack_Traffic.csv"
    if not ctu13_path.exists():
        print("CTU-13 data not found.")
        return
    
    df_ctu13 = pd.read_csv(ctu13_path)
    
    print(f"Translating {len(df_ctu13)} CTU-13 flows...")
    translator = CTU13Translator()
    translated_df, fidelity = translator.translate(df_ctu13)
    print(f"Translation Fidelity: {fidelity.fidelity_score_pct}%")
    
    print("Extracting UCS Windows for Attack Traffic...")
    extractor = UCSExtractor()
    ctu_tensor = extractor.extract_model_tensor(translated_df, source_type="csv")
    
    print("Loading Benign Traffic from CICIDS2018 to create a mixed test set...")
    cicids_path = REPO_ROOT / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
    df_cicids = pd.read_parquet(cicids_path).sort_values("window_start_utc").reset_index(drop=True)
    
    # Extract benign windows
    benign_df = df_cicids[df_cicids['label_binary'] == 0]
    benign_tensor = benign_df[extractor.MODEL_INPUT_COLUMNS].to_numpy(dtype=np.float32)
    
    # We will take an equal number of benign windows to match the CTU-13 attack windows
    num_attack = len(ctu_tensor)
    num_benign = min(len(benign_tensor), num_attack)
    
    # Construct mixed sequence: Benign followed by Attack
    mixed_tensor = np.vstack([benign_tensor[:num_benign], ctu_tensor])
    mixed_labels = np.concatenate([np.zeros(num_benign), np.ones(num_attack)])
    
    print(f"Mixed Test Set Created: {num_benign} Benign windows, {num_attack} Attack windows. Total = {len(mixed_labels)}.")
    
    pipeline = ShadowcatPipeline()
    lookback = 30
    predictions_det = []
    
    device = pipeline.device
    
    for i in range(len(mixed_tensor)):
        start = max(0, i - lookback + 1)
        seq = mixed_tensor[start:i+1]
        
        if len(seq) < lookback:
            pad_count = lookback - len(seq)
            seq = np.pad(seq, ((pad_count, 0), (0, 0)), mode='edge')
            
        seq_t = torch.as_tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        det_prob = pipeline._predict_detection_ensemble(seq)
        
        pred_det = 1 if (det_prob is not None and det_prob >= 0.5) else 0
        predictions_det.append(pred_det)
        
    y_true = mixed_labels
    y_pred = np.array(predictions_det)
    
    print("\n--- CTU-13 Cross-Dataset Evaluation (Detection) ---")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
    print(f"FPR:       {fpr:.4f}")
    
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
        print(f"MCC:       {mcc:.4f}")
    except Exception as e:
        print(f"MCC:       Undefined ({e})")

if __name__ == '__main__':
    evaluate_ctu13()
