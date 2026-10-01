import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data-engineering"))

from src.ctu13_translator import CTU13Translator
from src.ucs_extractor import UCSExtractor
from backend.predict import ShadowcatPipeline

def finetune_eval():
    print("Loading datasets...")
    ctu13_path = REPO_ROOT / "scratch" / "CTU13_Attack_Traffic.csv"
    df_ctu13 = pd.read_csv(ctu13_path)
    translator = CTU13Translator()
    translated_df, _ = translator.translate(df_ctu13)
    
    extractor = UCSExtractor()
    ctu_tensor = extractor.extract_model_tensor(translated_df, source_type="csv")
    
    cicids_path = REPO_ROOT / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
    df_cicids = pd.read_parquet(cicids_path).sort_values("window_start_utc").reset_index(drop=True)
    benign_df = df_cicids[df_cicids['label_binary'] == 0]
    benign_tensor = benign_df[extractor.MODEL_INPUT_COLUMNS].to_numpy(dtype=np.float32)
    
    num_attack = len(ctu_tensor)
    num_benign = min(len(benign_tensor), num_attack)
    
    # We create the 50/50 dataset
    full_tensor = np.vstack([benign_tensor[:num_benign], ctu_tensor])
    full_labels = np.concatenate([np.zeros(num_benign), np.ones(num_attack)])
    
    # Shuffle for split
    np.random.seed(42)
    indices = np.arange(len(full_labels))
    np.random.shuffle(indices)
    
    split_idx = int(0.2 * len(indices))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train, y_train = full_tensor[train_idx], full_labels[train_idx]
    X_test, y_test = full_tensor[test_idx], full_labels[test_idx]
    
    print(f"Fine-tuning set: {len(X_train)} windows. Test set: {len(X_test)} windows.")
    
    pipeline = ShadowcatPipeline()
    
    # We fine-tune the base LR coefficients of the stacked models!
    # This directly adapts the existing model without breaking PyTorch graph.
    # We will use SGD to update the coefficients.
    
    # Get the base model coefficients
    print("Fine-tuning base LR model with SGD...")
    from sklearn.metrics import log_loss
    
    for fold_model in pipeline.stacked_detection_models:
        coef = fold_model.lr_coef.ravel()
        intercept = fold_model.lr_intercept
        mean = fold_model.scaler_mean
        scale = fold_model.scaler_scale
        
        lr = 0.001
        for epoch in range(10): # brief fine-tuning
            X_scaled = (X_train - mean) / (scale + 1e-8)
            logits = np.dot(X_scaled, coef) + intercept
            probs = 1 / (1 + np.exp(-np.clip(logits, -10, 10)))
            
            error = probs - y_train
            
            grad_coef = np.dot(X_scaled.T, error) / len(X_train)
            grad_intercept = np.mean(error)
            
            coef -= lr * grad_coef
            intercept -= lr * grad_intercept
            
        fold_model.lr_coef = coef.reshape(1, -1)
        fold_model.lr_intercept = intercept

    print("Evaluating fine-tuned model...")
    predictions_det = []
    lookback = 30
    
    for i in range(len(X_test)):
        start = max(0, i - lookback + 1)
        seq = X_test[start:i+1]
        if len(seq) < lookback:
            seq = np.pad(seq, ((lookback - len(seq), 0), (0, 0)), mode='edge')
            
        det_prob = pipeline._predict_detection_ensemble(seq)
        pred_det = 1 if (det_prob is not None and det_prob >= 0.5) else 0
        predictions_det.append(pred_det)
        
    y_pred = np.array(predictions_det)
    
    print("\n--- Few-Shot Fine-Tuned Evaluation ---")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
    print(f"FPR:       {fpr:.4f}")
    
    try:
        mcc = matthews_corrcoef(y_test, y_pred)
        print(f"MCC:       {mcc:.4f}")
    except Exception as e:
        print(f"MCC:       Undefined ({e})")

if __name__ == '__main__':
    finetune_eval()
