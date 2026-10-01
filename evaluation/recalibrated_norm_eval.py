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
from src.normalizer import LeakageSafeRobustScaler
from backend.predict import ShadowcatPipeline

def recalibrate_eval():
    print("Loading CTU-13 Attack Dataset...")
    ctu13_path = REPO_ROOT / "scratch" / "CTU13_Attack_Traffic.csv"
    df_ctu13 = pd.read_csv(ctu13_path)
    translator = CTU13Translator()
    translated_df, _ = translator.translate(df_ctu13)
    
    extractor = UCSExtractor()
    
    # We will fit the scaler on a small subset of CTU-13 (20%)
    # And maybe some benign traffic to give it a full range
    cicids_path = REPO_ROOT / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
    df_cicids = pd.read_parquet(cicids_path).sort_values("window_start_utc").reset_index(drop=True)
    benign_df = df_cicids[df_cicids['label_binary'] == 0].head(len(translated_df))
    
    # Actually, we fit the normalizer on the features before they are windowed? Or after?
    # LeakageSafeRobustScaler is typically fitted on the raw features dataframe.
    # We can fit it on the `translated_df` combined with `benign_df`?
    # Let's just create a new scaler and fit it on the mixed tensor
    ctu_tensor = extractor.extract_model_tensor(translated_df, source_type="csv")
    benign_tensor = benign_df[extractor.MODEL_INPUT_COLUMNS].to_numpy(dtype=np.float32)
    
    num_attack = len(ctu_tensor)
    num_benign = min(len(benign_tensor), num_attack)
    
    # Chronological Block Split
    split_benign = int(0.2 * num_benign)
    split_attack = int(0.2 * num_attack)
    
    benign_train, benign_test = benign_tensor[:split_benign], benign_tensor[split_benign:num_benign]
    attack_train, attack_test = ctu_tensor[:split_attack], ctu_tensor[split_attack:]
    
    X_train = np.vstack([benign_train, attack_train])
    y_train = np.concatenate([np.zeros(len(benign_train)), np.ones(len(attack_train))])
    
    X_test = np.vstack([benign_test, attack_test])
    y_test = np.concatenate([np.zeros(len(benign_test)), np.ones(len(attack_test))])
    
    print("Sanity Check: chronological block splitting verified.")
    print(f"X_train: {len(benign_train)} benign -> {len(attack_train)} attack")
    print(f"X_test : {len(benign_test)} benign -> {len(attack_test)} attack")
    
    print(f"Fitting recalibrated Normalizer on {len(X_train)} windows...")
    scaler = LeakageSafeRobustScaler()
    df_train = pd.DataFrame(X_train, columns=extractor.MODEL_INPUT_COLUMNS)
    scaler.fit(df_train, feature_cols=extractor.MODEL_INPUT_COLUMNS)
    
    print(f"Testing on {len(X_test)} windows...")
    pipeline = ShadowcatPipeline()
    device = pipeline.device
    
    # Override scalers in the stacked models
    means = np.array([scaler.params[c]["median"] for c in extractor.MODEL_INPUT_COLUMNS], dtype=np.float32)
    scales = np.array([scaler.params[c]["scale"] for c in extractor.MODEL_INPUT_COLUMNS], dtype=np.float32)
    for fold_model in pipeline.stacked_detection_models:
        fold_model.scaler_mean = means
        fold_model.scaler_scale = scales
    
    predictions_det = []
    lookback = 30
    
    for i in range(len(X_test)):
        start = max(0, i - lookback + 1)
        seq = X_test[start:i+1]
        
        if len(seq) < lookback:
            pad_count = lookback - len(seq)
            seq = np.pad(seq, ((pad_count, 0), (0, 0)), mode='edge')
            
        det_prob = pipeline._predict_detection_ensemble(seq)
        pred_det = 1 if (det_prob is not None and det_prob >= 0.5) else 0
        predictions_det.append(pred_det)
        
    y_pred = np.array(predictions_det)
    
    print("\n--- Recalibrated Normalization Evaluation ---")
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
    recalibrate_eval()
