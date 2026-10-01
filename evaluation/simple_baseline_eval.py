import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data-engineering"))

from src.ctu13_translator import CTU13Translator
from src.ucs_extractor import UCSExtractor

def simple_baseline_eval():
    print("Loading Training Dataset (CICIDS2018)...")
    extractor = UCSExtractor()
    cicids_path = REPO_ROOT / "data-engineering" / "data" / "ucs" / "ucs_windows.parquet"
    df_cicids = pd.read_parquet(cicids_path)
    
    # Train set: use the CICIDS2018 data
    # (Since it's 1M rows, we might sample it for speed if needed, but LR is fast)
    X_train = df_cicids[extractor.MODEL_INPUT_COLUMNS].to_numpy(dtype=np.float32)
    y_train = df_cicids['label_binary'].values
    
    print("Training Logistic Regression on full feature set...")
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    print("Loading Test Dataset (CTU-13)...")
    ctu13_path = REPO_ROOT / "scratch" / "CTU13_Attack_Traffic.csv"
    df_ctu13 = pd.read_csv(ctu13_path)
    translator = CTU13Translator()
    translated_df, _ = translator.translate(df_ctu13)
    ctu_tensor = extractor.extract_model_tensor(translated_df, source_type="csv")
    
    # We create the 50/50 dataset exactly like cross_dataset_eval.py
    df_cicids_sorted = df_cicids.sort_values("window_start_utc").reset_index(drop=True)
    benign_df = df_cicids_sorted[df_cicids_sorted['label_binary'] == 0]
    benign_tensor = benign_df[extractor.MODEL_INPUT_COLUMNS].to_numpy(dtype=np.float32)
    
    num_attack = len(ctu_tensor)
    num_benign = min(len(benign_tensor), num_attack)
    
    X_test = np.vstack([benign_tensor[:num_benign], ctu_tensor])
    y_test = np.concatenate([np.zeros(num_benign), np.ones(num_attack)])
    
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    
    print("\n--- Simple Baseline Evaluation ---")
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
    simple_baseline_eval()
