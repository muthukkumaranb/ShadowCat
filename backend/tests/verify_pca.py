
import pandas as pd
import numpy as np
import torch
import sys
import joblib
from pathlib import Path

sys.path.insert(0, ".")
# Make sure lstm.ucs is available for joblib unpickling if needed
# The user said "(needs `lstm.ucs` on the path to deserialize)"

def run_verify():
    print("Loading datasets...")
    df = pd.read_parquet("data-engineering/data/ucs/ucs_windows.parquet").sort_values("window_start_utc")
    
    ben_seq = df[df["has_malicious_flows"] == 0].head(30)
    mal_seq = df.tail(30).reset_index(drop=True)
    
    from data_engineering.src.ucs_extractor import UCSExtractor
    ext = UCSExtractor()
    
    ben_feats = ext.map_to_canonical(ben_seq).to_numpy(dtype=np.float32)
    mal_feats = ext.map_to_canonical(mal_seq).to_numpy(dtype=np.float32)
    
    pca_path = "ml1/artifacts/experiments/ucs_pca_20260904/pca_32/preprocessing_pca.joblib"
    if not Path(pca_path).exists():
        print("PCA artifact not found!")
        return
        
    print("Loading PCA...")
    try:
        pca = joblib.load(pca_path)
        print("PCA loaded successfully.")
    except Exception as e:
        print(f"Failed to load PCA: {e}")
        return
        
    ben_pca = pca.transform(ben_feats)
    mal_pca = pca.transform(mal_feats)
    
    # Also grab raw slice for comparison
    ben_raw = ben_feats[:, :32]
    
    model_path = "ml1/artifacts/lstm/hazard_head_v4/H1/model_fold_1.pt"
    print(f"Loading Model: {model_path}")
    from backend.models import LSTMClassifier
    model = LSTMClassifier(input_size=32, hidden_size=64, num_layers=1, dropout=0.2)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    
    with torch.no_grad():
        out_ben_pca = model.predict_proba(torch.as_tensor(ben_pca, dtype=torch.float32).unsqueeze(0)).item()
        out_mal_pca = model.predict_proba(torch.as_tensor(mal_pca, dtype=torch.float32).unsqueeze(0)).item()
        out_ben_raw = model.predict_proba(torch.as_tensor(ben_raw, dtype=torch.float32).unsqueeze(0)).item()
        
    print(f"OUTPUT BENIGN (PCA): {out_ben_pca:.4f}")
    print(f"OUTPUT MALICIOUS (PCA): {out_mal_pca:.4f}")
    print(f"OUTPUT BENIGN (RAW-SLICE): {out_ben_raw:.4f}")

if __name__ == "__main__":
    run_verify()

