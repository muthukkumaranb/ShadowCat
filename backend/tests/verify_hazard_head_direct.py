
import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, ".")
from backend.models import LSTMClassifier

def run_direct():
    print("Loading benign sequence...")
    df = pd.read_parquet("data-engineering/data/ucs/ucs_windows.parquet").sort_values("window_start_utc")
    benign_seq = df[df["has_malicious_flows"] == 0].head(30)
    
    # 406 features as in predict.py
    from data_engineering.src.ucs_extractor import UCSExtractor
    ext = UCSExtractor()
    features = ext.map_to_canonical(benign_seq)
    seq_30x406 = features.to_numpy(dtype=np.float32)
    
    # What predict.py does: slices first 32 columns
    seq_32 = seq_30x406[:, :32]
    print(f"Raw sequence min: {seq_32.min()}, max: {seq_32.max()}, mean: {seq_32.mean()}")
    
    seq_tensor = torch.as_tensor(seq_32, dtype=torch.float32).unsqueeze(0)
    
    ckpt_path = Path("ml1/artifacts/lstm/hazard_head_v4/H1/model_fold_1.pt")
    if not ckpt_path.exists():
        print("Checkpoint not found!")
        return
        
    print(f"Loading actual v4 checkpoint: {ckpt_path}")
    model = LSTMClassifier(input_size=32, hidden_size=64, num_layers=1, dropout=0.2)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    
    with torch.no_grad():
        out = model.predict_proba(seq_tensor).item()
        print(f"Output probability from true v4 checkpoint (H1, Fold 1): {out:.4f}")

if __name__ == "__main__":
    run_direct()

