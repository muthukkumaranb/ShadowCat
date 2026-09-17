import sys
import os
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('ml2-full/GNN_FINAL'))

from backend.predict import ShadowcatPipeline
from ml2.models.fusion import FusedModel
import pandas as pd

def test_fused_compatibility():
    # 1. Load pipeline and FusedModel
    print("[1] Loading models...")
    pipeline = ShadowcatPipeline()
    pipeline.world_model.eval()
    
    checkpoint_path = 'ml2-full/GNN_FINAL/ml2/results/ablation/fused/best_model.pt'
    if not os.path.exists(checkpoint_path):
        print(f"FAILED: Checkpoint {checkpoint_path} not found.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # We might need to inspect the model's initialization parameters from the checkpoint
    # FusedModel defaults: z_dim=64, node_feature_dim=11, graph_embedding_dim=64, output_dim=11
    # We can just instantiate FusedModel and load state_dict
    try:
        fused_model = FusedModel(z_dim=64, output_dim=64)
        if 'model_state_dict' in model_data:
            fused_model.load_state_dict(model_data['model_state_dict'], strict=False)
        else:
            fused_model.load_state_dict(model_data, strict=False)
        fused_model.to(device)
        fused_model.eval()
    except Exception as e:
        print(f"FAILED: Could not load FusedModel state dict: {e}")
        return
    
    # 2. Get real v4 z(t) encoder output
    print("[2] Extracting v4 z(t) from sample CSV...")
    df = pd.read_csv('backend/tests/fixtures/judge_demo_sample.csv')
    extracted_df = pipeline.extractor.extract(df, source_type='csv')
    
    feature_cols = pipeline.extractor.MODEL_INPUT_COLUMNS
    model_tensor = extracted_df[feature_cols].values
    
    import numpy as np
    pad_count = max(0, 30 - len(model_tensor))
    seq_30x406 = np.pad(model_tensor, ((pad_count, 0), (0, 0)), mode='edge')[-30:]
    seq_tensor = torch.as_tensor(seq_30x406, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        z_t = pipeline.world_model.extract_latent_z(seq_tensor)
        print(f'    z(t) shape: {z_t.shape}')
        # duplicate to B=2 to test difference
        z_t = torch.cat([z_t, z_t], dim=0)
        
    # 3. Create dummy graph matching the batch size
    # FusedModel expects: x, edge_index, batch, z_t
    # node_feature_dim=11 is the default
    B = z_t.shape[0]
    node_feature_dim = 11
    
    # Create 1 node per graph in batch for dummy check
    x = torch.randn(B, node_feature_dim, device=device)
    # No edges (self loop)
    edge_index = torch.zeros((2, B), dtype=torch.long, device=device)
    edge_index[0, :] = torch.arange(B)
    edge_index[1, :] = torch.arange(B)
    batch = torch.arange(B, device=device)
    
    # 4. Forward pass
    print("[3] Running FusedModel forward pass...")
    try:
        with torch.no_grad():
            fused_out = fused_model(x, edge_index, z_t, batch)
            
        print(f'    Fused output shape: {fused_out.shape}')
        print(f"    Output contains NaN: {torch.isnan(fused_out).any().item()}")
        
        # Check non-degenerate output
        print(f"    First window output:\n{fused_out[0].cpu().numpy()}")
        print(f"    Second window output:\n{fused_out[1].cpu().numpy()}")
        
        diff = torch.abs(fused_out[0] - fused_out[1]).sum().item()
        print(f"    Diff between W0 and W1: {diff:.4f}")
        if diff < 1e-4:
            print("[FAIL] Output is degenerate (constant for different inputs)")
        else:
            print("[PASS] Output is non-degenerate and compatible!")
            
    except Exception as e:
        print(f"[FAIL] Forward pass crashed: {e}")

if __name__ == "__main__":
    test_fused_compatibility()
