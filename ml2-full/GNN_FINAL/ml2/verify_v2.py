import sys
import torch
import numpy as np
import pandas as pd
import yaml
import json
from pathlib import Path

# Add ml2 to path
sys.path.append(str(Path('.').absolute()))

from training.z_encoder import Z_tEncoder

root = Path('.')
checkpoint_path = root / 'data' / 'ml1_artifacts' / 'gaussian_next_state_best_v2.pt'
scaler_path = root / 'data' / 'ml1_artifacts' / 'inference_scaler_v2.yaml'
windows_path = root.parent / 'data' / 'processed' / 'ucs_v1' / 'ucs_windows_raw.parquet'
if not windows_path.exists():
    windows_path = root / 'data' / 'ucs' / 'ucs_windows.parquet'

print("--- PART 1.2: Checkpoint Load Verification ---")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("model_state_dict", checkpoint)
lstm_state = {key: value for key, value in state_dict.items() if key.startswith("lstm.")}

encoder = Z_tEncoder()
missing, unexpected = encoder.load_state_dict(lstm_state, strict=False)
print(f"Clean load: missing={missing}, unexpected={unexpected}")

print("\n--- PART 1.2: Output Verification ---")
# Need metadata
metadata_path = root / 'data' / 'ml1_artifacts' / 'inference_feature_order_v2.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

features = metadata['features']
windows = pd.read_parquet(windows_path).sort_values("window_start_utc").reset_index(drop=True)
values = windows[features].to_numpy(dtype=np.float32)

z_outputs = []
encoder.eval()
with torch.no_grad():
    for end in range(29, min(29 + 100, len(windows))): # Verify on first 100 windows
        sequence = torch.from_numpy(values[end - 29 : end + 1]).unsqueeze(0)
        z = encoder(sequence).detach().numpy()
        z_outputs.append(z[0])

z_outputs = np.array(z_outputs)
print(f"Sample batch size: {len(z_outputs)}")
print(f"Variance (across all features and samples): {np.var(z_outputs):.6f}")
print(f"Minimum: {np.min(z_outputs):.6f}")
print(f"Maximum: {np.max(z_outputs):.6f}")
print(f"Mean: {np.mean(z_outputs):.6f}")
print(f"Standard deviation: {np.std(z_outputs):.6f}")
print(f"Varies across windows: {not np.allclose(z_outputs[0], z_outputs[1])}")
print(f"L2 distance between first two outputs: {np.linalg.norm(z_outputs[0] - z_outputs[1]):.6f}")
print("\nFirst eight values of the first verified z(t):")
print(np.array2string(z_outputs[0, :8], separator=', ', precision=8))

print("\n--- PART 1.3: Normalization Check ---")
with open(scaler_path, 'r') as f:
    scaler_v2 = yaml.safe_load(f)['features']

# Find columns with scale=1.0 and median=0.0
changed_columns = [f for f, p in scaler_v2.items() if p.get('scale') == 1.0 and p.get('median') == 0.0]
changed_columns = changed_columns[:12] # Limit to 12 as mentioned

print(f"Found {len(changed_columns)} changed columns (scale=1.0, median=0.0):")
for col in changed_columns:
    if col in windows.columns:
        raw_values = windows[col].to_numpy(dtype=float)
        print(f"  {col}: Parquet min={raw_values.min():.4f}, max={raw_values.max():.4f}, mean={raw_values.mean():.4f}")

# Spot check unchanged
unchanged = [f for f, p in scaler_v2.items() if p.get('scale') != 1.0 and f in windows.columns][:2]
print("\nSpot check unchanged columns:")
for col in unchanged:
    raw_values = windows[col].to_numpy(dtype=float)
    p = scaler_v2[col]
    print(f"  {col}: Parquet mean={raw_values.mean():.4f}, Scaler expected raw median={p['median']:.2f}")

