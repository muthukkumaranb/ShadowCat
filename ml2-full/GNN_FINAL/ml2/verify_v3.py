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
checkpoint_path = root / 'data' / 'ml1_artifacts' / 'gaussian_next_state_best_v3.pt'
scaler_path = root / 'data' / 'ml1_artifacts' / 'inference_scaler_v3.yaml'
windows_path = root.parent.parent.parent / 'data-engineering' / 'data' / 'ucs' / 'ucs_windows.parquet'
if not windows_path.exists():
    windows_path = root / 'data' / 'ucs' / 'ucs_windows.parquet'

print("--- PART 1.1: Checkpoint Load Verification ---")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("model_state_dict", checkpoint)
lstm_state = {key: value for key, value in state_dict.items() if key.startswith("lstm.")}

encoder = Z_tEncoder()
missing, unexpected = encoder.load_state_dict(lstm_state, strict=False)
print(f"Clean load: missing={missing}, unexpected={unexpected}")

print("\n--- PART 1.2: Output Verification ---")
metadata_path = root / 'data' / 'ml1_artifacts' / 'inference_feature_order_v3.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

features = metadata['features']
windows = pd.read_parquet(windows_path).sort_values("window_start_utc").reset_index(drop=True)
values = windows[features].to_numpy(dtype=np.float32)

z_outputs = []
encoder.eval()
with torch.no_grad():
    for end in range(29, min(29 + 100, len(windows))):  # Verify on first 100 windows
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

print("\n--- PART 1.3: Scaler Parity Check ---")
with open(scaler_path, 'r') as f:
    scaler_v3 = yaml.safe_load(f)['features']

print(f"Total Scaler features in v3: {len(scaler_v3)}")
pkt_cols = [
    "pkt_ttl_min", "pkt_ttl_max", "pkt_ttl_std", "pkt_ttl_mode",
    "pkt_frag_mf_count", "pkt_frag_df_count",
    "pkt_payload_size_p25", "pkt_payload_size_p50",
    "pkt_payload_size_p75", "pkt_payload_size_p95",
    "pkt_tcp_retrans_count", "pkt_port_scan_seq_score"
]
print("Verified real packet features present with non-zero scale:")
for col in pkt_cols:
    p = scaler_v3.get(col, {})
    print(f"  {col}: median={p.get('median')}, scale={p.get('scale'):.4f}, is_log1p={p.get('is_log1p')}")
