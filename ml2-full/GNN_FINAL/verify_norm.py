import pandas as pd
import yaml
from pathlib import Path

root = Path('ml2')
df = pd.read_parquet(root / 'data/ucs/ucs_windows.parquet')
scaler = yaml.safe_load((root / 'data/ucs/scaler_params.yaml').read_text())

# Check three features across different types
for feature in ['duration_microsec_mean', 'byte_count_fwd_mean', 'packet_count_fwd_mean']:
    raw_values = df[feature].to_numpy(dtype=float)
    params = scaler[feature]
    print(f"Feature: {feature}")
    print(f"  Parquet: min={raw_values.min():.4f}, max={raw_values.max():.4f}, mean={raw_values.mean():.4f}, std={raw_values.std():.4f}")
    print(f"  Scaler: median={params['median']:.2f}, scale={params['scale']:.2f}, is_log1p={params.get('is_log1p')}")
    print()
