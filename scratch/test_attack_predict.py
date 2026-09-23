import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.predict import predict

parquet_path = "data-engineering/data/ucs/ucs_windows.parquet"
df = pd.read_parquet(parquet_path)

ssh_indices = df[df["label_attack_type"] == "SSH-Bruteforce"].index
print(f"SSH-Bruteforce count: {len(ssh_indices)}, first index: {ssh_indices[0]}")

# Take a contiguous window slice around the first SSH-Bruteforce index
start_idx = max(0, ssh_indices[0] - 10)
end_idx = start_idx + 40
sample_df = df.iloc[start_idx:end_idx].copy()
print(f"Sample window range: [{start_idx} : {end_idx}]")

output = predict(sample_df, source_type="flows")
fc = output["forecast_trajectory"]

print("\n" + "=" * 60)
print("END-TO-END PREDICTION ON SSH-BRUTEFORCE WINDOW")
print("=" * 60)
print("Window ID:", output["window_id"])
print("Risk:", fc["risk"])
print("Stage:", fc["stage"])
print("Tactic ID:", fc["tactic_id"])
print("Technique ID:", fc["technique_id"])
print("Technique Name:", fc["technique_name"])
print("Is Heuristic Progression:", fc["is_heuristic_progression"])
print("Stage Attribution Source:", fc["stage_attribution_source"])

print("\nLiteral Raw Steps Output (First 2 Steps):")
print(json.dumps(fc["raw_steps"][:2], indent=2))
