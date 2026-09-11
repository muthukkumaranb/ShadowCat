import shutil
from pathlib import Path

ws = Path(r"c:\Users\Vicky\Documents\SIH_2026\LSTM-type-model-CICIDS2018-UCS")

src_ckpt = ws / "artifacts/lstm/probabilistic_world_model_v2/gaussian_next_state_best.pt"

dest1 = ws / "artifacts/lstm/probabilistic_world_model/gaussian_next_state_best_v2.pt"
dest2 = ws / "artifacts/lstm/gaussian_next_state_best_v2.pt"

shutil.copy(src_ckpt, dest1)
shutil.copy(src_ckpt, dest2)

print(f"[+] Copied v2 checkpoint to {dest1}")
print(f"[+] Copied v2 checkpoint to {dest2}")
