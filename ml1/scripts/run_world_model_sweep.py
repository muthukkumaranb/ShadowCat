import os
import sys
from pathlib import Path
import yaml
import json
import subprocess

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, str(workspace_dir))

from ml1.examples.train_probabilistic import run

def main():
    sweep_configs = [
        {"name": "hidden128_L1", "hidden": 128, "layers": 1, "lr": 0.001, "epochs": 50},
        {"name": "hidden64_L2", "hidden": 64, "layers": 2, "lr": 0.001, "epochs": 50},
        {"name": "hidden128_L2_low_lr", "hidden": 128, "layers": 2, "lr": 0.0005, "epochs": 60},
    ]

    input_path = workspace_dir / "data-engineering/data/ucs/ucs_windows.parquet"
    if not input_path.exists():
        input_path = workspace_dir / "data/ucs/ucs_windows.parquet"

    sweep_dir = workspace_dir / "ml1/artifacts/lstm/sweep_v1"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    config_path = sweep_dir / "temp_config.yaml"

    results = []

    for cfg in sweep_configs:
        print(f"\n========================================")
        print(f"Running config: {cfg['name']}")
        print(f"========================================")
        
        output_dir = sweep_dir / cfg['name']
        
        config_data = {
            "model": {
                "hidden_size": cfg["hidden"],
                "num_layers": cfg["layers"],
                "dropout": 0.2
            },
            "training": {
                "epochs": cfg["epochs"],
                "batch_size": 64,
                "learning_rate": cfg["lr"],
                "weight_decay": 0.0001,
                "early_stopping": {
                    "patience": 10,
                    "min_delta": 0.001
                }
            },
            "world_model": {
                "seed": 42,
                "rollout": {"horizons": [1, 2, 3]},
                "intervals": {"levels": [0.80, 0.95]}
            }
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
            
        try:
            run(input_path, output_dir, config_path)
            
            # Run horizon evaluation
            ckpt_path = output_dir / "gaussian_next_state_best.pt"
            eval_script = workspace_dir / "evaluation/horizon_analysis/horizon_analysis.py"
            
            print(f"Evaluating horizon for {cfg['name']}...")
            result = subprocess.run(
                ["python", str(eval_script), "--checkpoint", str(ckpt_path)],
                capture_output=True,
                text=True
            )
            
            eval_output = result.stdout
            print(eval_output)
            
            # Save eval output to a file for later parsing
            with open(output_dir / "horizon_eval.txt", "w") as f:
                f.write(eval_output)
                if result.stderr:
                    f.write("\n\nERRORS:\n")
                    f.write(result.stderr)
                    
        except Exception as e:
            print(f"Failed config {cfg['name']}: {e}")

if __name__ == "__main__":
    main()
