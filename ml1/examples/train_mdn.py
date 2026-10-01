import argparse
import json
import os
import sys
from pathlib import Path

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

import numpy as np
import pandas as pd
import torch
import yaml

from lstm.model import LSTMMixtureDensityWorldModel
from lstm.probabilistic import train_mdn
from lstm.ucs import UCSConfig, validate_ucs_windows, build_next_state_sequences, load_ucs_windows, purge_and_embargo
from lstm.utils import get_device, set_seed

def run(input_path: Path, output_dir: Path, config_path: Path | None = None) -> dict:
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path and config_path.exists() else {}
    world_settings = settings.get("world_model", {})
    model_settings = settings.get("model", {})
    training_settings = settings.get("training", {})
    early_settings = training_settings.get("early_stopping", {})
    seed = int(world_settings.get("seed", 42))
    set_seed(seed)
    
    config = UCSConfig()
    windows = load_ucs_windows(input_path)
    features = validate_ucs_windows(windows, config=config)
    purged = purge_and_embargo(windows, config=config)
    sequences = build_next_state_sequences(purged, features=features, config=config)
    
    train_set, val_set, test_set = sequences["train"], sequences["val"], sequences["test"]
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = LSTMMixtureDensityWorldModel(
        input_size=train_set.X.shape[-1], 
        hidden_size=int(model_settings.get("hidden_size", 64)), 
        state_dim=train_set.y.shape[-1], 
        num_layers=int(model_settings.get("num_layers", 1)), 
        dropout=float(model_settings.get("dropout", 0.2)),
        num_components=5
    )
    
    training = train_mdn(
        model,
        train_set.X,
        train_set.y,
        val_set.X,
        val_set.y,
        epochs=int(training_settings.get("epochs", 30)),
        batch_size=int(training_settings.get("batch_size", 64)),
        learning_rate=float(training_settings.get("learning_rate", 0.001)),
        weight_decay=float(training_settings.get("weight_decay", 0.0001)),
        patience=int(early_settings.get("patience", 5)),
        min_delta=float(early_settings.get("min_delta", 0.001)),
        seed=seed,
        device=str(device),
        checkpoint_path=output_dir / "mdn_next_state_best.pt",
    )
    
    report = {
        "best_epoch": training["best_epoch"],
        "best_validation_nll": training["best_validation_nll"],
    }
    
    (output_dir / "training_history.json").write_text(json.dumps(training["history"], indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MDN world model.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lstm/mdn_v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/model_config.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir, args.config), indent=2))

if __name__ == "__main__":
    main()
