import yaml
import os
from pathlib import Path

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "ml2_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
