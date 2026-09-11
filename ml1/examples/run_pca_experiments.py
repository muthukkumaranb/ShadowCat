from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.train_ucs import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled UCS baseline and PCA experiments.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments/ucs_pca"))
    parser.add_argument("--config", type=Path, default=Path("configs/model_config.yaml"))
    args = parser.parse_args()

    experiments = {"baseline_406": None, "pca_32": 32, "pca_64": 64, "pca_128": 128}
    summary = {}
    for name, components in experiments.items():
        metrics = run(args.input, args.output_root / name, args.config, components)
        metadata = json.loads((args.output_root / name / "lstm_ucs_experiment.json").read_text(encoding="utf-8"))
        summary[name] = {
            "features": metadata["feature_count"],
            "train_sequences": metadata["train_sequences"],
            "validation_sequences": metadata["validation_sequences"],
            "test_sequences": metadata["test_sequences"],
            "validation_pr_auc": metadata["best_validation_pr_auc"],
            "validation_f1_at_selected_threshold": metadata["validation_f1_at_selected_threshold"],
            "test_metrics": metrics,
            "pca": metadata["pca"],
        }
    winner = max(summary, key=lambda name: (summary[name]["validation_pr_auc"], summary[name]["validation_f1_at_selected_threshold"] or 0.0))
    output = {"selection_metric": "validation_pr_auc", "winner": winner, "experiments": summary}
    (args.output_root / "comparison.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"winner": winner, "experiments": summary}, indent=2))


if __name__ == "__main__":
    main()