from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lstm.ucs import (
    LOEO_ELIGIBLE_ATTACK_TYPES,
    LOEO_EXCLUDED_ATTACK_TYPES,
    LOEO_GROUPING_COLUMN,
    build_loeo_episode_folds,
    load_ucs_windows,
)


def validate(input_path: Path, output_dir: Path) -> dict:
    frame = load_ucs_windows(input_path).sort_values("window_start_utc").reset_index(drop=True)
    if "forecast_episode_id" not in frame.columns:
        raise ValueError("The updated UCS artifact must contain forecast_episode_id.")
    if frame[LOEO_GROUPING_COLUMN].equals(frame["forecast_episode_id"]):
        raise ValueError("episode_id and forecast_episode_id unexpectedly have identical semantics.")

    episode_summary = frame.groupby(LOEO_GROUPING_COLUMN).agg(
        attack_type=("label_attack_type", lambda values: "|".join(sorted(set(values.astype(str))))),
        window_count=("window_id", "count"),
        splits=("split", lambda values: "|".join(sorted(set(values.astype(str))))),
        first_timestamp=("window_start_utc", "min"),
        last_timestamp=("window_start_utc", "max"),
        source_days=("source_day", lambda values: "|".join(sorted(set(values.astype(str))))),
    ).reset_index()
    episode_summary["split_count"] = episode_summary["splits"].str.count(r"\|") + 1

    spanning = episode_summary[episode_summary["split_count"] > 1].copy()
    spanning.to_csv(output_dir / "split_spanning_episodes.csv", index=False)

    singleton_ids = set(episode_summary.loc[episode_summary["window_count"] == 1, LOEO_GROUPING_COLUMN])
    singleton = frame[frame[LOEO_GROUPING_COLUMN].isin(singleton_ids)].merge(
        episode_summary[[LOEO_GROUPING_COLUMN, "attack_type"]], on=LOEO_GROUPING_COLUMN, how="left"
    )
    singleton = singleton[[LOEO_GROUPING_COLUMN, "attack_type", "label_attack_type", "window_start_utc", "split", "source_day"]]
    singleton.to_csv(output_dir / "singleton_episodes.csv", index=False)

    folds = build_loeo_episode_folds(frame)
    manifest_rows = []
    isolation_rows = []
    for fold_id, fold in enumerate(folds):
        train_ids = set(fold["train_indices"])
        holdout_ids = set(fold["holdout_indices"])
        manifest_rows.append({
            "fold_id": fold_id,
            "held_out_episode_id": fold["held_out_episode"],
            "attack_type": fold["attack_type"],
            "train_row_count": len(train_ids),
            "holdout_row_count": len(holdout_ids),
            "train_episode_count": len(fold["train_episodes"]),
        })
        isolation_rows.append({
            "fold_id": fold_id,
            "held_out_episode_id": fold["held_out_episode"],
            "held_out_in_train": fold["held_out_episode"] in set(frame.loc[list(train_ids), LOEO_GROUPING_COLUMN]),
            "row_overlap_count": len(train_ids & holdout_ids),
        })
    manifest = pd.DataFrame(manifest_rows)
    isolation = pd.DataFrame(isolation_rows)
    manifest.to_csv(output_dir / "loeo_fold_manifest.csv", index=False)
    isolation.to_csv(output_dir / "fold_isolation.csv", index=False)

    attack_counts = (
        episode_summary[episode_summary["attack_type"].isin(LOEO_ELIGIBLE_ATTACK_TYPES)]
        .assign(attack_type=lambda values: values["attack_type"].str.split("|").str[0])
        .groupby("attack_type")[LOEO_GROUPING_COLUMN]
        .count()
        .to_dict()
    )
    excluded_counts = (
        episode_summary[episode_summary["attack_type"].isin(LOEO_EXCLUDED_ATTACK_TYPES)]
        .assign(attack_type=lambda values: values["attack_type"].str.split("|").str[0])
        .groupby("attack_type")[LOEO_GROUPING_COLUMN]
        .count()
        .to_dict()
    )
    if len(folds) != 37:
        raise ValueError(f"Expected 37 eligible LOEO folds, found {len(folds)}.")
    if isolation["held_out_in_train"].any() or isolation["row_overlap_count"].any():
        raise ValueError("LOEO fold isolation failed.")
    summary = {
        "protocol": "37-fold LOEO eligibility validation; evaluation not executed",
        "grouping_column": LOEO_GROUPING_COLUMN,
        "rejected_grouping_columns": ["forecast_episode_id", "source_day"],
        "total_episode_ids": int(len(episode_summary)),
        "split_spanning_episode_count": int(len(spanning)),
        "singleton_episode_count": int(len(singleton_ids)),
        "singleton_benign_count": int((singleton["attack_type"] == "Benign").sum()),
        "singleton_non_benign_count": int((singleton["attack_type"] != "Benign").sum()),
        "eligible_attack_types": list(LOEO_ELIGIBLE_ATTACK_TYPES),
        "excluded_attack_types": list(LOEO_EXCLUDED_ATTACK_TYPES),
        "eligible_attack_counts": attack_counts,
        "excluded_attack_counts": excluded_counts,
        "validated_fold_count": int(len(folds)),
        "fold_isolation_verified": True,
        "evaluation_executed": False,
    }
    (output_dir / "protocol_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the canonical episode-level LOEO protocol without model evaluation.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/protocol_validation"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(validate(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()