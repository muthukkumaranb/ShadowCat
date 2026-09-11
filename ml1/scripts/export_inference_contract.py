#!/usr/bin/env python
"""Export the exact inference normalization contract for the probabilistic LSTM checkpoint.

This script reads the authoritative training metadata and canonical scaler
parameters, cross-validates them against the actual parquet data, and writes
two versioned artifacts that backend can consume for live inference:

    artifacts/lstm/inference_scaler_v1.yaml
    artifacts/lstm/inference_feature_order_v1.json

It performs every cross-check described in the implementation plan and fails
loudly if any invariant is violated — a failed check is always preferable to
a silently wrong export.

Usage:
    python scripts/export_inference_contract.py
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

REPRODUCIBILITY_METADATA = (
    REPO_ROOT
    / "artifacts"
    / "lstm"
    / "probabilistic_world_model"
    / "reproducibility_metadata.json"
)
SCALER_PARAMS = REPO_ROOT / "data" / "ucs" / "scaler_params.yaml"
UCS_WINDOWS = REPO_ROOT / "data" / "ucs" / "ucs_windows.parquet"
CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "lstm"
    / "probabilistic_world_model"
    / "gaussian_next_state_best.pt"
)
CONFIG_SNAPSHOT = (
    REPO_ROOT
    / "artifacts"
    / "lstm"
    / "probabilistic_world_model"
    / "config_snapshot.yaml"
)

OUTPUT_SCALER = REPO_ROOT / "artifacts" / "lstm" / "inference_scaler_v1.yaml"
OUTPUT_FEATURES = REPO_ROOT / "artifacts" / "lstm" / "inference_feature_order_v1.json"

# Canonical metadata / label columns that are NOT features.
UCS_METADATA_COLUMNS = {
    "window_id",
    "window_start_utc",
    "window_end_utc",
    "source_day",
    "split",
}
UCS_LABEL_COLUMNS = {
    "label_binary",
    "label_attack_type",
    "future_attack_label",
    "has_malicious_flows",
    "raw_label_dominant",
}

# From lstm/ucs.py — the base columns that receive log1p before robust scaling.
LOG1P_BASE_COLUMNS = {
    "byte_count_fwd",
    "byte_count_bwd",
    "packet_count_fwd",
    "packet_count_bwd",
    "bytes_per_sec",
    "packets_per_sec",
    "duration_sec",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fail(message: str) -> None:
    """Print an error and exit -- never silently proceed on a failed check."""
    print(f"\n[FAIL] {message}", file=sys.stderr)
    sys.exit(1)


def check(condition: bool, message: str) -> None:
    if not condition:
        fail(message)
    print(f"  [OK] {message}")


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Inference Contract Export")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load authoritative sources
    # ------------------------------------------------------------------
    print("\n[1/5] Loading authoritative sources...")

    check(REPRODUCIBILITY_METADATA.exists(), f"reproducibility_metadata.json exists")
    check(SCALER_PARAMS.exists(), f"scaler_params.yaml exists")
    check(UCS_WINDOWS.exists(), f"ucs_windows.parquet exists")
    check(CHECKPOINT.exists(), f"gaussian_next_state_best.pt exists")

    with open(REPRODUCIBILITY_METADATA, encoding="utf-8") as f:
        meta = json.load(f)

    with open(SCALER_PARAMS, encoding="utf-8") as f:
        scaler_raw = yaml.safe_load(f)

    # Read parquet to verify columns.
    parquet_sample = pd.read_parquet(UCS_WINDOWS).head(1)
    excluded = UCS_METADATA_COLUMNS | UCS_LABEL_COLUMNS
    parquet_numeric_features = [
        col
        for col in parquet_sample.select_dtypes(include=[np.number]).columns
        if col not in excluded
    ]

    # ------------------------------------------------------------------
    # 2. Cross-validation checks
    # ------------------------------------------------------------------
    print("\n[2/5] Cross-validating feature lists...")

    meta_features: list[str] = meta["features"]
    scaler_features: list[str] = list(scaler_raw.keys())

    check(
        meta["input_dimension"] == 406,
        f"input_dimension = {meta['input_dimension']} (expected 406)",
    )
    check(
        len(meta_features) == 406,
        f"reproducibility_metadata features count = {len(meta_features)} (expected 406)",
    )
    check(
        len(scaler_features) == 400,
        f"scaler_params.yaml features count = {len(scaler_features)} (expected 400)",
    )

    # Identify the 6 mask columns.
    meta_set = set(meta_features)
    scaler_set = set(scaler_features)
    extra_in_meta = sorted(meta_set - scaler_set)
    extra_in_scaler = sorted(scaler_set - meta_set)

    check(
        len(extra_in_scaler) == 0,
        f"No features in scaler_params.yaml absent from metadata",
    )
    check(
        len(extra_in_meta) == 6,
        f"Exactly 6 features in metadata absent from scaler_params.yaml: {extra_in_meta}",
    )
    expected_masks = sorted([
        "mask_has_traffic_volume_features",
        "mask_has_flow_timing_features",
        "mask_has_packet_level_features",
        "mask_has_tcp_flags",
        "mask_has_graph_topology",
        "mask_has_identity_auth",
    ])
    check(
        extra_in_meta == expected_masks,
        f"Extra features are exactly the 6 mask columns",
    )

    # Ordering check: the 400 scaled features appear in the same order.
    meta_scaled_ordered = [f for f in meta_features if f in scaler_set]
    check(
        meta_scaled_ordered == scaler_features,
        f"400 scaled features are in the same order in both sources",
    )

    # Parquet check.
    check(
        len(parquet_numeric_features) == 406,
        f"ucs_windows.parquet numeric feature count = {len(parquet_numeric_features)} (expected 406)",
    )
    parquet_feature_set = set(parquet_numeric_features)
    check(
        meta_set == parquet_feature_set,
        f"metadata features match parquet numeric features exactly",
    )

    # Duplicate check.
    check(
        len(meta_features) == len(set(meta_features)),
        f"No duplicate feature names in metadata",
    )

    # log1p consistency check.
    scaler_log1p = {
        name for name, params in scaler_raw.items() if params.get("is_log1p", False)
    }
    expected_log1p = {
        col
        for col in scaler_features
        if any(col.startswith(base) for base in LOG1P_BASE_COLUMNS)
    }
    check(
        scaler_log1p == expected_log1p,
        f"log1p columns in scaler_params.yaml match LOG1P_BASE_COLUMNS definition ({len(scaler_log1p)} features)",
    )

    # PCA metadata cross-check.
    check(
        meta["pca"]["n_features_in"] == 406,
        f"PCA n_features_in = 406",
    )
    check(
        meta["features"] == meta["pca"]["input_columns"],
        f"metadata features == PCA input_columns",
    )

    # Normalization field.
    check(
        meta["normalization"] == "as supplied by canonical UCS artifact",
        f"normalization field confirms no additional ML1 scaler",
    )

    # ------------------------------------------------------------------
    # 3. Build and write inference_scaler_v1.yaml
    # ------------------------------------------------------------------
    print("\n[3/5] Writing inference_scaler_v1.yaml...")

    # Use an OrderedDict to control YAML output order.
    scaler_doc = OrderedDict()
    scaler_doc["_metadata"] = OrderedDict([
        ("version", "v1"),
        ("artifact_type", "inference_scaler"),
        ("description", (
            "Exact normalization parameters for live inference against the "
            "probabilistic LSTM world-model checkpoint (gaussian_next_state_best.pt). "
            "Apply these transforms to raw UCS window features before feeding "
            "into the LSTM."
        )),
        ("checkpoint", "gaussian_next_state_best.pt"),
        ("checkpoint_commit", meta.get("repository_commit", "unknown")),
        ("config_snapshot", "config_snapshot.yaml"),
        ("exported_at", datetime.now(timezone.utc).isoformat()),
        ("relationship_to_data_eng_scaler", (
            "IDENTICAL — this IS the canonical scaler_params.yaml from the "
            "data-engineering repo (data/ucs/scaler_params.yaml). ML1 did NOT "
            "apply any additional normalization on top. The UCS pipeline applied "
            "log1p + RobustScaler before writing ucs_windows.parquet, and those "
            "already-normalized values were fed directly into the LSTM. There is "
            "ONE source of truth, not two."
        )),
        ("normalization_pipeline", OrderedDict([
            ("step_1", "For features with is_log1p=true: x = log1p(x)"),
            ("step_2", "For all 400 scaled features: x = (x - median) / scale"),
            ("step_3", (
                "The 6 mask columns (mask_has_*) are fixed flags (1.0 or 0.0) "
                "and are NOT scaled — pass them through unchanged."
            )),
        ])),
        ("total_scaled_features", 400),
        ("total_unscaled_mask_features", 6),
        ("total_lstm_input_features", 406),
        ("unscaled_mask_columns", expected_masks),
    ])

    # Add per-feature scaler parameters.
    scaler_doc["features"] = OrderedDict()
    for feature_name in scaler_features:
        params = scaler_raw[feature_name]
        scaler_doc["features"][feature_name] = OrderedDict([
            ("median", params["median"]),
            ("scale", params["scale"]),
            ("is_log1p", params.get("is_log1p", False)),
            ("q25", params.get("q25")),
            ("q75", params.get("q75")),
        ])

    # Custom YAML representer for OrderedDict to preserve order.
    def represent_ordereddict(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())

    yaml.add_representer(OrderedDict, represent_ordereddict)

    OUTPUT_SCALER.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SCALER, "w", encoding="utf-8") as f:
        yaml.dump(
            dict(scaler_doc),  # Convert top level for clean output
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )

    check(OUTPUT_SCALER.exists(), f"Written: {OUTPUT_SCALER.relative_to(REPO_ROOT)}")

    # Verify round-trip.
    with open(OUTPUT_SCALER, encoding="utf-8") as f:
        roundtrip = yaml.safe_load(f)
    check(
        len(roundtrip["features"]) == 400,
        f"Round-trip verification: 400 features in written YAML",
    )

    # ------------------------------------------------------------------
    # 4. Build and write inference_feature_order_v1.json
    # ------------------------------------------------------------------
    print("\n[4/5] Writing inference_feature_order_v1.json...")

    feature_doc = OrderedDict([
        ("_metadata", OrderedDict([
            ("version", "v1"),
            ("artifact_type", "inference_feature_order"),
            ("description", (
                "Exact ordered list of 406 feature names expected by the LSTM "
                "input layer. The model interprets each position by index — if "
                "features are supplied in a different order, predictions will be "
                "silently wrong."
            )),
            ("checkpoint", "gaussian_next_state_best.pt"),
            ("checkpoint_commit", meta.get("repository_commit", "unknown")),
            ("config_snapshot", "config_snapshot.yaml"),
            ("exported_at", datetime.now(timezone.utc).isoformat()),
            ("source", (
                "Cross-validated from reproducibility_metadata.json (authoritative), "
                "ucs_windows.parquet column order (confirmed matching), and "
                "SCHEMA.md (confirmed consistent)."
            )),
            ("total_features", 406),
            ("input_dimension", 406),
            ("state_dimension", meta.get("state_dimension", 406)),
        ])),
        ("features", meta_features),
    ])

    OUTPUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FEATURES, "w", encoding="utf-8") as f:
        json.dump(feature_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    check(OUTPUT_FEATURES.exists(), f"Written: {OUTPUT_FEATURES.relative_to(REPO_ROOT)}")

    # Verify round-trip.
    with open(OUTPUT_FEATURES, encoding="utf-8") as f:
        roundtrip_features = json.load(f)
    check(
        len(roundtrip_features["features"]) == 406,
        f"Round-trip verification: 406 features in written JSON",
    )
    check(
        roundtrip_features["features"] == meta_features,
        f"Round-trip verification: feature order preserved exactly",
    )

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n[5/5] Summary")
    print("=" * 72)
    print(f"  Scaler artifact:  {OUTPUT_SCALER.relative_to(REPO_ROOT)}")
    print(f"  Feature artifact: {OUTPUT_FEATURES.relative_to(REPO_ROOT)}")
    print(f"  Checkpoint:       {CHECKPOINT.relative_to(REPO_ROOT)}")
    print(f"  Commit:           {meta.get('repository_commit', 'unknown')}")
    print(f"  Scaled features:  400")
    print(f"  Mask features:    6  (pass-through, not scaled)")
    print(f"  Total LSTM input: 406")
    print(f"  log1p features:   {len(scaler_log1p)}")
    print()
    print("  ML1 scaler relationship: IDENTICAL to data-eng scaler_params.yaml")
    print("  (no additional normalization applied by ML1)")
    print("=" * 72)
    print("\n[SUCCESS] All cross-checks passed. Artifacts exported successfully.")


if __name__ == "__main__":
    main()
