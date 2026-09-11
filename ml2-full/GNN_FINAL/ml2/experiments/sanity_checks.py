"""Protocol sanity checks for the chronological ML2 ablation."""

import pandas as pd
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINDOW_PATH = ROOT / "data" / "ucs" / "ucs_windows.parquet"
EDGE_PATH = ROOT / "data" / "ucs" / "ucs_graph_edgelists.parquet"
LOOKUP_PATH = ROOT / "data" / "ucs" / "node_lookup.parquet"
DEVIATION_PATH = ROOT / "data" / "ml1_artifacts" / "deviation_predictions.csv"


def chronological_sets(timestamps):
    ordered = sorted(pd.to_datetime(timestamps, utc=True).unique())
    n_train = max(1, int(len(ordered) * 0.70))
    n_val = max(1, int(len(ordered) * 0.15))
    return {
        "train": set(ordered[:n_train]),
        "val": set(ordered[n_train:n_train + n_val]),
        "test": set(ordered[n_train + n_val:]),
    }


def labeled_split_sets(windows):
    return {
        split: set(windows.loc[windows["split"] == split, "window_start_utc"])
        for split in ("train", "val", "test")
    }


def print_artifact_coverage(name, split_sets, deviation):
    print(f"{name} split unique-window counts: "
          f"train={len(split_sets['train'])}, "
          f"val={len(split_sets['val'])}, "
          f"test={len(split_sets['test'])}")
    for split_name, timestamps in split_sets.items():
        rows = deviation[deviation["timestamp"].isin(timestamps)]
        matched = rows["timestamp"].nunique()
        total = len(timestamps)
        zero_rows = int((rows["deviation_score"] == 0).sum())
        print(
            f"{name} {split_name}: rows={len(rows)}, "
            f"unique_windows={matched}, zero_score_rows={zero_rows}, "
            f"slot10_fill_rate={matched / total:.6f}"
        )


def own_slot_values(edges, timestamps):
    values = [[] for _ in range(10)]
    for timestamp in timestamps:
        window = edges[edges["window_start_utc"] == timestamp]
        active = sorted(set(window["src_node_id"]) | set(window["dst_node_id"]))
        for node in active:
            incoming = window[window["dst_node_id"] == node]
            outgoing = window[window["src_node_id"] == node]
            values[0].append(len(incoming))
            values[1].append(len(outgoing))
            values[2].append(incoming["flow_count"].sum())
            values[3].append(outgoing["flow_count"].sum())
            values[4].append(incoming["byte_count_sum"].sum())
            values[5].append(outgoing["byte_count_sum"].sum())
            values[6].append(incoming["packet_count_sum"].sum())
            values[7].append(outgoing["packet_count_sum"].sum())
            values[8].append(incoming["duration_mean_sec"].mean() if len(incoming) else 0.0)
            values[9].append(outgoing["duration_mean_sec"].mean() if len(outgoing) else 0.0)
    return values


def main():
    windows = pd.read_parquet(WINDOW_PATH)
    edges = pd.read_parquet(EDGE_PATH)
    node_lookup = pd.read_parquet(LOOKUP_PATH)
    windows["window_start_utc"] = pd.to_datetime(windows["window_start_utc"], utc=True)
    edges["window_start_utc"] = pd.to_datetime(edges["window_start_utc"], utc=True)
    deviation = pd.read_csv(DEVIATION_PATH)
    if "window_start_utc" in deviation.columns:
        deviation = deviation.rename(columns={"window_start_utc": "timestamp"})
    if "normalized_deviation_score" in deviation.columns:
        deviation = deviation.rename(
            columns={"normalized_deviation_score": "deviation_score"}
        )
    deviation["timestamp"] = pd.to_datetime(deviation["timestamp"], utc=True)

    print(f"Artifact rows: {len(deviation)}")
    print(f"Artifact unique windows: {deviation['timestamp'].nunique()}")
    print(f"Artifact zero deviation rows: {int((deviation['deviation_score'] == 0).sum())}")
    print(f"Canonical windows: {len(windows)}")
    print(f"Canonical split counts: {windows['split'].value_counts().to_dict()}")
    print(f"Canonical date range: {windows['window_start_utc'].min()} to "
          f"{windows['window_start_utc'].max()}")
    print(f"Artifact exact canonical-window matches: "
          f"{int(deviation['timestamp'].isin(set(windows['window_start_utc'])).sum())}")
    print(f"Canonical graph node IDs in node_lookup: "
          f"{set(edges['src_node_id']).union(edges['dst_node_id']).issubset(set(node_lookup['node_id']))}")

    full_splits = labeled_split_sets(windows)
    matched_splits = chronological_sets(deviation["timestamp"])
    print_artifact_coverage("full_graph_sequence", full_splits, deviation)
    print_artifact_coverage("matched_deviation_population", matched_splits, deviation)

    slot_values = own_slot_values(edges, set(deviation["timestamp"]))
    print("Own node-feature slots 0-9 over matched active nodes:")
    for index, values in enumerate(slot_values):
        quantiles = np.quantile(values, [0.01, 0.50, 0.99])
        print(
            f"slot{index}: min={min(values):.6g}, p01={quantiles[0]:.6g}, "
            f"median={quantiles[1]:.6g}, p99={quantiles[2]:.6g}, "
            f"max={max(values):.6g}, mean={np.mean(values):.6g}, "
            f"std={np.std(values):.6g}"
        )

    normalized_scores = deviation["deviation_score"].to_numpy(dtype=float)
    median = float(np.median(normalized_scores))
    mad = float(np.median(np.abs(normalized_scores - median)))
    print(
        f"slot10 deviation_score: min={deviation['deviation_score'].min():.6g}, "
        f"median={deviation['deviation_score'].median():.6g}, "
        f"max={deviation['deviation_score'].max():.6g}, "
        f"normalized_median={median:.6g}, normalized_MAD={mad:.6g}"
    )


if __name__ == "__main__":
    main()