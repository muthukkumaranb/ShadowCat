import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/intermediate/ugr16/validated_flows.parquet"
OUTPUT_FILE = "data/processed/ucs_v1/ucs_windows_raw.parquet"


def compute_windows(df):
    """
    Takes a validated-flows dataframe and returns the 1-minute windows
    dataframe (Section 14 logic). Pulled into a function so it can be
    imported and reused by leakage tests, not just run as a script.
    """
    df = df.copy()

    # --- Step 1: Assign each flow to its 1-minute window ---
    df["window_timestamp"] = df["timestamp"].dt.floor("min")

    # --- Step 2: Protocol normalization ---
    df["is_tcp"] = (df["protocol"] == 6).astype(int)
    df["is_udp"] = (df["protocol"] == 17).astype(int)
    df["is_icmp"] = (df["protocol"] == 1).astype(int)

    # --- Step 3: Group by minute (+ dataset + scope_id) ---
    grouped = df.groupby(["window_timestamp", "dataset", "scope_id"])

    windows = grouped.agg(
        flow_count=("timestamp", "size"),
        packet_count=("packet_count", "sum"),
        bytes_in=("byte_count", "sum"),
        unique_src=("source_id", "nunique"),
        unique_dst=("destination_id", "nunique"),
        tcp_count=("is_tcp", "sum"),
        udp_count=("is_udp", "sum"),
        icmp_count=("is_icmp", "sum"),
    ).reset_index()

    # --- Step 4: peer_count via distinct (source_id, destination_id) pairs ---
    peer_counts = (
        df.groupby(["window_timestamp", "dataset", "scope_id"])[["source_id", "destination_id"]]
        .apply(lambda g: g.drop_duplicates().shape[0])
        .reset_index(name="peer_count")
    )
    windows = windows.merge(peer_counts, on=["window_timestamp", "dataset", "scope_id"], how="left")

    # --- Step 5: Ratios and rate ---
    windows["tcp_ratio"] = windows["tcp_count"] / windows["flow_count"]
    windows["udp_ratio"] = windows["udp_count"] / windows["flow_count"]
    windows["icmp_ratio"] = windows["icmp_count"] / windows["flow_count"]
    windows["connection_rate"] = windows["flow_count"] / 60.0

    windows = windows.drop(columns=["tcp_count", "udp_count", "icmp_count"])

    # --- Step 6: Rename + fixed columns ---
    windows = windows.rename(columns={"window_timestamp": "timestamp"})
    windows["window_size_sec"] = 60

    # --- Step 7: Sort chronologically ---
    windows = windows.sort_values("timestamp").reset_index(drop=True)

    return windows


if __name__ == "__main__":
    print("Loading validated flows...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df):,} rows.")

    print("Computing 1-minute windows...")
    windows = compute_windows(df)

    print(f"\nTotal 1-minute windows created: {len(windows):,}")

    expected_minutes = pd.date_range(
        start=windows["timestamp"].min(),
        end=windows["timestamp"].max(),
        freq="min"
    )
    missing_minutes = expected_minutes.difference(windows["timestamp"])
    print(f"Expected minute-slots in range: {len(expected_minutes):,}")
    print(f"Actual windows produced:        {len(windows):,}")
    print(f"Missing minutes (zero-flow gaps): {len(missing_minutes):,}")
    if len(missing_minutes) > 0:
        print("First few missing minutes:", list(missing_minutes[:5]))

    print(f"\nColumns: {list(windows.columns)}")
    print("\nFirst 5 windows:")
    print(windows.head())

    os.makedirs("data/processed/ucs_v1", exist_ok=True)
    windows.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")