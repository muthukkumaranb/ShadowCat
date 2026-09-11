import pandas as pd
import numpy as np
from build_windows import compute_windows

INPUT_FILE = "data/intermediate/ugr16/validated_flows.parquet"
WINDOW_SIZE_SEC = 60    # must match build_windows.py's window length
HORIZON_SECONDS = 600   # 10 minutes, per feature_config.yaml forecast.horizon_sec


def corrupt_future_flows(df, t, window_size_sec, horizon_seconds):
    """
    Returns a copy of df where flows STRICTLY AFTER window t's own interval
    ends (i.e. at or after t + window_size_sec) get scrambled, up through
    the forecast horizon. Flows inside [t, t+window_size_sec) — window t's
    own data — are never touched.
    """
    df = df.copy()
    future_start = t + pd.Timedelta(seconds=window_size_sec)
    future_end = future_start + pd.Timedelta(seconds=horizon_seconds)

    mask = (df["timestamp"] >= future_start) & (df["timestamp"] < future_end)
    n_corrupted = mask.sum()

    rng = np.random.default_rng(seed=42)
    df.loc[mask, "source_id"] = "CORRUPTED_SRC_" + rng.integers(0, 999999, mask.sum()).astype(str)
    df.loc[mask, "destination_id"] = "CORRUPTED_DST_" + rng.integers(0, 999999, mask.sum()).astype(str)
    df.loc[mask, "protocol"] = 250
    df.loc[mask, "packet_count"] = rng.integers(1_000_000, 9_000_000, mask.sum())
    df.loc[mask, "byte_count"] = rng.integers(1_000_000, 9_000_000, mask.sum())

    return df, n_corrupted


def run_perturbation_test(df, test_timestamps, window_size_sec, horizon_seconds):
    print("Building windows from ORIGINAL (uncorrupted) flows...")
    windows_original = compute_windows(df)

    all_passed = True

    for t in test_timestamps:
        print(f"\n{'='*60}")
        print(f"Testing window t = {t}")

        corrupted_df, n_corrupted = corrupt_future_flows(df, t, window_size_sec, horizon_seconds)
        future_start = t + pd.Timedelta(seconds=window_size_sec)
        future_end = future_start + pd.Timedelta(seconds=horizon_seconds)
        print(f"  Corrupted {n_corrupted:,} flows in [{future_start}, {future_end})")

        if n_corrupted == 0:
            print("  NOTE: no future data exists in the file for this window "
                  "(likely the last window) — main check will trivially pass, "
                  "canary check is not meaningful here. Skipping canary requirement.")

        windows_corrupted = compute_windows(corrupted_df)

        row_orig = windows_original[windows_original["timestamp"] == t]
        row_corrupt = windows_corrupted[windows_corrupted["timestamp"] == t]

        if row_orig.empty or row_corrupt.empty:
            print(f"  SKIPPED: window {t} not found in output.")
            continue

        compare_cols = [c for c in row_orig.columns if c not in ("timestamp", "dataset", "scope_id")]
        row_orig_vals = row_orig[compare_cols].reset_index(drop=True)
        row_corrupt_vals = row_corrupt[compare_cols].reset_index(drop=True)

        identical = row_orig_vals.equals(row_corrupt_vals)
        print(f"  MAIN CHECK (window t unaffected by future corruption): {'PASS' if identical else 'FAIL'}")
        if not identical:
            all_passed = False
            print("  Differences found:")
            for col in compare_cols:
                v1, v2 = row_orig_vals[col].iloc[0], row_corrupt_vals[col].iloc[0]
                if v1 != v2:
                    print(f"    {col}: original={v1}  corrupted={v2}")

        if n_corrupted > 0:
            future_mask_orig = (windows_original["timestamp"] >= future_start) & (windows_original["timestamp"] < future_end)
            future_mask_corrupt = (windows_corrupted["timestamp"] >= future_start) & (windows_corrupted["timestamp"] < future_end)

            future_orig = windows_original[future_mask_orig][compare_cols].reset_index(drop=True)
            future_corrupt = windows_corrupted[future_mask_corrupt][compare_cols].reset_index(drop=True)

            canary_changed = not future_orig.equals(future_corrupt)
            print(f"  CANARY CHECK (corruption actually changed future windows): {'PASS' if canary_changed else 'FAIL — test may be meaningless for this t'}")
            if not canary_changed:
                all_passed = False

    print(f"\n{'='*60}")
    print("OVERALL RESULT:", "ALL TESTS PASSED" if all_passed else "AT LEAST ONE TEST FAILED — investigate above")
    return all_passed


if __name__ == "__main__":
    print("Loading validated flows...")
    df = pd.read_parquet(INPUT_FILE)
    df = df.sort_values("timestamp").reset_index(drop=True)

    windows_check = compute_windows(df)
    all_timestamps = windows_check["timestamp"].sort_values().reset_index(drop=True)

    attack_window = pd.Timestamp("2016-08-29 01:08:00", tz="UTC")

    test_timestamps = [
        all_timestamps.iloc[0],
        all_timestamps.iloc[len(all_timestamps) // 4],
        attack_window,
        all_timestamps.iloc[len(all_timestamps) // 2],
        all_timestamps.iloc[-2],   # second-to-last, so real future data exists
        all_timestamps.iloc[-1],   # last window, exercises the "no future data" edge case on purpose
    ]

    run_perturbation_test(df, test_timestamps, WINDOW_SIZE_SEC, HORIZON_SECONDS)