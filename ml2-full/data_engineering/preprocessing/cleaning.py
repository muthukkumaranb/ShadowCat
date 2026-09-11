import pandas as pd


def clean(df, dataset_config):
    """
    Takes the raw-column dataframe from ingestion.loaders.load_raw_flows()
    and returns a cleaned, canonically-named, chronologically-sorted
    dataframe. Does NOT read or write any files itself.
    """
    dataset_name = dataset_config["dataset"]["name"]   # "UGR16"
    scope_id = dataset_config["scope"]["id"]            # "UGR16:TEST"

    # --- Validate the raw dataset column before trusting it ---
    unique_datasets = set(df["dataset"].unique())
    expected = {"ugr16"}
    if unique_datasets != expected:
        raise ValueError(
            f"Dataset validation failed! Expected {expected}, "
            f"but found {unique_datasets}. Refusing to proceed."
        )
    print(f"Validation passed: raw 'dataset' column contains only {unique_datasets}")

    # --- Overwrite with the CANONICAL value from config (not the raw value) ---
    df = df.copy()
    df["dataset"] = dataset_name
    df["scope_id"] = scope_id

    # --- Rename raw columns to canonical UCS field names ---
    df = df.rename(columns={
        "ts_start": "timestamp",
        "ts_end": "timestamp_end",
        "src_ip": "source_id",
        "dst_ip": "destination_id",
        "src_port": "source_port",
        "dst_port": "destination_port",
        "l4_proto": "protocol",
        "packets": "packet_count",
        "bytes": "byte_count",
        "label_attack": "label_attack_type",
    })

    # --- Convert timestamps from Unix epoch seconds to UTC datetime ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["timestamp_end"] = pd.to_datetime(df["timestamp_end"], unit="s", utc=True)

    # --- Sort chronologically (Section 12 requirement) ---
    df = df.sort_values("timestamp").reset_index(drop=True)
    print("Data sorted by timestamp.")

    return df