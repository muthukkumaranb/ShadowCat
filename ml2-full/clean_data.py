    import pyarrow.parquet as pq
    import pandas as pd
    import yaml
    import os

    # --- Step 1: Load configuration (no hardcoded values in this script) ---
    with open("configs/dataset_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    RAW_FILE = config["paths"]["raw_file"]
    DATASET_NAME = config["dataset"]["name"]          # "UGR16"
    SCOPE_ID = config["scope"]["id"]                   # "UGR16:TEST"
    OUTPUT_DIR = config["paths"]["intermediate_dir"]

    # --- Step 2: Read only the usable raw columns (from field_mapping.csv) ---
    usable_columns = [
        "dataset", "src_ip", "dst_ip", "src_port", "dst_port", "l4_proto",
        "ts_start", "ts_end", "duration", "packets", "bytes",
        "tcp_flags", "label_binary", "label_attack"
    ]

    parquet_file = pq.ParquetFile(RAW_FILE)
    table = parquet_file.read(columns=usable_columns)
    df = table.to_pandas()

    print(f"Loaded {len(df):,} rows with {len(usable_columns)} usable columns.")

    # --- Step 3: VALIDATE the raw dataset column before trusting it ---
    unique_datasets = set(df["dataset"].unique())
    expected = {"ugr16"}
    if unique_datasets != expected:
        raise ValueError(
            f"Dataset validation failed! Expected {expected}, "
            f"but found {unique_datasets}. Refusing to proceed."
        )
    print(f"Validation passed: raw 'dataset' column contains only {unique_datasets}")

    # --- Step 4: Overwrite with the CANONICAL value from config (not the raw value) ---
    df["dataset"] = DATASET_NAME          # "UGR16" (normalized casing)
    df["scope_id"] = SCOPE_ID             # injected from config, not hardcoded

    # --- Step 5: Rename raw columns to canonical UCS field names ---
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

    # --- Step 6: Convert timestamp from Unix epoch seconds to UTC datetime ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["timestamp_end"] = pd.to_datetime(df["timestamp_end"], unit="s", utc=True)

    # --- Step 7: Sort chronologically (Section 12 requirement) ---
    df = df.sort_values("timestamp").reset_index(drop=True)
    print("Data sorted by timestamp.")

    # --- Step 8: Save as the intermediate output ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "cleaned_sorted.parquet")
    df.to_parquet(output_path, index=False)

    print(f"Saved cleaned, sorted intermediate file to: {output_path}")
    print(f"Final row count: {len(df):,}")
    print(f"Final columns: {list(df.columns)}")