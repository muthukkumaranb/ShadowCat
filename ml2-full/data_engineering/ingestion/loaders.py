import pyarrow.parquet as pq

# Columns confirmed usable per reports/field_mapping.csv — everything else
# in the raw file is either 0%-filled or not part of the frozen P0 contract.
USABLE_COLUMNS = [
    "dataset", "src_ip", "dst_ip", "src_port", "dst_port", "l4_proto",
    "ts_start", "ts_end", "duration", "packets", "bytes",
    "tcp_flags", "label_binary", "label_attack"
]


def load_raw_flows(dataset_config):
    """
    Loads only the usable raw columns from the configured raw parquet file.
    Returns a plain pandas DataFrame with raw (not yet canonical) column
    names — canonicalization happens in preprocessing.cleaning.clean().
    """
    raw_file = dataset_config["paths"]["raw_file"]

    parquet_file = pq.ParquetFile(raw_file)
    table = parquet_file.read(columns=USABLE_COLUMNS)
    df = table.to_pandas()

    print(f"Loaded {len(df):,} rows with {len(USABLE_COLUMNS)} usable columns.")
    return df