"""
Stage 1: Input Ingestion Module
Ingests raw CSE-CIC-IDS2018 CSV files with encoding fallback, header sanitation,
removal of embedded header rows, and provenance metadata tracking.
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def extract_day_from_filename(filename: str) -> str:
    """Extract standard date string (e.g., '14-02-2018') from filename."""
    match = re.search(r"(\d{2}-\d{2}-\d{4})", filename)
    if match:
        return match.group(1)
    return os.path.splitext(os.path.basename(filename))[0]


def ingest_csv_file(
    filepath: str,
    encoding: str = "latin1",
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """
    Ingests a single CSE-CIC-IDS2018 CSV file, applying:
    1. Latin-1 encoding fallback to prevent decoding errors.
    2. Column header whitespace stripping.
    3. Detection and removal of repeated mid-file header lines.
    4. Attachment of source_file and source_day provenance metadata.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    filename = os.path.basename(filepath)
    source_day = extract_day_from_filename(filename)

    # Read CSV
    df = pd.read_csv(
        filepath,
        encoding=encoding,
        low_memory=False,
        skip_blank_lines=True,
        on_bad_lines="skip",
    )

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # Check for and filter out embedded duplicate header rows
    if "Label" in df.columns:
        header_mask = df["Label"].astype(str).str.strip() == "Label"
        dropped_headers = int(header_mask.sum())
        if dropped_headers > 0:
            df = df[~header_mask].copy()
    else:
        dropped_headers = 0

    # Add provenance columns
    df["source_file"] = filename
    df["source_day"] = source_day

    return df


def load_dataset_day(filepath: str) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Load a single day file and return DataFrame + ingestion metadata audit info.
    """
    filename = os.path.basename(filepath)
    raw_size_bytes = os.path.getsize(filepath)

    df = ingest_csv_file(filepath)
    
    audit_info = {
        "filename": filename,
        "source_day": df["source_day"].iloc[0] if len(df) > 0 else "",
        "raw_size_bytes": raw_size_bytes,
        "initial_row_count": len(df),
        "columns_count": len(df.columns),
    }

    return df, audit_info
