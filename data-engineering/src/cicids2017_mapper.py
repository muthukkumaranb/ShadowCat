import pandas as pd
from typing import Dict, Tuple, Optional
from src.canonical_mapper import load_canonical_mapping, KNOWN_CIC_ALIASES, _normalize_identifier

def map_cicids2017_schema(df: pd.DataFrame, config_path: str = "configs/canonical_mapping.yaml") -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Schema-mapping layer for CIC-IDS2017 dataset.
    Normalizes CIC-IDS2017 into the canonical shape expected by the pipeline.
    Documents columns with no 2018 equivalent.
    """
    mapping_config = load_canonical_mapping(config_path)
    col_map = mapping_config.get("mapping", {})
    canonical_targets = set(col_map.values())
    
    rename_dict = {}
    unmapped_raw_cols = []
    
    # Track 2017 specific columns with no 2018 equivalent
    unmapped_2017_specific = []

    for raw_col in df.columns:
        stripped = raw_col.strip()
        norm = _normalize_identifier(stripped)
        
        # Exact raw column match in mapping
        if stripped in col_map:
            rename_dict[raw_col] = col_map[stripped]
        # Already canonical name
        elif stripped in canonical_targets:
            rename_dict[raw_col] = stripped
        # CICFlowMeter naming aliases (covers many 2017 variations)
        elif norm in KNOWN_CIC_ALIASES:
            rename_dict[raw_col] = KNOWN_CIC_ALIASES[norm]
        elif raw_col in ("source_file", "source_day", "source_dataset"):
            rename_dict[raw_col] = raw_col
        else:
            # Document missing equivalent
            unmapped_raw_cols.append(raw_col)
            unmapped_2017_specific.append(raw_col)

    df_mapped = df.rename(columns=rename_dict).copy()
    
    # Remove duplicate columns if they appear
    if df_mapped.columns.duplicated().any():
        df_mapped = df_mapped.loc[:, ~df_mapped.columns.duplicated(keep="first")].copy()
        
    audit = {
        "mapped_columns_count": len(rename_dict),
        "unmapped_columns": unmapped_raw_cols,
        "unmapped_2017_specific": unmapped_2017_specific,
        "final_columns": list(df_mapped.columns),
    }

    return df_mapped, audit
