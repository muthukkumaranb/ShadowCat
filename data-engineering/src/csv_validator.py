"""
CSV Canonical-Mapper Validator for CICFlowMeter Inputs
SIH 2026 - Unified Cyber State (UCS) Ingestion Pipeline

Validates raw input DataFrames against the CICFlowMeter CSV contract before extraction.
Categorizes inputs into:
- EXACT_MATCH: All expected raw CICFlowMeter columns present with exact casing/names.
- MAPPED_VARIANT: Recognizable variants (case differences, underscores, canonical names) mapped cleanly.
- REJECTED: Incompatible schema, missing critical fields, or non-coercible data.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import yaml


class ValidationStatus(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    MAPPED_VARIANT = "MAPPED_VARIANT"
    REJECTED = "REJECTED"


@dataclass
class ValidationResult:
    status: ValidationStatus
    is_valid: bool
    mapped_df: Optional[pd.DataFrame] = None
    missing_required_columns: List[str] = field(default_factory=list)
    variant_mappings: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class CICFlowMeterValidator:
    """
    Validates and standardizes CICFlowMeter CSV inputs against the canonical schema specification.
    """

    # Core required functional fields that MUST be present to build valid UCS windows
    CRITICAL_FLOW_FIELDS = [
        "destination_port",
        "protocol",
        "raw_timestamp",
        "duration_microsec",
        "packet_count_fwd",
        "packet_count_bwd",
        "byte_count_fwd",
        "byte_count_bwd",
    ]

    def __init__(self, mapping_config_path: Optional[str] = None):
        base_dir = Path(__file__).resolve().parent.parent
        config_path = Path(mapping_config_path) if mapping_config_path else base_dir / "configs" / "canonical_mapping.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Mapping config not found at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.mapping_config = yaml.safe_load(f)

        self.raw_to_canonical: Dict[str, str] = self.mapping_config.get("mapping", {})
        self.canonical_to_raw: Dict[str, str] = {v: k for k, v in self.raw_to_canonical.items()}
        self.canonical_columns: Set[str] = set(self.raw_to_canonical.values())
        self.expected_raw_columns: Set[str] = set(self.raw_to_canonical.keys())

    @staticmethod
    def _normalize_identifier(name: str) -> str:
        """Strip, lowercase, and replace spaces and hyphens with underscores."""
        return name.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "")

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validates input DataFrame against CICFlowMeter CSV schema.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to validate.

        Returns
        -------
        ValidationResult
            Validation outcome with status, error details, and mapped DataFrame if valid.
        """
        if df is None or len(df) == 0:
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                is_valid=False,
                errors=["Input DataFrame is empty or None."],
            )

        col_names = [str(c).strip() for c in df.columns]
        df_working = df.copy()
        df_working.columns = col_names

        # 1. Check for EXACT_MATCH
        # All expected raw columns present and exact names match
        missing_exact_raw = [col for col in self.expected_raw_columns if col not in col_names]
        if len(missing_exact_raw) == 0:
            # Check type coercibility for sample
            type_errors = self._validate_data_types(df_working, is_raw=True)
            if type_errors:
                return ValidationResult(
                    status=ValidationStatus.REJECTED,
                    is_valid=False,
                    errors=type_errors,
                )
            return ValidationResult(
                status=ValidationStatus.EXACT_MATCH,
                is_valid=True,
                mapped_df=df_working,
            )

        # 2. Check for MAPPED_VARIANT
        # Attempt to map input columns using fuzzy/variant normalization
        variant_map: Dict[str, str] = {}
        mapped_df_cols: Dict[str, str] = {}
        warnings: List[str] = []

        # Precompute normalized lookups
        norm_to_raw = {self._normalize_identifier(k): k for k in self.expected_raw_columns}
        norm_to_canonical = {self._normalize_identifier(v): v for v in self.canonical_columns}

        for col in col_names:
            norm = self._normalize_identifier(col)
            # Already exact raw name
            if col in self.expected_raw_columns:
                mapped_df_cols[col] = col
            # Direct canonical name
            elif col in self.canonical_columns:
                mapped_df_cols[col] = self.canonical_to_raw[col]
                variant_map[col] = self.canonical_to_raw[col]
                warnings.append(f"Mapped canonical column '{col}' to raw schema '{self.canonical_to_raw[col]}'")
            # Normalized match to raw column
            elif norm in norm_to_raw:
                target_raw = norm_to_raw[norm]
                mapped_df_cols[col] = target_raw
                variant_map[col] = target_raw
                warnings.append(f"Mapped variant column '{col}' -> '{target_raw}'")
            # Normalized match to canonical column
            elif norm in norm_to_canonical:
                canonical_target = norm_to_canonical[norm]
                target_raw = self.canonical_to_raw[canonical_target]
                mapped_df_cols[col] = target_raw
                variant_map[col] = target_raw
                warnings.append(f"Mapped variant column '{col}' -> canonical '{canonical_target}' -> '{target_raw}'")

        # Check whether critical required fields can be resolved
        resolved_raw_cols = set(mapped_df_cols.values())
        resolved_canonical_cols = {self.raw_to_canonical[r] for r in resolved_raw_cols if r in self.raw_to_canonical}

        missing_critical = [
            f for f in self.CRITICAL_FLOW_FIELDS if f not in resolved_canonical_cols
        ]

        if missing_critical:
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                is_valid=False,
                missing_required_columns=missing_critical,
                variant_mappings=variant_map,
                errors=[f"Incompatible schema: missing critical flow fields {missing_critical}"],
                warnings=warnings,
            )

        # Standardize columns to expected raw names
        df_standardized = df_working.rename(columns=mapped_df_cols)

        # Validate types
        type_errors = self._validate_data_types(df_standardized, is_raw=True)
        if type_errors:
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                is_valid=False,
                missing_required_columns=missing_critical,
                variant_mappings=variant_map,
                errors=type_errors,
                warnings=warnings,
            )

        return ValidationResult(
            status=ValidationStatus.MAPPED_VARIANT,
            is_valid=True,
            mapped_df=df_standardized,
            variant_mappings=variant_map,
            warnings=warnings,
        )

    def _validate_data_types(self, df: pd.DataFrame, is_raw: bool = True) -> List[str]:
        """Validates that timestamp and numeric fields are coercible."""
        errors: List[str] = []

        # Find timestamp column
        ts_col = "Timestamp" if is_raw and "Timestamp" in df.columns else "raw_timestamp" if "raw_timestamp" in df.columns else None
        if ts_col:
            sample_ts = df[ts_col].dropna().head(20)
            parsed = pd.to_datetime(sample_ts, format="%d/%m/%Y %H:%M:%S", errors="coerce")
            if parsed.isna().all() and len(sample_ts) > 0:
                # Try generic datetime parsing fallback
                parsed_fallback = pd.to_datetime(sample_ts, errors="coerce")
                if parsed_fallback.isna().all():
                    errors.append(f"Timestamp column '{ts_col}' cannot be parsed as valid datetime.")

        # Check numeric coercibility on a few critical columns
        check_cols = ["Dst Port", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts"] if is_raw else ["destination_port", "duration_microsec"]
        for col in check_cols:
            if col in df.columns:
                series = df[col].dropna()
                # Replace known infinity strings
                cleaned_series = series.astype(str).str.strip().replace(
                    {"Infinity": "nan", "Inf": "nan", "-Infinity": "nan", "-Inf": "nan"}
                )
                numeric_converted = pd.to_numeric(cleaned_series, errors="coerce")
                # If more than 50% became NaN from non-null string, it's corrupt
                if len(series) > 0 and numeric_converted.isna().sum() > 0.5 * len(series):
                    errors.append(f"Column '{col}' contains non-numeric invalid values.")

        return errors
