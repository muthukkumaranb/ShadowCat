"""
Test B-t1: ATT&CK Mapping Integrity
Validates that every label_attack_type in ucs_windows_models_v1.parquet is mapped
in family_to_attck.yaml, and every tactic ID exists in backend/mitre_kb.py.
"""
from pathlib import Path
import pandas as pd
import pytest
import yaml

from backend.mitre_kb import get_mitre_kb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "data-engineering" / "configs" / "family_to_attck.yaml"
PARQUET_PATH = REPO_ROOT / "data-engineering" / "data" / "ucs" / "ucs_windows_models_v1.parquet"


def test_family_to_attck_config_exists():
    assert CONFIG_PATH.exists(), f"Configuration file missing: {CONFIG_PATH}"


def test_all_parquet_families_mapped():
    assert PARQUET_PATH.exists(), f"Parquet file missing: {PARQUET_PATH}"
    df = pd.read_parquet(PARQUET_PATH)
    parquet_families = set(df["label_attack_type"].dropna().unique())

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    families = config.get("families", {})
    mapped_families = set(families.keys())

    missing = parquet_families - mapped_families
    assert len(missing) == 0, f"Families in parquet missing from mapping: {missing}"


def test_all_tactics_exist_in_mitre_kb():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    families = config.get("families", {})
    kb = get_mitre_kb()

    for family, mapping in families.items():
        assert "justification" in mapping, f"Missing justification for family: {family}"
        assert mapping["justification"], f"Empty justification for family: {family}"
        assert "url" in mapping, f"Missing URL for family: {family}"
        assert mapping["url"].startswith("http"), f"Invalid URL for family: {family}"

        tactic_id = mapping.get("tactic_id")
        tactic_name = mapping.get("tactic_name")

        if family == "Benign":
            assert tactic_id is None, "Benign should map to null/None tactic_id"
            assert tactic_name is None, "Benign should map to null/None tactic_name"
        else:
            assert tactic_id is not None, f"Attack family {family} must have a tactic_id"
            tac = kb.get_tactic(tactic_id)
            assert tac is not None, f"Tactic ID {tactic_id} for family {family} not found in mitre_kb"
            assert tac["name"] == tactic_name, (
                f"Tactic name mismatch for {tactic_id}: expected {tac['name']}, got {tactic_name}"
            )
