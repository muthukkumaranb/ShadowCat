"""s2b-t3: already-scaled UCS windows must not be scaled a second time.

source_type="windows" returns parquet windows unchanged; flow_count must keep its spread
across windows (re-scaling scaled values collapses it towards a constant).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data-engineering"))

from src.ucs_extractor import UCSExtractor  # noqa: E402

PARQUET = REPO / "data-engineering/data/ucs/ucs_windows_models_v1.parquet"


@pytest.fixture(scope="module")
def windows():
    if not PARQUET.exists():
        pytest.skip(f"missing {PARQUET}")
    return pd.read_parquet(PARQUET).sort_values("window_start_utc").head(40).reset_index(drop=True)


@pytest.fixture(scope="module")
def extractor():
    return UCSExtractor(schema_version="v3.0", config_dir=str(REPO / "data-engineering/configs"),
                        data_dir=str(REPO / "data-engineering/data/ucs"))


def test_flow_count_not_collapsed(windows, extractor):
    tensor = extractor.extract_model_tensor(windows, source_type="windows")
    fc = tensor[:, UCSExtractor.MODEL_INPUT_COLUMNS.index("flow_count")]
    assert float(np.std(fc)) > 0.1


def test_windows_returned_unchanged(windows, extractor):
    out = extractor.extract(windows, source_type="windows")
    cols = UCSExtractor.MASK_COLUMNS + UCSExtractor.MODEL_FEATURE_COLUMNS
    np.testing.assert_array_equal(out[cols].to_numpy(), windows[cols].to_numpy())
