"""
Unit tests for forecasting labels and chronological boundary split creation.
"""
import pytest
import pandas as pd
import numpy as np
from gnn.graph_dataset import create_chronological_splits


def test_chronological_splits_purge_and_embargo():
    n_graphs = 100
    splits = create_chronological_splits(
        n_graphs=n_graphs,
        train_frac=0.70,
        val_frac=0.15,
        window_size=5,
        horizon_windows=2,
    )

    train_start, train_end = splits["train"]
    val_start, val_end = splits["val"]
    test_start, test_end = splits["test"]

    # Check strictly chronological ordering
    assert train_start == 0
    assert train_end < val_start
    assert val_start <= val_end
    assert val_end < test_start
    assert test_start <= test_end
    assert test_end == n_graphs - 1

    # Check purge gap exists between train_end and val cut point (index 70)
    assert train_end <= 70 - 2

    # Check embargo exists
    assert val_start >= 70 + 5
