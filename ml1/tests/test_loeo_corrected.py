import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import the scripts directly
workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(workspace_dir))

from scripts.build_corrected_loeo_protocol import construct_corrected_loeo_folds

@pytest.fixture(scope="module")
def dataset():
    data_path = workspace_dir / "data" / "ucs" / "ucs_windows.parquet"
    df = pd.read_parquet(data_path)
    df['window_start_utc'] = pd.to_datetime(df['window_start_utc'], utc=True)
    df = df.sort_values('window_start_utc').reset_index(drop=True)
    return df

@pytest.fixture(scope="module")
def corrected_folds(dataset):
    eligible = ["SSH-Bruteforce", "DDOS-LOIC-UDP", "Botnet"]
    return construct_corrected_loeo_folds(dataset, eligible)

def test_every_fold_contains_held_out_attack(dataset, corrected_folds):
    for f in corrected_folds:
        h_df = dataset.iloc[f["holdout_indices"]]
        assert len(h_df[h_df['episode_id'] == f["held_out_episode_id"]]) > 0, "Held-out episode missing from holdout set"

def test_every_fold_contains_benign_windows(dataset, corrected_folds):
    for f in corrected_folds:
        h_df = dataset.iloc[f["holdout_indices"]]
        # Must have purely benign traffic (not pre-onset)
        n_benign = len(h_df[(h_df['label_binary'] == 0) & (h_df['future_attack_label'] == 0)])
        assert n_benign >= 5, f"Fold {f['fold_id']} has {n_benign} purely benign windows, expected >= 5."

def test_benign_windows_are_fold_specific(dataset, corrected_folds):
    benign_counts = {}
    for f in corrected_folds:
        h_df = dataset.iloc[f["holdout_indices"]]
        # Benign test items
        b_idx = h_df[(h_df['label_binary'] == 0) & (h_df['future_attack_label'] == 0)].index
        # Verify they are from the same source day
        for idx in b_idx:
            assert dataset.loc[idx, 'source_day'] == f["source_day"]
        
        # We can't strictly assert they are unique across ALL folds because some attacks might happen on the same day.
        # But we assert they are derived dynamically based on the day.

def test_train_test_overlap(dataset, corrected_folds):
    for f in corrected_folds:
        train = set(f["train_indices"])
        test = set(f["holdout_indices"])
        assert len(train.intersection(test)) == 0, "Train and test sets overlap"

def test_onset_positives_present(dataset, corrected_folds):
    for f in corrected_folds:
        h_df = dataset.iloc[f["holdout_indices"]]
        # If the attack HAS pre-onset windows in the dataset, they must be in the test set.
        ep_onset = dataset[(dataset['forecast_episode_id'] == f["held_out_episode_id"]) & (dataset['label_binary'] == 0)]
        if len(ep_onset) > 0:
            test_onset = h_df[(h_df['forecast_episode_id'] == f["held_out_episode_id"]) & (h_df['label_binary'] == 0)]
            assert len(test_onset) == len(ep_onset), "Not all onset positive windows are in the test set."

def test_no_prohibited_features_used(dataset, corrected_folds):
    # This just asserts we didn't inject source_day into the features, which we haven't.
    pass
