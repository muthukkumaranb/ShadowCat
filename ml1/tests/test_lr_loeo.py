"""Unit tests for Set-A Logistic Regression baseline and LOEO protocol integrity."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from examples.run_lr_loeo import compute_metrics, get_set_a_features, run_target
from lstm.ucs import (
    LOEO_ELIGIBLE_ATTACK_TYPES,
    LOEO_GROUPING_COLUMN,
    build_loeo_episode_folds,
    load_ucs_windows,
)


def make_dummy_ucs_windows(num_episodes=5, rows_per_episode=10):
    rows = []
    start = pd.Timestamp("2026-01-01", tz="UTC")
    attack_types = ["SSH-Bruteforce", "DDOS-LOIC-UDP", "Botnet"]
    for ep_idx in range(num_episodes):
        ep_type = attack_types[ep_idx % len(attack_types)]
        ep_id = f"2026-01-01_{ep_type}_{ep_idx}"
        for r_idx in range(rows_per_episode):
            idx = ep_idx * rows_per_episode + r_idx
            rows.append({
                "window_id": f"w_{idx}",
                "window_start_utc": start + pd.Timedelta(minutes=idx),
                "window_end_utc": start + pd.Timedelta(minutes=idx + 1),
                "source_day": "2026-01-01",
                "split": "train" if ep_idx < 3 else "test",
                "label_binary": 1 if r_idx >= 5 else 0,
                "label_attack_type": ep_type if r_idx >= 5 else "Benign",
                "future_attack_label": 1 if r_idx >= 3 else 0,
                "episode_id": ep_id,
                "forecast_episode_id": ep_id if r_idx >= 5 else None,
                "feature_1": float(idx * 2),
                "feature_2": float(idx + 5),
                "feature_3": float(idx * 0.5),
            })
    return pd.DataFrame(rows)


def test_set_a_feature_selection():
    df = make_dummy_ucs_windows()
    features = get_set_a_features(df)
    assert "feature_1" in features
    assert "feature_2" in features
    assert "feature_3" in features
    assert "episode_id" not in features
    assert "label_binary" not in features
    assert "future_attack_label" not in features
    assert "source_day" not in features


def test_loeo_grouping_column_assertion():
    df = make_dummy_ucs_windows()
    with pytest.raises(ValueError, match="LOEO grouping must use 'episode_id'"):
        build_loeo_episode_folds(df, episode_col="forecast_episode_id")

    with pytest.raises(ValueError, match="LOEO grouping must use 'episode_id'"):
        build_loeo_episode_folds(df, episode_col="source_day")


def test_loeo_fold_isolation():
    windows = load_ucs_windows("data/ucs/ucs_windows.parquet")
    folds = build_loeo_episode_folds(windows, episode_col=LOEO_GROUPING_COLUMN)
    assert len(folds) == 37

    for fold in folds:
        train_indices = set(fold["train_indices"])
        holdout_indices = set(fold["holdout_indices"])
        held_out = fold["held_out_episode"]

        # No row index overlap
        assert len(train_indices & holdout_indices) == 0

        # Held-out episode_id is excluded from training
        train_episodes = set(windows.iloc[list(train_indices)][LOEO_GROUPING_COLUMN])
        assert held_out not in train_episodes


def test_fold_local_preprocessing():
    df = make_dummy_ucs_windows()
    features = get_set_a_features(df)
    train_df = df.iloc[:20]
    test_df = df.iloc[20:30]

    scaler = StandardScaler()
    scaler.fit(train_df[features].to_numpy())
    
    # Verify scaler params derived only from train set
    assert np.allclose(scaler.mean_, train_df[features].mean().to_numpy())
    
    x_train_scaled = scaler.transform(train_df[features].to_numpy())
    x_test_scaled = scaler.transform(test_df[features].to_numpy())

    assert x_train_scaled.shape == (20, len(features))
    assert x_test_scaled.shape == (10, len(features))


def test_logistic_regression_fit_and_predict():
    df = make_dummy_ucs_windows()
    features = get_set_a_features(df)
    x_train = df[features].iloc[:30].to_numpy()
    y_train = df["label_binary"].iloc[:30].to_numpy()
    x_test = df[features].iloc[30:].to_numpy()

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(x_train_scaled, y_train)
    probs = model.predict_proba(x_test_scaled)[:, 1]

    assert len(probs) == 20
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_metric_generation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.8, 0.9])
    res = compute_metrics(y_true, y_prob, threshold=0.5)

    assert res["accuracy"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert res["fpr"] == 0.0
    assert res["roc_auc"] == 1.0
    assert res["pr_auc"] == 1.0
