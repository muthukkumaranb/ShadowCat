import numpy as np
import pandas as pd
import pytest

import torch

from examples.train_probabilistic import write_rollout_plot
from lstm.model import LSTMGaussianWorldModel, LSTMWorldModel
from lstm.probabilistic import (
    ablate_history,
    deviation_scores,
    feature_groups,
    gaussian_nll,
    gaussian_nll_per_sample,
    interval_coverage,
    prediction_interval,
    sigma_diagnostics,
    summarize_nll,
    top_nll_outliers,
)
from lstm.ucs import (
    UCSConfig,
    build_loeo_episode_folds,
    build_next_state_sequences,
    fit_lagged_linear_baseline,
    fit_pca_lagged_linear_baseline,
    apply_training_only_pca,
    make_next_state_targets,
    pca_rollout_predictions,
    persistence_baseline,
    purge_and_embargo,
    rollout_targets,
    rollout_predictions,
)


@pytest.fixture
def synthetic_windows():
    rows = []
    start = pd.Timestamp("2026-01-01", tz="UTC")
    for split, offset in (("train", 0), ("val", 40), ("test", 80)):
        for index in range(offset, offset + 40):
            rows.append(
                {
                    "window_start_utc": start + pd.Timedelta(minutes=index),
                    "split": split,
                    "future_attack_label": (index % 3) > 1,
                    "label_attack_type": "Botnet",
                    "feature_a": float(index),
                    "feature_b": float(index * 2),
                }
            )
    return pd.DataFrame(rows)


def test_world_model_output_shapes():
    model = LSTMWorldModel(input_size=7, hidden_size=8, state_dim=7, dropout=0.2)
    x = np.random.randn(4, 30, 7).astype(np.float32)
    pred = model(x)
    assert pred.shape == (4, 7)


def test_gaussian_world_model_outputs_positive_std():
    model = LSTMGaussianWorldModel(input_size=4, hidden_size=8, state_dim=4, dropout=0.0)
    mean, std = model(np.random.randn(3, 5, 4).astype(np.float32))
    assert mean.shape == (3, 4)
    assert std.shape == (3, 4)
    assert torch.all(std > 0)


def test_gaussian_nll_intervals_and_coverage_are_finite():
    mean = torch.zeros(2, 3, requires_grad=True)
    std = torch.ones(2, 3)
    target = torch.zeros(2, 3)
    loss = gaussian_nll(mean, std, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert mean.grad is not None
    lower, upper = prediction_interval(np.zeros((2, 3)), np.ones((2, 3)), 0.95)
    assert np.all(lower < upper)
    assert interval_coverage(np.zeros((2, 3)), np.ones((2, 3)), np.zeros((2, 3)), 0.95) == 1.0


def test_nll_distribution_and_sigma_diagnostics_are_reproducible():
    mean = np.zeros((3, 2), dtype=np.float32)
    std = np.ones((3, 2), dtype=np.float32)
    target = np.array([[0, 0], [1, 0], [4, 0]], dtype=np.float32)
    nll = gaussian_nll_per_sample(mean, std, target)
    summary = summarize_nll(nll)
    assert summary["median"] == np.median(nll)
    assert summary["p90"] == np.percentile(nll, 90)
    assert sigma_diagnostics(std)["minimum"] == 1.0
    outliers = top_nll_outliers(nll, mean, std, target, ["a", "b", "c"], top_n=2)
    assert [row["index"] for row in outliers] == [2, 1]


def test_feature_group_ablation_and_deviation():
    groups = feature_groups(["byte_count_fwd_mean", "fin_flag_cnt_mean", "duration_sec_mean"])
    history = np.ones((2, 4, 3), dtype=np.float32)
    masked = ablate_history(history, groups["traffic_volume"])
    assert np.all(masked[:, :, 0] == 0)
    assert np.all(masked[:, :, 1:] == 1)
    assert np.allclose(deviation_scores(np.zeros((2, 3)), np.ones((2, 3))), 1.0)


def test_build_next_state_sequences_targets_next_window(synthetic_windows):
    config = UCSConfig(lookback_windows=5, horizon_windows=2)
    sequences = build_next_state_sequences(
        synthetic_windows,
        config=config,
        features=["feature_a", "feature_b"],
    )

    train = sequences["train"]
    assert train.X.shape[1:] == (5, 2)
    assert train.y.shape[1:] == (2,)
    assert train.y[0].tolist() == [float(5), float(10)]


def test_persistence_baseline_matches_last_observed_state(synthetic_windows):
    config = UCSConfig(lookback_windows=5, horizon_windows=2)
    sequences = build_next_state_sequences(
        synthetic_windows,
        config=config,
        features=["feature_a", "feature_b"],
    )
    preds = persistence_baseline(sequences["test"].X)
    assert preds.shape == sequences["test"].y.shape
    assert np.allclose(preds, sequences["test"].X[:, -1, :])


def test_rollout_uses_recursive_predictions_not_future_data(synthetic_windows):
    config = UCSConfig(lookback_windows=5, horizon_windows=2)
    sequences = build_next_state_sequences(
        synthetic_windows,
        config=config,
        features=["feature_a", "feature_b"],
    )

    def predictor(history):
        return history[:, -1, :] + 1.0

    rolled = rollout_predictions(sequences["test"].X[:2], predictor, k=3)
    assert rolled.shape == (2, 3, 2)
    assert np.allclose(rolled[:, 0], sequences["test"].X[:2, -1, :] + 1.0)
    assert np.allclose(rolled[:, 1], rolled[:, 0] + 1.0)
    assert np.allclose(rolled[:, 2], rolled[:, 1] + 1.0)


def test_lagged_linear_baseline_uses_last_observed_state(synthetic_windows):
    config = UCSConfig(lookback_windows=5, horizon_windows=2)
    sequences = build_next_state_sequences(
        synthetic_windows,
        config=config,
        features=["feature_a", "feature_b"],
    )
    model = fit_lagged_linear_baseline(sequences["train"])
    predictions = model.predict(sequences["test"].X[:, -1, :])
    assert predictions.shape == sequences["test"].y.shape


def test_pca_lagged_linear_rollout_is_finite_for_k_1_2_3(synthetic_windows):
    config = UCSConfig(lookback_windows=5, horizon_windows=2)
    purged = purge_and_embargo(synthetic_windows, config=config)
    raw_sequences = build_next_state_sequences(purged, config=config, features=["feature_a", "feature_b"])
    transformed, pca = apply_training_only_pca(purged, ["feature_a", "feature_b"], 2)
    reduced = build_next_state_sequences(transformed, config=config, features=["pca_0", "pca_1"])
    model = fit_pca_lagged_linear_baseline(reduced["train"], pca)
    for horizon in (1, 2, 3):
        rolled = pca_rollout_predictions(raw_sequences["test"].X, model, pca, k=horizon)
        assert rolled.shape == (len(raw_sequences["test"].X), horizon, 2)
        assert np.isfinite(rolled).all()


def test_rollout_plot_generation(tmp_path):
    rows = [
        {"k": 1, "model": "persistence", "rmse": 1.0},
        {"k": 1, "model": "lagged_linear_pca", "rmse": 2.0},
        {"k": 1, "model": "lstm_gaussian", "rmse": 0.5},
    ]
    path = write_rollout_plot(tmp_path, rows)
    assert path.exists()
    assert path.stat().st_size > 0


def test_rollout_targets_are_contiguous_and_future_only(synthetic_windows):
    config = UCSConfig(lookback_windows=5, horizon_windows=3)
    histories, targets, timestamps = rollout_targets(
        synthetic_windows,
        config=config,
        features=["feature_a", "feature_b"],
        k=3,
        split="test",
    )
    assert histories.shape[1:] == (5, 2)
    assert targets.shape[1:] == (3, 2)
    assert np.allclose(targets[:, 0], histories[:, -1, :] + np.array([1.0, 2.0]))
    assert len(timestamps) == len(histories)


def test_forbidden_time_fields_are_not_model_features(synthetic_windows):
    columns = ["feature_a", "feature_b"]
    assert "window_start_utc" not in columns
    assert "window_end_utc" not in columns
    assert "source_day" not in columns
    assert "future_attack_label" not in columns


def test_loeo_folds_exclude_held_out_episode_from_training(synthetic_windows):
    frame = synthetic_windows.copy()
    frame["episode_id"] = ["episode_a" if index < 60 else "episode_b" for index in range(len(frame))]
    folds = build_loeo_episode_folds(frame)
    assert len(folds) == 2
    for fold in folds:
        train_episodes = set(frame.loc[fold["train_indices"], "episode_id"])
        assert fold["held_out_episode"] not in train_episodes


def test_loeo_rejects_source_day_as_episode_identifier(synthetic_windows):
    frame = synthetic_windows.copy()
    frame["source_day"] = "2018-02-14"
    with pytest.raises(ValueError, match="source_day"):
        build_loeo_episode_folds(frame, episode_col="source_day")


def test_loeo_rejects_forecast_episode_as_grouping_identifier(synthetic_windows):
    frame = synthetic_windows.copy()
    frame["forecast_episode_id"] = "forecast"
    with pytest.raises(ValueError, match="episode_id"):
        build_loeo_episode_folds(frame, episode_col="forecast_episode_id")


def test_loeo_filters_to_named_attack_types(synthetic_windows):
    frame = synthetic_windows.copy()
    frame["episode_id"] = ["episode_a" if index < 60 else "episode_b" for index in range(len(frame))]
    frame["label_attack_type"] = ["Botnet" if index < 60 else "DDOS-HOIC" for index in range(len(frame))]
    folds = build_loeo_episode_folds(frame)
    assert [fold["held_out_episode"] for fold in folds] == ["episode_a"]


def test_world_model_training_accepts_multitarget_regression():
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(12, 5, 4)).astype(np.float32)
    y_train = rng.normal(size=(12, 4)).astype(np.float32)
    X_val = rng.normal(size=(6, 5, 4)).astype(np.float32)
    y_val = rng.normal(size=(6, 4)).astype(np.float32)

    model = LSTMWorldModel(input_size=4, hidden_size=8, state_dim=4, dropout=0.0)
    from lstm.trainer import train

    result = train(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=2,
        batch_size=4,
        learning_rate=0.001,
    )

    assert result["best_epoch"] > 0
    assert len(result["history"]["train_loss"]) == 2
    assert len(result["history"]["validation_loss"]) == 2
