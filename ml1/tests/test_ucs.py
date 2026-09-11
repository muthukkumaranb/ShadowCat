import numpy as np
import pandas as pd
import joblib

from lstm.ucs import (
    UCSConfig,
    UCSRobustScaler,
    build_lstm_sequences,
    prepare_lstm_inputs,
    purge_and_embargo,
    apply_training_only_pca,
)


def make_windows(count=90):
    start = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for split, offset in (("train", 0), ("val", count), ("test", count * 2)):
        for index in range(offset, offset + count):
            rows.append(
                {
                    "window_start_utc": start + pd.Timedelta(minutes=index),
                    "split": split,
                    "future_attack_label": index % 2,
                    "flow_count": float(index),
                    "duration_sec_mean": float(index + 1),
                }
            )
    return pd.DataFrame(rows)


def test_ucs_sequences_are_boundary_safe_and_aligned():
    config = UCSConfig(lookback_windows=5, horizon_windows=2)
    windows = make_windows()
    purged = purge_and_embargo(windows, config=config)
    sequences = build_lstm_sequences(purged, config=config)

    assert all(item.X.ndim == 3 for item in sequences.values())
    assert all(item.X.shape[1:] == (5, 2) for item in sequences.values())
    assert all(len(item.y) == len(item.timestamps) for item in sequences.values())
    assert len(set(sequences["train"].timestamps) & set(sequences["val"].timestamps)) == 0
    assert len(set(sequences["val"].timestamps) & set(sequences["test"].timestamps)) == 0

    train_windows = purged.loc[purged["split"] == "train"].reset_index(drop=True)
    assert sequences["train"].timestamps[0] == train_windows.loc[4, "window_start_utc"]
    assert sequences["train"].y[0] == train_windows.loc[4, "future_attack_label"]


def test_features_exclude_current_and_future_labels():
    windows = make_windows()
    sequences, columns, _, _ = prepare_lstm_inputs(
        windows,
        config=UCSConfig(lookback_windows=5, horizon_windows=2),
    )

    assert "label_binary" not in columns
    assert "future_attack_label" not in columns
    assert all(item.X.shape[2] == len(columns) for item in sequences.values())


def test_robust_scaler_is_fit_on_training_values_only():
    windows = make_windows()
    columns = ["flow_count", "duration_sec_mean"]
    scaler = UCSRobustScaler(columns).fit(windows.loc[windows["split"] == "train"])
    transformed = scaler.transform_frame(windows)

    assert np.isfinite(transformed[columns].to_numpy()).all()
    assert scaler.get_params()["columns"] == columns
    assert scaler.get_params()["median"][0] < 100


def test_sequences_do_not_bridge_timestamp_gaps():
    windows = make_windows(count=40)
    gap_index = 12
    windows.loc[gap_index:, "window_start_utc"] += pd.Timedelta(minutes=3)
    config = UCSConfig(lookback_windows=5, horizon_windows=2)
    purged = purge_and_embargo(windows, config=config)
    sequences = build_lstm_sequences(purged, config=config)

    for sequence_set in sequences.values():
        if len(sequence_set.timestamps) > 1:
            assert all(
                (sequence_set.timestamps[index] - sequence_set.timestamps[index - 1])
                >= pd.Timedelta(minutes=1)
                for index in range(1, len(sequence_set.timestamps))
            )


def test_training_only_pca_preserves_sequences_labels_and_timestamps():
    windows = make_windows(count=60)
    config = UCSConfig(lookback_windows=5, horizon_windows=2)
    purged = purge_and_embargo(windows, config=config)
    features = ["flow_count", "duration_sec_mean"]
    baseline = build_lstm_sequences(purged, features=features, config=config)
    transformed, pca = apply_training_only_pca(purged, features, 2)
    reduced = build_lstm_sequences(transformed, features=["pca_0", "pca_1"], config=config)

    assert pca.get_metadata()["n_features_in"] == 2
    for split in ("train", "val", "test"):
        assert reduced[split].X.shape[0] == baseline[split].X.shape[0]
        assert reduced[split].y.tolist() == baseline[split].y.tolist()
        assert reduced[split].timestamps.equals(baseline[split].timestamps)


def test_pca_artifact_reload_preserves_transform(tmp_path):
    windows = make_windows(count=60)
    columns = ["flow_count", "duration_sec_mean"]
    purged = purge_and_embargo(windows, config=UCSConfig(lookback_windows=5, horizon_windows=2))
    pca_frame, pca = apply_training_only_pca(
        purged,
        columns,
        2,
    )
    path = tmp_path / "pca.joblib"
    joblib.dump(pca, path)
    reloaded = joblib.load(path)
    assert np.allclose(
        reloaded.transform(purged)[["pca_0", "pca_1"]].to_numpy(),
        pca_frame[["pca_0", "pca_1"]].to_numpy(),
    )
