import numpy as np
import pytest

from lstm.preprocessing import (
    StandardScaler,
    validate_sequences,
    fit_scaler,
    transform_data,
)


def create_dummy_data(
    samples=16,
    sequence_length=10,
    features=5,
):
    rng = np.random.default_rng(42)

    return rng.normal(
        size=(samples, sequence_length, features)
    ).astype(np.float32)


def test_validate_sequences():

    X = create_dummy_data()

    validate_sequences(X)


def test_validate_rejects_wrong_dimensions():

    X = np.random.randn(16, 10)

    with pytest.raises(ValueError):
        validate_sequences(X)


def test_validate_rejects_empty_samples():

    X = np.empty((0, 10, 5), dtype=np.float32)

    with pytest.raises(ValueError):
        validate_sequences(X)


def test_validate_rejects_empty_sequence():

    X = np.empty((16, 0, 5), dtype=np.float32)

    with pytest.raises(ValueError):
        validate_sequences(X)


def test_validate_rejects_empty_features():

    X = np.empty((16, 10, 0), dtype=np.float32)

    with pytest.raises(ValueError):
        validate_sequences(X)


def test_validate_rejects_nan():

    X = create_dummy_data()

    X[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        validate_sequences(X)


def test_validate_rejects_infinity():

    X = create_dummy_data()

    X[0, 0, 0] = np.inf

    with pytest.raises(ValueError):
        validate_sequences(X)


def test_scaler_fit():

    X = create_dummy_data()

    scaler = StandardScaler()

    scaler.fit(X)

    assert scaler.fitted is True
    assert scaler.mean_ is not None
    assert scaler.std_ is not None


def test_scaler_transform_shape():

    X = create_dummy_data()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    assert X_scaled.shape == X.shape


def test_scaler_output_mean():

    X = create_dummy_data(
        samples=100,
        sequence_length=20,
        features=5,
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    means = X_scaled.mean(axis=(0, 1))

    assert np.allclose(
        means,
        0.0,
        atol=1e-5,
    )


def test_scaler_output_std():

    X = create_dummy_data(
        samples=100,
        sequence_length=20,
        features=5,
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    stds = X_scaled.std(axis=(0, 1))

    assert np.allclose(
        stds,
        1.0,
        atol=1e-5,
    )


def test_scaler_transform_before_fit():

    X = create_dummy_data()

    scaler = StandardScaler()

    with pytest.raises(RuntimeError):
        scaler.transform(X)


def test_scaler_inverse_transform():

    X = create_dummy_data()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_restored = scaler.inverse_transform(
        X_scaled
    )

    assert np.allclose(
        X,
        X_restored,
        atol=1e-5,
    )


def test_scaler_different_feature_counts():

    X_train = create_dummy_data(
        features=5
    )

    X_test = create_dummy_data(
        features=5
    )

    scaler = StandardScaler()

    scaler.fit(X_train)

    X_scaled = scaler.transform(X_test)

    assert X_scaled.shape == X_test.shape


def test_scaler_rejects_wrong_feature_count():

    X_train = create_dummy_data(
        features=5
    )

    X_test = create_dummy_data(
        features=7
    )

    scaler = StandardScaler()

    scaler.fit(X_train)

    with pytest.raises(ValueError):
        scaler.transform(X_test)


def test_fit_scaler_helper():

    X = create_dummy_data()

    scaler = fit_scaler(X)

    assert isinstance(
        scaler,
        StandardScaler,
    )

    assert scaler.fitted is True


def test_transform_data_helper():

    X_train = create_dummy_data()

    X_test = create_dummy_data(
        samples=8
    )

    scaler = fit_scaler(X_train)

    X_transformed = transform_data(
        scaler,
        X_test,
    )

    assert X_transformed.shape == X_test.shape


def test_constant_feature():

    X = create_dummy_data()

    # Make the first feature constant.
    X[:, :, 0] = 5.0

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Constant feature should remain numerically stable.
    assert np.isfinite(X_scaled).all()

    assert np.allclose(
        X_scaled[:, :, 0],
        0.0,
    )