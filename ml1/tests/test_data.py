import numpy as np
import pytest
import torch

from lstm.data import (
    LSTMDataError,
    validate_data,
    to_tensors,
    create_dataloader,
    split_data,
)


def create_dummy_data(
    samples=32,
    sequence_length=10,
    features=5,
):
    X = np.random.randn(
        samples,
        sequence_length,
        features,
    ).astype(np.float32)

    y = np.random.randint(
        0,
        2,
        size=samples,
    ).astype(np.float32)

    return X, y


def test_validate_data():

    X, y = create_dummy_data()

    validate_data(X, y)


def test_validate_data_rejects_wrong_X_dimensions():

    X = np.random.randn(
        32,
        5,
    ).astype(np.float32)

    y = np.random.randint(
        0,
        2,
        size=32,
    ).astype(np.float32)

    with pytest.raises(LSTMDataError):
        validate_data(X, y)


def test_validate_data_rejects_wrong_y_dimensions():

    X, y = create_dummy_data()

    y = y.reshape(-1, 1)

    with pytest.raises(LSTMDataError):
        validate_data(X, y)


def test_validate_data_rejects_mismatched_samples():

    X, y = create_dummy_data()

    y = y[:-1]

    with pytest.raises(LSTMDataError):
        validate_data(X, y)


def test_validate_data_rejects_nan():

    X, y = create_dummy_data()

    X[0, 0, 0] = np.nan

    with pytest.raises(LSTMDataError):
        validate_data(X, y)


def test_validate_data_rejects_infinity():

    X, y = create_dummy_data()

    X[0, 0, 0] = np.inf

    with pytest.raises(LSTMDataError):
        validate_data(X, y)


def test_to_tensors():

    X, y = create_dummy_data()

    X_tensor, y_tensor = to_tensors(
        X,
        y,
    )

    assert isinstance(
        X_tensor,
        torch.Tensor,
    )

    assert isinstance(
        y_tensor,
        torch.Tensor,
    )

    assert X_tensor.dtype == torch.float32
    assert y_tensor.dtype == torch.float32

    assert X_tensor.shape == X.shape
    assert y_tensor.shape == y.shape


def test_dataloader_shape():

    X, y = create_dummy_data(
        samples=32,
        sequence_length=10,
        features=5,
    )

    loader = create_dataloader(
        X,
        y,
        batch_size=8,
        shuffle=False,
    )

    batch_X, batch_y = next(
        iter(loader)
    )

    assert batch_X.shape == (
        8,
        10,
        5,
    )

    assert batch_y.shape == (8,)


def test_dataloader_contains_all_samples():

    X, y = create_dummy_data(
        samples=32,
    )

    loader = create_dataloader(
        X,
        y,
        batch_size=8,
        shuffle=False,
    )

    total_samples = 0

    for batch_X, batch_y in loader:
        total_samples += len(batch_X)

    assert total_samples == 32


def test_invalid_batch_size():

    X, y = create_dummy_data()

    with pytest.raises(LSTMDataError):
        create_dataloader(
            X,
            y,
            batch_size=0,
        )


def test_split_data():

    X, y = create_dummy_data(
        samples=100,
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = split_data(
        X,
        y,
        validation_fraction=0.2,
    )

    assert X_train.shape[0] == 80
    assert y_train.shape[0] == 80

    assert X_validation.shape[0] == 20
    assert y_validation.shape[0] == 20


def test_split_preserves_sequence_shape():

    X, y = create_dummy_data(
        samples=100,
        sequence_length=20,
        features=7,
    )

    (
        X_train,
        _,
        X_validation,
        _,
    ) = split_data(
        X,
        y,
        validation_fraction=0.2,
    )

    assert X_train.shape[1:] == (
        20,
        7,
    )

    assert X_validation.shape[1:] == (
        20,
        7,
    )


def test_split_preserves_order():

    X = np.arange(
        100,
        dtype=np.float32,
    ).reshape(
        20,
        5,
        1,
    )

    y = np.arange(
        20,
        dtype=np.float32,
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = split_data(
        X,
        y,
        validation_fraction=0.2,
    )

    assert y_train[-1] == 15
    assert y_validation[0] == 16

    assert X_train[-1, 0, 0] == X[15, 0, 0]
    assert X_validation[0, 0, 0] == X[16, 0, 0]