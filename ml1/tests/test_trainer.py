import numpy as np
import torch

from lstm.model import LSTMClassifier
from lstm.trainer import (
    validate_data,
    create_dataloader,
    train,
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


def test_dataloader_shape():

    X, y = create_dummy_data()

    loader = create_dataloader(
        X,
        y,
        batch_size=8,
    )

    X_batch, y_batch = next(
        iter(loader)
    )

    assert X_batch.shape == (
        8,
        10,
        5,
    )

    assert y_batch.shape == (8,)


def test_training_runs():

    X_train, y_train = create_dummy_data(
        samples=32,
        sequence_length=10,
        features=5,
    )

    X_validation, y_validation = create_dummy_data(
        samples=16,
        sequence_length=10,
        features=5,
    )

    model = LSTMClassifier(
        input_size=5,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
    )

    result = train(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation,
        epochs=2,
        batch_size=8,
        learning_rate=0.001,
    )

    assert result["best_epoch"] > 0

    assert len(
        result["history"]["train_loss"]
    ) == 2

    assert len(
        result["history"]["validation_loss"]
    ) == 2


def test_different_feature_counts():

    X_train, y_train = create_dummy_data(
        samples=20,
        sequence_length=8,
        features=12,
    )

    X_validation, y_validation = create_dummy_data(
        samples=10,
        sequence_length=8,
        features=12,
    )

    model = LSTMClassifier(
        input_size=12,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
    )

    result = train(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation,
        epochs=1,
        batch_size=5,
    )

    assert result["best_epoch"] == 1


def test_invalid_dimensions():

    X = np.random.randn(
        20,
        10,
    ).astype(np.float32)

    y = np.random.randint(
        0,
        2,
        size=20,
    ).astype(np.float32)

    try:
        validate_data(X, y)
        assert False
    except ValueError:
        assert True


def test_mismatched_samples():

    X = np.random.randn(
        20,
        10,
        5,
    ).astype(np.float32)

    y = np.random.randint(
        0,
        2,
        size=19,
    ).astype(np.float32)

    try:
        validate_data(X, y)
        assert False
    except ValueError:
        assert True