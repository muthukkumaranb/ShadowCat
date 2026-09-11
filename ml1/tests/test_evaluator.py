import numpy as np
import pytest
import torch

from lstm.model import LSTMClassifier
from lstm.evaluator import (
    evaluate,
    validate_evaluation_data,
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


def create_model(features=5):
    return LSTMClassifier(
        input_size=features,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
    )


def test_validate_evaluation_data():

    X, y = create_dummy_data()

    validate_evaluation_data(
        X,
        y,
    )


def test_invalid_X_dimensions():

    X = np.random.randn(
        32,
        10,
    ).astype(np.float32)

    y = np.random.randint(
        0,
        2,
        size=32,
    )

    with pytest.raises(ValueError):

        validate_evaluation_data(
            X,
            y,
        )


def test_invalid_y_dimensions():

    X, _ = create_dummy_data()

    y = np.random.randint(
        0,
        2,
        size=(32, 1),
    )

    with pytest.raises(ValueError):

        validate_evaluation_data(
            X,
            y,
        )


def test_mismatched_samples():

    X, _ = create_dummy_data(
        samples=32,
    )

    y = np.random.randint(
        0,
        2,
        size=16,
    )

    with pytest.raises(ValueError):

        validate_evaluation_data(
            X,
            y,
        )


def test_nan_rejected():

    X, y = create_dummy_data()

    X[0, 0, 0] = np.nan

    with pytest.raises(ValueError):

        validate_evaluation_data(
            X,
            y,
        )


def test_infinity_rejected():

    X, y = create_dummy_data()

    X[0, 0, 0] = np.inf

    with pytest.raises(ValueError):

        validate_evaluation_data(
            X,
            y,
        )


def test_invalid_labels():

    X, y = create_dummy_data()

    y[0] = 2

    with pytest.raises(ValueError):

        validate_evaluation_data(
            X,
            y,
        )


def test_evaluation_output():

    X, y = create_dummy_data()

    model = create_model()

    results = evaluate(
        model,
        X,
        y,
    )

    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    assert "roc_auc" in results
    assert "pr_auc" in results


def test_probability_shape():

    X, y = create_dummy_data(
        samples=24,
    )

    model = create_model()

    results = evaluate(
        model,
        X,
        y,
    )

    assert results["probabilities"].shape == (24,)


def test_prediction_shape():

    X, y = create_dummy_data(
        samples=24,
    )

    model = create_model()

    results = evaluate(
        model,
        X,
        y,
    )

    assert results["predictions"].shape == (24,)


def test_probability_range():

    X, y = create_dummy_data()

    model = create_model()

    results = evaluate(
        model,
        X,
        y,
    )

    assert np.all(
        results["probabilities"] >= 0
    )

    assert np.all(
        results["probabilities"] <= 1
    )


def test_prediction_values():

    X, y = create_dummy_data()

    model = create_model()

    results = evaluate(
        model,
        X,
        y,
    )

    assert np.all(
        np.isin(
            results["predictions"],
            [0, 1],
        )
    )


def test_invalid_threshold():

    X, y = create_dummy_data()

    model = create_model()

    with pytest.raises(ValueError):

        evaluate(
            model,
            X,
            y,
            threshold=1.5,
        )


def test_different_feature_counts():

    X, y = create_dummy_data(
        features=12,
    )

    model = create_model(
        features=12,
    )

    results = evaluate(
        model,
        X,
        y,
    )

    assert results["probabilities"].shape == (
        X.shape[0],
    )


def test_single_class_evaluation():

    X, _ = create_dummy_data()

    y = np.ones(
        X.shape[0],
        dtype=np.float32,
    )

    model = create_model()

    results = evaluate(
        model,
        X,
        y,
    )

    assert results["roc_auc"] is None
    assert results["pr_auc"] is None