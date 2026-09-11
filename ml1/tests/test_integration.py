from pathlib import Path

import numpy as np
import torch

from lstm.data import split_data
from lstm.evaluator import evaluate
from lstm.model import LSTMClassifier
from lstm.preprocessing import fit_scaler, transform_data
from lstm.trainer import train
from lstm.utils import load_model, save_model, set_seed


def create_synthetic_dataset(
    samples: int = 300,
    sequence_length: int = 20,
    features: int = 6,
    seed: int = 42,
):
    """
    Create a generic binary sequence dataset.

    No domain-specific assumptions are used.
    """

    rng = np.random.default_rng(seed)

    X = rng.normal(
        size=(
            samples,
            sequence_length,
            features,
        )
    ).astype(np.float32)

    signal = X[:, :, 0].mean(axis=1)

    y = (
        signal > 0.0
    ).astype(np.float32)

    return X, y


def test_complete_lstm_pipeline(tmp_path):
    """
    Verify that the complete standalone LSTM pipeline works.
    """

    set_seed(42)

    # ---------------------------------------------------------
    # 1. Create generic data
    # ---------------------------------------------------------

    X, y = create_synthetic_dataset()

    assert X.ndim == 3
    assert y.ndim == 1

    # ---------------------------------------------------------
    # 2. Train / validation / test split
    # ---------------------------------------------------------

    (
        X_train,
        y_train,
        X_remaining,
        y_remaining,
    ) = split_data(
        X,
        y,
        validation_fraction=0.4,
    )

    (
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = split_data(
        X_remaining,
        y_remaining,
        validation_fraction=0.5,
    )

    assert len(X_train) == 180
    assert len(X_validation) == 60
    assert len(X_test) == 60

    # ---------------------------------------------------------
    # 3. Fit scaler on training data only
    # ---------------------------------------------------------

    scaler = fit_scaler(
        X_train
    )

    X_train_scaled = transform_data(
        scaler,
        X_train,
    )

    X_validation_scaled = transform_data(
        scaler,
        X_validation,
    )

    X_test_scaled = transform_data(
        scaler,
        X_test,
    )

    assert X_train_scaled.shape == X_train.shape
    assert X_validation_scaled.shape == X_validation.shape
    assert X_test_scaled.shape == X_test.shape

    # ---------------------------------------------------------
    # 4. Create model
    # ---------------------------------------------------------

    model = LSTMClassifier(
        input_size=6,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
    )

    # ---------------------------------------------------------
    # 5. Train
    # ---------------------------------------------------------

    result = train(
        model=model,
        X_train=X_train_scaled,
        y_train=y_train,
        X_validation=X_validation_scaled,
        y_validation=y_validation,
        epochs=3,
        batch_size=32,
        learning_rate=0.001,
        seed=42,
        device="cpu",
    )

    assert result["best_epoch"] > 0
    assert len(
        result["history"]["train_loss"]
    ) == 3
    assert len(
        result["history"]["validation_loss"]
    ) == 3

    # ---------------------------------------------------------
    # 6. Evaluate
    # ---------------------------------------------------------

    evaluation = evaluate(
        model=model,
        X_test=X_test_scaled,
        y_test=y_test,
        threshold=0.5,
        device=torch.device("cpu"),
    )

    assert evaluation["probabilities"].shape == (
        len(X_test),
    )

    assert evaluation["predictions"].shape == (
        len(X_test),
    )

    assert 0.0 <= evaluation["accuracy"] <= 1.0
    assert 0.0 <= evaluation["precision"] <= 1.0
    assert 0.0 <= evaluation["recall"] <= 1.0
    assert 0.0 <= evaluation["f1"] <= 1.0

    # ---------------------------------------------------------
    # 7. Save model
    # ---------------------------------------------------------

    model_path = (
        tmp_path
        / "artifacts"
        / "models"
        / "integration_model.pt"
    )

    saved_path = save_model(
        model=model,
        path=model_path,
        metadata={
            "input_size": 6,
            "sequence_length": 20,
        },
    )

    assert saved_path.exists()

    # ---------------------------------------------------------
    # 8. Load model
    # ---------------------------------------------------------

    loaded_model = LSTMClassifier(
        input_size=6,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
    )

    loaded_model, metadata = load_model(
        loaded_model,
        model_path,
        device="cpu",
    )

    assert metadata["input_size"] == 6
    assert metadata["sequence_length"] == 20

    # ---------------------------------------------------------
    # 9. Compare original and loaded predictions
    # ---------------------------------------------------------

    model.eval()
    loaded_model.eval()

    X_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
    )

    with torch.no_grad():

        original_predictions = model(
            X_tensor
        )

        loaded_predictions = loaded_model(
            X_tensor
        )

    assert torch.allclose(
        original_predictions,
        loaded_predictions,
        atol=1e-6,
    )


def test_pipeline_supports_different_feature_count(
    tmp_path,
):
    """
    Verify that the entire pipeline works with a different
    feature count without changing the implementation.
    """

    set_seed(123)

    X, y = create_synthetic_dataset(
        samples=150,
        sequence_length=15,
        features=12,
        seed=123,
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = split_data(
        X,
        y,
        validation_fraction=0.2,
    )

    scaler = fit_scaler(
        X_train
    )

    X_train = transform_data(
        scaler,
        X_train,
    )

    X_test = transform_data(
        scaler,
        X_test,
    )

    model = LSTMClassifier(
        input_size=12,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
    )

    result = train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_test,
        y_validation=y_test,
        epochs=1,
        batch_size=16,
        learning_rate=0.001,
        seed=123,
        device="cpu",
    )

    assert result["best_epoch"] == 1

    results = evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=0.5,
        device=torch.device("cpu"),
    )

    assert results["probabilities"].shape == (
        len(X_test),
    )