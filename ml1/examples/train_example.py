from __future__ import annotations

from pathlib import Path

import numpy as np

from lstm.data import split_data
from lstm.evaluator import evaluate, print_evaluation_results
from lstm.model import LSTMClassifier
from lstm.preprocessing import fit_scaler, transform_data
from lstm.trainer import train
from lstm.utils import (
    count_parameters,
    get_device,
    model_summary,
    save_model,
    set_seed,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEED = 42

NUM_SAMPLES = 1000
SEQUENCE_LENGTH = 30
NUM_FEATURES = 8

VALIDATION_FRACTION = 0.2
TEST_FRACTION = 0.2

HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.2

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "models"
    / "example_model.pt"
)


# ---------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------

def create_synthetic_dataset(
    samples: int,
    sequence_length: int,
    features: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a generic binary sequence dataset.

    This function has no dataset-specific assumptions.

    X shape:
        (samples, sequence_length, features)

    y shape:
        (samples,)
    """

    rng = np.random.default_rng(seed)

    X = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(
            samples,
            sequence_length,
            features,
        ),
    ).astype(np.float32)

    # Construct a simple synthetic signal.
    #
    # The model can learn that sequences with a stronger
    # positive signal are more likely to belong to class 1.
    sequence_signal = X[:, :, 0].mean(axis=1)

    probability = 1.0 / (
        1.0 + np.exp(-sequence_signal)
    )

    y = (
        probability >= 0.5
    ).astype(np.float32)

    return X, y


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main() -> None:

    print("=" * 60)
    print("STANDALONE LSTM EXAMPLE")
    print("=" * 60)

    set_seed(SEED)

    # -----------------------------------------------------
    # 1. Create generic dataset
    # -----------------------------------------------------

    X, y = create_synthetic_dataset(
        samples=NUM_SAMPLES,
        sequence_length=SEQUENCE_LENGTH,
        features=NUM_FEATURES,
        seed=SEED,
    )

    print()
    print("DATASET")
    print("-" * 60)
    print(f"Samples:          {X.shape[0]}")
    print(f"Sequence length:  {X.shape[1]}")
    print(f"Features:         {X.shape[2]}")
    print(f"Labels:           {y.shape}")

    # -----------------------------------------------------
    # 2. Split into train / validation+test
    # -----------------------------------------------------

    split_fraction = (
        VALIDATION_FRACTION
        + TEST_FRACTION
    )

    (
        X_train,
        y_train,
        X_remaining,
        y_remaining,
    ) = split_data(
        X,
        y,
        validation_fraction=split_fraction,
    )

    # Split remaining data into validation and test.
    test_fraction_of_remaining = (
        TEST_FRACTION
        / split_fraction
    )

    (
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = split_data(
        X_remaining,
        y_remaining,
        validation_fraction=test_fraction_of_remaining,
    )

    print()
    print("SPLIT")
    print("-" * 60)
    print(f"Train:       {X_train.shape}")
    print(f"Validation:  {X_validation.shape}")
    print(f"Test:        {X_test.shape}")

    # -----------------------------------------------------
    # 3. Fit preprocessing ONLY on training data
    # -----------------------------------------------------

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

    print()
    print("PREPROCESSING")
    print("-" * 60)
    print("Scaler fitted using training data only.")
    print(
        f"Training mean: "
        f"{X_train_scaled.mean():.6f}"
    )
    print(
        f"Training std:  "
        f"{X_train_scaled.std():.6f}"
    )

    # -----------------------------------------------------
    # 4. Build LSTM
    # -----------------------------------------------------

    model = LSTMClassifier(
        input_size=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    print()
    print("MODEL")
    print("-" * 60)

    print(
        f"Parameters: "
        f"{count_parameters(model):,}"
    )

    print(
        model_summary(model)
    )

    # -----------------------------------------------------
    # 5. Train
    # -----------------------------------------------------

    print()
    print("TRAINING")
    print("-" * 60)

    device = get_device()

    result = train(
        model=model,
        X_train=X_train_scaled,
        y_train=y_train,
        X_validation=X_validation_scaled,
        y_validation=y_validation,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        device=str(device),
    )

    print(
        f"Best epoch: "
        f"{result['best_epoch']}"
    )

    print(
        f"Best validation loss: "
        f"{result['best_validation_loss']:.6f}"
    )

    # -----------------------------------------------------
    # 6. Evaluate
    # -----------------------------------------------------

    print()
    print("EVALUATION")
    print("-" * 60)

    evaluation = evaluate(
        model=model,
        X_test=X_test_scaled,
        y_test=y_test,
        threshold=0.5,
        device=device,
    )

    print_evaluation_results(
        evaluation
    )

    # -----------------------------------------------------
    # 7. Save model
    # -----------------------------------------------------

    save_model(
        model=model,
        path=MODEL_PATH,
        metadata={
            "input_size": NUM_FEATURES,
            "sequence_length": SEQUENCE_LENGTH,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "seed": SEED,
        },
    )

    print()
    print(
        f"Model saved to:\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()