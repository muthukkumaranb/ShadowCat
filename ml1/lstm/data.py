from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class LSTMDataError(ValueError):
    """Raised when LSTM input data is invalid."""
    pass


def validate_data(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Validate generic sequence data for LSTM training.

    Expected:
        X: (samples, sequence_length, features)
        y: (samples,)

    X must contain finite numerical values.
    y must contain one label per sample.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 3:
        raise LSTMDataError(
            "X must be a 3-dimensional array with shape "
            "(samples, sequence_length, features)."
        )

    if y.ndim != 1:
        raise LSTMDataError(
            "y must be a 1-dimensional array with shape (samples,)."
        )

    if X.shape[0] != y.shape[0]:
        raise LSTMDataError(
            f"X and y contain different numbers of samples: "
            f"X={X.shape[0]}, y={y.shape[0]}."
        )

    if X.shape[0] == 0:
        raise LSTMDataError("X cannot contain zero samples.")

    if X.shape[1] == 0:
        raise LSTMDataError(
            "Sequence length must be greater than zero."
        )

    if X.shape[2] == 0:
        raise LSTMDataError(
            "Number of features must be greater than zero."
        )

    if not np.issubdtype(X.dtype, np.number):
        raise LSTMDataError(
            "X must contain numerical values."
        )

    if not np.isfinite(X).all():
        raise LSTMDataError(
            "X contains NaN or infinite values."
        )

    if not np.issubdtype(y.dtype, np.number):
        raise LSTMDataError(
            "y must contain numerical labels."
        )

    if not np.isfinite(y).all():
        raise LSTMDataError(
            "y contains NaN or infinite values."
        )


def to_tensors(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert validated NumPy arrays to PyTorch tensors.

    X -> float32
    y -> float32

    Float labels are intentional because the trainer uses
    binary classification loss.
    """

    validate_data(X, y)

    X_tensor = torch.as_tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.as_tensor(
        y,
        dtype=torch.float32,
    )

    return X_tensor, y_tensor


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader from generic sequence data.

    Parameters
    ----------
    X:
        Array with shape (samples, sequence_length, features).

    y:
        Array with shape (samples,).

    batch_size:
        Number of samples per batch.

    shuffle:
        Whether to shuffle samples.

    Returns
    -------
    DataLoader
    """

    if not isinstance(batch_size, int):
        raise LSTMDataError(
            "batch_size must be an integer."
        )

    if batch_size <= 0:
        raise LSTMDataError(
            "batch_size must be greater than zero."
        )

    X_tensor, y_tensor = to_tensors(X, y)

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    validation_fraction: float = 0.2,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Split sequence data into training and validation sets.

    The split is performed in sample order.

    This function does not assume anything about the dataset,
    labels, timestamps, or domain.

    Returns
    -------
    X_train, y_train, X_validation, y_validation
    """

    validate_data(X, y)

    if not isinstance(
        validation_fraction,
        (float, int),
    ):
        raise LSTMDataError(
            "validation_fraction must be a number."
        )

    validation_fraction = float(
        validation_fraction
    )

    if not 0.0 < validation_fraction < 1.0:
        raise LSTMDataError(
            "validation_fraction must be between 0 and 1."
        )

    n_samples = X.shape[0]

    validation_size = max(
        1,
        int(
            round(
                n_samples * validation_fraction
            )
        ),
    )

    train_size = n_samples - validation_size

    if train_size <= 0:
        raise LSTMDataError(
            "Not enough samples for a training split."
        )

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_validation = X[train_size:]
    y_validation = y[train_size:]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
    )