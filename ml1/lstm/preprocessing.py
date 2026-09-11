from __future__ import annotations

import numpy as np


class StandardScaler:
    """
    Standardizes features using:

        z = (x - mean) / std

    Statistics are learned only from training data.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.fitted: bool = False

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                "X must have shape (samples, sequence_length, features)."
            )

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        if not np.isfinite(X).all():
            raise ValueError(
                "X contains NaN or infinite values."
            )

        # Calculate statistics across samples and timesteps.
        self.mean_ = np.mean(
            X,
            axis=(0, 1),
            keepdims=True,
        )

        self.std_ = np.std(
            X,
            axis=(0, 1),
            keepdims=True,
        )

        # Prevent division by zero for constant features.
        self.std_[self.std_ < 1e-8] = 1.0

        self.fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError(
                "Scaler has not been fitted."
            )

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                "X must have shape (samples, sequence_length, features)."
            )

        if X.shape[2] != self.mean_.shape[2]:
            raise ValueError(
                "Feature count does not match fitted scaler."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "X contains NaN or infinite values."
            )

        return (
            X - self.mean_
        ) / self.std_

    def fit_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError(
                "Scaler has not been fitted."
            )

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                "X must have shape (samples, sequence_length, features)."
            )

        if X.shape[2] != self.mean_.shape[2]:
            raise ValueError(
                "Feature count does not match fitted scaler."
            )

        return (
            X * self.std_
        ) + self.mean_


def validate_sequences(
    X: np.ndarray,
) -> None:
    """
    Validate generic LSTM input sequences.

    Expected shape:
        (samples, sequence_length, features)
    """

    X = np.asarray(X)

    if X.ndim != 3:
        raise ValueError(
            "X must have shape (samples, sequence_length, features)."
        )

    if X.shape[0] == 0:
        raise ValueError(
            "X must contain at least one sample."
        )

    if X.shape[1] == 0:
        raise ValueError(
            "Sequence length must be greater than zero."
        )

    if X.shape[2] == 0:
        raise ValueError(
            "Feature count must be greater than zero."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "X contains NaN or infinite values."
        )


def fit_scaler(
    X_train: np.ndarray,
) -> StandardScaler:
    """
    Fit a StandardScaler using training data only.
    """

    validate_sequences(X_train)

    scaler = StandardScaler()
    scaler.fit(X_train)

    return scaler


def transform_data(
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    """
    Transform data using an already-fitted scaler.
    """

    validate_sequences(X)

    return scaler.transform(X)