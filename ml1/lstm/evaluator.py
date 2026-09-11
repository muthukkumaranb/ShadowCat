from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def validate_evaluation_data(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Validate input data before evaluation.

    Parameters
    ----------
    X : np.ndarray
        Sequence data with shape:
        (samples, sequence_length, features)

    y : np.ndarray
        Binary labels with shape:
        (samples,)
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a NumPy array.")

    if X.ndim != 3:
        raise ValueError(
            "X must be a 3-dimensional array with shape "
            "(samples, sequence_length, features)."
        )

    if y.ndim != 1:
        raise ValueError(
            "y must be a 1-dimensional array with shape (samples,)."
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    if X.shape[0] == 0:
        raise ValueError("X and y cannot be empty.")

    if X.shape[1] == 0:
        raise ValueError("Sequence length cannot be zero.")

    if X.shape[2] == 0:
        raise ValueError("Number of features cannot be zero.")

    if not np.isfinite(X).all():
        raise ValueError(
            "X contains NaN or infinite values."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "y contains NaN or infinite values."
        )

    unique_labels = np.unique(y)

    if not np.isin(unique_labels, [0, 1]).all():
        raise ValueError(
            "y must contain only binary labels 0 and 1."
        )


def _extract_probabilities(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Generate prediction probabilities from the model.

    The model is expected to return probabilities in [0, 1].
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    probabilities = []

    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            batch_size,
        ):
            end = start + batch_size

            batch = X_tensor[start:end].to(device)

            output = model(batch)

            output = output.detach().cpu().numpy()

            output = np.asarray(output).reshape(-1)

            probabilities.append(output)

    probabilities = np.concatenate(
        probabilities,
        axis=0,
    )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Model produced NaN or infinite probabilities."
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(
            "Model output must contain probabilities "
            "between 0 and 1."
        )

    return probabilities


def evaluate(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Evaluate a trained LSTM classification model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained LSTM classifier.

    X_test : np.ndarray
        Test sequences with shape:
        (samples, sequence_length, features).

    y_test : np.ndarray
        Binary test labels with shape:
        (samples,).

    threshold : float
        Classification threshold used to convert
        probabilities into predictions.

    batch_size : int
        Number of samples processed at once.

    device : torch.device, optional
        Device used for inference.

    Returns
    -------
    Dict
        Evaluation metrics and prediction information.
    """

    validate_evaluation_data(
        X_test,
        y_test,
    )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    probabilities = _extract_probabilities(
        model=model,
        X=X_test,
        batch_size=batch_size,
        device=device,
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    y_true = y_test.astype(np.int64)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    unique_labels = np.unique(y_true)

    if len(unique_labels) == 2:
        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_true,
            probabilities,
        )

    else:
        roc_auc = None

        # PR-AUC is not statistically meaningful when
        # the evaluation set contains only one class.
        pr_auc = None

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": (
            None
            if roc_auc is None
            else float(roc_auc)
        ),
        "pr_auc": (
            None
            if pr_auc is None
            else float(pr_auc)
        ),
        "confusion_matrix": cm,
        "probabilities": probabilities,
        "predictions": predictions,
        "threshold": float(threshold),
    }


def print_evaluation_results(
    results: Dict,
) -> None:
    """
    Print evaluation results in a readable format.
    """

    print("=" * 60)
    print("LSTM EVALUATION")
    print("=" * 60)

    print()
    print("CLASSIFICATION RESULTS")
    print("-" * 60)

    print(
        f"Accuracy:         "
        f"{results['accuracy']:.6f}"
    )

    print(
        f"Precision:        "
        f"{results['precision']:.6f}"
    )

    print(
        f"Recall:           "
        f"{results['recall']:.6f}"
    )

    print(
        f"F1 Score:         "
        f"{results['f1']:.6f}"
    )

    if results["roc_auc"] is None:
        print("ROC-AUC:          N/A")
    else:
        print(
            f"ROC-AUC:          "
            f"{results['roc_auc']:.6f}"
        )

    if results["pr_auc"] is None:
        print("PR-AUC:           N/A")
    else:
        print(
            f"PR-AUC:           "
            f"{results['pr_auc']:.6f}"
        )

    print()
    print("CONFUSION MATRIX")
    print("-" * 60)

    cm = results["confusion_matrix"]

    print("                Predicted")
    print("                 0     1")

    print(
        f"Actual 0       "
        f"{cm[0, 0]:5d} "
        f"{cm[0, 1]:5d}"
    )

    print(
        f"Actual 1       "
        f"{cm[1, 0]:5d} "
        f"{cm[1, 1]:5d}"
    )

    print()
    print("PREDICTION DISTRIBUTION")
    print("-" * 60)

    probabilities = results["probabilities"]
    predictions = results["predictions"]

    print(
        f"Probability min:  "
        f"{probabilities.min():.6f}"
    )

    print(
        f"Probability max:  "
        f"{probabilities.max():.6f}"
    )

    print(
        f"Probability mean: "
        f"{probabilities.mean():.6f}"
    )

    print(
        f"Predicted 0:      "
        f"{(predictions == 0).sum()}"
    )

    print(
        f"Predicted 1:      "
        f"{(predictions == 1).sum()}"
    )

    print()
    print(
        f"Threshold: "
        f"{results['threshold']:.4f}"
    )