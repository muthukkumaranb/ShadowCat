from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_data(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Validate generic sequence classification data.

    Expected:
        X -> [samples, sequence_length, features]
        y -> [samples]
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")

    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array.")

    if X.ndim != 3:
        raise ValueError(
            "X must have shape "
            "[samples, sequence_length, features]."
        )

    if y.ndim not in (1, 2):
        raise ValueError(
            "y must have shape [samples] or [samples, targets]."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    if len(X) == 0:
        raise ValueError("Dataset is empty.")

    if not np.isfinite(X).all():
        raise ValueError(
            "X contains NaN or infinite values."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "y contains NaN or infinite values."
        )


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """
    Convert numpy arrays into a PyTorch DataLoader.
    """

    validate_data(X, y)

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Apply a loss function to either 1D classification or 2D regression targets."""
    if targets.ndim == 1:
        return criterion(logits.reshape_as(targets), targets)
    if logits.shape != targets.shape:
        raise ValueError(
            f"Model output shape {tuple(logits.shape)} does not match target shape {tuple(targets.shape)}."
        )
    return criterion(logits, targets)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model.forward_logits(X_batch) if hasattr(model, "forward_logits") else model(X_batch)
        loss = _compute_loss(logits, y_batch, criterion)

        loss.backward()

        optimizer.step()

        batch_size = X_batch.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    return total_loss / total_samples


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate model loss without updating weights.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model.forward_logits(X_batch) if hasattr(model, "forward_logits") else model(X_batch)
            loss = _compute_loss(logits, y_batch, criterion)

            batch_size = X_batch.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

    return total_loss / total_samples


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    path: str | Path,
) -> None:
    """
    Save a model checkpoint.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    seed: int = 42,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    weight_decay: float = 0.0,
    pos_weight: float | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
) -> dict:
    """
    Generic LSTM training function.

    This function is completely dataset-independent.

    Parameters
    ----------
    X_train:
        Shape [samples, sequence_length, features]

    y_train:
        Shape [samples]

    X_validation:
        Shape [samples, sequence_length, features]

    y_validation:
        Shape [samples]
    """

    validate_data(
        X_train,
        y_train,
    )

    validate_data(
        X_validation,
        y_validation,
    )

    if X_train.shape[1:] != X_validation.shape[1:]:
        raise ValueError(
            "Training and validation feature dimensions "
            "must match."
        )

    set_seed(seed)

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)

    model = model.to(device)

    train_loader = create_dataloader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = create_dataloader(
        X_validation,
        y_validation,
        batch_size=batch_size,
        shuffle=False,
    )

    classification_task = y_train.ndim == 1
    if classification_task:
        if pos_weight is None:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(float(pos_weight), device=device),
            )
    else:
        if pos_weight is not None:
            raise ValueError("pos_weight is only supported for binary classification targets.")
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_precision": [],
        "validation_recall": [],
        "validation_f1": [],
        "validation_pr_auc": [],
        "validation_roc_auc": [],
        "learning_rate": [],
    }

    best_validation_loss = float("inf")
    best_validation_metric = float("-inf") if classification_task else float("inf")
    best_epoch = 0
    best_state_dict = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        validation_loss = evaluate_loss(
            model,
            validation_loader,
            criterion,
            device,
        )

        if classification_task:
            model.eval()
            validation_probabilities = []
            validation_labels = []
            with torch.no_grad():
                for X_batch, y_batch in validation_loader:
                    logits = model.forward_logits(X_batch.to(device)) if hasattr(model, "forward_logits") else model(X_batch.to(device))
                    validation_probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
                    validation_labels.extend(y_batch.numpy().reshape(-1))
            validation_probabilities = np.asarray(validation_probabilities)
            validation_labels = np.asarray(validation_labels).astype(np.int64)
            validation_predictions = (validation_probabilities >= 0.5).astype(np.int64)
            precision = precision_score(validation_labels, validation_predictions, zero_division=0)
            recall = recall_score(validation_labels, validation_predictions, zero_division=0)
            f1 = f1_score(validation_labels, validation_predictions, zero_division=0)
            pr_auc = average_precision_score(validation_labels, validation_probabilities)
            roc_auc = roc_auc_score(validation_labels, validation_probabilities) if len(np.unique(validation_labels)) == 2 else None
        else:
            precision = recall = f1 = pr_auc = roc_auc = None

        history["train_loss"].append(
            train_loss
        )

        history["validation_loss"].append(
            validation_loss
        )
        history["validation_precision"].append(None if precision is None else float(precision))
        history["validation_recall"].append(None if recall is None else float(recall))
        history["validation_f1"].append(None if f1 is None else float(f1))
        history["validation_pr_auc"].append(None if pr_auc is None else float(pr_auc))
        history["validation_roc_auc"].append(None if roc_auc is None else float(roc_auc))
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))

        if classification_task:
            should_save = pr_auc > best_validation_metric + early_stopping_min_delta
            if should_save:
                best_validation_metric = pr_auc
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
                if checkpoint_path is not None:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        validation_loss=validation_loss,
                        path=checkpoint_path,
                    )
            else:
                stale_epochs += 1
                if early_stopping_patience is not None and stale_epochs >= early_stopping_patience:
                    break
        else:
            should_save = validation_loss < best_validation_loss - early_stopping_min_delta
            if should_save:
                best_validation_metric = validation_loss
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
                if checkpoint_path is not None:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        validation_loss=validation_loss,
                        path=checkpoint_path,
                    )
            else:
                stale_epochs += 1
                if early_stopping_patience is not None and stale_epochs >= early_stopping_patience:
                    break
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_metric": best_validation_metric,
        "stopped_epoch": epoch,
        "pos_weight": pos_weight,
        "weight_decay": weight_decay,
        "device": str(device),
    }