from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed:
        Integer random seed.
    """

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make CUDA behavior as deterministic as possible.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(
    preferred: str | None = None,
) -> torch.device:
    """
    Select the PyTorch device.

    Parameters
    ----------
    preferred:
        Optional device requested by the user.

        Examples:
            "cpu"
            "cuda"

        If None, CUDA is used when available,
        otherwise CPU is used.
    """

    if preferred is None:
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if not isinstance(preferred, str):
        raise TypeError(
            "preferred device must be a string or None."
        )

    preferred = preferred.lower()

    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    if preferred == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device: {preferred}. "
        "Use 'cpu', 'cuda', or None."
    )


def save_model(
    model: torch.nn.Module,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Save a PyTorch model checkpoint.

    Parameters
    ----------
    model:
        Trained PyTorch model.

    path:
        Destination checkpoint path.

    metadata:
        Optional metadata to save alongside the model state.

    Returns
    -------
    Path
        Path to the saved checkpoint.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "model must be an instance of torch.nn.Module."
        )

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    if metadata is not None:
        if not isinstance(metadata, dict):
            raise TypeError(
                "metadata must be a dictionary or None."
            )

        checkpoint["metadata"] = metadata

    torch.save(
        checkpoint,
        path,
    )

    return path


def load_model(
    model: torch.nn.Module,
    path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Load model weights from a checkpoint.

    Parameters
    ----------
    model:
        Already-constructed model with the same architecture
        as the saved checkpoint.

    path:
        Checkpoint path.

    device:
        Device on which the model should be loaded.

    Returns
    -------
    tuple
        Loaded model and metadata dictionary.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "model must be an instance of torch.nn.Module."
        )

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}"
        )

    if device is None:
        target_device = get_device()
    elif isinstance(device, torch.device):
        target_device = device
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        raise TypeError(
            "device must be a string, torch.device, or None."
        )

    checkpoint = torch.load(
        path,
        map_location=target_device,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Invalid model checkpoint format."
        )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(target_device)

    metadata = checkpoint.get(
        "metadata",
        {},
    )

    if not isinstance(metadata, dict):
        metadata = {}

    return model, metadata


def count_parameters(
    model: torch.nn.Module,
) -> int:
    """
    Return the number of trainable parameters.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "model must be an instance of torch.nn.Module."
        )

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def model_summary(
    model: torch.nn.Module,
) -> dict[str, Any]:
    """
    Return basic model information.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "model must be an instance of torch.nn.Module."
        )

    return {
        "model_class": model.__class__.__name__,
        "trainable_parameters": count_parameters(model),
        "device": str(
            next(model.parameters()).device
        ),
        "training_mode": model.training,
    }