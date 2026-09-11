from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def gaussian_nll(mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean diagonal-Gaussian negative log likelihood over samples and state dimensions."""
    if mean.shape != std.shape or mean.shape != target.shape:
        raise ValueError("mean, std, and target must have identical shapes.")
    if torch.any(std <= 0):
        raise ValueError("std must be strictly positive.")
    variance = std.square()
    return 0.5 * (torch.log(torch.tensor(2.0 * np.pi, device=mean.device)) + 2.0 * torch.log(std) + (target - mean).square() / variance).mean()


def gaussian_nll_per_sample(mean: np.ndarray, std: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return one diagonal-Gaussian NLL per transition window."""
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mean.shape != std.shape or mean.shape != target.shape or mean.ndim != 2:
        raise ValueError("mean, std, and target must be equal-shaped 2D arrays.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not np.isfinite(target).all():
        raise ValueError("mean, std, and target must be finite.")
    if np.any(std <= 0):
        raise ValueError("std must be strictly positive.")
    terms = 0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(std) + ((target - mean) / std) ** 2)
    return terms.mean(axis=1)


def summarize_nll(values: np.ndarray) -> dict[str, float]:
    """Summarize per-window NLL values without discarding their distribution."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("NLL values must be a non-empty finite 1D array.")
    percentiles = np.percentile(values, [50, 75, 90, 95, 99])
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "p50": float(percentiles[0]),
        "p75": float(percentiles[1]),
        "p90": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
    }


def sigma_diagnostics(std: np.ndarray) -> dict[str, float | int]:
    """Summarize predicted scales and count small-scale values."""
    values = np.asarray(std, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("std must be a finite positive 2D array.")
    flattened = values.ravel()
    result: dict[str, float | int] = {
        "minimum": float(np.min(flattened)),
        "median": float(np.median(flattened)),
        "mean": float(np.mean(flattened)),
        "maximum": float(np.max(flattened)),
    }
    for threshold in (1e-6, 1e-5, 1e-4):
        count = int(np.sum(flattened < threshold))
        result[f"count_below_{threshold:g}"] = count
        result[f"fraction_below_{threshold:g}"] = float(count / len(flattened))
    return result


def top_nll_outliers(
    nll: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    target: np.ndarray,
    timestamps: Sequence[object],
    top_n: int = 50,
) -> list[dict[str, object]]:
    """Extract the highest-loss windows with their diagnostic arrays."""
    nll = np.asarray(nll)
    order = np.argsort(nll)[::-1][:top_n]
    rows = []
    for rank, index in enumerate(order, start=1):
        rows.append({
            "rank": rank,
            "index": int(index),
            "timestamp": str(timestamps[index]),
            "nll": float(nll[index]),
            "prediction_error_mae": float(np.mean(np.abs(mean[index] - target[index]))),
            "target_magnitude": float(np.linalg.norm(target[index])),
            "predicted_mean": mean[index].tolist(),
            "predicted_std": std[index].tolist(),
            "target": target[index].tolist(),
        })
    return rows


def prediction_interval(mean: np.ndarray, std: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Return dimension-wise Gaussian prediction intervals for a nominal level."""
    critical_values = {0.80: 1.2815515655446004, 0.95: 1.959963984540054}
    if level not in critical_values:
        raise ValueError("Supported interval levels are 0.80 and 0.95.")
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != std.shape or np.any(std <= 0):
        raise ValueError("mean and std must have equal shapes and positive std values.")
    margin = critical_values[level] * std
    return mean - margin, mean + margin


def regression_metrics(mean: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.asarray(mean) - np.asarray(target)
    return {
        "mse": float(np.mean(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
    }


def interval_coverage(mean: np.ndarray, std: np.ndarray, target: np.ndarray, level: float) -> float:
    lower, upper = prediction_interval(mean, std, level)
    return float(np.mean((np.asarray(target) >= lower) & (np.asarray(target) <= upper)))


def feature_groups(columns: Iterable[str]) -> dict[str, list[int]]:
    """Group UCS columns by stable feature-family prefixes for ablation."""
    groups: dict[str, list[int]] = {}
    for index, column in enumerate(columns):
        name = str(column)
        if name.startswith("mask_"):
            group = "masks"
        elif name.startswith("byte_count") or name.startswith("bytes_per_sec"):
            group = "traffic_volume"
        elif name.startswith("packet_count") or name.startswith("packets_per_sec"):
            group = "packet_volume"
        elif name.startswith("duration") or "iat" in name or name.startswith("active_") or name.startswith("idle_"):
            group = "timing"
        elif "flag" in name:
            group = "tcp_flags"
        elif name.startswith("window_size") or name.startswith("fwd_seg_size"):
            group = "tcp_window_segment"
        elif name.startswith("down_up_ratio"):
            group = "direction_ratio"
        else:
            group = "other"
        groups.setdefault(group, []).append(index)
    return groups


def ablate_history(history: np.ndarray, indices: Iterable[int], value: float = 0.0) -> np.ndarray:
    result = np.asarray(history, dtype=np.float32).copy()
    result[:, :, list(indices)] = value
    return result


def deviation_scores(mean: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-sample mean absolute transition deviation in the model representation."""
    return np.mean(np.abs(np.asarray(target) - np.asarray(mean)), axis=-1)


def train_gaussian(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    patience: int = 5,
    min_delta: float = 0.001,
    seed: int = 42,
    device: str = "cpu",
    checkpoint_path=None,
) -> dict:
    torch.manual_seed(seed)
    device_obj = torch.device(device)
    model.to(device_obj)
    train_loader = DataLoader(TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.as_tensor(X_validation, dtype=torch.float32), torch.as_tensor(y_validation, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = {"train_nll": [], "validation_nll": []}
    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        for features, target in train_loader:
            optimizer.zero_grad()
            mean, std = model(features.to(device_obj))
            loss = gaussian_nll(mean, std, target.to(device_obj))
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(features)
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for features, target in val_loader:
                mean, std = model(features.to(device_obj))
                val_total += gaussian_nll(mean, std, target.to(device_obj)).item() * len(features)
        train_loss = train_total / len(train_loader.dataset)
        val_loss = val_total / len(val_loader.dataset)
        history["train_nll"].append(train_loss)
        history["validation_nll"].append(val_loss)
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "validation_nll": val_loss, "epoch": epoch}, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_epoch": int(np.argmin(history["validation_nll"]) + 1), "best_validation_nll": float(best_loss), "stopped_epoch": epoch, "device": str(device_obj)}
