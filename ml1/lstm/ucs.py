from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


UCS_METADATA_COLUMNS = {
    "window_id",
    "window_start_utc",
    "window_end_utc",
    "source_day",
    "split",
}
UCS_LABEL_COLUMNS = {
    "label_binary",
    "label_attack_type",
    "future_attack_label",
    "has_malicious_flows",
    "raw_label_dominant",
}
LOG1P_BASE_COLUMNS = {
    "byte_count_fwd",
    "byte_count_bwd",
    "packet_count_fwd",
    "packet_count_bwd",
    "bytes_per_sec",
    "packets_per_sec",
    "duration_sec",
}
SPLITS = ("train", "val", "test")
LOEO_GROUPING_COLUMN = "episode_id"
LOEO_ELIGIBLE_ATTACK_TYPES = (
    "SSH-Bruteforce",
    "DDOS-LOIC-UDP",
    "Botnet",
)
LOEO_EXCLUDED_ATTACK_TYPES = (
    "DDOS-HOIC",
    "Infiltration-Compromise",
    "Infiltration-Portscan",
)


@dataclass(frozen=True)
class UCSConfig:
    """Authoritative temporal settings from the canonical UCS pipeline."""

    window_seconds: int = 60
    lookback_windows: int = 30
    horizon_windows: int = 5
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    @property
    def purge_embargo_width(self) -> int:
        return self.lookback_windows + self.horizon_windows


@dataclass
class SequenceSet:
    X: np.ndarray
    y: np.ndarray
    timestamps: pd.DatetimeIndex


def load_ucs_windows(path: str | Path) -> pd.DataFrame:
    """Load canonical ``ucs_windows`` output without parsing raw flows."""
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("UCS windows must be a .parquet or .csv file.")


def feature_columns(
    frame: pd.DataFrame,
    *,
    explicit: Sequence[str] | None = None,
) -> list[str]:
    """Return numeric UCS inputs while excluding labels and metadata."""
    if explicit is not None:
        columns = list(explicit)
        forbidden = UCS_LABEL_COLUMNS | UCS_METADATA_COLUMNS
        if forbidden.intersection(columns):
            raise ValueError("Feature columns cannot contain metadata or labels.")
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing UCS feature columns: {missing}")
        return columns

    excluded = UCS_METADATA_COLUMNS | UCS_LABEL_COLUMNS
    columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    if not columns:
        raise ValueError("UCS windows contain no numeric input features.")
    return columns


def validate_ucs_windows(
    frame: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    config: UCSConfig = UCSConfig(),
) -> list[str]:
    """Validate the canonical UCS window contract and return input columns."""
    required = {"window_start_utc", "split", "future_attack_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"UCS windows are missing required columns: {missing}")
    if not frame["split"].isin(SPLITS).all():
        raise ValueError("UCS split must contain only train, val, and test.")
    timestamps = pd.to_datetime(frame["window_start_utc"], utc=True, errors="coerce")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("window_start_utc must be valid and chronological.")
    if frame["future_attack_label"].isna().any() or not frame["future_attack_label"].isin([0, 1]).all():
        raise ValueError("future_attack_label must contain only 0 and 1.")
    if config.lookback_windows <= 0 or config.horizon_windows <= 0:
        raise ValueError("lookback_windows and horizon_windows must be positive.")
    columns = feature_columns(frame, explicit=features)
    if frame[columns].isna().any().any() or not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError("UCS input features must be finite and non-null.")
    return columns


def purge_and_embargo(
    frame: pd.DataFrame,
    *,
    config: UCSConfig = UCSConfig(),
) -> pd.DataFrame:
    """Apply canonical positional purge/embargo around train/val/test cuts."""
    validate_ucs_windows(frame, config=config)
    ordered = frame.sort_values("window_start_utc").reset_index(drop=True)
    width = config.purge_embargo_width
    keep = np.ones(len(ordered), dtype=bool)
    split_values = ordered["split"].to_numpy()
    for left, right in zip(SPLITS, SPLITS[1:]):
        left_positions = np.flatnonzero(split_values == left)
        right_positions = np.flatnonzero(split_values == right)
        if not len(left_positions) or not len(right_positions):
            raise ValueError(f"Every split must contain windows: missing {left} or {right}.")
        keep[left_positions[-min(width, len(left_positions)):]] = False
        keep[right_positions[:min(width, len(right_positions))]] = False
    return ordered.loc[keep].reset_index(drop=True)


def build_lstm_sequences(
    purged_frame: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    config: UCSConfig = UCSConfig(),
    target_col: str = "future_attack_label",
    require_contiguous: bool = True,
) -> dict[str, SequenceSet]:
    """Build deterministic, within-split UCS sequences and end timestamps."""
    columns = validate_ucs_windows(purged_frame, features=features, config=config)
    if target_col not in {"future_attack_label", "label_binary"}:
        raise ValueError("target_col must be future_attack_label or label_binary.")
    result: dict[str, SequenceSet] = {}
    ordered = purged_frame.sort_values("window_start_utc")
    for split in SPLITS:
        subset = ordered.loc[ordered["split"] == split].reset_index(drop=True)
        values = subset[columns].to_numpy(dtype=np.float32)
        labels = subset[target_col].to_numpy(dtype=np.float32)
        times = pd.DatetimeIndex(pd.to_datetime(subset["window_start_utc"], utc=True))
        window_delta = pd.Timedelta(seconds=config.window_seconds)
        valid_starts = range(max(0, len(subset) - config.lookback_windows + 1))
        if require_contiguous:
            valid_starts = [
                start for start in valid_starts
                if (times[start + config.lookback_windows - 1] - times[start])
                == window_delta * (config.lookback_windows - 1)
                and (times[start + 1:start + config.lookback_windows].to_series().diff().dropna() == window_delta).all()
            ]
        else:
            valid_starts = list(valid_starts)
        count = len(valid_starts)
        X = np.empty((count, config.lookback_windows, len(columns)), dtype=np.float32)
        y = np.empty(count, dtype=np.float32)
        ends = []
        for output_index, start in enumerate(valid_starts):
            end = start + config.lookback_windows
            X[output_index] = values[start:end]
            y[output_index] = labels[end - 1]
            ends.append(times[end - 1])
        result[split] = SequenceSet(X=X, y=y, timestamps=pd.DatetimeIndex(ends, tz="UTC"))
    return result


def build_next_state_sequences(
    purged_frame: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    config: UCSConfig = UCSConfig(),
    require_contiguous: bool = True,
) -> dict[str, SequenceSet]:
    """Build next-state regression sequences where target is the immediately following UCS state.

    This is the deterministic approximation to p(S(t+1)|S(t)) requested for the world-model core.
    """
    columns = validate_ucs_windows(purged_frame, features=features, config=config)
    result: dict[str, SequenceSet] = {}
    ordered = purged_frame.sort_values("window_start_utc")
    for split in SPLITS:
        subset = ordered.loc[ordered["split"] == split].reset_index(drop=True)
        values = subset[columns].to_numpy(dtype=np.float32)
        times = pd.DatetimeIndex(pd.to_datetime(subset["window_start_utc"], utc=True))
        window_delta = pd.Timedelta(seconds=config.window_seconds)
        valid_starts = list(range(max(0, len(subset) - config.lookback_windows)))
        if require_contiguous:
            valid_starts = [
                start for start in valid_starts
                if (times[start + config.lookback_windows - 1] - times[start])
                == window_delta * (config.lookback_windows - 1)
                and (times[start + 1:start + config.lookback_windows].to_series().diff().dropna() == window_delta).all()
            ]
        count = len(valid_starts)
        X = np.empty((count, config.lookback_windows, len(columns)), dtype=np.float32)
        y = np.empty((count, len(columns)), dtype=np.float32)
        timestamps = []
        for output_index, start in enumerate(valid_starts):
            input_end = start + config.lookback_windows
            target_index = input_end
            if target_index >= len(subset):
                continue
            X[output_index] = values[start:input_end]
            y[output_index] = values[target_index]
            timestamps.append(times[target_index])
        result[split] = SequenceSet(X=X[: len(timestamps)], y=y[: len(timestamps)], timestamps=pd.DatetimeIndex(timestamps, tz="UTC"))
    return result


def make_next_state_targets(
    sequence_set: SequenceSet,
) -> np.ndarray:
    """Return the actual next UCS state targets for a next-state sequence set."""
    if sequence_set.y.ndim != 2:
        raise ValueError("Expected next-state targets with shape (samples, state_dim).")
    return sequence_set.y.copy()


def build_loeo_episode_folds(
    frame: pd.DataFrame,
    *,
    episode_col: str = "episode_id",
    eligible_attack_types: Sequence[str] = LOEO_ELIGIBLE_ATTACK_TYPES,
) -> list[dict[str, Any]]:
    """Build episode-wise LOEO folds with the held-out episode excluded from training.

    Each fold records the training rows, the held-out rows, and the episode identifier.
    This is the explicit retraining protocol required for LOEO evaluation.
    """
    if episode_col != LOEO_GROUPING_COLUMN:
        raise ValueError(
            f"LOEO grouping must use '{LOEO_GROUPING_COLUMN}'; "
            "forecast_episode_id and source_day are not grouping keys."
        )
    if episode_col not in frame.columns:
        raise ValueError(f"LOEO fold construction requires the '{episode_col}' column.")
    if frame[episode_col].isna().any():
        raise ValueError("episode_id contains null values.")
    if "label_attack_type" not in frame.columns:
        raise ValueError("LOEO fold construction requires label_attack_type.")
    episode_metadata = frame.groupby(episode_col).agg(
        attack_types=("label_attack_type", lambda values: sorted(set(values.astype(str)))),
    )
    inconsistent = episode_metadata[episode_metadata["attack_types"].map(len) != 1]
    if not inconsistent.empty:
        raise ValueError("Each episode_id must map to exactly one label_attack_type.")
    eligible = set(eligible_attack_types)
    episodes = sorted(
        episode_metadata.loc[
            episode_metadata["attack_types"].map(lambda values: values[0] in eligible)
        ].index.tolist()
    )
    folds: list[dict[str, Any]] = []
    for episode in episodes:
        holdout_mask = frame[episode_col] == episode
        train_mask = ~holdout_mask
        if not train_mask.any() or not holdout_mask.any():
            continue
        train_episodes = sorted(frame.loc[train_mask, episode_col].dropna().unique().tolist())
        folds.append(
            {
                "held_out_episode": episode,
                "attack_type": episode_metadata.loc[episode, "attack_types"][0],
                "train_episodes": train_episodes,
                "train_indices": frame.index[train_mask].tolist(),
                "holdout_indices": frame.index[holdout_mask].tolist(),
            }
        )
    return folds


def persistence_baseline(history: np.ndarray) -> np.ndarray:
    """Persistence baseline: S(t+1) ≈ S(t)."""
    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 3:
        raise ValueError("history must have shape (samples, lookback, features).")
    return history[:, -1, :].copy()


def fit_lagged_linear_baseline(
    train_sequences: SequenceSet,
) -> LinearRegression:
    """Fit the next-state lagged-linear baseline on training sequences only."""
    if train_sequences.X.ndim != 3 or train_sequences.y.ndim != 2:
        raise ValueError("Expected X with shape (samples, lookback, features) and 2D next-state targets.")
    model = LinearRegression()
    model.fit(train_sequences.X[:, -1, :], train_sequences.y)
    return model


def fit_pca_lagged_linear_baseline(
    train_sequences: SequenceSet,
    pca: UCSPCA,
) -> LinearRegression:
    """Fit lagged linear dynamics on sequences already in PCA space."""
    if pca.pca is None:
        raise RuntimeError("PCA must be fitted before fitting the lagged baseline.")
    model = LinearRegression()
    model.fit(train_sequences.X[:, -1, :], train_sequences.y)
    return model


def pca_rollout_predictions(
    history: np.ndarray,
    predictor: LinearRegression,
    pca: UCSPCA,
    *,
    k: int,
) -> np.ndarray:
    """Recursively roll out a PCA lagged-linear model and invert to UCS space."""
    if pca.pca is None:
        raise RuntimeError("PCA must be fitted before rolling out predictions.")
    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 3 or k <= 0:
        raise ValueError("history must be 3D and k must be positive.")
    current = pca.pca.transform(history.reshape(-1, history.shape[-1])).reshape(history.shape[0], history.shape[1], -1)
    rollout = np.empty((history.shape[0], k, history.shape[-1]), dtype=np.float32)
    for step in range(k):
        predicted_pca = np.asarray(predictor.predict(current[:, -1, :]), dtype=np.float64)
        predicted = pca.pca.inverse_transform(predicted_pca).astype(np.float32)
        if not np.isfinite(predicted).all():
            raise FloatingPointError("PCA lagged-linear rollout produced non-finite values.")
        rollout[:, step, :] = predicted
        if step < k - 1:
            current = np.concatenate([current[:, 1:, :], predicted_pca[:, None, :]], axis=1)
    return rollout


def rollout_targets(
    purged_frame: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    config: UCSConfig = UCSConfig(),
    k: int = 3,
    split: str = "test",
) -> tuple[np.ndarray, np.ndarray, list[pd.DatetimeIndex]]:
    """Build observed histories and contiguous future states for chronological rollouts."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    columns = validate_ucs_windows(purged_frame, features=features, config=config)
    subset = purged_frame.loc[purged_frame["split"] == split].sort_values("window_start_utc").reset_index(drop=True)
    values = subset[columns].to_numpy(dtype=np.float32)
    times = pd.DatetimeIndex(pd.to_datetime(subset["window_start_utc"], utc=True))
    window_delta = pd.Timedelta(seconds=config.window_seconds)
    starts = [
        start for start in range(max(0, len(subset) - config.lookback_windows - k + 1))
        if times[start + config.lookback_windows - 1] - times[start]
        == window_delta * (config.lookback_windows - 1)
        and (times[start + 1:start + config.lookback_windows].to_series().diff().dropna() == window_delta).all()
        and (times[start + config.lookback_windows:start + config.lookback_windows + k]
             .to_series().diff().dropna() == window_delta).all()
    ]
    histories = np.stack([values[start:start + config.lookback_windows] for start in starts], axis=0)
    targets = np.stack([
        values[start + config.lookback_windows:start + config.lookback_windows + k]
        for start in starts
    ], axis=0)
    timestamps = [pd.DatetimeIndex(
        times[start + config.lookback_windows:start + config.lookback_windows + k]
    ) for start in starts]
    return histories, targets, timestamps


def rollout_predictions(
    history: np.ndarray,
    predictor,
    *,
    k: int = 3,
) -> np.ndarray:
    """Recursive rollout for K future steps using the model's own predictions."""
    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 3:
        raise ValueError("history must have shape (samples, lookback, features).")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    current = history.copy()
    rollout = np.empty((history.shape[0], k, history.shape[2]), dtype=np.float32)
    for step in range(k):
        predicted = np.asarray(predictor(current), dtype=np.float32)
        if predicted.shape != (history.shape[0], history.shape[2]):
            raise ValueError(
                "Predictor must output a tensor with shape (samples, features) for rollout."
            )
        rollout[:, step, :] = predicted
        if step < k - 1:
            current = np.concatenate([current[:, 1:, :], predicted[:, None, :]], axis=1)
    return rollout


class UCSRobustScaler:
    """Canonical ``log1p`` plus training-only robust quantile scaler."""

    def __init__(self, columns: Sequence[str]):
        self.columns = list(columns)
        self.median_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.log1p_mask_ = np.array([any(column.startswith(base) for base in LOG1P_BASE_COLUMNS) for column in self.columns])

    def _log_transform(self, values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float64, copy=True)
        if np.any(values[:, self.log1p_mask_] < -1):
            raise ValueError("log1p UCS features cannot be less than -1.")
        values[:, self.log1p_mask_] = np.log1p(values[:, self.log1p_mask_])
        return values

    def fit(self, frame: pd.DataFrame) -> "UCSRobustScaler":
        values = frame[self.columns].to_numpy(dtype=float)
        values = self._log_transform(values)
        self.median_ = np.nanquantile(values, 0.50, axis=0)
        q25 = np.nanquantile(values, 0.25, axis=0)
        q75 = np.nanquantile(values, 0.75, axis=0)
        self.scale_ = q75 - q25
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.median_ is None or self.scale_ is None:
            raise RuntimeError("UCSRobustScaler has not been fitted.")
        result = frame.copy()
        values = self._log_transform(result[self.columns].to_numpy(dtype=float))
        result.loc[:, self.columns] = (values - self.median_) / self.scale_
        return result

    def fit_transform(self, train_frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_frame).transform_frame(train_frame)

    def get_params(self) -> Mapping[str, Any]:
        if self.median_ is None or self.scale_ is None:
            raise RuntimeError("UCSRobustScaler has not been fitted.")
        return {"columns": self.columns, "median": self.median_.tolist(), "scale": self.scale_.tolist(), "log1p_columns": [column for column, enabled in zip(self.columns, self.log1p_mask_) if enabled]}


class UCSPCA:
    """PCA fitted on UCS training windows only."""

    def __init__(self, n_components: int, random_state: int = 42):
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        self.n_components = n_components
        self.random_state = random_state
        self.columns: list[str] | None = None
        self.pca: PCA | None = None

    def fit(self, train_frame: pd.DataFrame, columns: Sequence[str]) -> "UCSPCA":
        columns = list(columns)
        values = train_frame[columns].to_numpy(dtype=np.float64)
        if self.n_components > min(values.shape):
            raise ValueError("n_components cannot exceed min(training rows, features).")
        self.columns = columns
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(values)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.pca is None or self.columns is None:
            raise RuntimeError("UCSPCA has not been fitted.")
        values = self.pca.transform(frame[self.columns].to_numpy(dtype=np.float64))
        result = frame.drop(columns=self.columns).copy()
        result.loc[:, [f"pca_{index}" for index in range(self.n_components)]] = values
        return result

    def fit_transform(self, train_frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        self.fit(train_frame, columns)
        return self.transform(train_frame)

    def get_metadata(self) -> dict[str, Any]:
        if self.pca is None or self.columns is None:
            raise RuntimeError("UCSPCA has not been fitted.")
        return {
            "n_components": self.n_components,
            "n_features_in": int(self.pca.n_features_in_),
            "explained_variance": self.pca.explained_variance_.tolist(),
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": np.cumsum(self.pca.explained_variance_ratio_).tolist(),
            "random_state": self.random_state,
            "input_columns": self.columns,
        }


def apply_training_only_pca(
    purged_frame: pd.DataFrame,
    columns: Sequence[str],
    n_components: int,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, UCSPCA]:
    """Transform all purged windows using PCA fitted on purged training rows."""
    pca = UCSPCA(n_components=n_components, random_state=random_state)
    train_frame = purged_frame.loc[purged_frame["split"] == "train"]
    transformed = pca.fit(train_frame, columns).transform(purged_frame)
    return transformed, pca


def prepare_lstm_inputs(
    windows: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    config: UCSConfig = UCSConfig(),
) -> tuple[dict[str, SequenceSet], list[str], UCSRobustScaler, pd.DataFrame]:
    """Purge, fit train-only normalization, and build LSTM-ready inputs."""
    columns = validate_ucs_windows(windows, features=features, config=config)
    purged = purge_and_embargo(windows, config=config)
    scaler = UCSRobustScaler(columns).fit(purged.loc[purged["split"] == "train"])
    normalized = scaler.transform_frame(purged)
    sequences = build_lstm_sequences(normalized, features=columns, config=config)
    return sequences, columns, scaler, normalized
