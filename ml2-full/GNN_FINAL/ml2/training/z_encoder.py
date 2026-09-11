"""Direct extraction of ML1's temporal latent z(t) for ML2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class Z_tEncoder(nn.Module):
    """Reimplements ML1's LSTM trunk and returns the final hidden state."""

    def __init__(
        self,
        input_size: int = 406,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 406 or x.shape[1] != 30:
            raise ValueError(f"Expected [batch, 30, 406], got {tuple(x.shape)}")
        output, _ = self.lstm(x)
        return output[:, -1, :]


def load_z_t_encoder(checkpoint_path: str | Path, device: str = "cpu") -> Z_tEncoder:
    """Load only the verified LSTM trunk from ML1's full checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("ML1 checkpoint does not contain a state dictionary")
    lstm_state = {key: value for key, value in state_dict.items() if key.startswith("lstm.")}
    if not lstm_state:
        raise ValueError("ML1 checkpoint contains no lstm.* weights")

    encoder = Z_tEncoder()
    missing, unexpected = encoder.load_state_dict(lstm_state, strict=False)
    if missing or unexpected:
        raise ValueError(f"Invalid ML1 LSTM state: missing={missing}, unexpected={unexpected}")
    encoder.eval()
    return encoder.to(device)


class TemporalFeatureStore:
    """Build causal, normalized 30-window inputs and cache z(t) by timestamp."""

    def __init__(
        self,
        windows_path: str | Path,
        metadata_path: str | Path,
        scaler_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self.device = device
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.features: List[str] = list(metadata["features"])
        if len(self.features) != 406 or metadata["lookback"] != 30:
            raise ValueError("ML1 metadata does not describe the required 30 x 406 input")

        windows = pd.read_parquet(windows_path).sort_values("window_start_utc").reset_index(drop=True)
        windows["window_start_utc"] = pd.to_datetime(windows["window_start_utc"], utc=True)
        missing = [feature for feature in self.features if feature not in windows.columns]
        if missing:
            raise ValueError(f"Canonical UCS is missing ML1 features: {missing[:10]}")
        if not windows["window_start_utc"].is_monotonic_increasing:
            raise ValueError("Canonical UCS timestamps must be chronological")

        values = windows[self.features].to_numpy(dtype=np.float32).copy()
        if not np.isfinite(values).all():
            raise ValueError("Canonical ML1 input contains non-finite feature values")
        if not Path(scaler_path).is_file():
            raise FileNotFoundError(f"Missing canonical scaler metadata: {scaler_path}")

        # ML1 uses the canonical parquet as-is when scaler_params.yaml exists;
        # applying the saved transform again would double-normalize the inputs.
        normalized = values

        encoder = load_z_t_encoder(checkpoint_path, device=device)
        self._z_by_timestamp: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for end in range(29, len(windows)):
                sequence = torch.from_numpy(normalized[end - 29 : end + 1]).unsqueeze(0).to(device)
                timestamp = str(windows.loc[end, "window_start_utc"])
                self._z_by_timestamp[timestamp] = encoder(sequence).detach()

    def get(self, timestamp: object) -> torch.Tensor:
        key = str(pd.to_datetime(timestamp, utc=True))
        if key not in self._z_by_timestamp:
            raise ValueError(f"No causal 30-window z(t) is available for {key}")
        return self._z_by_timestamp[key]

    @property
    def values(self) -> Dict[str, torch.Tensor]:
        return self._z_by_timestamp
