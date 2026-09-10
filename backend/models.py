"""
SHADOWCAT Backend - Neural Model Architectures
Verified PyTorch architectures for the LSTM World Model, Hazard Heads, and Stage Head.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class LSTMGaussianWorldModel(nn.Module):
    """
    Diagonal Gaussian approximation to p(S(t+1) | S(t-29), ..., S(t)).
    Continuous dynamics world model encoding sequence history into latent z(t)
    and forecasting next-step state mean and standard deviation.
    """

    def __init__(
        self,
        input_size: int = 406,
        hidden_size: int = 64,
        state_dim: int = 406,
        num_layers: int = 1,
        dropout: float = 0.2,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.epsilon = epsilon

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim),
        )
        self.raw_std_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim),
        )

    def forward(self, x: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs LSTM sequence and returns (predicted_mean, predicted_std).
        Input shape: (batch_size, sequence_length=30, input_size=406)
        """
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim != 3 or x.size(-1) != self.input_size:
            raise ValueError(f"Expected input shape (batch, seq_len, {self.input_size}), got {tuple(x.shape)}")

        output, _ = self.lstm(x)
        representation = self.dropout(output[:, -1, :])
        mean = self.mean_head(representation)
        std = torch.nn.functional.softplus(self.raw_std_head(representation)) + self.epsilon
        return mean, std

    def extract_latent_z(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Extracts the 64-dimensional latent state z(t) = output[:, -1, :]
        prior to mean/std prediction heads.
        """
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim != 3 or x.size(-1) != self.input_size:
            raise ValueError(f"Expected input shape (batch, seq_len, {self.input_size}), got {tuple(x.shape)}")
        output, _ = self.lstm(x)
        return output[:, -1, :]


class LSTMClassifier(nn.Module):
    """
    LSTM sequence classifier used for Hazard Head onset forecasting.
    """

    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if x.ndim != 3 or x.size(-1) != self.input_size:
            raise ValueError(f"Expected input shape (batch, seq, {self.input_size}), got {tuple(x.shape)}")
        output, _ = self.lstm(x)
        last_output = self.dropout(output[:, -1, :])
        logits = self.fc(last_output)
        return logits.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.forward_logits(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class StageClassificationHead(nn.Module):
    """
    Stage Classifier operating on predicted future state S_hat(t+1) (406 dims).
    Maps future network telemetry state to MITRE ATT&CK tactic stages.
    """

    STAGE_CLASSES = [
        "Command and Control",
        "Credential Access",
        "Discovery",
        "Impact",
        "Initial Access",
        "Unknown/Other",
    ]

    TACTIC_ID_MAP = {
        "Command and Control": "TA0011",
        "Credential Access": "TA0006",
        "Discovery": "TA0007",
        "Impact": "TA0040",
        "Initial Access": "TA0001",
        "Unknown/Other": "TA0000",
    }

    def __init__(
        self,
        state_dim: int = 406,
        num_classes: int = 6,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, predicted_state: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch_size, state_dim=406)
        Returns: logits shape (batch_size, num_classes=6)
        """
        predicted_state = torch.as_tensor(predicted_state, dtype=torch.float32)
        return self.net(predicted_state)

    def predict_probabilities(self, predicted_state: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(predicted_state)
            return torch.softmax(logits, dim=-1)
