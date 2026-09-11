from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LSTMWorldModel(nn.Module):
    """Deterministic next-state model: a deterministic approximation to p(S(t+1)|S(t))."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        state_dim: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("input_size must be a positive integer.")
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer.")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        if state_dim is None:
            state_dim = input_size
        if not isinstance(state_dim, int) or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.state_dim = state_dim
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
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, state_dim),
        )

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if not isinstance(x, (torch.Tensor, np.ndarray)):
            raise TypeError("Input must be a torch.Tensor or numpy.ndarray.")
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim != 3:
            raise ValueError(
                "Expected input with 3 dimensions (batch, sequence_length, features), "
                f"but received shape {tuple(x.shape)}."
            )
        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, but received {x.size(-1)}."
            )

        output, _ = self.lstm(x)
        last_output = self.dropout(output[:, -1, :])
        return self.mlp(last_output)

    def predict_next_state(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "state_dim": self.state_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
        }


class LSTMClassifier(nn.Module):
    """
    Generic LSTM-based binary sequence classifier.

    Expected input shape:
        (batch_size, sequence_length, input_size)

    Output shape:
        (batch_size,)

    Output values:
        Probability between 0 and 1.

    This model is dataset-independent.
    It can be used with any numerical sequential dataset
    as long as the input is converted to the expected shape.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Validate configuration
        # ---------------------------------------------------------

        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(
                "input_size must be a positive integer."
            )

        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                "hidden_size must be a positive integer."
            )

        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(
                "num_layers must be a positive integer."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        # ---------------------------------------------------------
        # LSTM
        # ---------------------------------------------------------

        # PyTorch only applies dropout between LSTM layers.
        # Therefore, disable internal LSTM dropout when num_layers=1.
        lstm_dropout = (
            dropout if num_layers > 1 else 0.0
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # ---------------------------------------------------------
        # External dropout
        # ---------------------------------------------------------

        self.dropout = nn.Dropout(
            p=dropout
        )

        # ---------------------------------------------------------
        # Classification head
        # ---------------------------------------------------------

        self.fc = nn.Linear(
            hidden_size,
            1,
        )

        # ---------------------------------------------------------
        # Probability conversion
        # ---------------------------------------------------------

        self.sigmoid = nn.Sigmoid()

        # Store configuration for reproducibility/debugging.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

    def forward_logits(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Tensor with shape:

                (batch, sequence_length, features)

        Returns
        -------
        torch.Tensor
            Probability for the positive class with shape:

                (batch,)
        """

        # ---------------------------------------------------------
        # Input validation
        # ---------------------------------------------------------

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "Input must be a torch.Tensor."
            )

        if x.ndim != 3:
            raise ValueError(
                "Expected input with 3 dimensions "
                "(batch, sequence_length, features), "
                f"but received shape {tuple(x.shape)}."
            )

        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, "
                f"but received {x.size(-1)}."
            )

        # ---------------------------------------------------------
        # LSTM
        # ---------------------------------------------------------

        output, _ = self.lstm(x)

        # ---------------------------------------------------------
        # Take the final timestep
        # ---------------------------------------------------------

        last_output = output[:, -1, :]

        # ---------------------------------------------------------
        # Dropout
        # ---------------------------------------------------------

        last_output = self.dropout(
            last_output
        )

        # ---------------------------------------------------------
        # Fully connected layer
        # ---------------------------------------------------------

        logits = self.fc(
            last_output
        )

        return logits.squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Return positive-class probabilities for compatibility."""
        return self.sigmoid(self.forward_logits(x))

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Generate binary predictions.

        Parameters
        ----------
        x:
            Input tensor with shape:

                (batch, sequence_length, features)

        threshold:
            Classification threshold between 0 and 1.

        Returns
        -------
        torch.Tensor
            Binary predictions with shape:

                (batch,)
        """

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        self.eval()

        with torch.no_grad():
            probabilities = self.forward(x)

        return (
            probabilities >= threshold
        ).long()

    def predict_proba(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return positive-class probabilities.

        This is useful during evaluation and inference.
        """

        self.eval()

        with torch.no_grad():
            probabilities = self.forward(x)

        return probabilities

    def get_config(self) -> dict:
        """
        Return model configuration.

        Useful for saving model metadata.
        """

        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
        }


class LSTMGaussianWorldModel(nn.Module):
    """Diagonal Gaussian approximation to p(S(t+1)|S(t-29), ..., S(t))."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        state_dim: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.2,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        if state_dim is None:
            state_dim = input_size
        if input_size <= 0 or hidden_size <= 0 or state_dim <= 0 or num_layers <= 0:
            raise ValueError("input_size, hidden_size, state_dim, and num_layers must be positive.")
        if not 0.0 <= dropout < 1.0 or epsilon <= 0.0:
            raise ValueError("dropout must be in [0, 1) and epsilon must be positive.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.epsilon = epsilon
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.mean_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, state_dim))
        self.raw_std_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, state_dim))

    def forward(self, x: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim != 3 or x.size(-1) != self.input_size:
            raise ValueError(f"Expected input shape (batch, sequence, {self.input_size}).")
        output, _ = self.lstm(x)
        representation = self.dropout(output[:, -1, :])
        mean = self.mean_head(representation)
        std = torch.nn.functional.softplus(self.raw_std_head(representation)) + self.epsilon
        return mean, std

    def predict_distribution(self, x: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x)

    def get_config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "state_dim": self.state_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "epsilon": self.epsilon,
            "distribution": "diagonal Gaussian",
        }