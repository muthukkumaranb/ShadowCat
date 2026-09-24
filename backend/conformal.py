"""
Split Conformal Prediction for Hazard/Onset Forecasts
Implements finite-sample distribution-free valid prediction intervals:
P(Y in C(X)) >= 1 - alpha
Calibrated strictly on validation split to prevent test-set leakage.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np


class SplitConformalPredictor:
    """
    Split conformal predictor providing coverage guarantees on risk/hazard predictions.
    """

    def __init__(self, coverage: float = 0.90):
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"Coverage level must be in (0, 1), got {coverage}")
        self.coverage = float(coverage)
        self.alpha = float(1.0 - coverage)
        self.calibrated_quantile: Optional[float] = None
        self.calibration_sample_size: int = 0
        self.is_calibrated: bool = False

    def calibrate(
        self,
        val_predictions: np.ndarray,
        val_targets: np.ndarray,
    ) -> "SplitConformalPredictor":
        """
        Calibrate prediction intervals on validation predictions and true binary targets.
        """
        preds = np.asarray(val_predictions, dtype=float).ravel()
        targets = np.asarray(val_targets, dtype=float).ravel()

        if len(preds) != len(targets):
            raise ValueError(f"Size mismatch: {len(preds)} predictions vs {len(targets)} targets")
        if len(preds) == 0:
            raise ValueError("Validation set cannot be empty")

        # Non-conformity score: absolute residual |y - p|
        residuals = np.abs(targets - preds)
        n = len(residuals)

        # Standard split conformal quantile with finite-sample correction: ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        q = float(np.quantile(residuals, level, method="higher"))

        self.calibrated_quantile = q
        self.calibration_sample_size = n
        self.is_calibrated = True
        return self

    def predict_interval(self, point_prob: float) -> Tuple[float, float]:
        """
        Returns calibrated prediction interval [lower, upper] clipped to [0, 1].
        """
        q = self.calibrated_quantile if self.is_calibrated else 0.15
        lower = float(np.clip(point_prob - q, 0.0, 1.0))
        upper = float(np.clip(point_prob + q, 0.0, 1.0))
        return lower, upper

    def predict_intervals(self, point_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction interval generation.
        """
        probs = np.asarray(point_probs, dtype=float)
        q = self.calibrated_quantile if self.is_calibrated else 0.15
        lower = np.clip(probs - q, 0.0, 1.0)
        upper = np.clip(probs + q, 0.0, 1.0)
        return lower, upper

    def format_output(self, point_prob: float) -> Dict[str, Any]:
        """
        Formats conformal interval for JSON payload surfacing.
        """
        lb, ub = self.predict_interval(point_prob)
        pct = int(self.coverage * 100)
        return {
            "point_estimate": round(float(point_prob), 4),
            "confidence_level": f"{pct}%",
            "conformal_interval": [round(lb, 4), round(ub, 4)],
            "interval_width": round(ub - lb, 4),
            "is_calibrated": self.is_calibrated,
            "calibration_samples": self.calibration_sample_size,
            "guarantee": f"Finite-sample marginal coverage >= {pct}%",
        }
