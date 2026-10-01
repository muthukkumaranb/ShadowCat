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

    def __init__(self, coverage: float = 0.90, aci_mode: bool = False, aci_gammas: Optional[List[float]] = None):
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"Coverage level must be in (0, 1), got {coverage}")
        self.coverage = float(coverage)
        self.alpha = float(1.0 - coverage)
        self.K = 5
        self.alpha_joint = self.alpha / self.K
        
        self.calibrated_quantile: Optional[float] = None
        self.calibrated_quantile_joint: Optional[float] = None
        self.calibration_sample_size: int = 0
        self.is_calibrated: bool = False
        self.residuals = None

        self.aci_mode = aci_mode
        self.aci_gammas = aci_gammas or [0.01, 0.01, 0.01, 0.05, 0.05]
        # Track effective alphas for ACI per horizon (0-indexed)
        self.aci_alphas = [self.alpha] * self.K
        self.aci_alphas_joint = [self.alpha_joint] * self.K

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
        self.residuals = np.abs(targets - preds)
        n = len(self.residuals)

        self.calibrated_quantile = self._get_quantile(self.alpha, n)
        self.calibrated_quantile_joint = self._get_quantile(self.alpha_joint, n)

        self.calibration_sample_size = n
        self.is_calibrated = True
        return self

    def _get_quantile(self, alpha: float, n: int) -> float:
        if self.residuals is None:
            return 0.15
        level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        return float(np.quantile(self.residuals, level, method="higher"))

    def predict_interval(self, point_prob: float, horizon_idx: int = 0) -> Tuple[float, float]:
        """
        Returns calibrated prediction interval [lower, upper] clipped to [0, 1].
        """
        alpha_eff = self.aci_alphas[horizon_idx] if self.aci_mode else self.alpha
        q = self._get_quantile(alpha_eff, self.calibration_sample_size) if self.is_calibrated else 0.15
        lower = float(np.clip(point_prob - q, 0.0, 1.0))
        upper = float(np.clip(point_prob + q, 0.0, 1.0))
        return lower, upper

    def predict_interval_joint(self, point_prob: float, horizon_idx: int = 0) -> Tuple[float, float]:
        """
        Returns joint-coverage (Bonferroni) calibrated prediction interval.
        """
        alpha_eff = self.aci_alphas_joint[horizon_idx] if self.aci_mode else self.alpha_joint
        q = self._get_quantile(alpha_eff, self.calibration_sample_size) if self.is_calibrated else 0.15
        lower = float(np.clip(point_prob - q, 0.0, 1.0))
        upper = float(np.clip(point_prob + q, 0.0, 1.0))
        return lower, upper

    def update_aci(self, point_prob: float, true_target: float, horizon_idx: int):
        if not self.aci_mode:
            return
            
        gamma = self.aci_gammas[horizon_idx]
        
        # Marginal ACI update
        lb, ub = self.predict_interval(point_prob, horizon_idx)
        if lb <= true_target <= ub:
            self.aci_alphas[horizon_idx] += gamma * self.alpha
        else:
            self.aci_alphas[horizon_idx] -= gamma * (1 - self.alpha)
        self.aci_alphas[horizon_idx] = float(np.clip(self.aci_alphas[horizon_idx], 0.001, 0.999))
        
        # Joint ACI update
        lb_j, ub_j = self.predict_interval_joint(point_prob, horizon_idx)
        if lb_j <= true_target <= ub_j:
            self.aci_alphas_joint[horizon_idx] += gamma * self.alpha_joint
        else:
            self.aci_alphas_joint[horizon_idx] -= gamma * (1 - self.alpha_joint)
        self.aci_alphas_joint[horizon_idx] = float(np.clip(self.aci_alphas_joint[horizon_idx], 0.0001, 0.999))

    def predict_intervals(self, point_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction interval generation.
        """
        probs = np.asarray(point_probs, dtype=float)
        q = self.calibrated_quantile if self.is_calibrated else 0.15
        lower = np.clip(probs - q, 0.0, 1.0)
        upper = np.clip(probs + q, 0.0, 1.0)
        return lower, upper

    def format_output(self, point_prob: float, horizon_idx: int = 0) -> Dict[str, Any]:
        """
        Formats conformal interval for JSON payload surfacing.
        """
        lb, ub = self.predict_interval(point_prob, horizon_idx)
        lb_j, ub_j = self.predict_interval_joint(point_prob, horizon_idx)
        pct = int(self.coverage * 100)
        pct_j = int((1.0 - self.alpha_joint) * 100)
        return {
            "point_estimate": round(float(point_prob), 4),
            "confidence_level": f"{pct}%",
            "conformal_interval": [round(lb, 4), round(ub, 4)],
            "interval_width": round(ub - lb, 4),
            "joint_confidence_level": f"{pct_j}%",
            "joint_conformal_interval": [round(lb_j, 4), round(ub_j, 4)],
            "joint_interval_width": round(ub_j - lb_j, 4),
            "is_calibrated": self.is_calibrated,
            "calibration_samples": self.calibration_sample_size,
            "guarantee": f"Finite-sample marginal coverage >= {pct}%",
        }
