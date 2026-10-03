"""
SHADOWCAT Backend - Explanatory Counterfactual Engine
Computes minimal, realistic perturbations to input features that would bring
a flagged window's hazard score below the calibrated alert threshold.

Safeguards:
1. Every candidate perturbation is strictly bounded to the empirical [1st, 99th]
   percentile range observed in the training split.
2. Inverts normalized representations back to original physical units (seconds,
   packets, bytes, retransmissions) via the empirical robust scaler parameters.
3. Explicit honesty disclosure: Discloses that results represent model-based
   counterfactuals under the learned decision boundary, not guaranteed attack prevention.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

# Resolve root paths
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
DATA_ENG_DIR = REPO_ROOT / "data-engineering"
MODELS_DIR = REPO_ROOT / "models"

HONESTY_LABEL = (
    "Model-based counterfactual under the hazard model's learned decision boundary — "
    "not a guarantee that this change would have prevented the actual attack, "
    "and not validated against real intervention data."
)


def get_feature_unit_label(feat_name: str) -> str:
    """Returns a readable physical unit label for a given feature name."""
    name = feat_name.lower()
    if "retrans" in name:
        return "retransmissions"
    if "duration_sec" in name:
        return "seconds"
    if "duration" in name or "iat" in name or "idle" in name or "active" in name:
        return "μs"
    if "byts_per_sec" in name or "bytes_per_sec" in name:
        return "bytes/sec"
    if "pkts_per_sec" in name:
        return "pkts/sec"
    if "totlen" in name or "payload" in name or "byte" in name or "size" in name:
        return "bytes"
    if "ttl" in name:
        return "TTL"
    if "win" in name:
        return "bytes"
    if "cnt" in name or "count" in name or "pkts" in name or "packet" in name:
        return "packets"
    if "ratio" in name or "skew" in name or "variance" in name or "entropy" in name:
        return "ratio"
    return "units"


class CounterfactualEngine:
    """
    Computes bounded explanatory counterfactual perturbations on sequence inputs.
    """

    def __init__(
        self,
        bounds_path: Optional[Union[str, Path]] = None,
        scaler_params_path: Optional[Union[str, Path]] = None,
    ):
        self.bounds_path = Path(bounds_path or (MODELS_DIR / "counterfactual_bounds.json"))
        self.scaler_params_path = Path(
            scaler_params_path or (DATA_ENG_DIR / "data" / "ucs" / "scaler_params.yaml")
        )
        self.bounds: Dict[str, Dict[str, float]] = {}
        self.scaler_params: Dict[str, Dict[str, Any]] = {}
        self._load_bounds()
        self._load_scaler_params()

    def _load_bounds(self) -> None:
        """Loads empirical [1st, 99th] percentile bounds and medians."""
        if self.bounds_path.exists():
            try:
                with open(self.bounds_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.bounds = data.get("bounds", {})
                logging.info(f"Loaded {len(self.bounds)} empirical feature bounds from {self.bounds_path}")
            except Exception as e:
                logging.warning(f"Failed to read counterfactual bounds from {self.bounds_path}: {e}")
        else:
            logging.warning(f"Counterfactual bounds file not found at {self.bounds_path}")

    def _load_scaler_params(self) -> None:
        """Loads robust scaler parameters for unit inversion."""
        if self.scaler_params_path.exists():
            try:
                with open(self.scaler_params_path, "r", encoding="utf-8") as f:
                    self.scaler_params = yaml.safe_load(f) or {}
                logging.info(f"Loaded scaler parameters for {len(self.scaler_params)} features")
            except Exception as e:
                logging.warning(f"Failed to read scaler params from {self.scaler_params_path}: {e}")

    def invert_units(self, feature_name: str, normalized_val: float) -> float:
        """
        Inverts a normalized/standardized feature value back to original physical units.
        RobustScaler inversion:
            x = z * scale + median
            if is_log1p: raw = exp(x) - 1
        """
        param = self.scaler_params.get(feature_name)
        if not param:
            return float(normalized_val)

        scale = float(param.get("scale", 1.0))
        median = float(param.get("median", 0.0))
        is_log1p = bool(param.get("is_log1p", False))

        unscaled = normalized_val * scale + median
        if is_log1p:
            raw = float(np.expm1(np.clip(unscaled, -20.0, 50.0)))
        else:
            raw = float(unscaled)
        return max(0.0, raw) if ("cnt" in feature_name or "len" in feature_name or "duration" in feature_name or "pkts" in feature_name) else raw

    def search(
        self,
        pipeline: Any,
        sequence_30x406: np.ndarray,
        top_features: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        max_features: int = 5,
        step_fractions: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Executes bounded perturbation search to find the minimal realistic change
        that reduces the hazard score below the calibrated alert threshold.

        Stopping condition: hazard < threshold.
        Percentile clipping: [p01, p99] strictly enforced.
        """
        target_threshold = float(
            threshold if threshold is not None else getattr(pipeline, "alert_threshold", 0.5)
        )
        if step_fractions is None:
            step_fractions = [0.15, 0.30, 0.50, 0.70, 0.85, 1.00]

        if pipeline is None:
            return {
                "status": "inconclusive",
                "found": False,
                "initial_hazard": 0.0,
                "summary": "Pipeline model not provided for counterfactual evaluation.",
                "honesty_label": HONESTY_LABEL,
            }

        # 1. Baseline Hazard Evaluation
        h0 = float(pipeline._predict_onset_probability(sequence_30x406))

        if h0 < target_threshold:
            return {
                "status": "already_below_threshold",
                "found": True,
                "initial_hazard": round(h0, 4),
                "counterfactual_hazard": round(h0, 4),
                "calibrated_threshold": target_threshold,
                "perturbed_features": [],
                "summary": (
                    f"Current window hazard ({h0:.3f}) is already below the alert threshold ({target_threshold:.3f}). "
                    "No perturbation required."
                ),
                "honesty_label": HONESTY_LABEL,
            }

        # 2. Resolve Candidate Features
        from backend.predict import UCSExtractor, get_feature_category
        cols = UCSExtractor.MODEL_INPUT_COLUMNS

        candidate_cols: List[str] = []
        if top_features:
            for f in top_features:
                clean_f = f.strip()
                if clean_f in cols and get_feature_category(clean_f) is not None:
                    candidate_cols.append(clean_f)
                else:
                    # Try matching case-insensitively or title-case
                    matched = [c for c in cols if c.lower() == clean_f.lower().replace(" ", "_")]
                    if matched and get_feature_category(matched[0]) is not None:
                        candidate_cols.append(matched[0])

        if not candidate_cols:
            # Rank by absolute deviation from median for non-mask features
            diffs = np.zeros(len(cols), dtype=np.float32)
            curr_win = sequence_30x406[-1]
            for i, c in enumerate(cols):
                if get_feature_category(c) is None:
                    diffs[i] = -1.0
                else:
                    med = self.bounds.get(c, {}).get("median", 0.0)
                    diffs[i] = abs(curr_win[i] - med)
            ranked_idx = np.argsort(diffs)[::-1]
            candidate_cols = [cols[i] for i in ranked_idx if diffs[i] >= 0][:max_features]
        else:
            candidate_cols = candidate_cols[:max_features]

        # 3. Search Phase 1: Single Feature Minimal Perturbation
        # Test if reducing any single top feature is sufficient
        best_hazard = h0
        for f in candidate_cols:
            f_idx = cols.index(f)
            orig_val = float(sequence_30x406[-1, f_idx])
            bound_meta = self.bounds.get(f, {"p01": -10.0, "median": 0.0, "p99": 10.0})
            p01 = bound_meta["p01"]
            p99 = bound_meta["p99"]
            baseline = bound_meta["median"]

            for alpha in step_fractions:
                # Target perturbation toward nominal baseline
                cand_val = orig_val + alpha * (baseline - orig_val)
                # REQUIRED SAFEGUARD: Empirical Percentile Clipping
                clipped_val = float(np.clip(cand_val, p01, p99))

                # Perturb input
                s_pert = sequence_30x406.copy()
                s_pert[-1, f_idx] = clipped_val

                h_pert = float(pipeline._predict_onset_probability(s_pert))
                if h_pert < best_hazard:
                    best_hazard = h_pert

                if h_pert < target_threshold:
                    # Minimal single-feature counterfactual found!
                    feat_delta = self._format_feature_change(
                        f, orig_val, clipped_val, baseline
                    )
                    summary = self._generate_summary(
                        [feat_delta], h0, h_pert, target_threshold
                    )
                    return {
                        "status": "feasible",
                        "found": True,
                        "initial_hazard": round(h0, 4),
                        "counterfactual_hazard": round(h_pert, 4),
                        "calibrated_threshold": target_threshold,
                        "perturbed_features": [feat_delta],
                        "summary": summary,
                        "honesty_label": HONESTY_LABEL,
                    }

        # 4. Search Phase 2: Multi-Feature Incremental Reduction
        # If no single feature can cross below threshold, greedily reduce features together
        s_accum = sequence_30x406.copy()
        accum_changes: List[Dict[str, Any]] = []

        for f in candidate_cols:
            f_idx = cols.index(f)
            orig_val = float(sequence_30x406[-1, f_idx])
            bound_meta = self.bounds.get(f, {"p01": -10.0, "median": 0.0, "p99": 10.0})
            p01 = bound_meta["p01"]
            p99 = bound_meta["p99"]
            baseline = bound_meta["median"]

            # Try step fractions for this feature while holding earlier reductions
            feature_improved = False
            for alpha in step_fractions:
                cand_val = orig_val + alpha * (baseline - orig_val)
                clipped_val = float(np.clip(cand_val, p01, p99))

                s_test = s_accum.copy()
                s_test[-1, f_idx] = clipped_val

                h_pert = float(pipeline._predict_onset_probability(s_test))
                if h_pert < best_hazard:
                    best_hazard = h_pert

                if h_pert < target_threshold:
                    # Crossed threshold!
                    feat_delta = self._format_feature_change(
                        f, orig_val, clipped_val, baseline
                    )
                    accum_changes.append(feat_delta)
                    summary = self._generate_summary(
                        accum_changes, h0, h_pert, target_threshold
                    )
                    return {
                        "status": "feasible",
                        "found": True,
                        "initial_hazard": round(h0, 4),
                        "counterfactual_hazard": round(h_pert, 4),
                        "calibrated_threshold": target_threshold,
                        "perturbed_features": accum_changes,
                        "summary": summary,
                        "honesty_label": HONESTY_LABEL,
                    }

            # If not yet below threshold, commit full reduction to baseline (within bounds)
            full_clipped = float(np.clip(baseline, p01, p99))
            s_accum[-1, f_idx] = full_clipped
            feat_delta = self._format_feature_change(f, orig_val, full_clipped, baseline)
            accum_changes.append(feat_delta)

        # 5. Search Inconclusive: Threshold could not be crossed within [p01, p99]
        summary = (
            f"No realistic minimal change found within observed data bounds [1st, 99th percentile]. "
            f"Even reducing top driving features ({', '.join(candidate_cols[:3])}) to baseline, "
            f"hazard score remained at {best_hazard:.3f} (above calibrated threshold {target_threshold:.3f})."
        )
        return {
            "status": "inconclusive",
            "found": False,
            "initial_hazard": round(h0, 4),
            "best_hazard_achieved": round(best_hazard, 4),
            "calibrated_threshold": target_threshold,
            "perturbed_features": [],
            "attempted_candidates": candidate_cols,
            "summary": summary,
            "honesty_label": HONESTY_LABEL,
        }

    def _format_feature_change(
        self,
        feature_name: str,
        orig_norm: float,
        pert_norm: float,
        baseline_norm: float,
    ) -> Dict[str, Any]:
        """Formats a feature perturbation with unit inversion and percentages."""
        from backend.predict import get_feature_category
        clean_name = feature_name.replace("_", " ").title()
        category = get_feature_category(feature_name) or "Flow Dynamics"
        unit_lbl = get_feature_unit_label(feature_name)

        raw_orig = self.invert_units(feature_name, orig_norm)
        raw_pert = self.invert_units(feature_name, pert_norm)
        raw_baseline = self.invert_units(feature_name, baseline_norm)

        bound_meta = self.bounds.get(feature_name, {"p01": 0.0, "median": 0.0, "p99": 0.0})
        p01_raw = self.invert_units(feature_name, bound_meta.get("p01", 0.0))
        p99_raw = self.invert_units(feature_name, bound_meta.get("p99", 0.0))
        range_display = f"[{p01_raw:g} – {p99_raw:g} {unit_lbl}]"

        raw_delta = raw_pert - raw_orig
        pct_change = (
            (raw_delta / (abs(raw_orig) + 1e-6)) * 100.0
            if abs(raw_orig) > 1e-6
            else 0.0
        )

        return {
            "feature": clean_name,
            "feature_id": feature_name,
            "category": category,
            "unit": unit_lbl,
            "original_value_raw": round(float(raw_orig), 2),
            "perturbed_value_raw": round(float(raw_pert), 2),
            "baseline_value_raw": round(float(raw_baseline), 2),
            "raw_delta": round(float(raw_delta), 2),
            "percent_change": round(float(pct_change), 1),
            "original_value_normalized": round(float(orig_norm), 4),
            "perturbed_value_normalized": round(float(pert_norm), 4),
            "empirical_p01_raw": round(float(p01_raw), 2),
            "empirical_p99_raw": round(float(p99_raw), 2),
            "empirical_range_display": range_display,
        }

    def _generate_summary(
        self,
        changes: List[Dict[str, Any]],
        h0: float,
        h_pert: float,
        threshold: float,
    ) -> str:
        """Constructs an analyst-ready plain-language explanation."""
        clauses = []
        for c in changes:
            feat = c["feature"]
            pct = abs(c["percent_change"])
            direction = "reducing" if c["raw_delta"] < 0 else "adjusting"
            orig = c["original_value_raw"]
            pert = c["perturbed_value_raw"]
            unit = c["unit"]
            clauses.append(f"{direction} {feat} by ~{pct:.1f}% (from {orig:g} to {pert:g} {unit})")

        clause_str = " and ".join(clauses)
        return (
            f"{clause_str.capitalize()} would have kept this window below the alert threshold "
            f"(hazard dropped from {h0:.3f} to {h_pert:.3f}, below calibrated threshold {threshold:.3f})."
        )


# Global singleton instance
_GLOBAL_ENGINE: Optional[CounterfactualEngine] = None


def get_counterfactual_engine() -> CounterfactualEngine:
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = CounterfactualEngine()
    return _GLOBAL_ENGINE
