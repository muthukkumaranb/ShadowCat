"""
SHADOWCAT Backend - Main Inference Entry Point (`predict()`)
End-to-End UCS Feature Extraction, LSTM World Model Dynamics, Hazard Forecasting,
Stage Classification, and Contract Formatting.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
import json
import hashlib
from datetime import datetime, timezone

# Resolve repository paths
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
DATA_ENG_DIR = REPO_ROOT / "data-engineering"
ML1_DIR = REPO_ROOT / "ml1"
ML2_DIR = REPO_ROOT / "ml2-full"
FRONTEND_DIR = REPO_ROOT / "frontend"
# Runtime outputs (alerts, lineage, audit chain, reports DB) are written here, never into tracked sources.
RUNTIME_DIR = Path(os.environ.get("SHADOWCAT_RUNTIME_DIR", REPO_ROOT / "runtime"))

# Ensure repository root, backend, and data-engineering are importable
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DATA_ENG_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_ENG_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from src.ucs_extractor import UCSExtractor

try:
    from backend.models import (
        LSTMGaussianWorldModel,
        ResidualLSTMClassifier,
        StageClassificationHead,
    )
except ImportError:
    from models import (
        LSTMGaussianWorldModel,
        ResidualLSTMClassifier,
        StageClassificationHead,
    )

try:
    from backend.fabric_bridge import notarize_model, notarize_alert, record_prediction_lineage
except ImportError:
    from fabric_bridge import notarize_model, notarize_alert, record_prediction_lineage

try:
    from backend.audit_chain import append_entry as append_audit_entry
except ImportError:
    from audit_chain import append_entry as append_audit_entry

try:
    from backend.conformal import SplitConformalPredictor
except ImportError:
    from conformal import SplitConformalPredictor

try:
    from backend.temporal_attribution import TemporalAttributor
except ImportError:
    from temporal_attribution import TemporalAttributor

try:
    from backend.graph_traversal import classify_endpoint_role, compute_graph_traversal
except ImportError:
    from graph_traversal import classify_endpoint_role, compute_graph_traversal

try:
    from backend.counterfactual import CounterfactualEngine, get_counterfactual_engine, HONESTY_LABEL
except ImportError:
    from counterfactual import CounterfactualEngine, get_counterfactual_engine, HONESTY_LABEL

try:
    from backend.reports_db import insert_report as _insert_report_record
except ImportError:
    try:
        from reports_db import insert_report as _insert_report_record
    except ImportError:
        _insert_report_record = None  # reports_db unavailable — skip DB writes silently



def get_feature_category(feat_name: str) -> Optional[str]:
    """
    Deterministically maps a feature name from the 406-dimensional vector
    to its accurate domain category.

    Returns None for presence masks (mask_has_*), which must be excluded
    from behavioral attribution rankings.
    Allowed categories: 'Temporal Rhythm', 'Flow Dynamics', 'Packet Header', 'Payload Stats'.
    'Graph Topology' is strictly excluded because the GNN branch was held back.
    """
    name_lower = feat_name.lower()
    # 1. Exclude presence masks
    if name_lower.startswith("mask_has_") or name_lower.startswith("mask_"):
        return None
    # 2. Payload Stats / Packet Length Distributions
    if (
        name_lower.startswith("pkt_payload_")
        or "payload" in name_lower
        or "pkt_len" in name_lower
        or "pkt_size" in name_lower
        or "seg_size" in name_lower
    ):
        return "Payload Stats"
    # 3. Packet Header & TCP Controls
    if (
        name_lower.startswith("pkt_ttl_")
        or name_lower.startswith("pkt_frag_")
        or name_lower.startswith("pkt_tcp_")
        or name_lower.startswith("pkt_port_scan_")
        or "flag" in name_lower
        or "header" in name_lower
        or "init_win" in name_lower
        or "window_size" in name_lower
        or "act_data_pkts" in name_lower
    ):
        return "Packet Header"
    # 4. Temporal Rhythm / Timing & Durations
    if (
        "iat" in name_lower
        or "idle" in name_lower
        or "active" in name_lower
        or "duration" in name_lower
    ):
        return "Temporal Rhythm"
    # 5. Flow Dynamics / Volume Counts & Rates
    if (
        "byte" in name_lower
        or "packet" in name_lower
        or "count" in name_lower
        or "rate" in name_lower
        or "per_sec" in name_lower
        or "flow_" in name_lower
        or "subflow" in name_lower
    ):
        return "Flow Dynamics"
    return "Flow Dynamics"


STAGE_MIN_CONFIDENCE = 0.5
NO_STAGE = "No stage determined"


class StackedFoldModel:
    """
    Encapsulates one fold of the trained Stacked & Temperature-Calibrated Residual LSTM.
    Matches the exact training mathematics:
        1. LR Base Model: z_base = x_scaled @ lr_coef.T + lr_intercept
        2. Residual LSTM: z_residual = fc(dropout(lstm(x_seq)))
        3. Combined Logit: z = z_base + z_residual
        4. Calibrated Probability: p = sigmoid(z / T)
    The 30-window history is projected with THIS fold's own PCA (checkpoint pca_mean /
    pca_components), as in evaluation/benchmark/reproduce_stacked_benchmark.py.
    """

    def __init__(
        self,
        fold_id: int,
        model: ResidualLSTMClassifier,
        lr_coef: np.ndarray,
        lr_intercept: float,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        temperature: float,
        target_name: str,
        checkpoint_path: str,
        held_out_episode_id: str = "",
        attack_type: str = "",
        val_predictions: Optional[List[float]] = None,
        val_targets: Optional[List[int]] = None,
        pca_mean: Optional[np.ndarray] = None,
        pca_components: Optional[np.ndarray] = None,
    ):
        self.fold_id = fold_id
        self.model = model
        self.lr_coef = np.asarray(lr_coef, dtype=np.float64).reshape(-1)
        self.lr_intercept = float(np.asarray(lr_intercept, dtype=np.float64).reshape(-1)[0])
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
        self.pca_mean = np.asarray(pca_mean, dtype=np.float64)
        self.pca_components = np.asarray(pca_components, dtype=np.float64)
        self.temperature = max(float(temperature), 0.05)
        self.target_name = target_name
        self.checkpoint_path = checkpoint_path
        self.held_out_episode_id = held_out_episode_id
        self.attack_type = attack_type
        self.val_predictions = val_predictions or []
        self.val_targets = val_targets or []

    def compute_base_logit(self, window_406: np.ndarray) -> float:
        x = np.asarray(window_406, dtype=np.float64).ravel()
        x_scaled = (x - self.scaler_mean) / self.scaler_scale
        return float(x_scaled @ self.lr_coef + self.lr_intercept)

    def project(self, seq_30x406: np.ndarray) -> np.ndarray:
        """(30, 406) history -> (30, 32) with this fold's own PCA (checkpoint pca_mean / pca_components)."""
        seq = np.asarray(seq_30x406, dtype=np.float64)
        return ((seq - self.pca_mean) @ self.pca_components.T).astype(np.float32)

    def predict_proba(self, seq_30x406: np.ndarray, device: str = "cpu") -> float:
        base_logit = self.compute_base_logit(seq_30x406[-1])
        seq_tensor = torch.as_tensor(self.project(seq_30x406)).unsqueeze(0).to(device)
        with torch.no_grad():
            b_tensor = torch.tensor([base_logit], dtype=torch.float32, device=device)
            p = self.model(seq_tensor, b_tensor, temperature=self.temperature).item()
        return float(p)


class StackedOnsetLogit(torch.nn.Module):
    """
    Differentiable view of the stacked onset ensemble for Integrated Gradients:
    forward_logits(x) with x of shape (B, 30, 406) returns the mean over folds of each fold's
    calibrated onset logit z / T, where z = LR(x[:, -1]) + LSTM residual(own-PCA(x)).
    sigmoid of each fold's term is that fold's probability; the dashboard risk is their mean.
    """

    def __init__(self, folds: List[StackedFoldModel]):
        super().__init__()
        self.models = torch.nn.ModuleList([f.model for f in folds])
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
        self.params = [
            (t(f.pca_mean), t(f.pca_components), t(f.scaler_mean), t(f.scaler_scale),
             t(f.lr_coef), float(f.lr_intercept), float(f.temperature))
            for f in folds
        ]

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for model, (pm, pc, sm, ss, coef, b, temp) in zip(self.models, self.params):
            base = ((x[:, -1, :] - sm.to(x.device)) / ss.to(x.device)) @ coef.to(x.device) + b
            seq = (x - pm.to(x.device)) @ pc.to(x.device).T
            out.append(model.forward_logits(seq, base) / temp)
        return torch.stack(out, dim=0).mean(dim=0)


class ShadowcatPipeline:
    """
    Production Inference Engine caching loaded weights and scalers.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        world_model_path: Optional[Path | str] = None,
        stage_head_path: Optional[Path | str] = None,
        hazard_head_dir: Optional[Path | str] = None,
        extractor_config_dir: Optional[Path | str] = None,
        extractor_data_dir: Optional[Path | str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Initialize UCSExtractor
        cfg_dir = str(extractor_config_dir or (DATA_ENG_DIR / "configs"))
        dat_dir = str(extractor_data_dir or (DATA_ENG_DIR / "data" / "ucs"))
        self.extractor = UCSExtractor(
            schema_version="v3.0",
            config_dir=cfg_dir,
            data_dir=dat_dir,
        )

        # 2. Load World Model (LSTM Gaussian continuous dynamics)
        # Prefers the full-packet-coverage retrain (all 3 headline attacks backed by
        # real extracted telemetry) at its own path; the original v3 checkpoint below
        # is left untouched on disk because it is hash-pinned by backend/audit_chain.json
        # (see ml1/artifacts/lstm/stage_head_v3/NOTE.md for why it is never overwritten in place).
        wm_path = world_model_path or (
            ML1_DIR / "artifacts" / "lstm" / "gaussian_next_state_best_v3_packetcov.pt"
        )
        if not Path(wm_path).exists():
            wm_path = ML1_DIR / "artifacts" / "lstm" / "gaussian_next_state_best_v3.pt"
        if not Path(wm_path).exists():
            wm_path = ML1_DIR / "artifacts" / "lstm" / "probabilistic_world_model_v3" / "gaussian_next_state_best.pt"
        if not Path(wm_path).exists():
            wm_path = ML1_DIR / "artifacts" / "lstm" / "gaussian_next_state_best_v2.pt"
        if not Path(wm_path).exists():
            wm_path = ML1_DIR / "artifacts" / "lstm" / "probabilistic_world_model_v2" / "gaussian_next_state_best.pt"

        self.world_model = LSTMGaussianWorldModel(
            input_size=406,
            hidden_size=64,
            state_dim=406,
            num_layers=1,
            dropout=0.2,
        )
        if Path(wm_path).exists():
            wm_ckpt = torch.load(wm_path, map_location=self.device, weights_only=False)
            state_dict = wm_ckpt.get("model_state_dict", wm_ckpt)
            self.world_model.load_state_dict(state_dict)
            self.world_model.to(self.device)
            self.world_model.eval()
            self.world_model_loaded = True
            
            # Notarize World Model: Fabric Primary -> SHA-256 Fallback
            fabric_ok = False
            try:
                fabric_ok = bool(notarize_model("lstm-gaussian-v3", str(wm_path), "v3.0"))
            except Exception as e:
                import logging
                logging.warning(f"Fabric Notarization Failed for model: {e}")
                fabric_ok = False

            if fabric_ok:
                import logging
                logging.info("World model notarized via: fabric")
            else:
                try:
                    from audit_chain import _load_chain
                    existing = _load_chain()
                    wm_filename = Path(wm_path).name
                    already_in_chain = any(e.get("artifact_type") == "model_checkpoint" and wm_filename in e.get("artifact_path", "") for e in existing)
                    if not already_in_chain:
                        append_audit_entry(
                            artifact_path=str(wm_path),
                            artifact_type="model_checkpoint",
                            description=f"LSTM world model ({wm_filename}) loaded into predict pipeline (SHA-256 fallback)",
                        )
                    import logging
                    logging.info("World model notarized via: sha256_fallback")
                except Exception as e:
                    import logging
                    logging.warning(f"SHA-256 fallback notarization failed for model: {e}")
        else:
            self.world_model_loaded = False

        # 2b. Canonical graph topology metadata for dynamic topology views
        self.fused_model_loaded = False
        self.fused_model = None
        dat_path = Path(dat_dir)
        edges_path = dat_path / "ucs_graph_edgelists.parquet"
        lookup_path = dat_path / "node_lookup.parquet"
        if edges_path.exists() and lookup_path.exists():
            self.ucs_edges = pd.read_parquet(edges_path)
            self.node_lookup = pd.read_parquet(lookup_path)
        else:
            self.ucs_edges = None
            self.node_lookup = None

        # 3. Load Stage Head (ATT&CK Stage Classifier)
        # Prefers the full-packet-coverage re-evaluation (paired with the world model
        # above); stage_head_v3/stage_head_best.pt and its report are left untouched
        # because stage_head_report.md is hash-pinned by backend/audit_chain.json.
        st_path = stage_head_path or (
            ML1_DIR / "artifacts" / "lstm" / "stage_head_v3" / "reeval_packetcov" / "stage_head_best.pt"
        )
        if not Path(st_path).exists():
            st_path = ML1_DIR / "artifacts" / "lstm" / "stage_head_v3" / "stage_head_best.pt"
        if not Path(st_path).exists():
            st_path = ML1_DIR / "artifacts" / "lstm" / "stage_head" / "stage_head_best.pt"

        self.stage_head = StageClassificationHead(
            state_dim=406,
            num_classes=6,
            hidden_dim=64,
            dropout=0.2,
        )
        if Path(st_path).exists():
            st_ckpt = torch.load(st_path, map_location=self.device, weights_only=False)
            self.stage_head.load_state_dict(st_ckpt)
            self.stage_head.to(self.device)
            self.stage_head.eval()
            self.stage_head_loaded = True
        else:
            self.stage_head_loaded = False

        # 4. Load Hazard Heads & Stacked Calibrated Residual Models
        # Check for Phase 1 Stacked & Temperature-Calibrated Residual Models (Winning Verified Model)
        self.use_stacked_model = True
        self.stacked_onset_models: List[StackedFoldModel] = []
        self.stacked_detection_models: List[StackedFoldModel] = []
        self.stacked_models_loaded = False

        stacked_base_dir = ML1_DIR / "artifacts" / "lstm" / "lstm_stacked"
        if not stacked_base_dir.exists():
            stacked_base_dir = ML1_DIR / "artifacts" / "lstm_stacked"
        self.stacked_base_dir = stacked_base_dir

        onset_dir = stacked_base_dir / "onset"
        detection_dir = stacked_base_dir / "detection"

        if onset_dir.exists():
            for f in sorted(onset_dir.glob("model_fold_*.pt")):
                try:
                    ckpt = torch.load(f, map_location=self.device, weights_only=False)
                    m = ResidualLSTMClassifier(input_size=32, hidden_size=64, dropout=0.20)
                    m.load_state_dict(ckpt["model_state_dict"])
                    m.to(self.device)
                    m.eval()
                    fold_obj = StackedFoldModel(
                        fold_id=ckpt.get("fold_id", 0),
                        model=m,
                        lr_coef=ckpt["lr_coef"],
                        lr_intercept=ckpt["lr_intercept"],
                        scaler_mean=ckpt["scaler_mean"],
                        scaler_scale=ckpt["scaler_scale"],
                        temperature=ckpt.get("temperature", 1.0),
                        target_name="onset",
                        checkpoint_path=str(f),
                        held_out_episode_id=ckpt.get("held_out_episode_id", ""),
                        attack_type=ckpt.get("attack_type", ""),
                        val_predictions=ckpt.get("val_predictions", []),
                        val_targets=ckpt.get("val_targets", []),
                        pca_mean=ckpt["pca_mean"],
                        pca_components=ckpt["pca_components"],
                    )
                    self.stacked_onset_models.append(fold_obj)
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to load stacked onset model from {f}: {e}")

        if detection_dir.exists():
            for f in sorted(detection_dir.glob("model_fold_*.pt")):
                try:
                    ckpt = torch.load(f, map_location=self.device, weights_only=False)
                    m = ResidualLSTMClassifier(input_size=32, hidden_size=64, dropout=0.20)
                    m.load_state_dict(ckpt["model_state_dict"])
                    m.to(self.device)
                    m.eval()
                    fold_obj = StackedFoldModel(
                        fold_id=ckpt.get("fold_id", 0),
                        model=m,
                        lr_coef=ckpt["lr_coef"],
                        lr_intercept=ckpt["lr_intercept"],
                        scaler_mean=ckpt["scaler_mean"],
                        scaler_scale=ckpt["scaler_scale"],
                        temperature=ckpt.get("temperature", 1.0),
                        target_name="detection",
                        checkpoint_path=str(f),
                        held_out_episode_id=ckpt.get("held_out_episode_id", ""),
                        attack_type=ckpt.get("attack_type", ""),
                        val_predictions=ckpt.get("val_predictions", []),
                        val_targets=ckpt.get("val_targets", []),
                        pca_mean=ckpt["pca_mean"],
                        pca_components=ckpt["pca_components"],
                    )
                    self.stacked_detection_models.append(fold_obj)
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to load stacked detection model from {f}: {e}")

        self.stacked_models_loaded = (len(self.stacked_onset_models) > 0)
        import logging
        if not self.stacked_models_loaded:
            raise RuntimeError(
                f"CRITICAL: Stacked & Calibrated Residual LSTM ensemble failed to load from {stacked_base_dir}! "
                "Silent degradation to legacy hazard heads has been completely removed."
            )
        logging.info(
            f"[DEPLOYED WINNING MODEL] Loaded Stacked & Temperature-Calibrated Residual LSTM ensemble: "
            f"{len(self.stacked_onset_models)} onset fold models, {len(self.stacked_detection_models)} detection fold models "
            f"from {stacked_base_dir}"
        )

        # hazard_head_v3 (H=1/2/5) is intentionally NOT loaded: its reported LOEO ROC-AUC
        # (0.789/0.843/0.770) only reproduces with a per-fold PCA refit on the 371b855 dataset,
        # not through models/pca_32.pkl. See evaluation/benchmark/hazard_v3_results.json.

        self.alert_threshold = 0.5

        # 4b. Load Current-Window Family / ATT&CK Stage Classifier (Phase B)
        self.family_classifier_models: List[Dict[str, Any]] = []
        family_dir = ML1_DIR / "artifacts" / "family_classifier"
        if family_dir.exists():
            import joblib
            for f in sorted(family_dir.glob("model_fold_*.joblib")):
                try:
                    art = joblib.load(f)
                    self.family_classifier_models.append(art)
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to load family classifier from {f}: {e}")

        # Load 5% FPR detection threshold from stacked_benchmark_results.json
        self.detection_threshold_5pct_fpr = 0.4014016389846802
        bench_path = REPO_ROOT / "evaluation" / "benchmark" / "stacked_benchmark_results.json"
        if bench_path.exists():
            try:
                with open(bench_path, "r", encoding="utf-8") as f:
                    bench_data = json.load(f)
                self.detection_threshold_5pct_fpr = float(
                    bench_data["detection"]["matched_fpr_5pct"]["stacked_ensemble"]["threshold"]
                )
            except Exception as e:
                import logging
                logging.warning(f"Failed to load detection 5% FPR threshold from {bench_path}: {e}")

        # Load family tactic macro F1 from family_results.json
        self.family_loeo_macro_f1 = 0.6402
        fam_res_path = REPO_ROOT / "evaluation" / "family" / "family_results.json"
        if fam_res_path.exists():
            try:
                with open(fam_res_path, "r", encoding="utf-8") as f:
                    fam_res_data = json.load(f)
                self.family_loeo_macro_f1 = float(
                    fam_res_data["loeo_evaluation"]["macro_f1_tactic"]
                )
            except Exception as e:
                import logging
                logging.warning(f"Failed to load family results from {fam_res_path}: {e}")

        # Load mapping from family_to_attck.yaml
        self.family_to_attck: Dict[str, Any] = {}
        mapping_path = DATA_ENG_DIR / "configs" / "family_to_attck.yaml"
        if mapping_path.exists():
            try:
                with open(mapping_path, "r", encoding="utf-8") as f:
                    mcfg = yaml.safe_load(f)
                self.family_to_attck = mcfg.get("families", {})
            except Exception as e:
                import logging
                logging.warning(f"Failed to load family_to_attck mapping: {e}")


        # 5. No shared PCA: each stacked fold projects its history with its own checkpoint PCA
        # (pca_32_stacked.pkl is no longer used; see docs/demo_check/STACKED_PCA_NOTE.md).
        if str(ML1_DIR) not in sys.path:
            sys.path.insert(0, str(ML1_DIR))

        # 6. Initialize Split Conformal Predictor with ACI and Joint Coverage
        self.conformal_coverage = 0.90
        # ACI gamma per-horizon: larger for K=4-5 (exploratory)
        aci_gammas = [0.01, 0.01, 0.01, 0.05, 0.05]
        # ACI is opt-in (disabled by default to preserve static calibration behavior)
        use_aci = os.environ.get("SHADOWCAT_ACI_MODE", "false").lower() == "true"
        self.conformal_predictor = SplitConformalPredictor(
            coverage=self.conformal_coverage,
            aci_mode=use_aci,
            aci_gammas=aci_gammas
        )
        self.conformal_cal_source = None

    def _get_conformal_calibration_data(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Retrieves real empirical validation predictions and true binary targets
        pooled across all 37 LOEO folds from training split with zero test leakage.
        """
        stacked_base_dir = getattr(self, "stacked_base_dir", ML1_DIR / "artifacts" / "lstm" / "lstm_stacked")
        candidate_files = [
            stacked_base_dir / "conformal_calibration_residuals.json",
            stacked_base_dir / "onset" / "val_residuals_pooled.json",
            stacked_base_dir / "detection" / "val_residuals_pooled.json",
        ]
        for c_file in candidate_files:
            if c_file.exists():
                try:
                    with open(c_file, "r", encoding="utf-8") as rf:
                        meta = json.load(rf)
                    if "onset" in meta and "val_predictions" in meta["onset"]:
                        p = np.asarray(meta["onset"]["val_predictions"], dtype=float)
                        y = np.asarray(meta["onset"]["val_targets"], dtype=float)
                    elif "val_predictions" in meta:
                        p = np.asarray(meta["val_predictions"], dtype=float)
                        y = np.asarray(meta["val_targets"], dtype=float)
                    else:
                        continue
                    if len(p) > 0 and len(p) == len(y):
                        source_desc = f"Pooled validation residuals across 37 LOEO folds from {c_file.name}"
                        return p, y, source_desc
                except Exception as e:
                    import logging
                    logging.warning(f"Error reading conformal calibration file {c_file}: {e}")

        # Pool directly from loaded fold checkpoints if master json not yet compiled
        pooled_p = []
        pooled_y = []
        models_to_check = self.stacked_onset_models or self.stacked_detection_models
        for fold in models_to_check:
            if hasattr(fold, "val_predictions") and hasattr(fold, "val_targets") and len(fold.val_predictions) > 0:
                pooled_p.extend(fold.val_predictions)
                pooled_y.extend(fold.val_targets)
        if len(pooled_p) > 0 and len(pooled_p) == len(pooled_y):
            source_desc = f"Pooled validation residuals across {len(models_to_check)} loaded LOEO fold checkpoints"
            return np.array(pooled_p, dtype=float), np.array(pooled_y, dtype=float), source_desc

        raise RuntimeError(
            "CRITICAL: No empirical validation residuals found for conformal calibration! "
            "Placeholder data is strictly prohibited per audit requirements."
        )

    def _predict_onset_probability(self, sequence_30x406: np.ndarray) -> float:
        """
        P(attack within the next 5 minutes): mean over the 37 stacked onset fold models
        (target = future_attack_label, i.e. an attack window within t+1..t+5).
        Each fold uses its own checkpoint PCA and LR scaler.
        """
        probs = [m.predict_proba(sequence_30x406, self.device) for m in self.stacked_onset_models]
        return float(np.mean(probs))

    def _predict_detection_ensemble(self, sequence_30x406: np.ndarray) -> Optional[float]:
        """
        Current-window attack probability: mean over the 37 stacked detection fold models,
        each using its own checkpoint PCA and LR scaler.
        """
        if not (self.use_stacked_model and len(self.stacked_detection_models) > 0):
            return None
        try:
            probs = [m.predict_proba(sequence_30x406, self.device) for m in self.stacked_detection_models]
            return float(np.mean(probs))
        except Exception as e:
            import logging
            logging.warning(f"Stacked detection ensemble error: {e}")
            return None

    def _predict_current_stage(
        self,
        window_406: np.ndarray,
        detection_prob: Optional[float],
    ) -> Dict[str, Any]:
        """
        Current-window ATT&CK stage prediction (Phase B).
        Gated by stacked detection probability >= 5% FPR threshold from stacked_benchmark_results.json.
        """
        det_p = float(detection_prob) if detection_prob is not None else 0.0
        source_str = f"family classifier (current window, LOEO macro-F1 = {self.family_loeo_macro_f1:.4f})"

        if det_p < self.detection_threshold_5pct_fpr:
            return {
                "current_stage": "No attack detected",
                "current_stage_confidence": None,
                "current_stage_source": source_str,
                "current_stage_family": None,
                "current_stage_tactic_id": None,
                "current_stage_url": None,
                "is_attack_detected": False,
            }

        if not self.family_classifier_models:
            raise RuntimeError("CRITICAL: Family classifier models failed to load! Silent fallback has been removed.")

        # Average probabilities across loaded fold models
        all_classes = set()
        for art in self.family_classifier_models:
            all_classes.update(art["classes"])
        canonical_classes = sorted(list(all_classes))
        class_to_idx = {c: i for i, c in enumerate(canonical_classes)}

        x_raw = np.asarray(window_406, dtype=np.float64).reshape(1, -1)
        ensemble_probs = np.zeros(len(canonical_classes), dtype=np.float64)

        for art in self.family_classifier_models:
            clf = art["model"]
            scaler = art["scaler"]
            classes = art["classes"]
            x_scaled = scaler.transform(x_raw)
            p = clf.predict_proba(x_scaled)[0]
            for local_idx, c in enumerate(classes):
                ensemble_probs[class_to_idx[c]] += p[local_idx]

        ensemble_probs /= len(self.family_classifier_models)
        best_idx = int(np.argmax(ensemble_probs))
        pred_family = canonical_classes[best_idx]
        confidence = float(ensemble_probs[best_idx])

        family_meta = self.family_to_attck.get(pred_family, {})
        tactic_name = family_meta.get("tactic_name") or pred_family
        tactic_id = family_meta.get("tactic_id")
        url = family_meta.get("url")

        return {
            "current_stage": tactic_name,
            "current_stage_confidence": round(float(confidence), 4),
            "current_stage_source": source_str,
            "current_stage_family": pred_family,
            "current_stage_tactic_id": tactic_id,
            "current_stage_url": url,
            "is_attack_detected": True,
        }

    def predict(
        self,
        raw_input: pd.DataFrame,
        source_type: str = "csv",
    ) -> Dict[str, Any]:
        """
        Main inference entry point.

        Steps:
        1. raw_input -> UCSExtractor.extract_model_tensor() -> (N, 406) float32 tensor.
           (Normalization is ALREADY applied internally by UCSExtractor).
        2. Sequence slicing with L=30 lookback, handling causal warm-up boundary.
        3. LSTM World Model continuous dynamics -> latent z(t) [64-dim], predicted state mean and std.
        4. Bypass fusion: z'(t) = z(t) (GNN held back per verified decision).
        5. Stacked onset ensemble -> P(attack within the next 5 minutes) (single probability).
        6. Stage Head -> MITRE ATT&CK stage label and tactic ID.
        7. Novelty / deviation score: observed vs predicted.
        8. Format complete payload adhering strictly to INTERFACE.md.
        """
        if raw_input is None or len(raw_input) == 0:
            raise ValueError("Input DataFrame is empty or None.")

        # Step 1: Feature Extraction
        window_df = self.extractor.extract(raw_input, source_type=source_type)
        model_tensor = self.extractor.extract_model_tensor(raw_input, source_type=source_type)
        n_windows = len(model_tensor)

        # Basic metadata
        window_id = str(window_df["window_id"].iloc[-1]) if "window_id" in window_df.columns else "WIN-20260910-01"
        window_start = str(window_df["window_start_utc"].iloc[-1]) if "window_start_utc" in window_df.columns else "2026-09-10 12:00:00 UTC"

        # Step 2: Sequence Slicing with L=30 Lookback
        lookback = 30
        if n_windows < lookback:
            # Handle causal warm-up: Pad or note warm-up status
            # When fewer than 30 windows, repeat earliest window for causal warm-up pad
            pad_count = lookback - n_windows
            pad_tensor = np.repeat(model_tensor[:1], pad_count, axis=0)
            seq_30x406 = np.concatenate([pad_count * [model_tensor[0]] if pad_count > 0 else [], model_tensor], axis=0)
            seq_30x406 = np.pad(model_tensor, ((pad_count, 0), (0, 0)), mode='edge')
            is_warmup = True
        else:
            seq_30x406 = model_tensor[-lookback:].copy()  # Exact last 30 windows
            is_warmup = False

        seq_30x406 = np.ascontiguousarray(seq_30x406, dtype=np.float32)
        seq_tensor = torch.as_tensor(seq_30x406, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Step 3 & 4: World Model Forward Pass + z'(t) = z(t) Pass-Through
        with torch.no_grad():
            pred_mean_t1, pred_std_t1 = self.world_model(seq_tensor)
            z_t = self.world_model.extract_latent_z(seq_tensor)  # 64-dim latent

        # Network Topology Construction (real host & interaction topology independent of neural model)
        nodes_info = []
        edges_info = []
        graph_built = False
        try:
            has_ip_flows = isinstance(raw_input, pd.DataFrame) and any(
                c in raw_input.columns for c in ("Src IP", "src_ip", "Dst IP", "dst_ip")
            )

            # Path B (Preferred for raw flows): Live Dynamic Topology Construction from input flows (GraphTopologyBuilder)
            if has_ip_flows and len(raw_input) > 0:
                from src.graph_builder import GraphTopologyBuilder
                gtb = GraphTopologyBuilder()
                flows_df = raw_input.copy()
                if "timestamp_utc" not in flows_df.columns:
                    if "Timestamp" in flows_df.columns:
                        flows_df["timestamp_utc"] = pd.to_datetime(flows_df["Timestamp"], errors="coerce", utc=True)
                    elif "timestamp" in flows_df.columns:
                        flows_df["timestamp_utc"] = pd.to_datetime(flows_df["timestamp"], errors="coerce", utc=True)
                    else:
                        base_ref = pd.Timestamp("2026-09-18 14:00:00", tz="UTC")
                        flows_df["timestamp_utc"] = [base_ref + pd.Timedelta(seconds=i*15) for i in range(len(flows_df))]
                if flows_df["timestamp_utc"].isna().any():
                    base_ref = pd.Timestamp("2026-09-18 14:00:00", tz="UTC")
                    flows_df["timestamp_utc"] = flows_df["timestamp_utc"].fillna(
                        pd.Series([base_ref + pd.Timedelta(seconds=i*15) for i in range(len(flows_df))])
                    )
                
                edge_df, node_lookup_df, _ = gtb.build_window_edge_lists(flows_df, interval_sec=60)
                if len(node_lookup_df) > 0 and len(edge_df) > 0:
                    id_to_ep = dict(zip(node_lookup_df["node_id"], node_lookup_df["endpoint_identifier"]))
                    edge_subset = edge_df.head(45)
                    active_nids = sorted(set(edge_subset["src_node_id"]).union(set(edge_subset["dst_node_id"])))
                    nodes_info = []
                    for nid in active_nids:
                        ep = id_to_ep.get(nid, f"node-{nid}")
                        ep_str = str(ep)
                        role = classify_endpoint_role(ep_str)
                        nodes_info.append({
                            "id": ep_str,
                            "name": ep_str,
                            "ip": ep_str,
                            "role": role,
                        })

                    edges_info = []
                    for row in edge_subset.itertuples(index=False):
                        src_ep = str(id_to_ep.get(row.src_node_id, row.src_node_id))
                        dst_ep = str(id_to_ep.get(row.dst_node_id, row.dst_node_id))
                        edges_info.append({
                            "source": src_ep,
                            "target": dst_ep,
                            "flow_count": int(row.flow_count),
                            "byte_count": float(row.byte_count_sum),
                            "packet_count": float(row.packet_count_sum),
                            "label": f"{int(row.flow_count)} flows ({float(row.byte_count_sum)/1024:.1f} KB)",
                        })
                    graph_built = True

            # Path A: Pre-computed canonical UCS graph edgelists (for windowed parquet datasets)
            if not graph_built and getattr(self, 'ucs_edges', None) is not None and getattr(self, 'node_lookup', None) is not None:
                w_df = window_df.tail(1).copy()
                w_id = w_df["window_id"].iloc[0]
                window_edges = self.ucs_edges[self.ucs_edges["window_id"] == w_id]
                if len(window_edges) > 0:
                    id_to_ep = dict(zip(self.node_lookup["node_id"], self.node_lookup["endpoint_identifier"]))
                    edge_subset = window_edges.head(45)
                    active_nids = sorted(set(edge_subset["src_node_id"]).union(set(edge_subset["dst_node_id"])))
                    nodes_info = []
                    for nid in active_nids:
                        ep_str = str(id_to_ep.get(nid, f"node-{nid}"))
                        role = classify_endpoint_role(ep_str)
                        nodes_info.append({
                            "id": ep_str,
                            "name": ep_str,
                            "ip": ep_str,
                            "role": role,
                        })
                    edges_info = []
                    for row in edge_subset.itertuples(index=False):
                        src_ep = str(id_to_ep.get(row.src_node_id, row.src_node_id))
                        dst_ep = str(id_to_ep.get(row.dst_node_id, row.dst_node_id))
                        fc = int(getattr(row, "flow_count", 1))
                        bc = float(getattr(row, "byte_count_sum", 0.0))
                        pk = float(getattr(row, "packet_count_sum", 0.0))
                        edges_info.append({
                            "source": src_ep,
                            "target": dst_ep,
                            "flow_count": fc,
                            "byte_count": bc,
                            "packet_count": pk,
                            "label": f"{fc} flows ({bc/1024:.1f} KB)" if bc > 0 else f"{fc} flows",
                        })
                    graph_built = True
        except Exception as e:
            import logging
            logging.warning(f"Graph topology construction failed at runtime: {e}")

        graph_topology = {
            "status": "active" if graph_built else "empty",
            "nodes_count": len(nodes_info),
            "edges_count": len(edges_info),
            "graph_nodes": nodes_info,
            "graph_edges": edges_info,
        }

        # 5-step forward simulation (Rollout K=1..5)
        rollout_means = []
        rollout_stds = []
        current_seq = seq_tensor.clone()

        with torch.no_grad():
            for k in range(5):
                k_mean, k_std = self.world_model(current_seq)
                rollout_means.append(k_mean.cpu().numpy()[0])
                rollout_stds.append(k_std.cpu().numpy()[0])
                # Autoregressive update: slide window and append forecast mean
                next_step = k_mean.unsqueeze(1)  # (1, 1, 406)
                current_seq = torch.cat([current_seq[:, 1:, :], next_step], dim=1)

        # Novelty / Forecast Deviation Score from World Model sequence transition
        obs_state = model_tensor[-1]
        pred_state = rollout_means[0]
        feature_deviations = np.abs(obs_state - pred_state) / (rollout_stds[0] + 1e-4)
        raw_novelty = float(np.mean(feature_deviations))
        novelty_score = float(1.0 / (1.0 + np.exp(-0.5 * (raw_novelty - 1.0))))
        novelty_score = float(np.clip(novelty_score, 0.05, 0.95))

        # Step 5: Onset probability from the 37-fold stacked onset ensemble.
        # Target = future_attack_label: an attack window within t+1..t+5 (1-minute windows).
        # This is the only risk number the dashboard shows; no per-horizon values are produced.
        p_onset = self._predict_onset_probability(seq_30x406)

        # Step 6: Stage Head Classification on Predicted Future State S_hat(t+1)
        # Grounded against the real MITRE ATT&CK STIX 2.1 knowledge base.
        # The model's real predicted stage and confidence are always reported honestly,
        # for every class — no scripted fallback or heuristic substitution.
        stage_names = []
        tactic_ids = []
        mitre_details = []
        heuristic_progression_flags = []
        attribution_sources = []

        from backend.mitre_kb import get_mitre_kb
        kb = get_mitre_kb()

        with torch.no_grad():
            for k in range(5):
                s_hat_k = torch.as_tensor(rollout_means[k], dtype=torch.float32).unsqueeze(0).to(self.device)
                stage_logits = self.stage_head(s_hat_k)
                stage_probs = torch.softmax(stage_logits, dim=-1).cpu().numpy()[0]
                stage_idx = int(np.argmax(stage_probs))
                pred_stage = StageClassificationHead.STAGE_CLASSES[stage_idx]
                pred_conf = float(stage_probs[stage_idx])

                is_heuristic = False
                attr_source = f"StageClassificationHead (top class {pred_stage}, p={pred_conf:.2f})"

                # A stage is shown only if the head is confident (p > 0.5) in a real ATT&CK
                # class. "Unknown/Other" has no ATT&CK ID and is never mapped to one.
                if pred_conf > STAGE_MIN_CONFIDENCE and pred_stage != "Unknown/Other":
                    selected_stage = pred_stage
                    stage_info = kb.resolve_stage(selected_stage)
                    stage_info["likely_next_techniques"] = kb.predict_likely_next_techniques(stage_info["technique_id"])
                else:
                    selected_stage = NO_STAGE
                    stage_info = {
                        "tactic_id": None, "tactic_name": None, "shortname": None, "description": None,
                        "url": None, "technique_id": None, "technique_name": None,
                        "technique_full_name": None, "technique_description": None,
                        "technique_url": None, "is_active_attack": False,
                        "likely_next_techniques": [],
                    }
                stage_info["is_heuristic_progression"] = is_heuristic
                stage_info["attribution_source"] = attr_source
                stage_info["stage_confidence"] = pred_conf
                stage_info["stage_head_top_class"] = pred_stage

                stage_names.append(selected_stage)
                tactic_ids.append(stage_info["tactic_id"])
                mitre_details.append(stage_info)
                heuristic_progression_flags.append(is_heuristic)
                attribution_sources.append(attr_source)

        # Split conformal interval for the onset probability, calibrated on the pooled
        # validation predictions of the same 37 onset fold models (no test windows).
        if not self.conformal_predictor.is_calibrated:
            real_val_preds, real_val_targets, self.conformal_cal_source = self._get_conformal_calibration_data()
            self.conformal_predictor.calibrate(real_val_preds, real_val_targets)
        c_lb, c_ub = self.conformal_predictor.predict_interval(p_onset, horizon_idx=0)
        c_lb_j, c_ub_j = self.conformal_predictor.predict_interval_joint(p_onset, horizon_idx=0)
        conformal_interval = [round(c_lb, 4), round(c_ub, 4)]
        conformal_interval_joint = [round(c_lb_j, 4), round(c_ub_j, 4)]
        onset_alert = bool(p_onset >= self.alert_threshold)

        # Step 8 & 9: Integrated Gradients on the stacked ONSET logit (the number shown as risk).
        # Target: mean over the 37 onset folds of the calibrated logit z/T; baseline: all-zero
        # (scaled) 30-window history; 15 interpolation steps.
        model_cols = UCSExtractor.MODEL_INPUT_COLUMNS
        attributions = []
        try:
            ig = TemporalAttributor(steps=15).attribute(
                model=StackedOnsetLogit(self.stacked_onset_models),
                x_sequence=seq_30x406,
                feature_names=model_cols,
                device=self.device,
            )
            behavioural = [f for f in ig["feature_totals"] if get_feature_category(f["feature"]) is not None]
            for f in sorted(behavioural, key=lambda f: f["abs_share"], reverse=True)[:5]:
                attributions.append({
                    "feature": f["feature"],
                    "contribution": round(f["abs_share"], 4),
                    "signed_attribution": round(f["signed_sum"], 4),
                    "category": get_feature_category(f["feature"]),
                    "method": "Integrated Gradients",
                })
            temporal_attributions = {
                "timestep_attributions": ig["timesteps"],
                "top_temporal_events": ig["top_temporal_events"][:5],
                "attribution_sum": round(ig["attribution_sum"], 4),
                "lookback_windows": lookback,
                "engine": "Integrated Gradients",
                "target": "mean calibrated onset logit of the 37 stacked fold models",
                "baseline": "all-zero scaled 30-window history",
                "contribution_definition": "share of total |attribution| across all 406 inputs and 30 windows",
            }
        except Exception as e:
            temporal_attributions = {"engine": "Integrated Gradients", "error": str(e)}

        # Step 10: Flows of the last window ranked by the onset LR stage (see _extract_flagged_flows)
        flagged_flows = self._extract_flagged_flows(raw_input, source_type)

        # Window-level contributions to the onset LR-stage logit (coef x scaled value, fold mean)
        lr_c = self._lr_contributions(seq_30x406[-1])
        lr_order = [i for i in np.argsort(np.abs(lr_c))[::-1] if get_feature_category(model_cols[i]) is not None][:10]
        lr_contributions = [
            {"feature": model_cols[i], "contribution": round(float(lr_c[i]), 4),
             "scaled_value": round(float(seq_30x406[-1][i]), 4)}
            for i in lr_order
        ]

        # Step 10b: Real Graph-Propagation Traversal over Actual Edge Data
        from backend.graph_traversal import compute_graph_traversal
        graph_traversal = compute_graph_traversal(
            graph_nodes=nodes_info,
            graph_edges=edges_info,
            flagged_flows=flagged_flows,
            max_k=5,
        )

        # Step 10c: Explanatory Counterfactuals (Bounded Perturbation Search)
        counterfactual_result = {}
        try:
            cf_engine = get_counterfactual_engine()
            top_cand_names = [a.get("feature", "").lower().replace(" ", "_") for a in attributions[:5]]
            counterfactual_result = cf_engine.search(
                pipeline=self,
                sequence_30x406=seq_30x406,
                top_features=top_cand_names,
                threshold=self.alert_threshold,
                max_features=5,
            )
        except Exception as e:
            import logging
            logging.warning(f"Explanatory counterfactual generation failed: {e}")
            counterfactual_result = {
                "status": "error",
                "found": False,
                "summary": f"Counterfactual generation skipped: {e}",
                "honesty_label": HONESTY_LABEL,
            }

        # Step 11: Construct Output Payload per INTERFACE.md
        # Detection probability for current-window gating
        p_detection = (
            float(self._predict_detection_ensemble(seq_30x406))
            if (self.use_stacked_model and self.stacked_models_loaded and len(self.stacked_detection_models) > 0)
            else None
        )
        current_stage_info = self._predict_current_stage(seq_30x406[-1], p_detection)

        # Stage fields below come from the stage head applied to the world-model rollout
        # S_hat(t+k), k=1..5. They carry no probability of attack.
        rollout_labels = [f"t+{k} (world-model rollout S_hat(t+{k}))" for k in range(1, 6)]

        forecast_trajectory = {
            "window_id": window_id,
            "onset_probability": round(float(p_onset), 6),
            "onset_probability_label": "P(attack within the next 5 minutes)",
            "onset_model": (
                "Stacked calibrated residual LSTM, mean of "
                f"{len(self.stacked_onset_models)} LOEO fold models (target: attack window in t+1..t+5)"
            ),
            # Current-window ATT&CK stage (Phase B)
            "current_stage": current_stage_info["current_stage"],
            "current_stage_confidence": current_stage_info["current_stage_confidence"],
            "current_stage_source": current_stage_info["current_stage_source"],
            "current_stage_family": current_stage_info["current_stage_family"],
            "current_stage_tactic_id": current_stage_info["current_stage_tactic_id"],
            "current_stage_url": current_stage_info["current_stage_url"],
            # World-model rollout stage prediction (renamed to rollout_stage_k per Phase B)
            "rollout_stage_k": stage_names,
            "rollout_stage_k_note": "world-model rollout, not validated",
            # Single-element list kept for consumers that read risk[0]; there is no per-horizon risk.
            "risk": [round(float(p_onset), 6)],
            "onset_alert": onset_alert,
            "alert_threshold": self.alert_threshold,
            "rollout_labels": rollout_labels,
            "stage": stage_names,
            "tactic_id": tactic_ids,
            "mitre_details": mitre_details,
            "technique_id": [m["technique_id"] for m in mitre_details],
            "technique_name": [m["technique_name"] for m in mitre_details],
            "technique_full_name": [m["technique_full_name"] for m in mitre_details],
            "technique_description": [m["technique_description"] for m in mitre_details],
            "technique_url": [m["technique_url"] for m in mitre_details],
            "tactic_url": [m["url"] for m in mitre_details],
            "is_heuristic_progression": heuristic_progression_flags,
            "stage_attribution_source": attribution_sources,
            "likely_next_techniques": [m["likely_next_techniques"] for m in mitre_details],
            "stage_confidence": [m["stage_confidence"] for m in mitre_details],
            "stage_head_top_class": [m["stage_head_top_class"] for m in mitre_details],
            "stage_rule": f"stage shown only if stage-head top probability > {STAGE_MIN_CONFIDENCE} and class is not Unknown/Other",
            "conformal_interval": conformal_interval,
            "conformal_interval_joint": conformal_interval_joint,
            "conformal_coverage": self.conformal_coverage,
            "conformal_coverage_joint": 1.0 - self.conformal_predictor.alpha_joint,
            "conformal_calibration_source": self.conformal_cal_source,
            "conformal_sample_size": self.conformal_predictor.calibration_sample_size,
            "conformal_quantile": round(float(self.conformal_predictor.calibrated_quantile), 4),
            "model_architecture": "Stacked & Calibrated Residual LSTM",
            "active_model_folds": len(self.stacked_onset_models),
            "checkpoint_dir": (self.stacked_base_dir.relative_to(REPO_ROOT) / "onset").as_posix(),
            "is_mock": False,
        }

        raw_steps = []
        for k in range(5):
            raw_steps.append({
                "step": k + 1,
                "horizon": rollout_labels[k],
                "stage": stage_names[k],
                "tactic_id": tactic_ids[k],
                "tactic_name": mitre_details[k]["tactic_name"],
                "tactic_description": mitre_details[k]["description"],
                "tactic_url": mitre_details[k]["url"],
                "technique_id": mitre_details[k]["technique_id"],
                "technique_name": mitre_details[k]["technique_name"],
                "technique_full_name": mitre_details[k]["technique_full_name"],
                "technique_description": mitre_details[k]["technique_description"],
                "technique_url": mitre_details[k]["technique_url"],
                "is_heuristic_progression": heuristic_progression_flags[k],
                "attribution_source": attribution_sources[k],
            })
        forecast_trajectory["raw_steps"] = raw_steps

        # Input statistics: only values computed from the input; None when the column is absent
        cols = raw_input.columns if isinstance(raw_input, pd.DataFrame) else []
        active_endpoints = None
        ip_cols = [c for c in ("Src IP", "src_ip", "Dst IP", "dst_ip") if c in cols]
        if ip_cols:
            active_endpoints = int(pd.unique(pd.concat([raw_input[c] for c in ip_cols]).dropna()).size)

        syn_ack_ratio = None
        if "SYN Flag Cnt" in cols and "ACK Flag Cnt" in cols:
            syn_c = float(pd.to_numeric(raw_input["SYN Flag Cnt"], errors="coerce").sum())
            ack_c = float(pd.to_numeric(raw_input["ACK Flag Cnt"], errors="coerce").sum())
            syn_ack_ratio = round(syn_c / ack_c, 4) if ack_c > 0 else None

        packets_analyzed = None
        if "Tot Fwd Pkts" in cols and "Tot Bwd Pkts" in cols:
            packets_analyzed = int(pd.to_numeric(raw_input["Tot Fwd Pkts"], errors="coerce").sum()
                                   + pd.to_numeric(raw_input["Tot Bwd Pkts"], errors="coerce").sum())

        novelty_payload = {
            "state_id": "S(t)",
            # sigmoid(0.5 * (mean |S(t) - S_hat(t)| / sigma - 1)), clipped to [0.05, 0.95]
            "novelty_score": round(novelty_score, 2),
            "novelty_definition": "world-model deviation: sigmoid of mean |S(t) - S_hat(t)| / sigma over 406 features",
            "active_endpoints": active_endpoints,
            "syn_ack_ratio": syn_ack_ratio,
            "flows_analyzed": len(raw_input),
            "input_rows_are": "UCS windows" if source_type == "windows" else "flows",
            "packets_analyzed": packets_analyzed,
            "is_mock": False,
        }

        analysis_metadata = {
            "source": f"{source_type} input",
            "window": "1-minute windows, 30-window lookback",
            "window_duration": "60s",
            "duration_sec": 60,
            "lookback_windows": 30,
            "timestamp": window_start,
            "is_mock": False,
        }

        payload = {
            "window_id": window_id,
            "is_warmup": is_warmup,
            "analysis_metadata": analysis_metadata,
            "forecast_trajectory": forecast_trajectory,
            "novelty_score": novelty_payload,
            "attributions": attributions,
            "flagged_flows": flagged_flows,
            "lr_contributions": lr_contributions,
            "lr_contribution_sum": round(float(lr_c.sum()), 4),
            "current_stage": current_stage_info["current_stage"],
            "current_stage_confidence": current_stage_info["current_stage_confidence"],
            "current_stage_source": current_stage_info["current_stage_source"],
            "current_stage_family": current_stage_info["current_stage_family"],
            "current_stage_tactic_id": current_stage_info["current_stage_tactic_id"],
            "current_stage_url": current_stage_info["current_stage_url"],
            "rollout_stage_k": stage_names,
            "rollout_stage_k_note": "world-model rollout, not validated",
            "stage_predictions": stage_names,
            "risk_scores": [round(float(p_onset), 6)],
            "graph_topology": graph_topology,
            "graph_traversal": graph_traversal,
            "temporal_attributions": temporal_attributions,
            "conformal_forecast": {
                "coverage": self.conformal_coverage,
                "coverage_joint": 1.0 - self.conformal_predictor.alpha_joint,
                "intervals": [conformal_interval],
                "intervals_joint": [conformal_interval_joint],
                "point_estimates": [round(float(p_onset), 6)],
                "method": "Split Conformal Prediction (37-Fold LOEO Validation-Calibrated)",
                "sample_size": self.conformal_predictor.calibration_sample_size,
                "quantile": round(float(self.conformal_predictor.calibrated_quantile), 4),
                "calibration_source": self.conformal_cal_source,
                "aci_mode": self.conformal_predictor.aci_mode,
            },
            "detection_probability": round(p_detection, 4) if p_detection is not None else None,
            "detection_alert": bool(p_detection >= 0.5) if p_detection is not None else False,
            "counterfactual": counterfactual_result,
        }

        # Fabric Notarization Hook with SHA-256 Fallback
        if onset_alert:
            import hashlib
            import logging
            max_hazard_val = float(p_onset)
            alert_str = f"{window_id}-{window_start}-{max_hazard_val}"
            alert_hash = hashlib.sha256(alert_str.encode()).hexdigest()[:16]
            severity = "HIGH" if max_hazard_val > 0.8 else "MEDIUM"

            fabric_ok = False
            try:
                fabric_ok = bool(notarize_alert(alert_hash, severity, window_start))
            except Exception as e:
                logging.warning(f"Fabric Notarization Failed for alert: {e}")
                fabric_ok = False

            if fabric_ok:
                logging.info(f"Alert {alert_hash} notarized via: fabric")
                payload["notarization_mechanism"] = "fabric"
                payload["notarized_via"] = "fabric"
                # Persist alert to reports DB (fabric path)
                try:
                    if _insert_report_record is not None:
                        _fabric_alert_record = {
                            "alert_hash": alert_hash,
                            "window_id": str(window_id),
                            "window_start": str(window_start),
                            "max_risk": round(float(max_hazard_val), 4),
                            "severity": severity,
                            "notarized_via": "fabric",
                        }
                        _insert_report_record("alert", _fabric_alert_record)
                except Exception as _db_err:
                    logging.warning(f"reports_db alert insert failed (fabric path): {_db_err}")
            else:
                # Fall back to SHA-256 hash-chaining in backend/audit_chain.py
                try:
                    runtime_dir = Path(os.environ.get("SHADOWCAT_RUNTIME_DIR", RUNTIME_DIR))
                    alerts_dir = runtime_dir / "alerts"
                    alerts_dir.mkdir(parents=True, exist_ok=True)
                    alert_file = alerts_dir / f"alert_{alert_hash}.json"
                    alert_record = {
                        "alert_hash": alert_hash,
                        "window_id": str(window_id),
                        "window_start": str(window_start),
                        "max_risk": round(float(max_hazard_val), 4),
                        "severity": severity,
                        "notarized_via": "sha256_fallback",
                    }
                    with open(alert_file, "w", encoding="utf-8") as f:
                        json.dump(alert_record, f, indent=2)

                    # Persist alert to reports DB (sha256_fallback path)
                    try:
                        if _insert_report_record is not None:
                            _insert_report_record("alert", alert_record)
                    except Exception as _db_err:
                        logging.warning(f"reports_db alert insert failed (sha256_fallback path): {_db_err}")

                    from audit_chain import _load_chain
                    existing = _load_chain()
                    already_in_chain = any(alert_hash in e.get("description", "") for e in existing)
                    if not already_in_chain:
                        append_audit_entry(
                            artifact_path=str(alert_file),
                            artifact_type="hazard_alert",
                            description=f"Hazard alert {alert_hash} [{severity}] (SHA-256 fallback)",
                        )
                    logging.info(f"Alert {alert_hash} notarized via: sha256_fallback")
                    payload["notarization_mechanism"] = "sha256_fallback"
                    payload["notarized_via"] = "sha256_fallback"
                except Exception as e:
                    logging.warning(f"SHA-256 fallback notarization failed for alert: {e}")
                    payload["notarization_mechanism"] = "failed"
                    payload["notarized_via"] = "failed"
        else:
            payload["notarization_mechanism"] = "none"
            payload["notarized_via"] = "none"

        # Real 4-Stage Pipeline Lineage (Atomic Ledger Record + Chaincode Auto-Trigger)
        # Stage 1: raw_data_hash (SHA-256 of raw input data)
        # Stage 2: feature_hash (SHA-256 of 406-dim canonical normalized feature sequence)
        #          RATIONALE: We select the 406-dim base engineered feature sequence (seq_30x406)
        #          because it is the foundational, loss-free engineered representation consumed by the
        #          primary pipeline (World Model, Stage Head, Novelty, Attributor), guaranteeing complete
        #          forensic auditability without PCA projection loss.
        # Stage 3: model_id ("lstm_world_model_v4", referencing the on-chain model provenance record)
        # Stage 4: prediction_hash (SHA-256 of risk forecast, predicted stages, and novelty score)
        import hashlib
        import logging

        raw_csv_bytes = raw_input.to_csv(index=False).encode("utf-8") if isinstance(raw_input, pd.DataFrame) else str(raw_input).encode("utf-8")
        raw_data_hash = hashlib.sha256(raw_csv_bytes).hexdigest()
        feature_hash = hashlib.sha256(seq_30x406.tobytes()).hexdigest()
        model_id = "lstm_world_model_v4"
        pred_dict = {
            "risk": [round(float(p_onset), 6)],
            "stages": stage_names,
            "novelty": round(float(novelty_score), 4),
        }
        prediction_hash = hashlib.sha256(json.dumps(pred_dict, sort_keys=True).encode("utf-8")).hexdigest()

        # Determine severity and target node
        max_r = float(p_onset)
        if max_r > 0.8:
            lineage_severity = "HIGH"
        elif max_r > 0.4:
            lineage_severity = "MEDIUM"
        else:
            lineage_severity = "LOW"

        target_node = ""  # no host is known unless the input names one
        if len(flagged_flows) > 0 and flagged_flows[0].get("source"):
            target_node = str(flagged_flows[0]["source"])
        elif graph_traversal.get("start_node"):
            target_node = str(graph_traversal["start_node"])

        lineage_id = f"lin_{window_id}_{raw_data_hash[:8]}"

        # Record atomic lineage on Fabric ledger
        # Go chaincode autonomously creates an IncidentResponseRecord if severity is HIGH or CRITICAL
        lineage_ok = False
        try:
            lineage_ok = bool(record_prediction_lineage(
                lineage_id=lineage_id,
                raw_data_hash=raw_data_hash,
                feature_hash=feature_hash,
                model_id=model_id,
                prediction_hash=prediction_hash,
                severity=lineage_severity,
                timestamp=window_start,
                target_node=target_node,
            ))
        except Exception as e:
            logging.warning(f"Fabric Lineage Notarization Failed: {e}")
            lineage_ok = False

        lineage_payload = {
            "lineage_id": lineage_id,
            "raw_data_hash": raw_data_hash,
            "feature_hash": feature_hash,
            "feature_vector_type": "406-dim canonical engineered features (loss-free audit baseline)",
            "model_id": model_id,
            "prediction_hash": prediction_hash,
            "severity": lineage_severity,
            "target_node": target_node,
            "timestamp": window_start,
            "notarized_via": "fabric" if lineage_ok else "sha256_fallback",
        }

        # Persist lineage to reports DB (regardless of fabric/sha256 path)
        try:
            if _insert_report_record is not None:
                _insert_report_record("lineage", lineage_payload)
        except Exception as _db_err:
            logging.warning(f"reports_db lineage insert failed: {_db_err}")

        if not lineage_ok:
            # Fall back to SHA-256 hash chaining in backend/alerts and audit_chain
            try:
                runtime_dir = Path(os.environ.get("SHADOWCAT_RUNTIME_DIR", RUNTIME_DIR))
                alerts_dir = runtime_dir / "alerts"
                alerts_dir.mkdir(parents=True, exist_ok=True)
                lin_file = alerts_dir / f"lineage_{lineage_id}.json"
                with open(lin_file, "w", encoding="utf-8") as f:
                    json.dump(lineage_payload, f, indent=2)

                from audit_chain import _load_chain
                existing = _load_chain()
                already_in_chain = any(lineage_id in e.get("description", "") for e in existing)
                if not already_in_chain:
                    append_audit_entry(
                        artifact_path=str(lin_file),
                        artifact_type="prediction_lineage",
                        description=f"Prediction lineage {lineage_id} [{lineage_severity}] (SHA-256 fallback)",
                    )
                logging.info(f"Lineage {lineage_id} notarized via: sha256_fallback")
            except Exception as e:
                logging.warning(f"SHA-256 fallback lineage notarization failed: {e}")

        payload["lineage"] = lineage_payload
        forecast_trajectory["lineage"] = lineage_payload

        try:
            runtime_dir = Path(os.environ.get("SHADOWCAT_RUNTIME_DIR", RUNTIME_DIR))
            forecasts_dir = runtime_dir / "forecasts"
            forecasts_dir.mkdir(parents=True, exist_ok=True)
            payload_str = json.dumps(payload, indent=2)
            payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()[:8]
            now_utc = datetime.now(timezone.utc)
            ts_str = now_utc.strftime("%Y%m%d_%H%M%S_%f")
            forecast_filename = f"forecast_{window_id}_{ts_str}_{payload_hash}.json"
            forecast_path = str(forecasts_dir / forecast_filename)
            with open(forecast_path, "x", encoding="utf-8") as f:
                f.write(payload_str)
            append_audit_entry(
                artifact_path=forecast_path,
                artifact_type="forecast_payload",
                description=f"Inference forecast for {window_id}"
            )
        except Exception as e:
            import logging
            logging.warning(f"Failed to append forecast to audit chain: {e}")

        return payload

    def _lr_contributions(self, window_406: np.ndarray) -> np.ndarray:
        """Per-feature contribution to the onset LR-stage logit for one window:
        coef_j * (x_j - mean_j) / scale_j, averaged over the 37 onset folds (logit units)."""
        x = np.asarray(window_406, dtype=np.float64)
        return np.mean([f.lr_coef * (x - f.scaler_mean) / f.scaler_scale for f in self.stacked_onset_models], axis=0)

    def _extract_flagged_flows(self, raw_input: pd.DataFrame, source_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Ranks the flows of the input's last 1-minute window by the onset LR stage applied to
        each flow's own values: for every window feature "<column>_mean" whose flow column is
        present, the flow value is scaled exactly as the window feature is (UCS scaler, then the
        fold's LR scaler) and multiplied by the fold's coefficient; the score is the sum over
        those features, averaged over the 37 onset folds (logit units, higher = pushes the
        onset probability up). Window input has no flows, so nothing is returned.
        """
        if source_type == "windows" or not isinstance(raw_input, pd.DataFrame) or "Timestamp" not in raw_input.columns:
            return []
        ts = pd.to_datetime(raw_input["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        if ts.isna().all():
            return []
        minute = ts.dt.floor("min")
        last = raw_input[minute == minute.max()]

        mapping = self.extractor.canonical_mapping.get("mapping", {})
        cols = list(UCSExtractor.MODEL_INPUT_COLUMNS)
        feats, raw_cols = [], []
        for raw_col, canon in mapping.items():
            name = f"{canon}_mean"
            if raw_col in last.columns and name in cols:
                feats.append(name)
                raw_cols.append(raw_col)
        if not feats:
            return []
        idx = [cols.index(f) for f in feats]
        vals = last[raw_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        vals.columns = feats
        ucs = self.extractor.scaler.transform(vals)[feats].to_numpy(dtype=np.float64)  # (n_flows, k)
        contrib = np.mean([f.lr_coef[idx] * (ucs - f.scaler_mean[idx]) / f.scaler_scale[idx]
                           for f in self.stacked_onset_models], axis=0)
        scores = contrib.sum(axis=1)
        order = np.argsort(scores)[::-1][:limit]

        def col(*names):
            return next((n for n in names if n in last.columns), None)

        src_c, dst_c = col("Src IP", "src_ip"), col("Dst IP", "dst_ip")
        sport_c, dport_c, proto_c = col("Src Port", "src_port"), col("Dst Port", "dst_port"), col("Protocol", "protocol")
        flows = []
        for r in order:
            row = last.iloc[r]
            top = np.argsort(contrib[r])[::-1][:3]
            flows.append({
                "row": int(last.index[r]),
                "timestamp": str(row["Timestamp"]),
                "source": str(row[src_c]) if src_c else None,
                "sport": str(row[sport_c]) if sport_c else None,
                "destination": str(row[dst_c]) if dst_c else None,
                "dport": str(row[dport_c]) if dport_c else None,
                "protocol": str(row[proto_c]) if proto_c else None,
                "lr_score": round(float(scores[r]), 4),
                "top_features": ", ".join(f"{feats[j]} {contrib[r, j]:+.3f}" for j in top),
                "dataset_label": str(row["Label"]) if "Label" in last.columns else None,
            })
        return flows


# Global singleton pipeline instance
_GLOBAL_PIPELINE: Optional[ShadowcatPipeline] = None


def get_pipeline() -> ShadowcatPipeline:
    """Returns initialized singleton pipeline."""
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = ShadowcatPipeline()
    return _GLOBAL_PIPELINE


def predict(raw_input: pd.DataFrame, source_type: str = "csv") -> Dict[str, Any]:
    """
    Main entry point function matching the specification in Master Prompt.
    """
    pipeline = get_pipeline()
    return pipeline.predict(raw_input=raw_input, source_type=source_type)
