"""
SHADOWCAT - Authoritative Data Provider & Decoupled Access Boundary
Single access interface connecting the UI layer to ML/DE pipeline models and live inference.
Every accessor returns values from the active backend.predict result or a committed results file.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Project root directory for checkpoint artifact resolution
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

# Add backend and data-engineering to path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))
if str(REPO_ROOT / "data-engineering") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "data-engineering"))


# -----------------------------------------------------------------------------
# Persistent Reports Database Access (SQLite-backed, survives restarts)
# -----------------------------------------------------------------------------
try:
    from backend.reports_db import query_reports as _query_reports_db
except ImportError:
    try:
        import importlib
        _reports_db_mod = importlib.import_module("reports_db")
        _query_reports_db = _reports_db_mod.query_reports
    except (ImportError, AttributeError):
        _query_reports_db = None  # reports_db unavailable — get_report_history returns []


def get_report_history(
    severity: Optional[str] = None,
    record_type: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Retrieve durable report history from the SQLite database.
    Thin wrapper around reports_db.query_reports().
    Returns [] if the DB is empty, missing, or the module is unavailable.
    Never crashes the dashboard.
    """
    if _query_reports_db is None:
        return []
    try:
        return _query_reports_db(severity=severity, record_type=record_type, limit=limit)
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Cached Live Inference Pipeline Invocation
# -----------------------------------------------------------------------------
_CACHED_LIVE_PREDICTION: Optional[Dict[str, Any]] = None


def _get_live_prediction() -> Dict[str, Any]:
    """Runs or retrieves cached live inference from backend.predict.
    Prioritizes session_state ML results (set by the Ingestion page's Run button)
    so all cockpit views display the same live prediction output."""
    global _CACHED_LIVE_PREDICTION

    # Priority 1: Use session_state prediction if user has run ML inference
    try:
        import streamlit as st
        if "ml_prediction_result" in st.session_state and st.session_state["ml_prediction_result"] is not None:
            _CACHED_LIVE_PREDICTION = st.session_state["ml_prediction_result"]
            return _CACHED_LIVE_PREDICTION
    except Exception:
        pass

    if _CACHED_LIVE_PREDICTION is not None:
        return _CACHED_LIVE_PREDICTION

    # Default before the user loads anything: the committed benign demo slice
    # (32 real UCS windows, see frontend/demo_data/slices.json).
    try:
        from backend.predict import predict
        df = load_demo_slice(DEFAULT_DEMO_SLICE)
        _CACHED_LIVE_PREDICTION = predict(df, source_type="windows")
        return _CACHED_LIVE_PREDICTION
    except Exception:
        return {}


DEMO_DIR = PROJECT_ROOT / "demo_data"
DEFAULT_DEMO_SLICE = "benign"


def get_demo_slices() -> Dict[str, Any]:
    """Metadata of the committed demo slices (frontend/demo_data/slices.json)."""
    path = DEMO_DIR / "slices.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("slices", {})


def load_demo_slice(key: str) -> pd.DataFrame:
    """The 32 UCS windows of a demo slice (rows of ucs_windows_models_v1.parquet)."""
    meta = get_demo_slices()[key]
    return pd.read_parquet(REPO_ROOT / meta["windows"])


def get_active_source() -> str:
    """Human-readable description of the data behind the active prediction."""
    try:
        import streamlit as st
        if st.session_state.get("active_source"):
            return st.session_state["active_source"]
    except Exception:
        pass
    meta = get_demo_slices().get(DEFAULT_DEMO_SLICE)
    if not meta:
        return "no input loaded"
    return f"Demo slice '{DEFAULT_DEMO_SLICE}' ({meta['time_range_utc'][0]} to {meta['time_range_utc'][1]} UTC)"


def set_active_prediction(prediction_dict: Dict[str, Any]) -> None:
    """Explicitly sets the active live prediction across all views."""
    global _CACHED_LIVE_PREDICTION
    _CACHED_LIVE_PREDICTION = prediction_dict


def run_core_ml_inference(input_df: pd.DataFrame, source_type: str = "csv") -> Dict[str, Any]:
    """
    Executes the core backend ML pipeline on an input DataFrame.
    Pre-normalizes timestamps and column aliases for UCSExtractor,
    invokes LSTM Gaussian world model and heads, and caches the result.
    """
    global _CACHED_LIVE_PREDICTION
    from backend.predict import predict

    df = input_df.copy()
    # Deduplicate existing columns in input_df
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()

    if source_type == "windows":
        # Already-built, already-scaled UCS windows: no flow-column normalization.
        try:
            pred = predict(df, source_type="windows")
            _CACHED_LIVE_PREDICTION = pred
            return pred
        except Exception as e:
            import traceback
            return {"_inference_error": str(e), "_inference_traceback": traceback.format_exc(),
                    "_critical_schema_failure": True}

    # Normalize timestamp column for UCSExtractor without duplicate keys
    ts_cols = [c for c in df.columns if c.lower() in ("timestamp", "timestamp_utc", "time", "ts", "date", "starttime")]
    if not ts_cols:
        return {"_inference_error": "Input has no timestamp column; flows cannot be windowed.",
                "_critical_schema_failure": True}
    ts_col = ts_cols[0]
    # CICFlowMeter writes day-first timestamps (dd/mm/YYYY); parse that format first so
    # 02/03/2018 stays 2 March and is not read as 3 February.
    ts_series = pd.to_datetime(df[ts_col], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    if ts_series.isna().all():
        ts_series = pd.to_datetime(df[ts_col], dayfirst=True, errors="coerce")
    if ts_series.isna().all():
        return {"_inference_error": f"Could not parse timestamps in column '{ts_col}'.",
                "_critical_schema_failure": True}
    df["Timestamp"] = ts_series.dt.strftime("%d/%m/%Y %H:%M:%S")
    # Drop redundant alternative timestamp columns to prevent duplicate mapping
    for c in ts_cols:
        if c != "Timestamp" and c in df.columns:
            df = df.drop(columns=[c])

    # Normalize flow duration if present
    if "flow_duration_s" in df.columns and "Flow Duration" not in df.columns:
        df["Flow Duration"] = (df["flow_duration_s"] * 1000000).astype(int)
        df = df.drop(columns=["flow_duration_s"])
    elif "Dur" in df.columns and "Flow Duration" not in df.columns:
        df["Flow Duration"] = (df["Dur"] * 1000000).astype(int)
        df = df.drop(columns=["Dur"])

    # Normalize packet count and byte count columns
    if "tot_fwd_pkts" in df.columns and "Tot Fwd Pkts" not in df.columns:
        df["Tot Fwd Pkts"] = df["tot_fwd_pkts"]
        df = df.drop(columns=["tot_fwd_pkts"])
    elif "TotPkts" in df.columns and "Tot Fwd Pkts" not in df.columns:
        df["Tot Fwd Pkts"] = df["TotPkts"]
    
    if "TotBytes" in df.columns and "TotLen Fwd Pkts" not in df.columns:
        df["TotLen Fwd Pkts"] = df["TotBytes"]
    elif "flow_byts_s" in df.columns and "TotLen Fwd Pkts" not in df.columns:
        df["TotLen Fwd Pkts"] = df["flow_byts_s"]

    if "tot_bwd_pkts" in df.columns and "Tot Bwd Pkts" not in df.columns:
        df["Tot Bwd Pkts"] = df["tot_bwd_pkts"]
        df = df.drop(columns=["tot_bwd_pkts"])
        
    if "src_ip" in df.columns and "Src IP" not in df.columns:
        df["Src IP"] = df["src_ip"]
        df = df.drop(columns=["src_ip"])
    elif "SrcAddr" in df.columns and "Src IP" not in df.columns:
        df["Src IP"] = df["SrcAddr"]
        df = df.drop(columns=["SrcAddr"])
        
    if "dst_ip" in df.columns and "Dst IP" not in df.columns:
        df["Dst IP"] = df["dst_ip"]
        df = df.drop(columns=["dst_ip"])
    elif "DstAddr" in df.columns and "Dst IP" not in df.columns:
        df["Dst IP"] = df["DstAddr"]
        df = df.drop(columns=["DstAddr"])
        
    if "src_port" in df.columns and "Src Port" not in df.columns:
        df["Src Port"] = df["src_port"]
        df = df.drop(columns=["src_port"])
    elif "Sport" in df.columns and "Src Port" not in df.columns:
        df["Src Port"] = df["Sport"]
        df = df.drop(columns=["Sport"])
        
    if "dst_port" in df.columns and "Dst Port" not in df.columns:
        df["Dst Port"] = df["dst_port"]
        df = df.drop(columns=["dst_port"])
    elif "Dport" in df.columns and "Dst Port" not in df.columns:
        df["Dst Port"] = df["Dport"]
        df = df.drop(columns=["Dport"])
        
    if "protocol" in df.columns and "Protocol" not in df.columns:
        df["Protocol"] = df["protocol"].apply(lambda p: 6 if str(p).upper() == "TCP" else (17 if str(p).upper() == "UDP" else 6))
        df = df.drop(columns=["protocol"])
    elif "Proto" in df.columns and "Protocol" not in df.columns:
        df["Protocol"] = df["Proto"].apply(lambda p: 6 if str(p).upper() == "TCP" else (17 if str(p).upper() == "UDP" else 6))
        df = df.drop(columns=["Proto"])

    # Final safeguard against any duplicate column names
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()

    try:
        pred = predict(df, source_type=source_type)
        _CACHED_LIVE_PREDICTION = pred
        return pred
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        
        # [DEFENSIVE UI BEHAVIOR] If it's a schema validation error, do NOT fallback to confident cached data.
        if "SchemaValidationError" in str(type(e)):
            return {"_inference_error": str(e), "_inference_traceback": tb, "_critical_schema_failure": True}
        
        # Still fall back to cached prediction for other runtime errors, but attach error info so UI can show it
        fallback = _get_live_prediction()
        fallback["_inference_error"] = str(e)
        fallback["_inference_traceback"] = tb
        _CACHED_LIVE_PREDICTION = fallback
        return fallback


# -----------------------------------------------------------------------------
# 2. Typed Accessor Functions (UI Consumer API)
# -----------------------------------------------------------------------------

def get_analysis_metadata() -> dict:
    """
    Analysis metadata of the active prediction (input type, window size, last window time).
    """
    return dict(_get_live_prediction().get("analysis_metadata", {}))


def get_forecast_trajectory(window_id: str = None) -> dict:
    """
    Returns the forecast block of the active inference result: the stacked onset
    probability P(attack within the next 5 minutes), its conformal interval and the
    stage head output on the world-model rollout. Empty dict if no inference ran.
    Consumed by: views/03_Forecast.py
    """
    pred = _get_live_prediction()
    fc = dict(pred.get("forecast_trajectory", {}))
    if window_id:
        fc["window_id"] = window_id
    return fc


def get_detection_probability() -> Optional[float]:
    """Stacked detection ensemble probability for the current window, or None."""
    return _get_live_prediction().get("detection_probability")


def get_graph_topology() -> Optional[dict]:
    """
    Returns real network graph topology (nodes and edges) constructed from flow data.
    """
    pred = _get_live_prediction()
    return pred.get("graph_topology")


def get_graph_traversal(max_k: int = 5) -> dict:
    """
    Returns real, explainable, weight-based attack propagation traversal over actual network graph edges.
    Consumed by: views/04_Attack_Graph.py, components/cytoscape_attack_graph.py
    """
    pred = _get_live_prediction()
    if "graph_traversal" in pred and pred["graph_traversal"] is not None:
        return pred["graph_traversal"]

    topology = pred.get("graph_topology") or {}
    nodes = topology.get("graph_nodes", [])
    edges = topology.get("graph_edges", [])
    flagged_flows = pred.get("flagged_flows", [])

    from backend.graph_traversal import compute_graph_traversal
    return compute_graph_traversal(
        graph_nodes=nodes,
        graph_edges=edges,
        flagged_flows=flagged_flows,
        max_k=max_k,
    )


def get_fusion_experimental() -> Optional[dict]:
    """
    Deprecated alias: returns graph_topology for backward compatibility.
    """
    return get_graph_topology()



def get_host_risk_graph(episode_id: str = None, k_step: int = 2) -> dict:
    """
    Returns enterprise host graph topology, communication edges, and
    forward-simulated host risk scores h_v(t) across rollout horizons.
    Consumed by: components/attack_graph.py
    """
    UI_NODE_ROLES = {
        "10.0.2.15": "Workstation (Patient Zero)",
        "10.0.3.50": "Internal File Share",
        "10.0.4.10": "SSH Jump Host",
        "10.0.4.21": "Internal Auth Cluster",
        "10.0.5.1": "Domain Controller (Critical Asset)",
    }

    HOST_TELEMETRY = {
        "10.0.2.15": {
            "role": "Workstation (Patient Zero)",
            "criticality_tier": "Tier 3 (User Endpoint)",
            "criticality_level": 3,
            "subnet": "10.0.2.0/24 (User Endpoint Tier)",
            "active_ports": "TCP/22 (SSH Outbound), TCP/445 (SMB Outbound), TCP/5355 (LLMNR)",
            "driving_indicators": "Sustained high SYN packet bursts, anomalous subprocess socket spawning, credential memory read attempts.",
            "containment_stance": "Isolate endpoint network adapter and revoke active Kerberos session tickets.",
        },
        "10.0.3.50": {
            "role": "Internal File Share",
            "criticality_tier": "Tier 3 (Storage Share)",
            "criticality_level": 3,
            "subnet": "10.0.3.0/24 (Enterprise Storage Tier)",
            "active_ports": "TCP/445 (SMB/CIFS), TCP/139 (NetBIOS Session), TCP/2049 (NFS)",
            "driving_indicators": "Rapid sequential SMB directory enumeration, mass file metadata queries, volume shadow copy inspect probes.",
            "containment_stance": "Enable strict SMB signing, enforce share ACL restrictions, and audit shadow copy access.",
        },
        "10.0.4.10": {
            "role": "SSH Jump Host",
            "criticality_tier": "Tier 2 (Management Gateway)",
            "criticality_level": 2,
            "subnet": "10.0.4.0/24 (DMZ Management Tier)",
            "active_ports": "TCP/22 (OpenSSH Ingress), TCP/88 (Kerberos Client), TCP/514 (Syslog)",
            "driving_indicators": "Repeated SSH authentication failure bursts (18 attempts/min), brute-force credential spray, privileged session forwarding.",
            "containment_stance": "Terminate active jump host sessions, restrict SSH ingress to bastion management CIDRs.",
        },
        "10.0.4.21": {
            "role": "Internal Auth Cluster",
            "criticality_tier": "Tier 2 (Auth Infrastructure)",
            "criticality_level": 2,
            "subnet": "10.0.4.0/24 (DMZ Management Tier)",
            "active_ports": "TCP/88 (Kerberos KDC), TCP/389 (LDAP Auth), TCP/636 (LDAPS), TCP/3268 (GC)",
            "driving_indicators": "Anomalous Kerberos Ticket Granting Service (TGS) request spike from jump host, unusual RC4 ticket encryption negotiation.",
            "containment_stance": "Enforce AES-256 Kerberos ticket encryption and rotate service account credentials.",
        },
        "10.0.5.1": {
            "role": "Domain Controller (Critical Asset)",
            "criticality_tier": "Tier 1 (Critical Asset)",
            "criticality_level": 1,
            "subnet": "10.0.5.0/24 (Core Identity Tier)",
            "active_ports": "TCP/389 (Active Directory LDAP), TCP/88 (Kerberos TGT), RPC/135 (DCSync Endpoint)",
            "driving_indicators": "Privileged directory replication request (DCSync pattern), high-volume directory object queries from internal auth nodes.",
            "containment_stance": "Apply pre-emptive access control filters on directory replication RPC endpoints.",
        },
    }

    ROLLOUT_DATA = {
        0: {
            "label": "Window t (Current Observed)",
            "horizon_code": "t",
            "uncertainty_sigma": 0.02,
            "uncertainty_label": "±2% (Observed Window)",
            "uncertainty_tier": "Nominal Telemetry",
            "uncertainty_color": "#2FB872",
            "summary": "Attacker observed scanning ports on 10.0.4.10 from 10.0.2.15.",
            "active_edges": [("10.0.2.15", "10.0.4.10", "Port 22/TCP")],
            "host_risks": {
                "10.0.2.15": {"risk": 0.85, "uncertainty": 0.02},
                "10.0.4.10": {"risk": 0.38, "uncertainty": 0.03},
                "10.0.4.21": {"risk": 0.12, "uncertainty": 0.02},
                "10.0.3.50": {"risk": 0.08, "uncertainty": 0.02},
                "10.0.5.1": {"risk": 0.05, "uncertainty": 0.01},
            },
        },
        1: {
            "label": "Horizon t+1 (+1 min Rollout)",
            "horizon_code": "t+1",
            "uncertainty_sigma": 0.05,
            "uncertainty_label": "±5% (Autoregressive Step 1)",
            "uncertainty_tier": "Low Epistemic Drift",
            "uncertainty_color": "#8A8A8A",
            "summary": "SSH brute-force activity intensifies against jump host 10.0.4.10.",
            "active_edges": [
                ("10.0.2.15", "10.0.4.10", "Port 22/TCP [High Rate]"),
                ("10.0.2.15", "10.0.3.50", "Port 445/SMB [Probe]"),
            ],
            "host_risks": {
                "10.0.2.15": {"risk": 0.92, "uncertainty": 0.04},
                "10.0.4.10": {"risk": 0.54, "uncertainty": 0.06},
                "10.0.4.21": {"risk": 0.18, "uncertainty": 0.05},
                "10.0.3.50": {"risk": 0.14, "uncertainty": 0.04},
                "10.0.5.1": {"risk": 0.08, "uncertainty": 0.03},
            },
        },
        2: {
            "label": "Horizon t+2 (+2 min Rollout)",
            "horizon_code": "t+2",
            "uncertainty_sigma": 0.09,
            "uncertainty_label": "±9% (Autoregressive Step 2)",
            "uncertainty_tier": "Controlled Epistemic Drift",
            "uncertainty_color": "#8A8A8A",
            "summary": "Anticipated credential extraction on 10.0.4.10; reconnaissance directed at Auth Cluster.",
            "active_edges": [
                ("10.0.2.15", "10.0.4.10", "Compromised"),
                ("10.0.4.10", "10.0.4.21", "Port 88/Kerberos"),
            ],
            "host_risks": {
                "10.0.4.10": {"risk": 0.76, "uncertainty": 0.09},
                "10.0.2.15": {"risk": 0.94, "uncertainty": 0.05},
                "10.0.4.21": {"risk": 0.46, "uncertainty": 0.09},
                "10.0.5.1": {"risk": 0.19, "uncertainty": 0.07},
                "10.0.3.50": {"risk": 0.15, "uncertainty": 0.06},
            },
        },
        3: {
            "label": "Horizon t+3 (+3 min Rollout)",
            "horizon_code": "t+3",
            "uncertainty_sigma": 0.16,
            "uncertainty_label": "±16% (Autoregressive Step 3)",
            "uncertainty_tier": "Compounding Rollout Error",
            "uncertainty_color": "#E0982B",
            "summary": "Anticipated lateral pivot: Kerberos ticket reuse toward Domain Controller 10.0.5.1.",
            "active_edges": [
                ("10.0.4.10", "10.0.4.21", "Auth Spray"),
                ("10.0.4.21", "10.0.5.1", "Port 389/LDAP Pivot"),
            ],
            "host_risks": {
                "10.0.4.21": {"risk": 0.79, "uncertainty": 0.15},
                "10.0.5.1": {"risk": 0.67, "uncertainty": 0.17},
                "10.0.4.10": {"risk": 0.88, "uncertainty": 0.12},
                "10.0.2.15": {"risk": 0.95, "uncertainty": 0.08},
                "10.0.3.50": {"risk": 0.18, "uncertainty": 0.11},
            },
        },
        4: {
            "label": "Horizon t+4 (+4 min Rollout)",
            "horizon_code": "t+4",
            "uncertainty_sigma": 0.27,
            "uncertainty_label": "±27% (Epistemic Drift Limit)",
            "uncertainty_tier": "Exploratory Horizon Bound",
            "uncertainty_color": "#E5484D",
            "summary": "Privileged persistence on Domain Controller; replication traffic initiated.",
            "active_edges": [
                ("10.0.4.21", "10.0.5.1", "DCSync Replication"),
                ("10.0.5.1", "10.0.3.50", "Volume Shadow Copy"),
            ],
            "host_risks": {
                "10.0.5.1": {"risk": 0.86, "uncertainty": 0.27},
                "10.0.4.21": {"risk": 0.82, "uncertainty": 0.22},
                "10.0.4.10": {"risk": 0.91, "uncertainty": 0.18},
                "10.0.2.15": {"risk": 0.96, "uncertainty": 0.12},
                "10.0.3.50": {"risk": 0.42, "uncertainty": 0.24},
            },
        },
        5: {
            "label": "Horizon t+5 (+5 min Rollout)",
            "horizon_code": "t+5",
            "uncertainty_sigma": 0.32,
            "uncertainty_label": "±32% (Exploratory Epistemic Bound)",
            "uncertainty_tier": "Maximum Horizon Bound",
            "uncertainty_color": "#E5484D",
            "summary": "Full domain compromise / data exfiltration impact across enterprise core.",
            "active_edges": [
                ("10.0.5.1", "10.0.3.50", "Exfiltration Stream"),
            ],
            "host_risks": {
                "10.0.5.1": {"risk": 0.89, "uncertainty": 0.31},
                "10.0.4.21": {"risk": 0.85, "uncertainty": 0.25},
                "10.0.4.10": {"risk": 0.93, "uncertainty": 0.20},
                "10.0.2.15": {"risk": 0.97, "uncertainty": 0.14},
                "10.0.3.50": {"risk": 0.51, "uncertainty": 0.29},
            },
        },
    }

    return {
        "episode_id": episode_id or "EPISODE-INFIL-01",
        "node_roles": UI_NODE_ROLES,
        "host_telemetry": HOST_TELEMETRY,
        "rollout_steps": ROLLOUT_DATA,
        "active_k_step": k_step,
        "disclaimer": "Demonstrated on the single infiltration case study (n = 1). Not a general lateral-movement forecasting capability.",
        "is_mock": True,  # scripted story; removed in s2b-t8
    }


def get_attributions(window_id: str = None) -> list[dict]:
    """
    Top input features by Integrated Gradients on the stacked onset logit for the active
    prediction (empty list if none was computed).
    Consumed by: views/06_Explainability.py, components/layered_explanation.py
    """
    return list(_get_live_prediction().get("attributions", []))


def get_temporal_attributions() -> dict:
    """
    Integrated Gradients on the stacked onset logit, decomposed over the 30 lookback windows.
    Consumed by: views/06_Explainability.py
    """
    pred = _get_live_prediction()
    return pred.get("temporal_attributions", {})


def get_conformal_forecast() -> dict:
    """
    Split-conformal interval for the onset probability of the active prediction ({} if none).
    Consumed by: views/06_Explainability.py
    """
    cf = (_get_live_prediction() or {}).get("conformal_forecast")
    return cf if isinstance(cf, dict) else {}


def get_conformal_credibility(window_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns the conformal credibility and out-of-distribution (OOD) diagnostic status.
    Consumed by: components/layered_explanation.py, views/03_Forecast.py, views/05_Alerts.py

    CONTRACT SPECIFICATION (For integration with branch 'feature/conformal-mimo' by Person 1):
    Expected schema in prediction payload or data provider:
    {
        "conformal_credibility": {
            "status": "in_distribution" | "out_of_distribution",
            "is_in_distribution": bool,          # True = nominal, False = OOD drift detected
            "credibility_score": float,         # Conformal p-value in [0, 1]
            "confidence_level": float,          # Nominal confidence e.g. 0.95
            "drift_score": float,               # Non-conformity score / Mahalanobis distance
            "drift_threshold": float,           # Rejection threshold
            "badge_label": str,                 # "MODEL CONFIDENCE: IN-DISTRIBUTION" | "OUT-OF-DISTRIBUTION"
            "advisory": str,                    # SOC action advisory text
            "is_stub": bool                     # True while waiting for feature/conformal-mimo
        }
    }
    """
    pred = _get_live_prediction() or {}
    # No out-of-distribution check is implemented in backend.predict; nothing is shown
    # unless a real one populates the payload.
    if isinstance(pred.get("conformal_credibility"), dict):
        cred = dict(pred["conformal_credibility"])
        cred["is_stub"] = False
        return cred
    return {"is_stub": True}


def get_entity_risk_summary(
    threshold: float = 100.0,
    lookback_hours: int = 24,
    group_by: str = "host",
) -> Dict[str, Any]:
    """
    Returns entity-level accumulated risk and threshold crossing alert state.
    Consumed by: views/05_Alerts.py, views/04_Attack_Graph.py, components/risk_accumulator.py
    """
    from components.risk_accumulator import calculate_entity_risk
    return calculate_entity_risk(threshold=threshold, lookback_hours=lookback_hours, group_by=group_by)


def get_counterfactual(window_id: str = None) -> dict:
    """
    Returns explanatory counterfactual recommendations for the current flagged window.
    Consumed by: views/06_Explainability.py
    """
    pred = _get_live_prediction()
    cf = pred.get("counterfactual")
    if not cf:
        return {
            "status": "inconclusive",
            "found": False,
            "summary": "No counterfactual analysis available for this telemetry window.",
            "honesty_label": (
                "Model-based counterfactual under the hazard model's learned decision boundary — "
                "not a guarantee that this change would have prevented the actual attack, "
                "and not validated against real intervention data."
            ),
            "is_mock": False,
        }
    cf["is_mock"] = False
    return cf



def get_novelty_score(window_id: str = None) -> dict:
    """
    Returns current observed network state S(t) telemetry, novelty score,
    behavioral envelope classification, and sparkline metrics.
    Consumed by: views/03_Forecast.py, views/06_Explainability.py, components/state.py
    """
    pred = _get_live_prediction()
    nv = pred.get("novelty_score", {})
    return nv


def get_flagged_flows(window_id: str = None, limit: int = None) -> list[dict]:
    """
    Flows of the active input ranked by the backend's flow score (empty for window input).
    """
    flows = list(_get_live_prediction().get("flagged_flows", []))
    return flows[:limit] if limit else flows


def get_validation_data() -> dict:
    """
    Returns the LOEO 37-fold validation results generated by
    frontend/scripts/build_validation_json.py from committed sidecar / benchmark files.
    Empty dict if the generated file is missing (no fallback numbers).
    Consumed by: views/07_Validation_Trust.py
    """
    json_path = PROJECT_ROOT / "models" / "loeo_37fold_results.json"
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_mitre_data() -> list[dict]:
    """
    Stage-head output of the active prediction, one entry per world-model rollout step
    t+1..t+5. A stage (with its real ATT&CK tactic / technique) is present only when the
    stage head's top probability is > 0.5 and the class is not Unknown/Other; otherwise
    stage = "No stage determined" and all IDs are None.
    """
    fc = _get_live_prediction().get("forecast_trajectory", {})
    stages = fc.get("stage", [])
    out = []
    for k, md in enumerate(fc.get("mitre_details", []), start=1):
        out.append({
            "step": f"t+{k}",
            "stage": stages[k - 1] if k - 1 < len(stages) else None,
            "stage_head_top_class": md.get("stage_head_top_class"),
            "confidence": md.get("stage_confidence"),
            "tactic_id": md.get("tactic_id"),
            "tactic_name": md.get("tactic_name"),
            "tactic_url": md.get("url"),
            "technique_id": md.get("technique_id"),
            "technique_name": md.get("technique_name"),
            "technique_full_name": md.get("technique_full_name"),
            "technique_url": md.get("technique_url"),
            "description": md.get("technique_description"),
            "likely_next_techniques": md.get("likely_next_techniques", []),
        })
    return out


def get_audit_chain_status() -> dict:
    """
    Returns the status and entries of the blockchain-inspired tamper-evident audit chain.
    Consumed by: views/07_Validation_Trust.py
    """
    chain_file = REPO_ROOT / "backend" / "audit_chain.json"
    if chain_file.exists():
        try:
            with open(chain_file, "r", encoding="utf-8") as f:
                chain = json.load(f)
            from audit_chain import verify_chain
            is_valid, issues = verify_chain(str(chain_file))
            return {
                "length": len(chain),
                "is_valid": is_valid,
                "issues": issues,
                "entries": chain,
            }
        except Exception as e:
            return {"length": 0, "is_valid": False, "issues": [str(e)], "entries": []}
    return {"length": 0, "is_valid": False, "issues": ["audit_chain.json not found"], "entries": []}


def get_live_notarization_status() -> dict:
    """
    How the last prediction's alert / lineage record was notarized ("fabric",
    "sha256_fallback", "none" or "failed"), as reported by backend.predict.
    Consumed by: views/07_Validation_Trust.py, views/02_Overview.py
    """
    pred = _get_live_prediction() or {}
    mech = pred.get("notarization_mechanism", "none")
    lineage = pred.get("lineage") or {}
    return {
        "active_mechanism": mech,
        "fabric_active": mech == "fabric",
        "fallback_engaged": mech == "sha256_fallback",
        "lineage_id": lineage.get("lineage_id"),
        "lineage_notarized_via": lineage.get("notarized_via"),
    }
    return incidents
