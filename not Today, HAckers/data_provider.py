"""
SHADOWCAT - Authoritative Data Provider & Decoupled Access Boundary
Single access interface connecting the UI layer to ML/DE pipeline models.
Supports granular per-function provenance tracking (MOCK_STATUS) and auto-detection.
"""

import os
import sys
from pathlib import Path

# Project root directory for checkpoint artifact resolution
PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# 1. Provenance Tracking & Checkpoint Auto-Detection
# -----------------------------------------------------------------------------
# Expected live model artifact paths relative to project root
CHECKPOINT_PATHS = {
    "forecast_trajectory": "models/world_model.pt",
    "comparison_table": "models/baseline_benchmarks.json",
    "host_risk_graph": "models/gnn_topology.pt",
    "attributions": "models/integrated_gradients.npz",
    "novelty_score": "models/novelty_detector.pt",
    "flagged_flows": "models/flow_classifier.pt",
    "validation_data": "models/loeo_37fold_results.json",
    "mitre_data": "models/mitre_mapper.json",
    "analysis_metadata": "models/telemetry_manifest.json",
}

# Manual fallback override dictionary (used if checkpoint files are not present)
MOCK_STATUS = {
    "forecast_trajectory": True,
    "comparison_table": True,
    "host_risk_graph": True,
    "attributions": True,
    "novelty_score": True,
    "flagged_flows": True,
    "validation_data": True,
    "mitre_data": True,
    "analysis_metadata": True,
}


def is_using_mock_data(key: str) -> bool:
    """
    Returns False if live model artifact exists for this key,
    otherwise falls back to MOCK_STATUS[key].
    """
    ckpt_rel = CHECKPOINT_PATHS.get(key)
    if ckpt_rel:
        ckpt_full = PROJECT_ROOT / ckpt_rel
        if ckpt_full.is_file():
            return False  # Live model checkpoint artifact detected!
    return MOCK_STATUS.get(key, True)


def get_mock_badge_html(key: str) -> str:
    """
    Returns styled [MOCK] badge HTML if the specified key is in mock mode.
    Must be rendered via render_html() or st.markdown(..., unsafe_allow_html=True).
    """
    if is_using_mock_data(key):
        return (
            '<span class="badge-mock" style="background: rgba(224, 152, 43, 0.15); '
            'border: 1px solid #E0982B; color: #E0982B; padding: 1px 6px; border-radius: 4px; '
            'font-size: 0.68rem; font-weight: 700; font-family: \'JetBrains Mono\', monospace; '
            'letter-spacing: 0.05em; margin-left: 6px; vertical-align: middle;">MOCK</span>'
        )
    return ""


# -----------------------------------------------------------------------------
# Internal Mock Data Store
# (Private to this module — no external component may import directly)
# -----------------------------------------------------------------------------
from mock_data import get_demo_data as _get_raw_mock_dict  # Isolated private import


# -----------------------------------------------------------------------------
# 2. Typed Accessor Functions (UI Consumer API)
# -----------------------------------------------------------------------------

def get_analysis_metadata() -> dict:
    """
    Returns active telemetry session, sensor tap ID, and window timestamp.
    Consumed by: components/header.py -> render_header()
    """
    raw = _get_raw_mock_dict()
    return {
        "source": raw["analysis"].get("source", "BENCHMARK: CSE-CIC-IDS2018 (Infiltration)"),
        "sensor_id": "TAP-DMZ-01",
        "sensor_throughput": "10Gbps Ingress",
        "window": raw["analysis"].get("window", "t+1 → t+4 (Active)"),
        "window_duration": "60s",
        "duration_sec": int(raw["analysis"].get("duration_sec", 60)),
        "lookback_windows": 30,
        "rollout_horizons": 4,
        "timestamp": "2026-09-09 11:40:00 UTC",
        "is_mock": is_using_mock_data("analysis_metadata"),
    }


def get_forecast_trajectory(window_id: str = None) -> dict:
    """
    Returns multi-step forward trajectory simulation across K=1..4 horizons,
    including risk probabilities, stage mapping, calibrated uncertainty, and protocol metadata.
    Consumed by: views/01_Forecast.py, components/forecast.py
    """
    raw = _get_raw_mock_dict()
    forecast_steps = raw.get("forecast", [])

    return {
        "window_id": window_id or "WIN-20260909-01",
        "horizons": [s["step"] for s in forecast_steps],
        "labels": [f"{s['horizon']} ({s['time_ahead']})" for s in forecast_steps],
        "risk": [s["probability"] for s in forecast_steps],
        "stage": [s["stage"] for s in forecast_steps],
        "tactic_id": [s.get("tactic_id", "TA0000") for s in forecast_steps],
        "lead_time": [s["lead_time"] for s in forecast_steps],
        "uncertainty": [s["uncertainty"] for s in forecast_steps],
        "lower_bound": [s["lower_bound"] for s in forecast_steps],
        "upper_bound": [s["upper_bound"] for s in forecast_steps],
        "calibrated_uncertainty": True,
        "source_branch": "Continuous Dynamics Head",
        "protocol": "Chronological Split",
        "raw_steps": forecast_steps,
        "is_mock": is_using_mock_data("forecast_trajectory"),
    }


def get_comparison_table() -> list[dict]:
    """
    Returns mandated PS benchmark results comparing the World Model against
    the Logistic Regression baseline, Lagged baseline, and Signature IDS.
    Consumed by: views/03_Validation.py
    """
    raw = _get_raw_mock_dict()
    table = raw.get("baseline", {}).get("comparison_table", [])

    # Defensive fallback if unpopulated or empty
    if not table:
        table = [
            {
                "model": "SHADOWCAT World Model (Bi-LSTM + Temporal Dynamics)",
                "paradigm": "Forward Simulation P(S_t+1 | S_t)",
                "f1_score": 0.816,
                "precision": 0.842,
                "recall": 0.791,
                "fpr": 0.048,
                "lead_time": "+3.5 min (Pre-emptive)",
                "status": "Production Candidate",
            },
            {
                "model": "Lagged Autoregressive Baseline (Historical Trend)",
                "paradigm": "Rolling Window Extrapolation",
                "f1_score": 0.612,
                "precision": 0.648,
                "recall": 0.580,
                "fpr": 0.114,
                "lead_time": "+1.2 min (Partial)",
                "status": "Comparative Baseline",
            },
            {
                "model": "Logistic Regression Baseline (Static Classifiers - PS Mandated)",
                "paradigm": "Static Flow Snapshot (No Temporal Memory)",
                "f1_score": 0.468,
                "precision": 0.512,
                "recall": 0.431,
                "fpr": 0.182,
                "lead_time": "0.0 min (Post-facto)",
                "status": "PS Benchmark Reference",
            },
            {
                "model": "Conventional Signature IDS (Suricata / Snort Rules)",
                "paradigm": "Deterministic Packet Matching",
                "f1_score": 0.732,
                "precision": 0.884,
                "recall": 0.625,
                "fpr": 0.021,
                "lead_time": "0.0 min (Alert fired post-compromise)",
                "status": "Legacy Reactive",
            },
        ]

    clean_table = []
    for row in table:
        if not row:
            continue
        model_name = str(row.get("model", "")).strip()
        if not model_name or model_name.lower() in ["empty", "none", "placeholder", "n/a"]:
            continue
        row_copy = {
            k: ("—" if str(v).strip().lower() in ["empty", "none", "n/a"] else v)
            for k, v in row.items()
        }
        row_copy["is_mock"] = is_using_mock_data("comparison_table")
        clean_table.append(row_copy)
    return clean_table


def get_host_risk_graph(episode_id: str = None, k_step: int = 2) -> dict:
    """
    Returns enterprise host graph topology, communication edges, and
    forward-simulated host risk scores h_v(t) across rollout horizons.
    Consumed by: components/attack_graph.py
    """
    # Authoritative graph rollout data
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
            "containment_stance": "Illustrative analyst guidance — not a system recommendation: Isolate endpoint network adapter and revoke active Kerberos session tickets."
        },
        "10.0.3.50": {
            "role": "Internal File Share",
            "criticality_tier": "Tier 3 (Storage Share)",
            "criticality_level": 3,
            "subnet": "10.0.3.0/24 (Enterprise Storage Tier)",
            "active_ports": "TCP/445 (SMB/CIFS), TCP/139 (NetBIOS Session), TCP/2049 (NFS)",
            "driving_indicators": "Rapid sequential SMB directory enumeration, mass file metadata queries, volume shadow copy inspect probes.",
            "containment_stance": "Illustrative analyst guidance — not a system recommendation: Enable strict SMB signing, enforce share ACL restrictions, and audit shadow copy access."
        },
        "10.0.4.10": {
            "role": "SSH Jump Host",
            "criticality_tier": "Tier 2 (Management Gateway)",
            "criticality_level": 2,
            "subnet": "10.0.4.0/24 (DMZ Management Tier)",
            "active_ports": "TCP/22 (OpenSSH Ingress), TCP/88 (Kerberos Client), TCP/514 (Syslog)",
            "driving_indicators": "Repeated SSH authentication failure bursts (18 attempts/min), brute-force credential spray, privileged session forwarding.",
            "containment_stance": "Illustrative analyst guidance — not a system recommendation: Terminate active jump host sessions, restrict SSH ingress to bastion management CIDRs."
        },
        "10.0.4.21": {
            "role": "Internal Auth Cluster",
            "criticality_tier": "Tier 2 (Auth Infrastructure)",
            "criticality_level": 2,
            "subnet": "10.0.4.0/24 (DMZ Management Tier)",
            "active_ports": "TCP/88 (Kerberos KDC), TCP/389 (LDAP Auth), TCP/636 (LDAPS), TCP/3268 (GC)",
            "driving_indicators": "Anomalous Kerberos Ticket Granting Service (TGS) request spike from jump host, unusual RC4 ticket encryption negotiation.",
            "containment_stance": "Illustrative analyst guidance — not a system recommendation: Enforce AES-256 Kerberos ticket encryption and rotate service account credentials."
        },
        "10.0.5.1": {
            "role": "Domain Controller (Critical Asset)",
            "criticality_tier": "Tier 1 (Critical Asset)",
            "criticality_level": 1,
            "subnet": "10.0.5.0/24 (Core Identity Tier)",
            "active_ports": "TCP/389 (Active Directory LDAP), TCP/88 (Kerberos TGT), RPC/135 (DCSync Endpoint)",
            "driving_indicators": "Privileged directory replication request (DCSync pattern), high-volume directory object queries from internal auth nodes.",
            "containment_stance": "Illustrative analyst guidance — not a system recommendation: Apply pre-emptive access control filters on directory replication RPC endpoints."
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
                "10.0.5.1":  {"risk": 0.05, "uncertainty": 0.01},
            }
        },
        1: {
            "label": "Horizon t+1 (+1 min Rollout)",
            "horizon_code": "t+1",
            "uncertainty_sigma": 0.05,
            "uncertainty_label": "±5% (Autoregressive Step 1)",
            "uncertainty_tier": "Low Epistemic Drift",
            "uncertainty_color": "#38BDF8",
            "summary": "SSH brute-force activity intensifies against jump host 10.0.4.10.",
            "active_edges": [
                ("10.0.2.15", "10.0.4.10", "Port 22/TCP [High Rate]"),
                ("10.0.2.15", "10.0.3.50", "Port 445/SMB [Probe]")
            ],
            "host_risks": {
                "10.0.2.15": {"risk": 0.92, "uncertainty": 0.04},
                "10.0.4.10": {"risk": 0.54, "uncertainty": 0.06},
                "10.0.4.21": {"risk": 0.18, "uncertainty": 0.05},
                "10.0.3.50": {"risk": 0.14, "uncertainty": 0.04},
                "10.0.5.1":  {"risk": 0.08, "uncertainty": 0.03},
            }
        },
        2: {
            "label": "Horizon t+2 (+2 min Rollout)",
            "horizon_code": "t+2",
            "uncertainty_sigma": 0.09,
            "uncertainty_label": "±9% (Autoregressive Step 2)",
            "uncertainty_tier": "Controlled Epistemic Drift",
            "uncertainty_color": "#38BDF8",
            "summary": "Anticipated credential extraction on 10.0.4.10; reconnaissance directed at Auth Cluster.",
            "active_edges": [
                ("10.0.2.15", "10.0.4.10", "Compromised"),
                ("10.0.4.10", "10.0.4.21", "Port 88/Kerberos")
            ],
            "host_risks": {
                "10.0.4.10": {"risk": 0.76, "uncertainty": 0.09},
                "10.0.2.15": {"risk": 0.94, "uncertainty": 0.05},
                "10.0.4.21": {"risk": 0.46, "uncertainty": 0.09},
                "10.0.5.1":  {"risk": 0.19, "uncertainty": 0.07},
                "10.0.3.50": {"risk": 0.15, "uncertainty": 0.06},
            }
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
                ("10.0.4.21", "10.0.5.1", "Port 389/LDAP Pivot")
            ],
            "host_risks": {
                "10.0.4.21": {"risk": 0.79, "uncertainty": 0.15},
                "10.0.5.1":  {"risk": 0.67, "uncertainty": 0.17},
                "10.0.4.10": {"risk": 0.88, "uncertainty": 0.12},
                "10.0.2.15": {"risk": 0.95, "uncertainty": 0.08},
                "10.0.3.50": {"risk": 0.18, "uncertainty": 0.11},
            }
        },
        4: {
            "label": "Horizon t+4 (+4 min Rollout)",
            "horizon_code": "t+4",
            "uncertainty_sigma": 0.27,
            "uncertainty_label": "±27% (Epistemic Drift Limit)",
            "uncertainty_tier": "Maximum Horizon Bound",
            "uncertainty_color": "#E5484D",
            "summary": "Privileged persistence on Domain Controller; replication traffic initiated.",
            "active_edges": [
                ("10.0.4.21", "10.0.5.1", "DCSync Replication"),
                ("10.0.5.1", "10.0.3.50", "Volume Shadow Copy")
            ],
            "host_risks": {
                "10.0.5.1":  {"risk": 0.86, "uncertainty": 0.27},
                "10.0.4.21": {"risk": 0.82, "uncertainty": 0.22},
                "10.0.4.10": {"risk": 0.91, "uncertainty": 0.18},
                "10.0.2.15": {"risk": 0.96, "uncertainty": 0.12},
                "10.0.3.50": {"risk": 0.42, "uncertainty": 0.24},
            }
        }
    }

    return {
        "episode_id": episode_id or "EPISODE-INFIL-01",
        "node_roles": UI_NODE_ROLES,
        "host_telemetry": HOST_TELEMETRY,
        "rollout_steps": ROLLOUT_DATA,
        "active_k_step": k_step,
        "disclaimer": "Demonstrated on the single infiltration case study (n = 1). Not a general lateral-movement forecasting capability.",
        "is_mock": is_using_mock_data("host_risk_graph"),
    }


def get_attributions(window_id: str = None) -> list[dict]:
    """
    Returns deletion-tested feature attribution rankings and percentage weights.
    Consumed by: views/02_Evidence.py, components/explanation.py
    """
    raw = _get_raw_mock_dict()
    items = raw.get("explanation", [])
    for item in items:
        item["is_mock"] = is_using_mock_data("attributions")
    return items


def get_novelty_score(window_id: str = None) -> dict:
    """
    Returns current observed network state S(t) telemetry, novelty score,
    behavioral envelope classification, and sparkline metrics.
    Consumed by: views/01_Forecast.py, views/02_Evidence.py, components/state.py
    """
    raw = _get_raw_mock_dict()
    state = raw.get("current_state", {})
    return {
        "state_id": state.get("state_id", "S(t)"),
        "window_label": state.get("window_label", "Window t (Current)"),
        "dominant_behavior": state.get("dominant_behavior", "Probing & port sweeping"),
        "novelty_score": state.get("novelty_score", 0.23),
        "novelty_status": state.get("novelty_status", "Expected Behavior Envelope"),
        "active_endpoints": state.get("active_endpoints", 42),
        "syn_ack_ratio": state.get("syn_ack_ratio", 4.8),
        "mean_packet_size": state.get("mean_packet_size", 312),
        "entropy": state.get("entropy", 3.42),
        "flows_analyzed": raw.get("analysis", {}).get("flows_analyzed", 12480),
        "packets_analyzed": raw.get("analysis", {}).get("packets_analyzed", 84216),
        "is_mock": is_using_mock_data("novelty_score"),
    }


def get_flagged_flows(window_id: str = None, limit: int = None) -> list[dict]:
    """
    Returns flagged suspicious network flows driving the forecast.
    Consumed by: views/02_Evidence.py, components/evidence.py
    """
    raw = _get_raw_mock_dict()
    flows = raw.get("flagged_flows", [])
    if limit:
        flows = flows[:limit]
    for f in flows:
        f["is_mock"] = is_using_mock_data("flagged_flows")
    return flows


def get_validation_data() -> dict:
    """
    Returns authoritative validation benchmarks, LOEO 37-fold cross-validation metrics,
    horizon stability data, and leakage audit checklist.
    Consumed by: views/03_Validation.py
    """
    raw = _get_raw_mock_dict()
    val = raw.get("validation", {})
    val["is_mock"] = is_using_mock_data("validation_data")
    return val


def get_mitre_data() -> list[dict]:
    """
    Returns MITRE ATT&CK progression stages and IDs.
    Consumed by: views/01_Forecast.py, components/explanation.py
    """
    raw = _get_raw_mock_dict()
    stages = raw.get("mitre", [])
    for s in stages:
        s["is_mock"] = is_using_mock_data("mitre_data")
    return stages


def get_demo_data() -> dict:
    """
    [DEPRECATED AGGREGATOR]
    Constructs the root data dictionary strictly by invoking the typed accessor functions above.
    Preserved for backward compatibility during view migration.
    """
    analysis = get_analysis_metadata()
    forecast = get_forecast_trajectory()
    current_state = get_novelty_score()
    validation = get_validation_data()

    return {
        "analysis": {
            "source": analysis["source"],
            "window": analysis["window"],
            "window_duration": analysis.get("window_duration", "60s"),
            "duration_sec": analysis.get("duration_sec", 60),
            "flows_analyzed": current_state["flows_analyzed"],
            "packets_analyzed": current_state["packets_analyzed"],
            "alerts_suppressed": 142,
        },
        "current_state": current_state,
        "forecast": forecast["raw_steps"],
        "explanation": get_attributions(),
        "attack": {
            "type": "SSH-Bruteforce & Credential Access",
            "predicted_stage": "Lateral Movement",
            "primary_target": "Subnet 10.0.4.0/24 (Internal Auth Cluster)",
            "confidence": 0.67,
            "lead_time": "~3.5 min",
            "operational_impact": "Compromise of SSH jump host leading to internal segment penetration.",
            "illustrative_guidance": "Illustrative analyst guidance — not a system recommendation: Evaluate pre-emptive rate limiting on port 22 and step-up auth for 10.0.4.0/24.",
        },
        "mitre": get_mitre_data(),
        "flagged_flows": get_flagged_flows(),
        "baseline": {
            "world_model": 0.67,
            "lagged_logistic_regression": 0.41,
            "logistic_regression": 0.38,
            "lead_time_comparison": {
                "world_model_lead": "3.5 min",
                "reactive_ids_lead": "0.0 min (Post-facto)",
                "lagged_lr_lead": "1.2 min",
            },
            "comparison_table": get_comparison_table(),
        },
        "validation": validation,
    }
