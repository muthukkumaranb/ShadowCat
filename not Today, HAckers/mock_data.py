"""
SHADOWCAT - Mock Telemetry & Inference Data Module
"""

DEMO_DATA = {
    "analysis": {
        "status": "Analysis Ready",
        "source": "CSE-CIC-IDS2018",
        "window": "10:30:00 – 10:31:00",
        "flows_analyzed": 12480,
        "packets_analyzed": 84216,
        "mode": "Offline Analysis (Air-Gapped)",
        "duration_sec": 60,
    },

    "current_state": {
        "state_id": "S(t)",
        "window_label": "Window t (Current)",
        "dominant_behavior": "Suspicious connection probing & port sweeping",
        "novelty_score": 0.23,  # "One model, two signals" — flags unseen behaviour
        "novelty_status": "Expected Behavior Envelope",
        "active_endpoints": 42,
        "syn_ack_ratio": 4.8,
        "mean_packet_size": 312,
        "entropy": 3.42,
    },

    # Multi-step autoregressive rollout: S(t+1) ... S(t+k)
    # Notice: Uncertainty grows monotonically with horizon K (reflecting compounding rollout error)
    "forecast": [
        {
            "step": 1,
            "horizon": "t+1",
            "time_ahead": "1 min",
            "stage": "Reconnaissance",
            "probability": 0.21,
            "uncertainty": 0.06,
            "lower_bound": 0.15,
            "upper_bound": 0.27,
            "tactic_id": "TA0043",
            "confidence_band": "High Confidence (±6%)",
            "lead_time": "1m 00s",
        },
        {
            "step": 2,
            "horizon": "t+2",
            "time_ahead": "2 min",
            "stage": "Initial Access",
            "probability": 0.38,
            "uncertainty": 0.11,
            "lower_bound": 0.27,
            "upper_bound": 0.49,
            "tactic_id": "TA0001",
            "confidence_band": "Moderate Confidence (±11%)",
            "lead_time": "2m 00s",
        },
        {
            "step": 3,
            "horizon": "t+3",
            "time_ahead": "3 min",
            "stage": "Lateral Movement",
            "probability": 0.67,
            "uncertainty": 0.18,
            "lower_bound": 0.49,
            "upper_bound": 0.85,
            "tactic_id": "TA0008",
            "confidence_band": "Elevated Risk / Widening Variance (±18%)",
            "lead_time": "3m 00s",
        },
        {
            "step": 4,
            "horizon": "t+4",
            "time_ahead": "4 min",
            "stage": "Impact",
            "probability": 0.81,
            "uncertainty": 0.27,
            "lower_bound": 0.54,
            "upper_bound": 1.00,
            "tactic_id": "TA0040",
            "confidence_band": "High Uncertainty / Compounding Drift (±27%)",
            "lead_time": "4m 00s",
        },
    ],

    # Explainability: Feature-group ablation & deletion validation
    "explanation": [
        {"feature": "SYN packet rate", "contribution": 0.34, "category": "Flow Dynamics", "delta": "+280%"},
        {"feature": "Connection attempt frequency", "contribution": 0.28, "category": "Graph Topology", "delta": "14 targets"},
        {"feature": "TTL variance", "contribution": 0.21, "category": "Packet Header", "delta": "Abnormal hops"},
        {"feature": "Inter-arrival timing jitter", "contribution": 0.12, "category": "Temporal Rhythm", "delta": "Paced pulses"},
        {"feature": "Payload entropy distribution", "contribution": 0.05, "category": "Payload Stats", "delta": "Baseline"},
    ],

    "attack": {
        "type": "SSH-Bruteforce & Credential Access",
        "predicted_stage": "Lateral Movement",
        "primary_target": "Subnet 10.0.4.0/24 (Internal Auth Cluster)",
        "confidence": 0.67,
        "lead_time": "~3–4 min",
        "operational_impact": "Compromise of SSH jump host leading to internal segment penetration and persistence.",
        "illustrative_guidance": "Illustrative analyst guidance — not a system recommendation: Evaluate pre-emptive rate limiting on port 22 and step-up auth for 10.0.4.0/24.",
    },

    "mitre": [
        {"stage": "Reconnaissance", "id": "TA0043", "description": "Port scan: 22, 80"},
        {"stage": "Initial Access", "id": "TA0001", "description": "Auth spray: boundary endpoints"},
        {"stage": "Lateral Movement", "id": "TA0008", "description": "SSH session pivot: internal hosts"},
        {"stage": "Impact", "id": "TA0040", "description": "Privileged service disruption"},
    ],

    "flagged_flows": [
        {
            "id": "FLW-10492",
            "source": "10.0.2.15",
            "sport": 54102,
            "destination": "10.0.4.21",
            "dport": 22,
            "protocol": "TCP",
            "reason": "SYN burst (140 pkts/s)",
            "risk": "High",
            "pkts": 412,
            "bytes": 26368,
        },
        {
            "id": "FLW-10518",
            "source": "10.0.2.18",
            "sport": 49120,
            "destination": "10.0.4.21",
            "dport": 22,
            "protocol": "TCP",
            "reason": "Repeated connection resets (RST)",
            "risk": "Medium",
            "pkts": 218,
            "bytes": 13952,
        },
        {
            "id": "FLW-10577",
            "source": "10.0.2.31",
            "sport": 58310,
            "destination": "10.0.4.10",
            "dport": 22,
            "protocol": "SSH",
            "reason": "Rapid SSH banner exchange failure",
            "risk": "High",
            "pkts": 594,
            "bytes": 48210,
        },
        {
            "id": "FLW-10602",
            "source": "10.0.2.15",
            "sport": 54118,
            "destination": "10.0.4.10",
            "dport": 443,
            "protocol": "TCP",
            "reason": "Target discovery reconnaissance probe",
            "risk": "Low",
            "pkts": 48,
            "bytes": 3120,
        },
    ],

    # Comparative evaluation: World Model vs baselines on CSE-CIC-IDS2018 (PS Evaluation Deliverable)
    "baseline": {
        "world_model": 0.67,
        "lagged_logistic_regression": 0.41,
        "logistic_regression": 0.38,
        "lead_time_comparison": {
            "world_model_lead": "3.5 min",
            "reactive_ids_lead": "0.0 min (Post-facto)",
            "lagged_lr_lead": "1.2 min",
        },
        "comparison_table": [
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
    },

    # Scientific validation metadata (Slide 3 & 4 accountability)
    "validation": {
        "metrics": {
            "precision": 0.842,
            "recall": 0.791,
            "f1_score": 0.816,
            "pr_auc": 0.835,
            "fpr": 0.048,
        },
        "loeo_summary": {
            "protocol": "Leave-One-Episode-Out (LOEO 37-Fold Cross-Validation)",
            "description": "Evaluated across 37 held-out episodic attack sequences to eliminate temporal and schedule leakage.",
            "folds_tested": 37,
            "mean_pr_auc": 0.821,
            "variance": 0.014,
        },
        "horizons": [
            {"horizon": "t+1 (1 min)", "status": "Reliable", "f1": 0.88, "error_growth": "Low", "verdict": "Validated for automated alerting"},
            {"horizon": "t+2 (2 min)", "status": "Reliable", "f1": 0.82, "error_growth": "Controlled", "verdict": "Validated for SOC queue prioritization"},
            {"horizon": "t+3 (3 min)", "status": "Informative", "f1": 0.74, "error_growth": "Moderate", "verdict": "Validated for human analyst escalation review"},
            {"horizon": "t+4 (4 min)", "status": "Exploratory", "f1": 0.63, "error_growth": "Compounding", "verdict": "Informational trend indicator only"},
            {"horizon": "t+5 (5 min) [H=5 Primary]", "status": "Benchmark Bound", "f1": 0.58, "error_growth": "Epistemic Limit", "verdict": "Primary DE evaluation horizon (H=5) under LOEO 37-fold"},
        ],
        "leakage_controls": [
            {"layer": "Temporal Partitioning", "check": "Passed", "detail": "Strict chronological split; no future packets in state window"},
            {"layer": "Graph Neighbor Sampling", "check": "Passed", "detail": "Sub-graph induction restricted to historical edges t_window"},
            {"layer": "Feature Standardization", "check": "Passed", "detail": "Scaler fitted strictly on training episodes; no test contamination"},
            {"layer": "State Normalization", "check": "Passed", "detail": "Flow rate bounds clamped using robust training quantiles"},
        ],
        "schedule_controls": [
            {"artifact": "Periodic cron traffic in IDS2018", "mitigation": "Timestamp ablation & jitter perturbation test completed"},
            {"artifact": "Subnet IP bias", "mitigation": "Topology permutation test validated with GraphSAGE"},
        ]
    }
}


def get_demo_data():
    """Returns the single-source demo data contract dictionary."""
    return DEMO_DATA
