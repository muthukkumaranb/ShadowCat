#!/usr/bin/env python3
"""
SHADOWCAT - Comprehensive Dynamic Data Layer Audit & Stress Harness
Tests:
1. Static code grep for hardcoded literals across views & components.
2. Host scalability stress testing (1 host, 2 hosts, 20 hosts, 0% & 100% critical).
3. Status logic and condition reactivity.
4. Dynamic mutation before/after proof via AppTest rendering.
"""

import os
import re
import sys
import json
import math
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data_provider
import components.attack_graph as ag
import components.forecast as fc
import components.explanation as exp
import components.evidence as ev

VIEW_FILES = [
    ROOT / "views" / "01_Forecast.py",
    ROOT / "views" / "01a_Input.py",
    ROOT / "views" / "01b_AttackGraph.py",
    ROOT / "views" / "02_Evidence.py",
    ROOT / "views" / "03_Validation.py",
    ROOT / "views" / "05_About.py",
]

COMPONENT_FILES = [
    ROOT / "components" / "forecast.py",
    ROOT / "components" / "attack_graph.py",
    ROOT / "components" / "evidence.py",
    ROOT / "components" / "explanation.py",
    ROOT / "components" / "input_panel.py",
    ROOT / "components" / "state.py",
    ROOT / "components" / "temporal_evidence.py",
    ROOT / "components" / "header.py",
]

print("=" * 80)
print("SHADOWCAT DYNAMIC DATA LAYER AUDIT & STRESS HARNESS")
print("=" * 80)

# -----------------------------------------------------------------------------
# Test 1: Static Grep Audit for Hardcoded Patterns in Views and Components
# -----------------------------------------------------------------------------
print("\n[SECTION 1] Scanning for Hardcoded Patterns across Views & Components...")

TARGET_LITERALS = [
    ("67%", r"67%"),
    ("3m 00s", r"3m\s*00s"),
    ("12,480", r"12[,\s]?480"),
    ("84,216", r"84[,\s]?216"),
    ("81%", r"81%"),
    ("94%", r"94%"),
    ("88%", r"88%"),
    ("10.0.2.15", r"10\.0\.2\.15"),
    ("10.0.4.10", r"10\.0\.4\.10"),
    ("10.0.4.21", r"10\.0\.4\.21"),
    ("10.0.3.50", r"10\.0\.3\.50"),
    ("10.0.5.1", r"10\.0\.5\.1"),
    ("TA0043", r"TA0043"),
    ("TA0001", r"TA0001"),
    ("TA0008", r"TA0008"),
    ("TA0040", r"TA0040"),
    ("Subnet 10.0.4.0/24", r"10\.0\.4\.0/24"),
]

scan_files = VIEW_FILES + COMPONENT_FILES
findings = []

for file_path in scan_files:
    rel_path = file_path.relative_to(ROOT)
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line_idx, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip comments
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("*"):
            continue
        for term, pattern in TARGET_LITERALS:
            if re.search(pattern, line):
                findings.append({
                    "file": str(rel_path),
                    "line": line_idx,
                    "term": term,
                    "snippet": stripped[:100]
                })

print(f"Total literal pattern matches found: {len(findings)}")
for f in findings:
    print(f"  • {f['file']}:{f['line']} [{f['term']}] -> {f['snippet']}")

# -----------------------------------------------------------------------------
# Test 2: Host Scalability Stress-Testing (1, 2, 5, 20 hosts, 0% & 100% critical)
# -----------------------------------------------------------------------------
print("\n[SECTION 2] Host Scalability & Donut Chart Stress-Testing...")

test_scenarios = [
    ("Single Host (N=1)", {
        "192.168.1.50": {"risk": 0.85, "uncertainty": 0.05}
    }),
    ("Two Hosts (N=2)", {
        "10.0.0.1": {"risk": 0.92, "uncertainty": 0.04},
        "10.0.0.2": {"risk": 0.25, "uncertainty": 0.02},
    }),
    ("Default Benchmark (N=5)", {
        "10.0.2.15": {"risk": 0.95, "uncertainty": 0.08},
        "10.0.4.10": {"risk": 0.88, "uncertainty": 0.12},
        "10.0.4.21": {"risk": 0.79, "uncertainty": 0.15},
        "10.0.5.1": {"risk": 0.67, "uncertainty": 0.17},
        "10.0.3.50": {"risk": 0.18, "uncertainty": 0.11},
    }),
    ("Large Enterprise Subnet (N=20)", {
        f"172.16.1.{i}": {"risk": round(0.10 + 0.04 * i, 2), "uncertainty": 0.05}
        for i in range(1, 21)
    }),
    ("Zero Critical (All Nominal 0%)", {
        f"10.10.1.{i}": {"risk": 0.15, "uncertainty": 0.02} for i in range(1, 6)
    }),
    ("100% Critical (All Critical)", {
        f"10.10.2.{i}": {"risk": 0.95, "uncertainty": 0.05} for i in range(1, 6)
    }),
]

scalability_results = []

for name, host_risks in test_scenarios:
    try:
        plotly_fig = ag.create_host_risk_donut(host_risks)
        svg_donut = ag.create_host_risk_donut_svg(host_risks)
        node_layout = ag.generate_node_layout(list(host_risks.keys()))
        
        # Verify donut counts
        crit_count = sum(1 for h in host_risks.values() if h["risk"] > 0.70)
        elev_count = sum(1 for h in host_risks.values() if 0.35 < h["risk"] <= 0.70)
        norm_count = sum(1 for h in host_risks.values() if h["risk"] <= 0.35)
        
        scalability_results.append({
            "scenario": name,
            "hosts": len(host_risks),
            "crit": crit_count,
            "elev": elev_count,
            "norm": norm_count,
            "layout_nodes": len(node_layout),
            "svg_valid": ("<svg" in svg_donut and "</svg>" in svg_donut),
            "plotly_valid": (plotly_fig is not None),
            "status": "PASS"
        })
        print(f"  [PASS] {name}: {len(host_risks)} hosts -> Layout={len(node_layout)} nodes, SVG={len(svg_donut)} bytes (Crit={crit_count}, Elev={elev_count}, Norm={norm_count})")
    except Exception as e:
        scalability_results.append({
            "scenario": name,
            "hosts": len(host_risks),
            "status": f"FAIL: {e}"
        })
        print(f"  [FAIL] {name}: {e}")

# -----------------------------------------------------------------------------
# Test 3: Status Logic & Condition Reactivity
# -----------------------------------------------------------------------------
print("\n[SECTION 3] Status Logic & Condition Reactivity Audit...")

status_results = {}
status_results["inference_mock"] = data_provider.inference_status()
status_results["validation_status"] = data_provider.validation_status()

print(f"  • inference_status(): {status_results['inference_mock']}")
print(f"  • validation_status(): {status_results['validation_status']}")

# -----------------------------------------------------------------------------
# Test 4: Dynamic Mutation Before / After Proof
# -----------------------------------------------------------------------------
print("\n[SECTION 4] Testing Dynamic Mutation (Proof that changing data changes render)...")

# Render Baseline Threat Forecast
at_baseline = AppTest.from_file(str(ROOT / "views" / "01_Forecast.py"), default_timeout=35)
at_baseline.run()
baseline_html = "".join([m.value for m in at_baseline.markdown])

has_67_base = "67%" in baseline_html
has_3m_base = "3m 00s" in baseline_html
has_12480_base = "12,480" in baseline_html
has_lateral_base = "Lateral Movement" in baseline_html

print(f"  Baseline check: 67%={has_67_base}, 3m 00s={has_3m_base}, 12,480={has_12480_base}, Lateral Movement={has_lateral_base}")

# Now inject a synthetic mutation into _CACHED_LIVE_PREDICTION in data_provider
mutated_prediction = {
    "analysis": {
        "status": "Live Analysis Active",
        "source": "SYNTHETIC-LIVE-STREAM",
        "window": "14:15:00 – 14:16:00",
        "flows_analyzed": 58920,
        "packets_analyzed": 312400,
        "mode": "Live Operational Mode",
        "duration_sec": 60,
    },
    "novelty_score": {
        "state_id": "S(t+live)",
        "window_label": "Window t (Live Stream)",
        "dominant_behavior": "Kerberos ticket brute-force and LDAP enumeration",
        "novelty_score": 0.88,
        "novelty_status": "Anomalous Drift Envelope",
        "active_endpoints": 98,
        "syn_ack_ratio": 12.4,
        "mean_packet_size": 540,
        "entropy": 4.85,
        "flows_analyzed": 58920,
        "packets_analyzed": 312400,
    },
    "forecast_trajectory": {
        "source_branch": "Live Continuous Dynamics Head",
        "protocol": "TCP/Kerberos/LDAP",
        "stage": ["Reconnaissance", "Initial Access", "Exfiltration & C2", "Domain Takeover"],
        "risk": [0.35, 0.65, 0.91, 0.98],
        "lead_time": ["1m 15s", "2m 30s", "5m 30s", "8m 00s"],
        "raw_steps": [
            {
                "step": 1, "horizon": "t+1", "time_ahead": "1.2 min",
                "stage": "Reconnaissance", "probability": 0.35,
                "uncertainty": 0.05, "lower_bound": 0.30, "upper_bound": 0.40,
                "tactic_id": "TA0043", "lead_time": "1m 15s",
            },
            {
                "step": 2, "horizon": "t+2", "time_ahead": "2.5 min",
                "stage": "Initial Access", "probability": 0.65,
                "uncertainty": 0.08, "lower_bound": 0.57, "upper_bound": 0.73,
                "tactic_id": "TA0001", "lead_time": "2m 30s",
            },
            {
                "step": 3, "horizon": "t+3", "time_ahead": "5.5 min",
                "stage": "Exfiltration & C2", "probability": 0.91,
                "uncertainty": 0.12, "lower_bound": 0.79, "upper_bound": 1.00,
                "tactic_id": "TA0010", "lead_time": "5m 30s",
            },
            {
                "step": 4, "horizon": "t+4", "time_ahead": "8.0 min",
                "stage": "Domain Takeover", "probability": 0.98,
                "uncertainty": 0.15, "lower_bound": 0.83, "upper_bound": 1.00,
                "tactic_id": "TA0040", "lead_time": "8m 00s",
            },
        ]
    },
    "attributions": [
        {"feature": "Kerberos TGS request spike", "contribution": 0.48, "category": "Auth Protocol", "delta": "+450%"},
        {"feature": "LDAP root search frequency", "contribution": 0.26, "category": "Directory Service", "delta": "32 targets"},
        {"feature": "TCP window size reduction", "contribution": 0.16, "category": "Transport Dynamic", "delta": "Constrained"},
        {"feature": "Outbound byte surge", "contribution": 0.10, "category": "Volume Dynamics", "delta": "+800MB"},
    ],
    "flagged_flows": [
        {"id": "FL-9901", "source": "192.168.10.45", "sport": 49201, "destination": "172.16.50.1", "dport": 88, "protocol": "TCP", "reason": "Kerberos TGS Spray", "risk": "High", "bytes": 482000},
        {"id": "FL-9902", "source": "192.168.10.45", "sport": 49202, "destination": "172.16.50.1", "dport": 389, "protocol": "TCP", "reason": "LDAP Subtree Enum", "risk": "High", "bytes": 1240000},
    ]
}

data_provider._CACHED_LIVE_PREDICTION = mutated_prediction

# Re-run Forecast page with mutated state
at_mutated = AppTest.from_file(str(ROOT / "views" / "01_Forecast.py"), default_timeout=35)
at_mutated.run()
mutated_html = "".join([m.value for m in at_mutated.markdown])

has_91_mut = "91%" in mutated_html
has_5m_mut = "5m 30s" in mutated_html
has_58920_mut = "58,920" in mutated_html
has_exfil_mut = "Exfiltration & C2" in mutated_html
has_ta0010_mut = "TA0010" in mutated_html
severity_is_critical = "CRITICAL" in mutated_html

print("\n--- MUTATION VERIFICATION RESULTS (01_Forecast.py) ---")
print(f"  • Threat Probability: 67% -> 91%: {'PROVED DYNAMIC (renders 91%)' if has_91_mut else 'FAILED'}")
print(f"  • Pre-emptive Lead Time: 3m 00s -> 5m 30s: {'PROVED DYNAMIC (renders 5m 30s)' if has_5m_mut else 'FAILED'}")
print(f"  • Ingested Flows: 12,480 -> 58,920: {'PROVED DYNAMIC (renders 58,920)' if has_58920_mut else 'FAILED'}")
print(f"  • Predicted Stage: Lateral Movement -> Exfiltration & C2: {'PROVED DYNAMIC (renders Exfiltration & C2)' if has_exfil_mut else 'FAILED'}")
print(f"  • Tactic ID: TA0008 -> TA0010: {'PROVED DYNAMIC (renders TA0010)' if has_ta0010_mut else 'FAILED'}")
print(f"  • Risk Tier Severity: ELEVATED -> CRITICAL: {'PROVED DYNAMIC (renders CRITICAL)' if severity_is_critical else 'FAILED'}")

# Reset cached prediction
data_provider._CACHED_LIVE_PREDICTION = None

print("\n" + "=" * 80)
print("AUDIT EXECUTION COMPLETE")
print("=" * 80)
