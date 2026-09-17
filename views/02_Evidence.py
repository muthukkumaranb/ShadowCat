"""
SHADOWCAT - Telemetry Evidence & Forensic Benchmarks
Operational Network Flows, Feature Attribution, and Baseline Evaluations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import streamlit as st
import plotly.graph_objects as go
from data_provider import (
    get_analysis_metadata,
    get_attributions,
    get_novelty_score,
    get_flagged_flows,
)
import importlib
import styles
importlib.reload(styles)
from styles import apply_custom_css, render_sidebar, COLORS, render_html, render_footer
import components.header
importlib.reload(components.header)
from components.header import render_header
import components.explanation
importlib.reload(components.explanation)
from components.explanation import create_attribution_chart
import components.evidence
importlib.reload(components.evidence)
from components.evidence import render_evidence_table
import components.temporal_evidence
importlib.reload(components.temporal_evidence)
from components.temporal_evidence import render_temporal_evidence

st.set_page_config(
    page_title="SHADOWCAT — Evidence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

apply_custom_css()

# Retrieve typed telemetry data
analysis = get_analysis_metadata()
attributions = get_attributions()
state = get_novelty_score()
flagged_flows = get_flagged_flows()

render_header({"analysis": analysis}, active_tab="Evidence & Attribution")

render_html("""
<div style="margin-bottom: 14px;">
    <h2 class="page-title">
        Telemetry Evidence & Forensic Attribution
    </h2>
    <div class="page-caption">
        Correlated flow evidence and feature attribution rankings.
    </div>
</div>
""")

# 1. Feature Attribution and Live Dual-Signal Evaluation (Two-Column Layout)
col_att, col_sig = st.columns([1.55, 1.45])

with col_att:
    render_html("""
    <div class="card-title" style="display: flex; align-items: center; gap: 8px;">
        <span>Feature Attribution</span>
        <span class="info-tooltip-wrapper">
            <span class="info-icon">i</span>
            <span class="info-tooltip-box tooltip-left">
                <span class="tooltip-header">Protocol: Deletion-Tested Attribution</span>
                Measures predictive drop when features are systematically ablated from input sequence windows, establishing certified causal importance rather than raw internal attention correlation.
            </span>
        </span>
    </div>
    """)
    st.plotly_chart(create_attribution_chart(attributions), width="stretch", config={"displayModeBar": False})

with col_sig:
    nov_val_raw = state.get("novelty_score", 0.23)
    try:
        nov_val = float(nov_val_raw) if nov_val_raw is not None else 0.0
        if math.isnan(nov_val):
            nov_val = 0.0
    except (TypeError, ValueError):
        nov_val = 0.0
    nov_val = max(0.0, min(1.0, nov_val))

    render_html(f"""
    <div class="card-title">
        <span>Dual-Signal Evaluation</span>
    </div>
    <div class="glass-card" style="height: 240px; display: flex; flex-direction: column; justify-content: space-between; padding: 18px 20px;">
        <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span class="metric-label" style="margin: 0;">Novelty Score</span>
                <span style="background: rgba(47, 184, 114, 0.12); border: 1px solid #2FB872; color: #2FB872; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.78rem;">
                    {nov_val:.2f} / 1.0 · Normal Baseline
                </span>
            </div>
            <div style="font-size: 1.10rem; font-weight: 800; color: #FFFFFF; line-height: 1.25; margin-top: 2px;">
                Known Attack Escalation
            </div>
            <div style="font-size: 0.78rem; color: #FFFFFF; margin-top: 10px; line-height: 1.55;">
                • <b style="color: #E5484D;">Predicted Threat:</b> Lateral pivot projected at horizon <b>t+3</b>.<br>
                • <b style="color: #2FB872;">Baseline Novelty:</b> Known credential spray pattern (within normal envelope).
            </div>
        </div>
        <div style="border-top: 1px solid #262626; padding-top: 8px; font-size: 0.72rem; color: #8A8A8A;">
            Separates familiar attack trajectories from unfamiliar baseline drift.
        </div>
    </div>
    """)

# 2. Temporal Context & Attention Distribution (Model Internals Only)
render_temporal_evidence({"analysis": analysis, "forecast": []})

# 3. Correlated Flagged Network Telemetry Flows
render_evidence_table({"flagged_flows": flagged_flows})

# 4. Dynamic Analyst Guidance Callout
high_risk = [f for f in flagged_flows if f.get("risk") == "High"]
top_flow = high_risk[0] if high_risk else (flagged_flows[0] if flagged_flows else {})
target_socket = f"{top_flow.get('destination', '10.0.4.21')}:{top_flow.get('dport', 22)}"
indicator = top_flow.get("reason", "SYN burst")

render_html(f"""
<div style="background: #171717; border: 1px solid #282828; border-left: 3px solid #38BDF8; border-radius: 6px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45); padding: 12px 16px; margin-top: 12px; margin-bottom: 20px;">
    <div style="font-size: 0.80rem; color: #FFFFFF; line-height: 1.5;">
        <b style="color: #38BDF8;">Containment guidance:</b>
        Evaluate ingress rate-limiting and host isolation for socket <b class="soc-mono">{target_socket}</b> triggered by {indicator.lower()}.
    </div>
</div>
""")

render_footer()
