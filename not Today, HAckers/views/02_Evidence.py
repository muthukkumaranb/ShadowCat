"""
SHADOWCAT - Telemetry Evidence & Forensic Benchmarks
Operational Network Flows, Feature Attribution, and Baseline Evaluations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go
from mock_data import get_demo_data
from styles import apply_custom_css, render_sidebar, COLORS, render_html, render_footer
from components.header import render_header
from components.explanation import create_attribution_chart
from components.evidence import render_evidence_table
from components.temporal_evidence import render_temporal_evidence

st.set_page_config(
    page_title="SHADOWCAT — Evidence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()
data = get_demo_data()
render_header(data)

render_html("""
<div style="margin-bottom: 18px;">
    <h2 style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF; margin: 0;">
        Telemetry Evidence & Forensic Attribution
    </h2>
    <div style="font-size: 0.84rem; color: #94A3B8; margin-top: 4px;">
        Correlated flow evidence, deletion-tested feature attribution, and operational baseline comparisons.
    </div>
</div>
""")

# 1. Feature Attribution, Baseline Comparison, and Live Dual-Signal Evaluation
col_att, col_base, col_sig = st.columns([1.35, 0.85, 1.15])

with col_att:
    render_html("""
    <div class="card-title">
        <span>Feature Attribution</span>
        <span class="badge">[Protocol: Deletion-Tested Attribution]</span>
    </div>
    """)
    st.plotly_chart(create_attribution_chart(data["explanation"]), use_container_width=True, config={"displayModeBar": False})

with col_base:
    render_html("""
    <div class="card-title">
        <span>Baseline PR-AUC Comparison</span>
        <span class="badge">[Protocol: Chronological Split]</span>
    </div>
    """)

    base_fig = go.Figure()
    base_fig.add_trace(go.Bar(
        x=["Forecasting Engine", "Lagged Baseline", "Standard Baseline"],
        y=[data["baseline"]["world_model"], data["baseline"]["lagged_logistic_regression"], data["baseline"]["logistic_regression"]],
        marker=dict(
            color=["#00E5FF", "#FFB300", "#64748B"],
            line=dict(color="rgba(255, 255, 255, 0.2)", width=1)
        ),
        text=[f"<b>{data['baseline']['world_model']:.2f}</b>", f"<b>{data['baseline']['lagged_logistic_regression']:.2f}</b>", f"<b>{data['baseline']['logistic_regression']:.2f}</b>"],
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11, family="'JetBrains Mono', monospace")
    ))
    base_fig.update_layout(
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        margin=dict(l=25, r=20, t=30, b=30),
        height=240,
        yaxis=dict(range=[0, 0.88], tickformat=".2f", gridcolor="rgba(255,255,255,0.05)", title=dict(text="PR-AUC", font=dict(color="#94A3B8", size=10))),
        xaxis=dict(tickfont=dict(color="#FFFFFF", size=10, family="'Plus Jakarta Sans', sans-serif"))
    )
    st.plotly_chart(base_fig, use_container_width=True, config={"displayModeBar": False})

with col_sig:
    state = data["current_state"]
    render_html(f"""
    <div class="card-title">
        <span>Dual-Signal Evaluation</span>
        <span class="badge">[Operational Telemetry]</span>
    </div>
    <div class="glass-card" style="height: 240px; display: flex; flex-direction: column; justify-content: space-between; padding: 18px 20px;">
        <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span class="metric-label" style="margin: 0;">Novelty Score</span>
                <span style="background: rgba(0, 230, 118, 0.12); border: 1px solid #00E676; color: #00E676; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.78rem;">
                    {state['novelty_score']:.2f} / 1.0 · Normal Baseline
                </span>
            </div>
            <div style="font-size: 1.12rem; font-weight: 800; color: #FFFFFF; line-height: 1.25; margin-top: 2px;">
                Known Attack Escalation
            </div>
            <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 10px; line-height: 1.55;">
                • <b style="color: #FF5252;">Risk (67%):</b> Lateral pivot predicted at horizon <b>t+3</b>.<br>
                • <b style="color: #00E676;">Novelty ({state['novelty_score']:.2f}):</b> Known credential spray pattern (within normal envelope).
            </div>
        </div>
        <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 8px; display: flex; justify-content: space-between; align-items: center; font-size: 0.72rem; color: #94A3B8;">
            <span>Separates familiar attack trajectories from unfamiliar baseline drift.</span>
            <span style="color: #64748B; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;">[Model Info]</span>
        </div>
    </div>
    """)

# 2. Temporal Context & Attention Distribution (Model Internals Only)
render_temporal_evidence(data)

# 3. Correlated Flagged Network Telemetry Flows
render_evidence_table(data)

# Persistent Executive Footer
render_footer()

