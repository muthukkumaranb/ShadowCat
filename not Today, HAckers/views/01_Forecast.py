"""
SHADOWCAT - Multi-Step Attack Forecasting
Autonomous Pre-Emptive Threat Trajectory Engine
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go
from mock_data import get_demo_data
from styles import apply_custom_css, render_sidebar, COLORS, render_html, render_footer
from components.header import render_header
from components.forecast import create_forecast_chart
from components.explanation import render_attack_stepper
from components.input_panel import render_input_panel
from components.attack_graph import render_attack_graph_panel

st.set_page_config(
    page_title="SHADOWCAT — Forecast",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()
data = get_demo_data()
render_header(data)

# Ingestion Panel (PCAP / CSV / Bundled Sample)
render_input_panel()

# 1. Actionable Metric Highlights
m1, m2, m3 = st.columns(3)
with m1:
    st.metric(
        label="Lead Time",
        value="~3 mins",
        delta="+3 min pre-emptive window",
        delta_color="normal"
    )
with m2:
    st.metric(
        label="Threat Risk",
        value="67%",
        delta="Horizon t+3 escalation (±18%)",
        delta_color="inverse"
    )
with m3:
    st.metric(
        label="Active Flows",
        value="12,480",
        delta="84,216 packets",
        delta_color="off"
    )

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

render_html("""
<div style="margin-bottom: 16px; display: flex; justify-content: space-between; align-items: flex-end;">
    <div>
        <h2 style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF; margin: 0;">
            Attack Trajectory Forecast
        </h2>
        <div style="font-size: 0.84rem; color: #94A3B8; margin-top: 4px;">
            Multi-step threat progression across horizons <code>t+1 → t+4</code> with compounding epistemic uncertainty bounds.
        </div>
    </div>
    <span class="badge" style="font-size: 0.72rem; padding: 4px 10px;">
        [Branch: Continuous Dynamics Head · Protocol: Chronological Split]
    </span>
</div>
""")

# 2. Main Trajectory Hero Chart (shaded uncertainty band conveys compounding variance)
st.plotly_chart(create_forecast_chart(data["forecast"]), use_container_width=True, config={"displayModeBar": False})

# 3. Interactive Horizon Deep-Dive Selector
render_html("""
<div class="card-title" style="margin-top: 24px; margin-bottom: 12px;">
    <span>Horizon Detail</span>
    <span class="badge">Step Inspection</span>
</div>
""")

step_options = [f"{s['horizon']} ({s['time_ahead']}) — {s['stage']}" for s in data["forecast"]]
selected_label = st.radio(
    "Select Rollout Horizon:",
    step_options,
    index=2,  # Default to t+3 (Lateral Movement)
    horizontal=True,
    label_visibility="collapsed"
)

selected_idx = step_options.index(selected_label)
sel = data["forecast"][selected_idx]

# Detailed Cards for Selected Step
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_html(f"""
    <div class="glass-card" style="min-height: 145px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div class="metric-label">Horizon</div>
            <div class="metric-value-huge" style="color: #00E5FF;">{sel['horizon']}</div>
        </div>
        <div style="font-size: 0.82rem; color: #E2E8F0;">
            Lead Time: <b>{sel['lead_time']}</b>
        </div>
    </div>
    """)

with col2:
    risk_color = "#00E676" if sel['probability'] < 0.25 else "#FFB300" if sel['probability'] < 0.50 else "#FF5252" if sel['probability'] < 0.75 else "#FF1744"
    render_html(f"""
    <div class="glass-card" style="min-height: 145px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div class="metric-label">Threat Risk</div>
            <div class="metric-value-huge" style="color: {risk_color};">{sel['probability']:.0%}</div>
        </div>
        <div style="font-size: 0.82rem; color: #E2E8F0;">
            Stage: <b>{sel['stage']}</b>
        </div>
    </div>
    """)

with col3:
    render_html(f"""
    <div class="glass-card" style="min-height: 145px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div class="metric-label">Uncertainty</div>
            <div class="metric-value-huge" style="color: #FFD54F;">±{sel['uncertainty']*100:.0f}%</div>
        </div>
        <div style="font-size: 0.82rem; color: #E2E8F0;">
            Bounds: <b>[{sel['lower_bound']:.0%} – {sel['upper_bound']:.0%}]</b>
        </div>
    </div>
    """)

with col4:
    error_mult = [1.0, 1.8, 3.0, 4.5][selected_idx]
    reliability = "High Confidence" if selected_idx <= 1 else "Elevated Risk" if selected_idx == 2 else "Exploratory"
    render_html(f"""
    <div class="glass-card" style="min-height: 145px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div class="metric-label">Drift Factor</div>
            <div class="metric-value-huge" style="color: #FFB300;">{error_mult:.1f}x</div>
        </div>
        <div style="font-size: 0.82rem; color: #E2E8F0;">
            Status: <b>{reliability}</b>
        </div>
    </div>
    """)

# 4. MITRE ATT&CK Killchain Pipeline (Task 2 - Kept on Forecast page)
render_html("""
<div class="card-title" style="margin-top: 24px; margin-bottom: 12px;">
    <span>MITRE ATT&CK Killchain Progression</span>
    <span class="badge">Pipeline Mapping</span>
</div>
""")
render_attack_stepper(data["mitre"], current_step_idx=2)

# 5. Dynamic Attack Graph with K-Step Rollout & Mandatory n=1 Caption
render_attack_graph_panel()

# 6. Response Lead Time vs Conventional IDS (Task 3 - redundant bar chart removed)
render_html("""
<div class="card-title" style="margin-top: 24px; margin-bottom: 12px;">
    <span>Response Lead Time vs Conventional IDS</span>
    <span class="badge">Pre-Emptive Window</span>
</div>
""")

render_html(f"""
<div class="glass-card">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px;">
        <div style="background: rgba(0, 229, 255, 0.08); border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 12px; padding: 18px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: 700; color: #FFFFFF; font-size: 1rem;">SHADOWCAT Engine</div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">Predicts attack trajectory before compromise</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.4rem; font-weight: 800; color: #00E5FF; font-family: 'JetBrains Mono', monospace;">+3.5 min</div>
                <div style="font-size: 0.72rem; color: #00E676; font-weight: 700; letter-spacing: 0.05em;">PRE-EMPTIVE</div>
            </div>
        </div>

        <div style="background: rgba(255, 179, 0, 0.08); border: 1px solid rgba(255, 179, 0, 0.25); border-radius: 12px; padding: 18px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: 700; color: #FFFFFF; font-size: 1rem;">Lagged Baseline</div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">Historical extrapolation without graph topology</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.4rem; font-weight: 800; color: #FFB300; font-family: 'JetBrains Mono', monospace;">+1.2 min</div>
                <div style="font-size: 0.72rem; color: #FFB300; font-weight: 700; letter-spacing: 0.05em;">PARTIAL LEAD</div>
            </div>
        </div>

        <div style="background: rgba(255, 23, 68, 0.08); border: 1px solid rgba(255, 23, 68, 0.25); border-radius: 12px; padding: 18px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: 700; color: #FFFFFF; font-size: 1rem;">Conventional IDS</div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">Reactive detection (alert fired post-compromise)</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.4rem; font-weight: 800; color: #FF1744; font-family: 'JetBrains Mono', monospace;">0.0 min</div>
                <div style="font-size: 0.72rem; color: #FF1744; font-weight: 700; letter-spacing: 0.05em;">POST-BREACH</div>
            </div>
        </div>
    </div>
</div>
""")

# Executive Persistent Footer
render_footer()

