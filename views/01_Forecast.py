"""
SHADOWCAT - Multi-Step Threat Trajectory & Operational Cockpit
Real-Time Forecasting Console with Interactive Rollout Inspection & Granular Provenance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import streamlit as st
import plotly.graph_objects as go
from data_provider import (
    get_analysis_metadata,
    get_forecast_trajectory,
    get_novelty_score,
    get_mitre_data,
)
from styles import apply_custom_css, render_sidebar, COLORS, render_html, render_footer
from components.header import render_header
from components.forecast import create_forecast_chart
from components.explanation import render_attack_stepper
import components.attack_graph
import importlib
importlib.reload(components.attack_graph)
from components.attack_graph import render_attack_graph_panel

st.set_page_config(
    page_title="SHADOWCAT — Threat Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Retrieve typed telemetry data
analysis = get_analysis_metadata()
forecast = get_forecast_trajectory()
current_state = get_novelty_score()
mitre = get_mitre_data()

# Render persistent header
render_header({"analysis": analysis})

# 0. Active Predicted Threat Alert Banner (High-Visibility Operational Cockpit Strip)
focal_idx = 2  # Active projected focal stage (Lateral Movement at t+3)
raw_steps = forecast.get("raw_steps", [])
if len(raw_steps) > focal_idx:
    focal_step = raw_steps[focal_idx]
elif raw_steps:
    focal_step = raw_steps[0]
else:
    focal_step = {}

pred_stage = focal_step.get("stage", "Lateral Movement")
pred_horizon = f"{focal_step.get('horizon', 't+3')} ({focal_step.get('time_ahead', '3 min')})"
pred_prob_raw = focal_step.get("probability", 0.67)
try:
    pred_prob = float(pred_prob_raw) if pred_prob_raw is not None else 0.0
    if math.isnan(pred_prob):
        pred_prob = 0.0
except (TypeError, ValueError):
    pred_prob = 0.0
pred_prob = max(0.0, min(1.0, pred_prob))
pred_lead_time = focal_step.get("lead_time", "+3.5 min")

# Color/severity thresholds: Critical >70%, Elevated >35%, Normal <=35%
if pred_prob > 0.70:
    severity_label = "CRITICAL"
    banner_bg = "rgba(229, 72, 77, 0.16)"
    banner_border = "#E5484D"
    badge_bg = "#E5484D"
    badge_text = "#FFFFFF"
    accent_color = "#FF453A"
elif pred_prob > 0.35:
    severity_label = "ELEVATED"
    banner_bg = "rgba(224, 152, 43, 0.14)"
    banner_border = "#E0982B"
    badge_bg = "#E0982B"
    badge_text = "#0A0E17"
    accent_color = "#FF9F0A"
else:
    severity_label = "NORMAL"
    banner_bg = "rgba(47, 184, 114, 0.12)"
    banner_border = "#2FB872"
    badge_bg = "#2FB872"
    badge_text = "#0A0E17"
    accent_color = "#30D158"

render_html(f"""
<div style="background: {banner_bg}; border: 1px solid {banner_border}; border-left: 5px solid {banner_border}; border-radius: 8px; padding: 12px 18px; margin-top: 8px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap; box-shadow: 0 4px 16px rgba(0,0,0,0.5);">
    <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
        <span style="background: {badge_bg}; color: {badge_text}; font-size: 0.70rem; font-weight: 800; padding: 3px 8px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.08em; text-transform: uppercase;">
            {severity_label}
        </span>
        <span style="font-size: 0.94rem; font-weight: 700; color: #FFFFFF; letter-spacing: 0.01em;">
            Active forecast &mdash; <span style="color: {accent_color}; font-weight: 800;">{pred_stage}</span> predicted at <span style="color: #FFFFFF; font-family: 'JetBrains Mono', monospace;">{pred_horizon}</span> &middot; <span style="color: {accent_color}; font-family: 'JetBrains Mono', monospace;">{pred_prob:.0%} probability</span> &middot; <span style="color: #8A8A8A; font-family: 'JetBrains Mono', monospace;">{pred_lead_time} pre-emptive window</span>
        </span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px; font-size: 0.72rem; color: #8A8A8A; font-family: 'JetBrains Mono', monospace;">
        <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: {accent_color}; box-shadow: 0 0 8px {accent_color};"></span>
        <span style="letter-spacing: 0.04em;">REAL-TIME FORECAST</span>
    </div>
</div>
""")

# 1. High-Density Borderless KPI Row (Divided by hairlines #262626)
render_html(f"""
<div style="display: flex; border: 1px solid #262626; border-radius: 8px; background: #141414; margin-top: 10px; margin-bottom: 14px;">
    <div style="flex: 1; padding: 14px 18px; border-right: 1px solid #262626;">
        <div style="font-size: 0.68rem; font-weight: 700; color: #8A8A8A; text-transform: uppercase; letter-spacing: 0.05em;">
            Pre-Emptive Lead Time
        </div>
        <div style="font-size: 1.65rem; font-weight: 800; color: #FFFFFF; font-family: 'JetBrains Mono', monospace; margin: 2px 0;">
            {pred_lead_time}
        </div>
        <div style="font-size: 0.70rem; color: #8A8A8A; font-weight: 500;">
            Active pre-breach window
        </div>
    </div>
    <div style="flex: 1; padding: 14px 18px; border-right: 1px solid #262626; background: rgba(224, 152, 43, 0.08);">
        <div style="font-size: 0.68rem; font-weight: 700; color: #E0982B; text-transform: uppercase; letter-spacing: 0.05em;">
            Projected Threat Risk
        </div>
        <div style="font-size: 1.65rem; font-weight: 800; color: #E0982B; font-family: 'JetBrains Mono', monospace; margin: 2px 0;">
            {pred_prob:.0%}
        </div>
        <div style="font-size: 0.70rem; color: #E0982B; font-weight: 600;">
            Escalation at Horizon {focal_step.get('horizon', 't+3')} (±{focal_step.get('uncertainty', 0.18)*100:.0f}%)
        </div>
    </div>
    <div style="flex: 1; padding: 14px 18px; border-right: 1px solid #262626;">
        <div style="font-size: 0.68rem; font-weight: 700; color: #8A8A8A; text-transform: uppercase; letter-spacing: 0.05em;">
            Active State Ingestion
        </div>
        <div style="font-size: 1.65rem; font-weight: 800; color: #FFFFFF; font-family: 'JetBrains Mono', monospace; margin: 2px 0;">
            {current_state['flows_analyzed']:,}
        </div>
        <div style="font-size: 0.70rem; color: #8A8A8A;">
            Flows · {current_state['packets_analyzed']:,} packets/s
        </div>
    </div>
    <div style="flex: 1; padding: 14px 18px;">
        <div style="font-size: 0.68rem; font-weight: 700; color: #8A8A8A; text-transform: uppercase; letter-spacing: 0.05em;">
            Predicted ATT&CK Stage
        </div>
        <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin: 4px 0;">
            {pred_stage}
        </div>
        <div style="font-size: 0.70rem; color: #8A8A8A; font-family: 'JetBrains Mono', monospace;">
            {focal_step.get('tactic_id', 'TA0008')} · Subnet 10.0.4.0/24
        </div>
    </div>
</div>
""")

# 2. History & Current State S(t) Strip (Panel §3 requirement)
render_html(f"""
<div style="display: flex; justify-content: space-between; align-items: center; background: #111111; border: 1px solid #262626; border-radius: 6px; padding: 10px 16px; margin-bottom: 16px;">
    <div style="display: flex; gap: 18px; align-items: center; flex-wrap: wrap;">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.80rem; font-weight: 700; color: #FFFFFF;">
            STATE {current_state['state_id']}: {current_state['dominant_behavior']}
        </span>
        <span style="color: #262626;">|</span>
        <span style="font-size: 0.76rem; color: #8A8A8A;">
            Active Endpoints: <b style="color: #FFFFFF;">{current_state['active_endpoints']}</b> · SYN/ACK Ratio: <b style="color: #E0982B;">{current_state['syn_ack_ratio']:.1f}x</b>
        </span>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 0.74rem; color: #8A8A8A;">Novelty Drift:</span>
        <span style="background: rgba(47, 184, 114, 0.12); border: 1px solid #2FB872; color: #2FB872; padding: 1px 7px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
            Nominal (In Envelope)
        </span>
    </div>
</div>
""")

# 3. Main Trajectory Chart Header
render_html(f"""
<div style="margin-bottom: 10px;">
    <div class="card-title">
        <span>Attack Risk Trajectory & Epistemic Uncertainty</span>
        <span class="badge">{forecast['source_branch']} · {forecast['protocol']}</span>
    </div>
    <div style="font-size: 0.78rem; color: #8A8A8A;">
        Multi-step forward trajectory simulation across horizons t+1 &rarr; t+4 with compounding uncertainty bounds.
    </div>
</div>
""")

# 4. Interactive Plotly Trajectory Chart
chart_fig = create_forecast_chart(forecast["raw_steps"])
chart_event = st.plotly_chart(
    chart_fig,
    use_container_width=True,
    on_select="rerun",
    selection_mode="points",
    key="trajectory_chart",
    config={"displayModeBar": False}
)

# Synchronize chart clicks with horizon state
if "selected_horizon_idx" not in st.session_state:
    st.session_state["selected_horizon_idx"] = 2  # default to t+3

if chart_event and hasattr(chart_event, "selection") and chart_event.selection and "points" in chart_event.selection:
    pts = chart_event.selection["points"]
    if pts and len(pts) > 0:
        pt_idx = pts[0].get("point_index", None)
        if pt_idx is not None and 1 <= pt_idx <= 4:
            st.session_state["selected_horizon_idx"] = pt_idx - 1

# 5. Horizon Deep-Dive Selector
step_options = [f"{s['horizon']} ({s['time_ahead']}) — {s['stage']}" for s in forecast["raw_steps"]]
selected_label = st.radio(
    "Select Rollout Horizon:",
    step_options,
    index=st.session_state["selected_horizon_idx"],
    horizontal=True,
    label_visibility="collapsed",
    key="horizon_radio_selector"
)
render_html("""
<div style="font-size: 0.72rem; color: #8A8A8A; margin-top: 4px; margin-bottom: 8px;">
    <i>Interactive rollout capped at t+4 &mdash; reliability drops to exploratory beyond this point (see Validation &rarr; Horizon Stability for H=5 benchmark bound).</i>
</div>
""")

selected_idx = step_options.index(selected_label)
st.session_state["selected_horizon_idx"] = selected_idx
sel = forecast["raw_steps"][selected_idx]

# 6. Horizon Metrics (Bordered high-density row with edge case sanitization)
sel_prob_raw = sel.get("probability", 0.0)
try:
    sel_prob = float(sel_prob_raw) if sel_prob_raw is not None else 0.0
    if math.isnan(sel_prob):
        sel_prob = 0.0
except (TypeError, ValueError):
    sel_prob = 0.0
sel_prob = max(0.0, min(1.0, sel_prob))

sel_unc_raw = sel.get("uncertainty", 0.0)
try:
    sel_unc = float(sel_unc_raw) if sel_unc_raw is not None else 0.0
    if math.isnan(sel_unc):
        sel_unc = 0.0
except (TypeError, ValueError):
    sel_unc = 0.0
sel_unc = max(0.0, min(1.0, sel_unc))

sel_lb_raw = sel.get("lower_bound", 0.0)
try:
    sel_lb = float(sel_lb_raw) if sel_lb_raw is not None else 0.0
    if math.isnan(sel_lb):
        sel_lb = 0.0
except (TypeError, ValueError):
    sel_lb = 0.0
sel_lb = max(0.0, min(1.0, sel_lb))

sel_ub_raw = sel.get("upper_bound", 1.0)
try:
    sel_ub = float(sel_ub_raw) if sel_ub_raw is not None else 1.0
    if math.isnan(sel_ub):
        sel_ub = 1.0
except (TypeError, ValueError):
    sel_ub = 1.0
sel_ub = max(0.0, min(1.0, sel_ub))

risk_color = "#2FB872" if sel_prob < 0.25 else "#E0982B" if sel_prob < 0.50 else "#E5484D"
error_mult = [1.0, 1.8, 3.0, 4.5][selected_idx]
reliability = "Automated Alerting" if selected_idx <= 1 else "Analyst Escalation" if selected_idx == 2 else "Exploratory Trend"

render_html(f"""
<div style="display: flex; border: 1px solid #262626; border-radius: 8px; background: #141414; margin-top: 10px; margin-bottom: 22px;">
    <div style="flex: 1; padding: 12px 16px; border-right: 1px solid #262626;">
        <div class="metric-label">Horizon</div>
        <div style="font-size: 1.45rem; font-weight: 800; color: #FFFFFF; font-family: 'JetBrains Mono', monospace;">{sel['horizon']}</div>
        <div style="font-size: 0.74rem; color: #8A8A8A;">Lead Time: <b style="color: #FFFFFF;">{sel['lead_time']}</b></div>
    </div>
    <div style="flex: 1; padding: 12px 16px; border-right: 1px solid #262626;">
        <div class="metric-label">Threat Risk</div>
        <div style="font-size: 1.45rem; font-weight: 800; color: {risk_color}; font-family: 'JetBrains Mono', monospace;">{sel_prob:.0%}</div>
        <div style="font-size: 0.74rem; color: #8A8A8A;">Stage: <b style="color: #FFFFFF;">{sel['stage']}</b></div>
    </div>
    <div style="flex: 1; padding: 12px 16px; border-right: 1px solid #262626;">
        <div class="metric-label">Epistemic Variance</div>
        <div style="font-size: 1.45rem; font-weight: 800; color: #E0982B; font-family: 'JetBrains Mono', monospace;">±{sel_unc*100:.0f}%</div>
        <div style="font-size: 0.74rem; color: #8A8A8A;">Bounds: <b style="color: #FFFFFF;">[{sel_lb:.0%} – {sel_ub:.0%}]</b></div>
    </div>
    <div style="flex: 1; padding: 12px 16px;">
        <div class="metric-label">Drift Multiplier</div>
        <div style="font-size: 1.45rem; font-weight: 800; color: #FFFFFF; font-family: 'JetBrains Mono', monospace;">{error_mult:.1f}x</div>
        <div style="font-size: 0.74rem; color: #8A8A8A; font-weight: 600;">{reliability}</div>
    </div>
</div>
""")

# 7. Continuous MITRE ATT&CK Killchain Pipeline
render_html("""
<div style="margin-top: 18px; margin-bottom: 8px;">
    <div class="card-title">
        <span>MITRE ATT&CK Killchain Progression</span>
    </div>
</div>
""")
render_attack_stepper(mitre, current_step_idx=2)

# 8. Dynamic Force-Directed Attack Graph (Agraph)
render_attack_graph_panel()

# 9. Response Lead Time vs Baselines
render_html("""
<div style="margin-top: 26px; margin-bottom: 10px;">
    <div class="card-title">
        <span>Pre-Emptive Response Lead Time vs Baselines</span>
    </div>
</div>
""")

render_html("""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin-bottom: 22px;">
    <div style="background: #141414; border: 1px solid #262626; border-radius: 8px; padding: 14px 16px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.92rem;">SHADOWCAT World Model</div>
            <div style="font-size: 0.74rem; color: #8A8A8A; margin-top: 2px;">Pre-lateral trajectory forecasting</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; font-family: 'JetBrains Mono', monospace;">+3.5 min</div>
            <div style="font-size: 0.68rem; color: #2FB872; font-weight: 700; letter-spacing: 0.04em;">PRE-EMPTIVE</div>
        </div>
    </div>

    <div style="background: #141414; border: 1px solid #262626; border-radius: 8px; padding: 14px 16px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.92rem;">Lagged Autoregressive</div>
            <div style="font-size: 0.74rem; color: #8A8A8A; margin-top: 2px;">History extrapolation without topology</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 1.35rem; font-weight: 800; color: #E0982B; font-family: 'JetBrains Mono', monospace;">+1.2 min</div>
            <div style="font-size: 0.68rem; color: #E0982B; font-weight: 700; letter-spacing: 0.04em;">PARTIAL LEAD</div>
        </div>
    </div>

    <div style="background: #141414; border: 1px solid #262626; border-radius: 8px; padding: 14px 16px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.92rem;">Conventional Static IDS</div>
            <div style="font-size: 0.74rem; color: #8A8A8A; margin-top: 2px;">Reactive post-compromise detection</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 1.35rem; font-weight: 800; color: #E5484D; font-family: 'JetBrains Mono', monospace;">0.0 min</div>
            <div style="font-size: 0.68rem; color: #E5484D; font-weight: 700; letter-spacing: 0.04em;">POST-BREACH</div>
        </div>
    </div>
</div>
""")

render_footer()
