"""
SHADOWCAT - Platform Overview & Mission Doctrine
Pre-Emptive Cyber Threat Forecasting Platform
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from mock_data import get_demo_data
from styles import apply_custom_css, render_sidebar, COLORS, render_html, render_footer
from components.header import render_header

st.set_page_config(
    page_title="SHADOWCAT — About",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()
data = get_demo_data()
render_header(data)

render_html("""
<div style="margin-bottom: 20px;">
    <h2 style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF; margin: 0;">
        About SHADOWCAT
    </h2>
    <div style="font-size: 0.84rem; color: #94A3B8; margin-top: 4px;">
        Autonomous pre-emptive cyber threat forecasting engine for air-gapped enterprise defense.
    </div>
</div>
""")

# 1. Mission Doctrine Overview
render_html("""
<div class="glass-card" style="margin-bottom: 22px;">
    <div class="card-title">
        <span>Executive Mission & Platform Vision</span>
        <span class="badge">Operational Doctrine</span>
    </div>
    <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.7;">
        <b>SHADOWCAT</b> represents a fundamental shift in network security operations: transitioning cybersecurity from 
        <b>reactive forensic alert triage</b> to <b>pre-emptive threat trajectory forecasting</b>.
        By modeling network traffic kinematics and host communication graphs as a continuous dynamical system, 
        SHADOWCAT predicts impending multi-step attack escalations up to <b>3 to 4 minutes before privilege escalation</b> occurs.
    </div>
</div>
""")

# 2. Paradigm Shift: Reactive Detection vs Pre-Emptive Forecasting (Moved from Evidence per Task 4)
render_html("""
<div class="glass-card" style="margin-bottom: 22px;">
    <div class="card-title">
        <span>Paradigm Shift: Reactive Detection vs Pre-Emptive Forecasting</span>
        <span class="badge">Comparative Analysis</span>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div style="background: rgba(255, 23, 68, 0.08); border: 1px solid rgba(255, 23, 68, 0.25); border-radius: 14px; padding: 20px;">
            <div style="font-weight: 800; color: #FF5252; font-size: 1.1rem; margin-bottom: 10px;">
                Conventional IDS / IPS
            </div>
            <ul style="font-size: 0.84rem; color: #CBD5E1; margin-left: -15px; line-height: 1.7;">
                <li><b>Posture:</b> Reactive (alerts fired post-compromise).</li>
                <li><b>Lead Time:</b> <b>0 minutes</b> (Zero pre-emptive response window).</li>
                <li><b>Detection:</b> Static signatures & single-point packet anomalies.</li>
                <li><b>SOC Impact:</b> High alert fatigue during active damage phase.</li>
            </ul>
        </div>
        <div style="background: rgba(0, 229, 255, 0.08); border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 14px; padding: 20px;">
            <div style="font-weight: 800; color: #00E5FF; font-size: 1.1rem; margin-bottom: 10px;">
                SHADOWCAT Forecasting Engine
            </div>
            <ul style="font-size: 0.84rem; color: #FFFFFF; margin-left: -15px; line-height: 1.7;">
                <li><b>Posture:</b> Pre-emptive (forecasts trajectory before breach).</li>
                <li><b>Lead Time:</b> <b>~3–4 minutes</b> pre-emptive intervention window.</li>
                <li><b>Detection:</b> Temporal dynamics + host graph topology.</li>
                <li><b>SOC Impact:</b> Automated rate-limiting & proactive host quarantine.</li>
            </ul>
        </div>
    </div>
</div>
""")

# 3. Key Operational Advantages
render_html("""
<div class="card-title" style="margin-bottom: 12px;">
    <span>Core Operational Capabilities</span>
    <span class="badge">Architecture Highlights</span>
</div>
""")

col_c1, col_c2, col_c3 = st.columns(3)

with col_c1:
    render_html("""
    <div class="glass-card" style="min-height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div style="font-size: 0.98rem; font-weight: 800; color: #00E5FF; margin-bottom: 6px;">
                Multi-Horizon Rollout
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.55;">
                Autoregressive state transition modeling forecasts cyber attack stages across horizons K=1..4 with calibrated lead times.
            </div>
        </div>
        <div style="font-size: 0.74rem; color: #00E5FF; font-weight: 700; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
            Horizons: t+1 → t+4
        </div>
    </div>
    """)

with col_c2:
    render_html("""
    <div class="glass-card" style="min-height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div style="font-size: 0.98rem; font-weight: 800; color: #FFB300; margin-bottom: 6px;">
                Compounding Uncertainty
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.55;">
                Epistemic uncertainty quantification explicitly reflects compounding drift over future rollout steps to guide analyst escalation.
            </div>
        </div>
        <div style="font-size: 0.74rem; color: #FFB300; font-weight: 700; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
            Calibrated Variance Bounds
        </div>
    </div>
    """)

with col_c3:
    render_html("""
    <div class="glass-card" style="min-height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div style="font-size: 0.98rem; font-weight: 800; color: #00E676; margin-bottom: 6px;">
                Air-Gapped Operation
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.55;">
                Zero cloud reliance, zero external telemetry egress. Engineered for strict mission-critical offline SOC deployments.
            </div>
        </div>
        <div style="font-size: 0.74rem; color: #00E676; font-weight: 700; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
            100% On-Premise Execution
        </div>
    </div>
    """)

# Persistent Executive Footer
render_footer()
