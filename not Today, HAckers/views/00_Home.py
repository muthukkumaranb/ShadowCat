"""
SHADOWCAT - Executive Home Portal
Autonomous Pre-Emptive Cyber Threat Forecasting Engine
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from data_provider import get_demo_data, is_using_mock_data, get_mock_badge_html
from styles import apply_custom_css, render_sidebar, COLORS, render_html, render_footer
from components.header import render_header

st.set_page_config(
    page_title="SHADOWCAT — Home",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()
data = get_demo_data()
render_header(data)

mock_badge_val = get_mock_badge_html("validation_data")

# -----------------------------------------------------------------------------
# 1. HERO
# -----------------------------------------------------------------------------
render_html("""
<div class="glass-card" style="margin-bottom: 20px; padding: 26px 30px; border-left: 4px solid #00E5FF;">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
        <div>
            <div style="font-size: 2.3rem; font-weight: 900; letter-spacing: 0.04em; color: #FFFFFF; line-height: 1.1;">
                SHADOWCAT
            </div>
            <div style="font-size: 1.05rem; font-weight: 600; color: #94A3B8; margin-top: 6px; letter-spacing: 0.01em;">
                Autonomous Pre-Emptive Cyber Threat Forecasting Engine
            </div>
            <div style="display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap; align-items: center;">
                <span class="badge-offline" style="padding: 3px 10px; font-size: 0.72rem;">
                    <span class="status-dot"></span>
                    Air-Gapped
                </span>
                <span class="badge" style="background: rgba(0, 229, 255, 0.1); border-color: rgba(0, 229, 255, 0.3); font-size: 0.72rem; padding: 3px 10px;">
                    MITRE ATT&CK Aligned
                </span>
            </div>
        </div>
    </div>
</div>
""")

col_hero_btn, col_hero_fill = st.columns([1.8, 4.2])
with col_hero_btn:
    if st.button("Open Forecast Cockpit", key="btn_hero_launch_demo", use_container_width=True, type="primary"):
        st.switch_page("views/01_Forecast.py")

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. KPI STRIP
# -----------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    render_html("""
    <div class="glass-card" style="padding: 16px 20px; text-align: center;">
        <div class="metric-value-huge" style="color: #00E5FF; margin: 0;">~3.5 min</div>
        <div class="metric-label" style="margin-top: 4px;">Pre-emptive lead time</div>
    </div>
    """)

with k2:
    render_html("""
    <div class="glass-card" style="padding: 16px 20px; text-align: center;">
        <div class="metric-value-huge" style="color: #FFFFFF; margin: 0;">K=4</div>
        <div class="metric-label" style="margin-top: 4px;">Forecast horizon</div>
    </div>
    """)

with k3:
    render_html(f"""
    <div class="glass-card" style="padding: 16px 20px; text-align: center;">
        <div class="metric-value-huge" style="color: #00E676; margin: 0; display: flex; align-items: center; justify-content: center; gap: 6px;">
            <span>0.835</span>
            {mock_badge_val}
        </div>
        <div class="metric-label" style="margin-top: 4px;">PR-AUC benchmark</div>
    </div>
    """)

with k4:
    render_html("""
    <div class="glass-card" style="padding: 16px 20px; text-align: center;">
        <div class="metric-value-huge" style="color: #00E5FF; margin: 0;">0.00%</div>
        <div class="metric-label" style="margin-top: 4px;">Temporal leakage</div>
    </div>
    """)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. PARADIGM SHIFT
# -----------------------------------------------------------------------------
render_html("""
<div class="glass-card" style="margin-bottom: 22px; padding: 20px 24px;">
    <div class="card-title" style="margin-bottom: 14px;">
        <span>Paradigm Shift</span>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 18px;">
        <div style="background: rgba(255, 23, 68, 0.08); border: 1px solid rgba(255, 23, 68, 0.3); border-radius: 12px; padding: 16px 18px;">
            <div style="font-weight: 800; color: #FF5252; font-size: 0.95rem; margin-bottom: 10px; letter-spacing: 0.04em;">
                CONVENTIONAL IDS
            </div>
            <ul style="font-size: 0.84rem; color: #CBD5E1; margin: 0; padding-left: 18px; line-height: 1.8;">
                <li>0 min lead time</li>
                <li>Static signatures</li>
                <li>Alerts post-compromise</li>
            </ul>
        </div>
        <div style="background: rgba(0, 229, 255, 0.08); border: 1px solid rgba(0, 229, 255, 0.35); border-radius: 12px; padding: 16px 18px;">
            <div style="font-weight: 800; color: #00E5FF; font-size: 0.95rem; margin-bottom: 10px; letter-spacing: 0.04em;">
                SHADOWCAT
            </div>
            <ul style="font-size: 0.84rem; color: #FFFFFF; margin: 0; padding-left: 18px; line-height: 1.8;">
                <li>~3.5 min lead time</li>
                <li>Continuous state kinematics</li>
                <li>Pre-emptive intervention</li>
            </ul>
        </div>
    </div>
</div>
""")

# -----------------------------------------------------------------------------
# 4. THREE-TIER FLOW
# -----------------------------------------------------------------------------
render_html("""
<div class="card-title" style="margin-bottom: 12px;">
    <span>System Pipeline</span>
</div>
""")

flow1, flow2, flow3 = st.columns(3)

with flow1:
    render_html("""
    <div class="glass-card" style="padding: 16px 18px; min-height: 85px;">
        <div style="font-weight: 800; color: #00E5FF; font-size: 0.94rem; margin-bottom: 4px;">
            Ingestion Pipeline
        </div>
        <div style="font-size: 0.82rem; color: #94A3B8;">
            Packet & flow signals → unified state
        </div>
    </div>
    """)

with flow2:
    render_html("""
    <div class="glass-card" style="padding: 16px 18px; min-height: 85px;">
        <div style="font-weight: 800; color: #FFB300; font-size: 0.94rem; margin-bottom: 4px;">
            World Model
        </div>
        <div style="font-size: 0.82rem; color: #94A3B8;">
            Learns state-transition dynamics
        </div>
    </div>
    """)

with flow3:
    render_html("""
    <div class="glass-card" style="padding: 16px 18px; min-height: 85px;">
        <div style="font-weight: 800; color: #00E676; font-size: 0.94rem; margin-bottom: 4px;">
            Intelligence Head
        </div>
        <div style="font-size: 0.82rem; color: #94A3B8;">
            Hazard score + ATT&CK stage
        </div>
    </div>
    """)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. LAUNCHPAD GRID
# -----------------------------------------------------------------------------
render_html("""
<div class="card-title" style="margin-bottom: 12px;">
    <span>System Launchpad</span>
</div>
""")

lp1, lp2, lp3, lp4 = st.columns(4)

with lp1:
    render_html("""
    <div class="glass-card" style="padding: 16px 18px; min-height: 85px; margin-bottom: 8px;">
        <div style="font-weight: 800; color: #00E5FF; font-size: 0.94rem; margin-bottom: 4px;">Forecast</div>
        <div style="font-size: 0.8rem; color: #94A3B8;">Multi-step trajectory simulation</div>
    </div>
    """)
    if st.button("Open Threat Forecast", key="btn_home_forecast", use_container_width=True):
        st.switch_page("views/01_Forecast.py")

with lp2:
    render_html("""
    <div class="glass-card" style="padding: 16px 18px; min-height: 85px; margin-bottom: 8px;">
        <div style="font-weight: 800; color: #00E5FF; font-size: 0.94rem; margin-bottom: 4px;">Evidence</div>
        <div style="font-size: 0.8rem; color: #94A3B8;">Feature attribution & telemetry</div>
    </div>
    """)
    if st.button("Inspect Evidence & Attribution", key="btn_home_evidence", use_container_width=True):
        st.switch_page("views/02_Evidence.py")

with lp3:
    render_html("""
    <div class="glass-card" style="padding: 16px 18px; min-height: 85px; margin-bottom: 8px;">
        <div style="font-weight: 800; color: #00E5FF; font-size: 0.94rem; margin-bottom: 4px;">Validation</div>
        <div style="font-size: 0.8rem; color: #94A3B8;">LOEO 37-fold benchmark rigor</div>
    </div>
    """)
    if st.button("Review Validation & Benchmarks", key="btn_home_validation", use_container_width=True):
        st.switch_page("views/03_Validation.py")

with lp4:
    render_html("""
    <div class="glass-card" style="padding: 16px 18px; min-height: 85px; margin-bottom: 8px;">
        <div style="font-weight: 800; color: #00E5FF; font-size: 0.94rem; margin-bottom: 4px;">Architecture</div>
        <div style="font-size: 0.8rem; color: #94A3B8;">Neural kinematics & GraphSAGE</div>
    </div>
    """)
    if st.button("Explore Architecture & UCS", key="btn_home_architecture", use_container_width=True):
        st.switch_page("views/00b_Architecture.py")

# -----------------------------------------------------------------------------
# 6. FOOTER
# -----------------------------------------------------------------------------
render_footer()
