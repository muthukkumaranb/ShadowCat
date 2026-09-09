"""
SHADOWCAT - Executive Persistent Header Component
"""

import streamlit as st
from styles import COLORS, render_html


def render_header(data):
    """Renders the slim, single-row persistent top bar with brand, air-gapped beacon, sensor tap, and stream timestamp."""
    analysis = data["analysis"]
    telemetry_source = st.session_state.get("telemetry_source", "BENCHMARK: CIC-IDS2018 (Infiltration)")
    window_str = st.session_state.get("window_str", analysis.get("window", "t+1 → t+4 (Active)"))

    render_html(f"""
    <div class="slim-header-bar">
        <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
            <span class="slim-header-brand">SHADOWCAT</span>
            <span class="badge-offline">
                <span class="status-dot"></span>
                AIR-GAPPED
            </span>
            <span style="color: #22304a; font-size: 0.85rem;">|</span>
            <span style="font-size: 0.78rem; color: #9aa7bd; font-weight: 500; font-family: 'JetBrains Mono', monospace;">
                SENSOR: <b style="color: #e8edf5;">TAP-DMZ-01</b> (10Gbps Ingress)
            </span>
        </div>
        <div style="display: flex; align-items: center; gap: 16px; font-size: 0.76rem; font-family: 'JetBrains Mono', monospace;">
            <span style="color: #9aa7bd;">FEED: <b style="color: #e8edf5;">{telemetry_source}</b></span>
            <span style="color: #38bdf8; font-weight: 600;">WINDOW: {window_str}</span>
        </div>
    </div>
    """)


