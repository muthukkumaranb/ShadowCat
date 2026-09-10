"""
SHADOWCAT - Executive Persistent Header Component
"""

import streamlit as st
from styles import COLORS, render_html
from data_provider import inference_status


def render_header(data):
    """Renders the sticky pinned status banner and slim persistent top bar."""
    analysis = data["analysis"]
    telemetry_source = st.session_state.get("telemetry_source", "BENCHMARK: CIC-IDS2018 (Infiltration)")
    window_str = st.session_state.get("window_str", analysis.get("window", "t+1 → t+4 (Active)"))

    short_feed = telemetry_source.replace("BENCHMARK: ", "").replace(" (Infiltration)", "")
    inf_status = inference_status()

    # Revision B: Pinned / sticky top banner across all pages
    if inf_status == "live":
        banner_html = """
        <div style="position: sticky; top: 0; z-index: 999; background: rgba(13, 20, 36, 0.96); backdrop-filter: blur(8px); border-bottom: 1px solid #2FB872; padding: 6px 16px; margin: -1rem -1rem 10px -1rem; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 12px rgba(0,0,0,0.35);">
            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.78rem; color: #E8EDF5;">
                <span style="background: rgba(47, 184, 114, 0.2); border: 1px solid #2FB872; color: #2FB872; font-size: 0.66rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; padding: 2px 7px; border-radius: 4px; letter-spacing: 0.05em;">
                    ● LIVE
                </span>
                <span>Live inference pipeline connected</span>
            </div>
            <div style="font-size: 0.70rem; color: #2FB872; font-family: 'JetBrains Mono', monospace; font-weight: 600;">
                ACTIVE STREAM
            </div>
        </div>
        """
    else:
        banner_html = """
        <div style="position: sticky; top: 0; z-index: 999; background: rgba(13, 20, 36, 0.96); backdrop-filter: blur(8px); border-bottom: 1px solid #E0982B; padding: 6px 16px; margin: -1rem -1rem 10px -1rem; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 12px rgba(0,0,0,0.35);">
            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.78rem; color: #E8EDF5;">
                <span style="background: rgba(224, 152, 43, 0.18); border: 1px solid #E0982B; color: #E0982B; font-size: 0.66rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; padding: 2px 7px; border-radius: 4px; letter-spacing: 0.05em;">
                    BENCHMARK MODE
                </span>
                <span>Running on benchmark data (CIC-IDS2018) &mdash; live inference not yet connected</span>
            </div>
            <div style="font-size: 0.70rem; color: #9AA7BD; font-family: 'JetBrains Mono', monospace;">
                LOEO 37-FOLD VALIDATED
            </div>
        </div>
        """

    render_html(f"""
    {banner_html}
    <div class="slim-header-bar">
        <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
            <span class="slim-header-brand">SHADOWCAT</span>
            <span class="badge-offline">
                <span class="status-dot"></span>
                AIR-GAPPED
            </span>
        </div>
        <div style="display: flex; align-items: center; gap: 16px; font-size: 0.76rem; font-family: 'JetBrains Mono', monospace;">
            <span style="color: #9aa7bd;">FEED: <b style="color: #e8edf5;">{short_feed}</b></span>
            <span style="color: #38bdf8; font-weight: 600;">WINDOW: {window_str}</span>
        </div>
    </div>
    """)


