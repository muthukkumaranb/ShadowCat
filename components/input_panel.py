"""
SHADOWCAT - Telemetry Ingestion & Input Selector Component
Supports PCAP upload, CSV flow ingestion, and 1-click Bundled Sample demo run.

PIPELINE INDICATORS:
- PCAP: [Active Pipeline: Full Unified Cyber State (UCS)]
- CSV:  [Active Pipeline: Flow-Only Features]
- Sample: [Active Pipeline: Bundled Reference Case (UCS)]
"""

import streamlit as st
from styles import render_html


def render_input_panel():
    """
    Renders the telemetry input and dataset selection panel with active pipeline indicators.
    """
    if "input_mode" not in st.session_state:
        st.session_state["input_mode"] = "Bundled Sample (Infiltration Case)"
    if "pipeline_name" not in st.session_state:
        st.session_state["pipeline_name"] = "Full Unified Cyber State (UCS)"
    if "demo_running" not in st.session_state:
        st.session_state["demo_running"] = False

    with st.expander("Telemetry Ingestion & Dataset Source", expanded=False):
        col_select, col_badge = st.columns([2.2, 1.2])

        with col_select:
            mode = st.radio(
                "Select Telemetry Ingestion Source:",
                options=[
                    "Bundled Sample (Infiltration Case)",
                    "Upload PCAP (.pcap, .pcapng)",
                    "Upload CSV (Flow-Only Schema)",
                ],
                index=[
                    "Bundled Sample (Infiltration Case)",
                    "Upload PCAP (.pcap, .pcapng)",
                    "Upload CSV (Flow-Only Schema)",
                ].index(st.session_state["input_mode"]),
                horizontal=True,
                key="telemetry_input_radio",
            )
            st.session_state["input_mode"] = mode

        with col_badge:
            if mode == "Upload CSV (Flow-Only Schema)":
                badge_text = "[Active Pipeline: Flow-Only Features]"
                badge_bg = "rgba(224, 152, 43, 0.12)"
                badge_color = "#E0982B"
                pipeline_desc = "Flow-level attributes (NetFlow/IPFIX summary, durations, byte counts)."
                st.session_state["pipeline_name"] = "Flow-Only Features"
            elif mode == "Upload PCAP (.pcap, .pcapng)":
                badge_text = "[Active Pipeline: Full Unified Cyber State (UCS)]"
                badge_bg = "rgba(255, 255, 255, 0.08)"
                badge_color = "#FFFFFF"
                pipeline_desc = "Full packet-level + flow-level extraction (TTL, TCP window size, flag bitmasks)."
                st.session_state["pipeline_name"] = "Full Unified Cyber State (UCS)"
            else:
                badge_text = "[Active Pipeline: Full Unified Cyber State (UCS)]"
                badge_bg = "rgba(47, 184, 114, 0.12)"
                badge_color = "#2FB872"
                pipeline_desc = "CSE-CIC-IDS2018 Infiltration Scenario episode with full packet + flow metadata."
                st.session_state["pipeline_name"] = "Full Unified Cyber State (UCS)"

            render_html(f"""
            <div class="glass-card" style="padding: 10px 14px; margin-top: 4px;">
                <div class="metric-label" style="margin-bottom: 4px;">Active telemetry pipeline</div>
                <div style="font-size: 0.76rem; font-weight: 700; color: {badge_color}; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin-bottom: 4px;">
                    {badge_text}
                </div>
                <div style="font-size: 0.70rem; color: #8A8A8A; line-height: 1.4;">
                    {pipeline_desc}
                </div>
            </div>
            """)

        # Input mode specific actions
        if mode == "Bundled Sample (Infiltration Case)":
            c_left, c_btn = st.columns([2.5, 1.5])
            with c_left:
                render_html("""
                <div style="font-size: 0.78rem; color: #8A8A8A; margin-top: 6px; line-height: 1.5;">
                    <b>Dataset:</b> CSE-CIC-IDS2018 (Scenario: Infiltration & SSH Brute Force)<br>
                    <b>Episode Length:</b> 60-second sliding windows (L=30 history lookback, K=4 forward rollout)
                </div>
                """)
            with c_btn:
                if st.button("Run Infiltration Episode (Video Mode)", use_container_width=True, type="primary"):
                    st.session_state["demo_running"] = True
                    st.toast("Bundled sample loaded: CSE-CIC-IDS2018 Infiltration episode active.")

        elif mode == "Upload PCAP (.pcap, .pcapng)":
            pcap_file = st.file_uploader(
                "Upload Raw Network Capture File (.pcap, .pcapng)",
                type=["pcap", "pcapng"],
                help="Offline processing: features extracted strictly using local Scapy/PyShark without cloud transmission."
            )
            if pcap_file:
                st.success(f"Loaded '{pcap_file.name}' ({pcap_file.size / 1024:.1f} KB). Extracting UCS packet + flow feature tensors...")

        elif mode == "Upload CSV (Flow-Only Schema)":
            csv_file = st.file_uploader(
                "Upload Flow Record CSV File (.csv)",
                type=["csv"],
                help="Requires CICFlowMeter or IPFIX/NetFlow schema with 80 flow statistics."
            )
            if csv_file:
                st.success(f"Loaded '{csv_file.name}' ({csv_file.size / 1024:.1f} KB). Mapping flow-only feature schema...")
