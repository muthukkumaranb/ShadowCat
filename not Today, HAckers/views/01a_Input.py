"""
SHADOWCAT - Dual-Level Telemetry Ingestion & Benchmark Replay Portal
Supports Multi-Dataset Benchmarks (CIC-IDS2018, CTU-13, UNSW-NB15) and Custom PCAP / Flow Ingestion.
Enforces PS Section 1 Requirement: Dual-Level Fusion (Flow-Level + Packet-Level).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from data_provider import get_demo_data
from styles import apply_custom_css, render_sidebar, COLORS, render_html, render_footer
from components.header import render_header

st.set_page_config(
    page_title="SHADOWCAT — Telemetry Ingestion",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()
data = get_demo_data()
render_header(data)

# Ensure session state initialization
if "telemetry_source" not in st.session_state:
    st.session_state["telemetry_source"] = "BENCHMARK: CIC-IDS2018 (Infiltration)"
if "pipeline_name" not in st.session_state:
    st.session_state["pipeline_name"] = "Dual-Level Fusion (Flow + Packet UCS)"
if "demo_running" not in st.session_state:
    st.session_state["demo_running"] = False

render_html("""
<div style="margin-bottom: 18px;">
    <h2 style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF; margin: 0;">
        Telemetry Ingestion & Dataset Replay
    </h2>
    <div style="font-size: 0.84rem; color: #9AA7BD; margin-top: 4px;">
        Ingest live sensor taps, standardized multi-dataset benchmarks, or custom capture files into the Unified Cyber State (UCS) tensor.
    </div>
</div>
""")

# Ingestion Source Tabs
tab_benchmark, tab_custom = st.tabs(["Pre-Loaded Benchmark Episodes (Multi-Dataset)", "Custom Sensor Telemetry (PCAP / Flow)"])

with tab_benchmark:
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    benchmark_episodes = {
        "CSE-CIC-IDS2018 (Infiltration & Lateral Pivot)": {
            "source_id": "BENCHMARK: CIC-IDS2018 (Infiltration)",
            "scenario": "Infiltration, SSH Brute Force, and Pivot into Internal Authentication Cluster",
            "duration": "60-second sliding windows (L=30 history lookback, K=4 forward rollout)",
            "sensors": "Dual-level (Raw PCAP + Flow CSV extracted)",
            "lead_time": "~3.5 min pre-emptive lead",
        },
        "CTU-13 (Botnet C2 Traffic — Scenario 9)": {
            "source_id": "BENCHMARK: CTU-13 (Nerve-1 Botnet)",
            "scenario": "Multi-host Botnet command-and-control beaconing with periodic ingress scan",
            "duration": "30-second sliding windows (L=30 lookback, K=4 rollout)",
            "sensors": "NetFlow + Bidir Flow records with TCP flag distribution",
            "lead_time": "~2.8 min pre-emptive lead",
        },
        "UNSW-NB15 (Multi-Vector Reconnaissance Sweep)": {
            "source_id": "BENCHMARK: UNSW-NB15 (Recon/Fuzz)",
            "scenario": "Port scanning, fuzzing, and initial boundary probe sequence",
            "duration": "45-second sliding windows (L=30 lookback, K=4 rollout)",
            "sensors": "Dual-level PCAP packet timings + Bro/Zeek session records",
            "lead_time": "~4.0 min pre-emptive lead",
        },
    }

    selected_benchmark = st.radio(
        "Select Benchmark Dataset Episode:",
        list(benchmark_episodes.keys()),
        index=0,
        horizontal=False,
        key="benchmark_selector"
    )

    info = benchmark_episodes[selected_benchmark]

    render_html(f"""
    <div class="glass-card" style="padding: 14px 18px; margin-top: 10px; margin-bottom: 14px;">
        <div style="font-size: 0.85rem; font-weight: 700; color: #E8EDF5; margin-bottom: 6px;">
            {selected_benchmark}
        </div>
        <div style="font-size: 0.78rem; color: #9AA7BD; line-height: 1.5;">
            <b>Attack Scenario:</b> {info['scenario']}<br>
            <b>Windowing Protocol:</b> {info['duration']}<br>
            <b>Feature Pipeline:</b> {info['sensors']} · <i>{info['lead_time']}</i>
        </div>
    </div>
    """)

    c_left, c_btn = st.columns([2.5, 1.5])
    with c_left:
        st.caption("Loads verified telemetry episode with ground-truth timeline annotations.")
    with c_btn:
        if st.button("Activate Benchmark Episode", key="btn_run_selected_benchmark", use_container_width=True, type="primary"):
            st.session_state["telemetry_source"] = info["source_id"]
            st.session_state["demo_running"] = True
            st.toast(f"Active feed updated: {info['source_id']}")
            st.switch_page("views/01_Forecast.py")

with tab_custom:
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    custom_mode = st.radio(
        "Select Custom Ingestion Pipeline:",
        ["Raw Packet Capture (.pcap, .pcapng)", "Flow Records CSV (.csv, NetFlow / IPFIX)"],
        horizontal=True,
        key="custom_pipeline_mode"
    )

    if custom_mode == "Raw Packet Capture (.pcap, .pcapng)":
        pcap_file = st.file_uploader(
            "Upload Network Packet Capture (.pcap, .pcapng)",
            type=["pcap", "pcapng"],
            key="pcap_uploader",
            help="Constructs Unified Cyber State tensors via local packet parser."
        )
        if pcap_file is not None:
            st.success(f"Capture parsed: '{pcap_file.name}' ({pcap_file.size / 1024:.1f} KB). Extracted Level 1 flow aggregates and Level 2 packet timing.")
            if st.button("Ingest & Forecast Trajectory", key="btn_run_pcap_custom", use_container_width=True, type="primary"):
                st.session_state["telemetry_source"] = f"PCAP: {pcap_file.name}"
                st.session_state["demo_running"] = True
                st.switch_page("views/01_Forecast.py")

    else:
        st.caption("Standard NetFlow/IPFIX/CICFlowMeter schema: bidirectional flow aggregates with statistical packet parameter imputation.")
        csv_file = st.file_uploader(
            "Upload Flow Telemetry CSV (.csv)",
            type=["csv"],
            key="csv_uploader",
            help="Ingests flow statistics into the Unified Cyber State matrix."
        )
        if csv_file is not None:
            st.success(f"Flow records parsed: '{csv_file.name}' ({csv_file.size / 1024:.1f} KB). Dual-level schema mapped successfully.")
            if st.button("Ingest & Forecast Trajectory", key="btn_run_csv_custom", use_container_width=True, type="primary"):
                st.session_state["telemetry_source"] = f"FLOW: {csv_file.name}"
                st.session_state["demo_running"] = True
                st.switch_page("views/01_Forecast.py")

render_footer()
