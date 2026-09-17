"""
SHADOWCAT - Platform Overview & Architecture Specifications
Air-Gapped Enterprise Network Defense Engine
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from data_provider import get_demo_data
import importlib
import styles
importlib.reload(styles)
from styles import apply_custom_css, render_sidebar, COLORS, render_html, render_footer
import components.header
importlib.reload(components.header)
from components.header import render_header

st.set_page_config(
    page_title="SHADOWCAT — Platform Specifications",
    layout="wide",
    initial_sidebar_state="collapsed"
)

apply_custom_css()
data = get_demo_data()
render_header(data, active_tab="Platform Specifications")

# 1. Executive Summary & Operational Doctrine
render_html("""
<div style="margin-bottom: 18px;">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
        <h2 style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin: 0;">
            Platform Specifications
        </h2>
        <span style="background: #222222; border: 1px solid #333333; color: #FFFFFF; font-size: 0.72rem; font-weight: 700; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 2px 8px; border-radius: 4px;">
            v1.0
        </span>
    </div>
    <div style="font-size: 0.84rem; color: #8A8A8A; line-height: 1.5;">
        Autonomous pre-emptive cyber threat forecasting engine for air-gapped enterprise network defense. Models causal state transitions from raw network telemetry to forecast attack progression before compromise completes.
    </div>
    <div style="display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap;">
        <span style="background: rgba(47, 184, 114, 0.12); border: 1px solid #2FB872; color: #2FB872; font-size: 0.70rem; font-weight: 700; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.04em;">
            AIR-GAPPED
        </span>
        <span style="background: #181818; border: 1px solid #262626; color: #8A8A8A; font-size: 0.70rem; font-weight: 600; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 3px 8px; border-radius: 4px;">
            MITRE ATT&CK aligned
        </span>
        <span style="background: #181818; border: 1px solid #262626; color: #8A8A8A; font-size: 0.70rem; font-weight: 600; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 3px 8px; border-radius: 4px;">
            World model architecture
        </span>
    </div>
</div>
""")

# 2. Core Operational Capabilities (3-Card Section - Task 3)
render_html("""
<div class="card-title" style="margin-bottom: 12px;">
    <span>Core operational capabilities</span>
</div>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 24px;">
    <div class="glass-card" style="padding: 16px;" title="Evaluates future risk progression across multi-step horizons prior to lateral breach completion.">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
            <div style="font-size: 0.90rem; font-weight: 700; color: #FFFFFF;">
                Multi-Horizon Trajectory Rollout
            </div>
            <span style="font-size: 0.68rem; color: #8A8A8A; border: 1px solid #333333; border-radius: 50%; width: 15px; height: 15px; display: inline-flex; align-items: center; justify-content: center; cursor: help;" title="Evaluates future risk progression across multi-step horizons prior to lateral breach completion.">i</span>
        </div>
        <div style="font-size: 0.78rem; color: #8A8A8A; line-height: 1.5;">
            Continuous forward simulation of state transition dynamics P(S_{t+1} | S_t) across future horizons t+1 &rarr; t+4.
        </div>
    </div>
    <div class="glass-card" style="padding: 16px;" title="Governs operational policies: horizons K=1–2 trigger automated mitigation, while K=3–4 trigger tiered SOC escalation.">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
            <div style="font-size: 0.90rem; font-weight: 700; color: #E0982B;">
                Compounding Epistemic Uncertainty
            </div>
            <span style="font-size: 0.68rem; color: #8A8A8A; border: 1px solid #333333; border-radius: 50%; width: 15px; height: 15px; display: inline-flex; align-items: center; justify-content: center; cursor: help;" title="Governs operational policies: horizons K=1–2 trigger automated mitigation, while K=3–4 trigger tiered SOC escalation.">i</span>
        </div>
        <div style="font-size: 0.78rem; color: #8A8A8A; line-height: 1.5;">
            Explicitly models compounding variance growth at each forward step to govern tiered response policies.
        </div>
    </div>
    <div class="glass-card" style="padding: 16px;" title="Local system fonts, disabled telemetry tracking, and in-memory feature tensor parsing for Critical Information Infrastructure.">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
            <div style="font-size: 0.90rem; font-weight: 700; color: #2FB872;">
                Air-Gapped High-Assurance Operation
            </div>
            <span style="font-size: 0.68rem; color: #8A8A8A; border: 1px solid #333333; border-radius: 50%; width: 15px; height: 15px; display: inline-flex; align-items: center; justify-content: center; cursor: help;" title="Local system fonts, disabled telemetry tracking, and in-memory feature tensor parsing for Critical Information Infrastructure.">i</span>
        </div>
        <div style="font-size: 0.78rem; color: #8A8A8A; line-height: 1.5;">
            Operates fully offline with zero outbound cloud dependencies, local fonts, and in-memory tensor parsing.
        </div>
    </div>
</div>
""")

# 3. Dual-Level Telemetry Fusion Architecture (Task 7 Trim)
render_html("""
<div class="card-title" style="margin-bottom: 12px;">
    <span>Dual-level telemetry fusion architecture</span>
</div>
<div class="glass-card" style="padding: 16px 18px; margin-bottom: 24px;">
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; font-size: 0.78rem; color: #8A8A8A;">
        <div>
            <b style="color: #FFFFFF;">Level 1: Flow Aggregates (NetFlow)</b><br>
            Bidirectional byte/packet ratios, TCP flag bitmasks, flow durations, and port distributions.
        </div>
        <div>
            <b style="color: #FFFFFF;">Level 2: Packet Dynamics (PCAP)</b><br>
            Micro-timing inter-arrival jitter, TTL variance, TCP window sizes, and payload entropy.
        </div>
        <div>
            <b style="color: #FFFFFF;">Unified Cyber State S(t)</b><br>
            Fused tensor representation feeding sequence dynamics and topological GraphSAGE embeddings.
        </div>
    </div>
    <div style="font-size: 0.72rem; color: #8A8A8A; margin-top: 10px; border-top: 1px solid #262626; padding-top: 8px;">
        Local offline Scapy extraction operates without cloud dependencies.
    </div>
</div>
""")

# 4. Authoritative Validation Link (Zero Duplicate Metrics - Task 3)
render_html("""
<div class="card-title" style="margin-bottom: 10px;">
    <span>Empirical validation & benchmarks</span>
</div>
<div class="glass-card" style="padding: 16px 18px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
    <div>
        <div style="font-size: 0.88rem; font-weight: 700; color: #FFFFFF;">
            Leave-One-Episode-Out (LOEO 37-Fold) Cross-Validation & Baseline Benchmarks
        </div>
        <div style="font-size: 0.76rem; color: #8A8A8A; margin-top: 2px;">
            Full quantitative metrics, schedule artifact controls, and horizon error bounds are maintained on the Validation console.
        </div>
    </div>
    <div>
        <a href="Validation" target="_self" class="soc-action-link" style="display: inline-flex; align-items: center; gap: 6px; font-size: 0.78rem; padding: 6px 12px; text-decoration: none;">
            <span>Open Model Validation & Benchmarks</span>
            <span style="font-size: 0.85rem;">&rarr;</span>
        </a>
    </div>
</div>
""")

render_footer()
