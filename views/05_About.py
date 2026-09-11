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
    initial_sidebar_state="expanded"
)

apply_custom_css()
data = get_demo_data()
render_header(data)

# 1. Executive Summary & Operational Doctrine
render_html("""
<div style="margin-bottom: 18px;">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
        <h2 style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin: 0;">
            Platform Specifications
        </h2>
        <span style="background: #222222; border: 1px solid #333333; color: #FFFFFF; font-size: 0.72rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; padding: 2px 8px; border-radius: 4px;">
            v1.0
        </span>
    </div>
    <div style="font-size: 0.84rem; color: #8A8A8A; line-height: 1.5;">
        Autonomous pre-emptive cyber threat forecasting engine for air-gapped enterprise network defense. Models causal state transitions from raw network telemetry to forecast attack progression before compromise completes.
    </div>
    <div style="display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap;">
        <span style="background: rgba(47, 184, 114, 0.12); border: 1px solid #2FB872; color: #2FB872; font-size: 0.70rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; padding: 3px 8px; border-radius: 4px;">
            AIR-GAPPED
        </span>
        <span style="background: #181818; border: 1px solid #262626; color: #8A8A8A; font-size: 0.70rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; padding: 3px 8px; border-radius: 4px;">
            MITRE ATT&CK ALIGNED
        </span>
        <span style="background: #181818; border: 1px solid #262626; color: #8A8A8A; font-size: 0.70rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; padding: 3px 8px; border-radius: 4px;">
            WORLD MODEL ARCHITECTURE
        </span>
    </div>
</div>
""")

# 2. Core Operational Capabilities (3-Card Section - Task 3)
render_html("""
<div class="card-title" style="margin-bottom: 12px;">
    <span>Core Operational Capabilities</span>
</div>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 24px;">
    <div class="glass-card" style="padding: 16px;">
        <div style="font-size: 0.90rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px;">
            Multi-Horizon Trajectory Rollout
        </div>
        <div style="font-size: 0.78rem; color: #8A8A8A; line-height: 1.5;">
            Continuous forward simulation of state transition dynamics P(S_{t+1} | S_t). Evaluates future risk progression across horizons t+1 through t+4 prior to lateral breach completion.
        </div>
    </div>
    <div class="glass-card" style="padding: 16px;">
        <div style="font-size: 0.90rem; font-weight: 700; color: #E0982B; margin-bottom: 6px;">
            Compounding Epistemic Uncertainty
        </div>
        <div style="font-size: 0.78rem; color: #8A8A8A; line-height: 1.5;">
            Explicitly models variance growth at each forward step. Governs operational policies: horizons K=1–2 trigger automated mitigation, while K=3–4 trigger tiered SOC escalation.
        </div>
    </div>
    <div class="glass-card" style="padding: 16px;">
        <div style="font-size: 0.90rem; font-weight: 700; color: #2FB872; margin-bottom: 6px;">
            Air-Gapped High-Assurance Operation
        </div>
        <div style="font-size: 0.78rem; color: #8A8A8A; line-height: 1.5;">
            Operates fully offline with zero outbound cloud API calls, local system fonts, disabled telemetry tracking, and in-memory feature tensor parsing for Critical Information Infrastructure.
        </div>
    </div>
</div>
""")

# 3. Dual-Level Telemetry Fusion Architecture (Task 7 Trim)
render_html("""
<div class="card-title" style="margin-bottom: 12px;">
    <span>Dual-Level Telemetry Fusion Architecture</span>
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
    <span>Empirical Validation & Benchmarks</span>
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
</div>
""")

try:
    st.page_link(
        "views/03_Validation.py",
        label="Open Model Validation & Benchmarks"
    )
except Exception:
    pass

render_footer()
