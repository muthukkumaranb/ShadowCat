"""
SHADOWCAT - Model Architecture & Telemetry Pipeline
Neural Kinematics, Topological Graph Induction, and Air-Gapped Inference
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from mock_data import get_demo_data
from styles import apply_custom_css, render_sidebar, COLORS, render_html, render_footer
from components.header import render_header

st.set_page_config(
    page_title="SHADOWCAT — Model Info",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()
data = get_demo_data()
render_header(data)

render_html("""
<div style="margin-bottom: 20px;">
    <h2 style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF; margin: 0;">
        Model Architecture & Ingestion Telemetry
    </h2>
    <div style="font-size: 0.84rem; color: #94A3B8; margin-top: 4px;">
        Deep sequence kinematics, topological graph induction, and air-gapped inference runtime specifications.
    </div>
</div>
""")

analysis = data["analysis"]
state = data["current_state"]

# 1. Ingestion Stream Telemetry
render_html("""
<div class="card-title">
    <span>Telemetry Ingestion Stream</span>
    <span class="badge">Input Configuration</span>
</div>
""")

col_ing1, col_ing2, col_ing3, col_ing4 = st.columns(4)

with col_ing1:
    render_html(f"""
    <div class="glass-card" style="padding: 16px; min-height: 120px;">
        <div class="metric-label">Dataset Source</div>
        <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; margin-top: 4px;">{analysis['source']}</div>
        <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 2px;">Standardized Network Benchmark</div>
    </div>
    """)

with col_ing2:
    render_html(f"""
    <div class="glass-card" style="padding: 16px; min-height: 120px;">
        <div class="metric-label">Window Duration</div>
        <div style="font-size: 1.25rem; font-weight: 800; color: #00E5FF; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">
            {analysis['duration_sec']}s (10s stride)
        </div>
        <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 2px;">Window: {analysis['window']}</div>
    </div>
    """)

with col_ing3:
    render_html(f"""
    <div class="glass-card" style="padding: 16px; min-height: 120px;">
        <div class="metric-label">Analyzed Throughput</div>
        <div style="font-size: 1.25rem; font-weight: 800; color: #00E676; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">
            {analysis['flows_analyzed']:,} flows
        </div>
        <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 2px;">{analysis['packets_analyzed']:,} total packets</div>
    </div>
    """)

with col_ing4:
    render_html(f"""
    <div class="glass-card" style="padding: 16px; min-height: 120px;">
        <div class="metric-label">Operational Mode</div>
        <div style="font-size: 1.2rem; font-weight: 800; color: #FFB300; margin-top: 4px;">Air-Gapped</div>
        <div style="font-size: 0.74rem; color: #00E676; font-weight: 600; margin-top: 2px;">● Zero external telemetry egress</div>
    </div>
    """)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# 2. Core Architecture Modules
render_html("""
<div class="card-title" style="margin-top: 16px; margin-bottom: 12px;">
    <span>Forecasting Engine Architecture</span>
    <span class="badge">Multi-Stage Pipeline</span>
</div>
""")

render_html(r"""
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px;">
    <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 1.05rem; font-weight: 800; color: #00E5FF;">1. Temporal Sequence Encoder</span>
            <span class="badge">Kinematics</span>
        </div>
        <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
            <b>Architecture:</b> Bidirectional LSTM with Multi-Head Self-Attention.<br>
            <b>Input:</b> Inter-arrival timing jitter, packet byte sizes, TCP flag transitions, and TTL hop decay across 60-second sliding windows.<br>
            <b>Representation:</b> 128-dimensional latent temporal vector capturing traffic kinematics.
        </div>
    </div>

    <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 1.05rem; font-weight: 800; color: #00E676;">2. Host Topology Inducer</span>
            <span class="badge">GraphSAGE</span>
        </div>
        <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
            <b>Architecture:</b> Inductive Graph Neural Network (GraphSAGE).<br>
            <b>Input:</b> Dynamic IP-to-IP bipartite communication graph with localized edge attributes and degree distributions.<br>
            <b>Representation:</b> 64-dimensional structural embedding capturing stealthy lateral connection spreading across subnets.
        </div>
    </div>

    <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 1.05rem; font-weight: 800; color: #FFB300;">3. State Dynamics Head</span>
            <span class="badge">Rollout & Bounds</span>
        </div>
        <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
            <b>Architecture:</b> Autoregressive Transition Model with Epistemic Uncertainty Estimation.<br>
            <b>Rollout:</b> Recurrent transition mapping <code>S(t+k) = f(S(t+k-1))</code> across multi-step horizons <b>K=1..4</b>.<br>
            <b>Uncertainty:</b> Monte Carlo Dropout variance sampling yielding calibrated prediction bounds <code>[μ ± σ]</code>.
        </div>
    </div>

    <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 1.05rem; font-weight: 800; color: #F43F5E;">4. Air-Gapped Edge Runtime</span>
            <span class="badge">Execution</span>
        </div>
        <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
            <b>Inference Latency:</b> &lt;45 ms per 10-second traffic window on standard x86 CPU.<br>
            <b>Memory Footprint:</b> &lt;850 MB peak RAM consumption with quantized weights.<br>
            <b>Isolation:</b> Fully air-gapped; requires zero outbound internet access or third-party APIs.
        </div>
    </div>
</div>
""")

# 3. Dual-Signal Evaluation Framework
render_html(f"""
<div class="glass-card" style="margin-bottom: 20px;">
    <div class="card-title">
        <span>Dual-Signal Decision Manifold</span>
        <span class="badge">Signature vs Novelty</span>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div style="background: rgba(0, 229, 255, 0.06); border: 1px solid rgba(0, 229, 255, 0.2); border-radius: 12px; padding: 16px;">
            <div style="font-size: 0.95rem; font-weight: 700; color: #00E5FF; margin-bottom: 6px;">
                Signal 1: Trajectory Risk Probability P(Attack)
            </div>
            <div style="font-size: 0.8rem; color: #CBD5E1; line-height: 1.55;">
                Measures alignment with learned multi-step killchain progressions (Reconnaissance → Initial Access → Lateral Movement). Triggers pre-emptive rate limiting and credential verification.
            </div>
        </div>
        <div style="background: rgba(0, 230, 118, 0.06); border: 1px solid rgba(0, 230, 118, 0.2); border-radius: 12px; padding: 16px;">
            <div style="font-size: 0.95rem; font-weight: 700; color: #00E676; margin-bottom: 6px;">
                Signal 2: Out-of-Distribution Novelty Score ({state['novelty_score']:.2f})
            </div>
            <div style="font-size: 0.8rem; color: #CBD5E1; line-height: 1.55;">
                Latent space density estimation flags anomalous deviations outside the baseline behavior envelope. Differentiates known attack escalation from benign operational changes.
            </div>
        </div>
    </div>
</div>
""")

# Persistent Executive Footer
render_footer()
