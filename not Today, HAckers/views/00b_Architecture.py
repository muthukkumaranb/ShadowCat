"""
SHADOWCAT - Model Architecture & Telemetry Pipeline
Neural Kinematics, Topological Graph Induction, and Air-Gapped Inference
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from data_provider import get_demo_data, is_using_mock_data, get_mock_badge_html
from styles import apply_custom_css, render_sidebar, COLORS, render_html, render_footer
from components.header import render_header

st.set_page_config(
    page_title="SHADOWCAT — Architecture",
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

analysis = data.get("analysis", {})
state = data.get("current_state", {})

mock_badge_meta = get_mock_badge_html("analysis_metadata")
mock_badge_nov = get_mock_badge_html("novelty_score")

# 1. Ingestion Stream Telemetry
render_html(f"""
<div class="card-title">
    <span>Telemetry Ingestion Stream {mock_badge_meta}</span>
</div>
""")

col_ing1, col_ing2, col_ing3, col_ing4 = st.columns(4)

source_val = analysis.get('source', 'BENCHMARK: CSE-CIC-IDS2018 (Infiltration)')
duration_val = analysis.get('duration_sec', 60)
window_val = analysis.get('window', 't+1 → t+4 (Active)')
flows_val = analysis.get('flows_analyzed', 12450)
packets_val = analysis.get('packets_analyzed', 84920)

with col_ing1:
    render_html(f"""
    <div class="glass-card" style="padding: 16px; min-height: 120px;">
        <div class="metric-label">Dataset Source</div>
        <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; margin-top: 4px;">{source_val}</div>
        <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 2px;">Standardized Network Benchmark</div>
    </div>
    """)

with col_ing2:
    render_html(f"""
    <div class="glass-card" style="padding: 16px; min-height: 120px;">
        <div class="metric-label">Window Duration</div>
        <div style="font-size: 1.25rem; font-weight: 800; color: #00E5FF; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">
            {duration_val}s (10s stride)
        </div>
        <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 2px;">Window: {window_val}</div>
    </div>
    """)

with col_ing3:
    render_html(f"""
    <div class="glass-card" style="padding: 16px; min-height: 120px;">
        <div class="metric-label">Analyzed Throughput</div>
        <div style="font-size: 1.25rem; font-weight: 800; color: #00E676; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">
            {flows_val:,} flows
        </div>
        <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 2px;">{packets_val:,} total packets</div>
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

# 2. Three-Tier Architecture Flow
render_html("""
<div class="card-title" style="margin-top: 14px; margin-bottom: 12px;">
    <span>Three-Tier Architectural Flow</span>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; margin-bottom: 24px;">
    <div class="glass-card" style="min-height: 175px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div style="font-size: 0.72rem; font-weight: 800; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
                Phase 01 · Ingestion & Topology
            </div>
            <div style="font-size: 1.02rem; font-weight: 800; color: #00E5FF; margin-bottom: 8px;">
                Dual-Signal Graph Induction
            </div>
            <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.55;">
                Fuses packet kinematics (TTL, window size, jitter) with host interaction graphs via 2-layer GraphSAGE into Unified Cyber State <b>S(t)</b>.
            </div>
        </div>
        <div style="font-size: 0.74rem; color: #00E5FF; font-weight: 700; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
            Dual Input: Live PCAP & NetFlow CSV
        </div>
    </div>

    <div class="glass-card" style="min-height: 175px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div style="font-size: 0.72rem; font-weight: 800; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
                Phase 02 · Learned Dynamics
            </div>
            <div style="font-size: 1.02rem; font-weight: 800; color: #FFB300; margin-bottom: 8px;">
                Autoregressive World Model
            </div>
            <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.55;">
                Simulates forward network evolution <b>S(t+1) → S(t+k)</b>. Eliminates future-shifted classifiers in favor of true dynamical state transitions.
            </div>
        </div>
        <div style="font-size: 0.74rem; color: #FFB300; font-weight: 700; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
            K-Step Forward Rollout Engine
        </div>
    </div>

    <div class="glass-card" style="min-height: 175px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div style="font-size: 0.72rem; font-weight: 800; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
                Phase 03 · Operational Intelligence
            </div>
            <div style="font-size: 1.02rem; font-weight: 800; color: #00E676; margin-bottom: 8px;">
                Hazard Head & ATT&CK Mapping
            </div>
            <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.55;">
                Predicts attack risk probability, ATT&CK escalation tactics, and calibrated epistemic uncertainty bands to guide analyst triage.
            </div>
        </div>
        <div style="font-size: 0.74rem; color: #00E676; font-weight: 700; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
            Uncertainty-Calibrated Alerts
        </div>
    </div>
</div>
""")

# 3. Core Architecture Modules
render_html("""
<div class="card-title" style="margin-top: 16px; margin-bottom: 12px;">
    <span>Forecasting Engine Architecture</span>
</div>
""")

render_html(r"""
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px;">
    <div class="glass-card">
        <div style="margin-bottom: 10px;">
            <span style="font-size: 1.05rem; font-weight: 800; color: #00E5FF;">1. Temporal Sequence Encoder</span>
        </div>
        <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
            <b>Architecture:</b> Bidirectional LSTM with Multi-Head Self-Attention.<br>
            <b>Input:</b> Inter-arrival timing jitter, packet byte sizes, TCP flag transitions, and TTL hop decay across 60-second sliding windows.<br>
            <b>Representation:</b> 128-dimensional latent temporal vector capturing traffic kinematics.
        </div>
    </div>

    <div class="glass-card">
        <div style="margin-bottom: 10px;">
            <span style="font-size: 1.05rem; font-weight: 800; color: #00E676;">2. Host Topology Inducer</span>
        </div>
        <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
            <b>Architecture:</b> Inductive Graph Neural Network (GraphSAGE).<br>
            <b>Input:</b> Dynamic IP-to-IP bipartite communication graph with localized edge attributes and degree distributions.<br>
            <b>Representation:</b> 64-dimensional structural embedding capturing stealthy lateral connection spreading across subnets.
        </div>
    </div>

    <div class="glass-card">
        <div style="margin-bottom: 10px;">
            <span style="font-size: 1.05rem; font-weight: 800; color: #FFB300;">3. State Dynamics Head</span>
        </div>
        <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
            <b>Architecture:</b> Autoregressive Transition Model with Epistemic Uncertainty Estimation.<br>
            <b>Rollout:</b> Recurrent transition mapping <code>S(t+k) = f(S(t+k-1))</code> across multi-step horizons <b>K=1..4</b>.<br>
            <b>Uncertainty:</b> Monte Carlo Dropout variance sampling yielding calibrated prediction bounds <code>[μ ± σ]</code>.
        </div>
    </div>

    <div class="glass-card">
        <div style="margin-bottom: 10px;">
            <span style="font-size: 1.05rem; font-weight: 800; color: #F43F5E;">4. Air-Gapped Edge Runtime</span>
        </div>
        <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.6;">
            <b>Inference Latency:</b> &lt;45 ms per 10-second traffic window on standard x86 CPU.<br>
            <b>Memory Footprint:</b> &lt;850 MB peak RAM consumption with quantized weights.<br>
            <b>Isolation:</b> Fully air-gapped; requires zero outbound internet access or third-party APIs.
        </div>
    </div>
</div>
""")

# 4. Dual-Signal Evaluation Framework
novelty_val = state.get('novelty_score', 0.23)

render_html(f"""
<div class="glass-card" style="margin-bottom: 20px;">
    <div class="card-title">
        <span>Dual-Signal Decision Manifold {mock_badge_nov}</span>
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
                Signal 2: Out-of-Distribution Novelty Score ({novelty_val:.2f})
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
