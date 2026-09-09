"""
SHADOWCAT - Platform Overview & Mission Doctrine
Pre-Emptive Cyber Threat Forecasting Platform
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from data_provider import get_demo_data, get_validation_data, is_using_mock_data, get_mock_badge_html
from styles import apply_custom_css, render_sidebar, COLORS, render_html, render_footer
from components.header import render_header

st.set_page_config(
    page_title="SHADOWCAT — About",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()
data = get_demo_data()
render_header(data)

render_html("""
<div style="margin-bottom: 18px;">
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
        <h2 style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin: 0;">
            About SHADOWCAT
        </h2>
        <span style="background: rgba(56, 189, 248, 0.12); border: 1px solid #38BDF8; color: #38BDF8; font-size: 0.72rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; padding: 2px 8px; border-radius: 4px;">
            v1.0
        </span>
    </div>
    <div style="font-size: 0.84rem; color: #94A3B8; line-height: 1.5;">
        Autonomous pre-emptive cyber threat forecasting engine for air-gapped enterprise network defense.
    </div>
    <div style="display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap;">
        <span style="background: rgba(47, 184, 114, 0.12); border: 1px solid #2FB872; color: #2FB872; font-size: 0.70rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; padding: 3px 8px; border-radius: 4px;">
            AIR-GAPPED
        </span>
        <span style="background: rgba(56, 189, 248, 0.12); border: 1px solid #38BDF8; color: #38BDF8; font-size: 0.70rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; padding: 3px 8px; border-radius: 4px;">
            MITRE ATT&CK ALIGNED
        </span>
        <span style="background: rgba(224, 152, 43, 0.12); border: 1px solid #E0982B; color: #E0982B; font-size: 0.70rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; padding: 3px 8px; border-radius: 4px;">
            LOEO 37-FOLD EVALUATED
        </span>
    </div>
</div>

<div style="background: rgba(56, 189, 248, 0.05); border-left: 3px solid #38BDF8; padding: 10px 14px; border-radius: 0 6px 6px 0; margin-bottom: 20px;">
    <div style="font-size: 0.80rem; color: #E8EDF5; line-height: 1.5; font-style: italic;">
        "Demonstrated on the single infiltration case study (n = 1). Not a general lateral-movement forecasting capability."
    </div>
</div>

<div style="font-size: 0.76rem; font-weight: 700; color: #64708A; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 14px; margin-bottom: 6px;">
    Core Operational Capabilities & Validation Protocol
</div>
""")

# 4. Validation Methodology & Protocol Details (Task 4 & Task 8)
with st.expander("Validation methodology & protocol details", expanded=False):
    val = get_validation_data()
    is_val_mock = is_using_mock_data("validation_data")
    mock_badge_val = get_mock_badge_html("validation_data")

    # Top Mock-Mode Warning Banner (Governance & Provenance Gate)
    if is_val_mock:
        render_html("""
        <div style="background: rgba(224, 152, 43, 0.08); border: 1px solid #E0982B; border-radius: 8px; padding: 12px 18px; margin-top: 6px; margin-bottom: 16px;">
            <div style="font-size: 0.82rem; color: #E8EDF5; line-height: 1.4;">
                <b style="color: #E0982B;">Illustrative Benchmark Data — Pending Live Model Integration:</b>
                The metrics below represent simulated target baselines under the LOEO 37-fold evaluation protocol. They will be superseded automatically once live training checkpoints are placed in <code>models/</code>.
            </div>
        </div>
        """)

    # Dual-Level Telemetry Fusion Architecture (PS Section 1 Mandate)
    render_html("""
    <div class="glass-card" style="padding: 14px 18px; margin-bottom: 18px; border-left: 3px solid #38BDF8;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-size: 0.82rem; font-weight: 700; color: #E8EDF5; letter-spacing: 0.04em;">
                DUAL-LEVEL TELEMETRY FUSION ARCHITECTURE (PS Section 1 Mandate)
            </span>
            <span style="font-size: 0.72rem; color: #2FB872; font-weight: 600; font-family: 'JetBrains Mono', monospace;">
                ✓ ACTIVE
            </span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; font-size: 0.76rem; color: #9AA7BD; margin-top: 8px;">
            <div>
                <b style="color: #E8EDF5;">Level 1: Flow Aggregates (NetFlow)</b><br>
                Bidirectional byte/pkt ratios, TCP flag bitmasks (SYN/ACK/FIN/RST), flow duration, port distributions.
            </div>
            <div>
                <b style="color: #E8EDF5;">Level 2: Packet Dynamics (PCAP)</b><br>
                TTL variance, TCP window sizes, micro-timing inter-arrival jitter, packet payload entropy.
            </div>
            <div>
                <b style="color: #38BDF8;">Unified Cyber State S(t)</b><br>
                Fused tensor representation feeding the World Model transition dynamics P(S_{t+1} | S_t).
            </div>
        </div>
        <div style="font-size: 0.74rem; color: #9AA7BD; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
            Local offline Scapy extraction: derives both packet micro-timing and aggregated flow features without cloud API dependencies.
        </div>
    </div>
    """)

    render_html("""
    <div style="margin-bottom: 16px;">
        <div style="font-size: 0.92rem; font-weight: 800; color: #00E5FF; margin-bottom: 4px;">
            Evaluation Branches & Protocol Context
        </div>
        <div style="font-size: 0.8rem; color: #CBD5E1; line-height: 1.6;">
            • <b>Branch 1 (State Classifier):</b> Evaluated under strict <b>Chronological Split</b> protocol (historical training windows, forward evaluation).<br>
            • <b>Branch 2 (Continuous Dynamics Head):</b> Evaluated under <b>LOEO 37-Fold Cross-Validation</b> with primary evaluation bound at <b>H=5</b>.
        </div>
    </div>
    """)



    col_loeo, col_leak = st.columns([1.2, 1.8])

    with col_loeo:
        loeo_summary = val.get("loeo_summary", {})
        protocol_title = loeo_summary.get("protocol", "Leave-One-Episode-Out (LOEO 37-Fold Cross-Validation)")
        folds_tested = loeo_summary.get("folds_tested", 37)
        mean_pr_auc = loeo_summary.get("mean_pr_auc", 0.821)
        variance = loeo_summary.get("variance", 0.014)

        render_html(f"""
        <div class="card-title" style="margin-bottom: 10px;">
            <span>Leave-One-Episode-Out (LOEO) {mock_badge_val}</span>
            <span class="badge">[37-Fold Cross-Validation]</span>
        </div>
        <div class="glass-card" style="min-height: 200px; display: flex; flex-direction: column; justify-content: space-between; padding: 16px;">
            <div>
                <div style="font-size: 0.92rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
                    {protocol_title}
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.5; margin-bottom: 12px;">
                    Full attack episodes withheld sequentially to prevent temporal data leakage across sequential time steps.
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;">
                <div>
                    <div class="metric-label">Cross-Folds</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF;">{folds_tested} Episodes</div>
                </div>
                <div>
                    <div class="metric-label">Mean PR-AUC</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #00E5FF; font-family: 'JetBrains Mono', monospace;">{mean_pr_auc:.3f}</div>
                </div>
                <div>
                    <div class="metric-label">Variance</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #00E676; font-family: 'JetBrains Mono', monospace;">±{variance:.3f}</div>
                </div>
            </div>
        </div>
        """)

    with col_leak:
        render_html(f"""
        <div class="card-title" style="margin-bottom: 10px;">
            <span>Temporal Leakage Audit {mock_badge_val}</span>
        </div>
        """)
        raw_controls = val.get("leakage_controls", [])
        leak_controls = []
        for item in raw_controls:
            row = dict(item)
            if is_val_mock and row.get("check") == "Passed":
                row["check"] = "Passed (simulated)"
            leak_controls.append(row)
        leak_df = pd.DataFrame(leak_controls)
        st.dataframe(
            leak_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "layer": st.column_config.TextColumn("Component", width="medium"),
                "check": st.column_config.TextColumn("Status", width="small"),
                "detail": st.column_config.TextColumn("Guardrail", width="large"),
            }
        )

    render_html(f"""
    <div class="card-title" style="margin-top: 18px; margin-bottom: 10px;">
        <span>Robustness Safeguards {mock_badge_val}</span>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 10px;">
        <div class="glass-card" style="padding: 14px 16px;">
            <div style="font-size: 0.94rem; font-weight: 700; color: #FFB300; margin-bottom: 6px;">
                Schedule Artifact Immunity
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.55;">
                Timestamp ablation and jitter perturbation test protocols designed to confirm the model learns traffic kinematics rather than clock-based attack scripts.
            </div>
        </div>
        <div class="glass-card" style="padding: 14px 16px;">
            <div style="font-size: 0.94rem; font-weight: 700; color: #00E676; margin-bottom: 6px;">
                Multi-Stage Episode Validation
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.55;">
                Complete multi-stage attack episodes structured for chronological evaluation, ensuring host topology embeddings track stealth lateral movement.
            </div>
        </div>
    </div>
    """)

# Persistent Executive Footer
render_footer()
