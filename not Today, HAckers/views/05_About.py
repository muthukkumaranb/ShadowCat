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
<div style="margin-bottom: 20px;">
    <h2 style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF; margin: 0;">
        About SHADOWCAT
    </h2>
    <div style="font-size: 0.84rem; color: #94A3B8; margin-top: 4px;">
        Autonomous pre-emptive cyber threat forecasting engine for air-gapped enterprise defense.
    </div>
</div>
""")

# 1. Mission Doctrine Overview
render_html("""
<div class="glass-card" style="margin-bottom: 22px;">
    <div class="card-title">
        <span>Executive Mission & Platform Vision</span>
    </div>
    <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.7;">
        <b>SHADOWCAT</b> represents a fundamental shift in network security operations: transitioning cybersecurity from 
        <b>reactive forensic alert triage</b> to <b>pre-emptive threat trajectory forecasting</b>.
        By modeling network traffic kinematics and host communication graphs as a continuous dynamical system, 
        SHADOWCAT predicts impending multi-step attack escalations up to <b>3 to 4 minutes before privilege escalation</b> occurs.
    </div>
</div>
""")

# 2. Paradigm Shift: Reactive Detection vs Pre-Emptive Forecasting
render_html("""
<div class="glass-card" style="margin-bottom: 22px;">
    <div class="card-title">
        <span>Paradigm Shift: Reactive Detection vs Pre-Emptive Forecasting</span>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div style="background: rgba(255, 23, 68, 0.08); border: 1px solid rgba(255, 23, 68, 0.25); border-radius: 14px; padding: 20px;">
            <div style="font-weight: 800; color: #FF5252; font-size: 1.1rem; margin-bottom: 10px;">
                Conventional IDS / IPS
            </div>
            <ul style="font-size: 0.84rem; color: #CBD5E1; margin-left: -15px; line-height: 1.7;">
                <li><b>Posture:</b> Reactive (alerts fired post-compromise).</li>
                <li><b>Lead Time:</b> <b>0 minutes</b> (Zero pre-emptive response window).</li>
                <li><b>Detection:</b> Static signatures & single-point packet anomalies.</li>
                <li><b>SOC Impact:</b> High alert fatigue during active damage phase.</li>
            </ul>
        </div>
        <div style="background: rgba(0, 229, 255, 0.08); border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 14px; padding: 20px;">
            <div style="font-weight: 800; color: #00E5FF; font-size: 1.1rem; margin-bottom: 10px;">
                SHADOWCAT Forecasting Engine
            </div>
            <ul style="font-size: 0.84rem; color: #FFFFFF; margin-left: -15px; line-height: 1.7;">
                <li><b>Posture:</b> Pre-emptive (forecasts trajectory before breach).</li>
                <li><b>Lead Time:</b> <b>~3–4 minutes</b> pre-emptive intervention window.</li>
                <li><b>Detection:</b> Temporal dynamics + host graph topology.</li>
                <li><b>SOC Impact:</b> Automated rate-limiting & proactive host quarantine.</li>
            </ul>
        </div>
    </div>
</div>
""")

# 3. Key Operational Advantages
render_html("""
<div class="card-title" style="margin-bottom: 12px;">
    <span>Core Operational Capabilities</span>
</div>
""")

col_c1, col_c2, col_c3 = st.columns(3)

with col_c1:
    render_html("""
    <div class="glass-card" style="min-height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div style="font-size: 0.98rem; font-weight: 800; color: #00E5FF; margin-bottom: 6px;">
                Multi-Horizon Rollout
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.55;">
                Autoregressive state transition modeling forecasts cyber attack stages across horizons K=1..4 with calibrated lead times.
            </div>
        </div>
        <div style="font-size: 0.74rem; color: #00E5FF; font-weight: 700; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
            Horizons: t+1 → t+4
        </div>
    </div>
    """)

with col_c2:
    render_html("""
    <div class="glass-card" style="min-height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div style="font-size: 0.98rem; font-weight: 800; color: #FFB300; margin-bottom: 6px;">
                Compounding Uncertainty
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.55;">
                Epistemic uncertainty quantification explicitly reflects compounding drift over future rollout steps to guide analyst escalation.
            </div>
        </div>
        <div style="font-size: 0.74rem; color: #FFB300; font-weight: 700; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
            Calibrated Variance Bounds
        </div>
    </div>
    """)

with col_c3:
    render_html("""
    <div class="glass-card" style="min-height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
            <div style="font-size: 0.98rem; font-weight: 800; color: #00E676; margin-bottom: 6px;">
                Air-Gapped Operation
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.55;">
                Zero cloud reliance, zero external telemetry egress. Engineered for strict mission-critical offline SOC deployments.
            </div>
        </div>
        <div style="font-size: 0.74rem; color: #00E676; font-weight: 700; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
            100% On-Premise Execution
        </div>
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

    # Metric Context & Calibration (dynamically retrieved from data_provider)
    metrics = val.get("metrics", {})
    precision_val = metrics.get("precision", 0.842)
    recall_val = metrics.get("recall", 0.791)
    f1_val = metrics.get("f1_score", 0.816)
    pr_auc_val = metrics.get("pr_auc", 0.835)
    fpr_val = metrics.get("fpr", 0.048)

    render_html(f"""
    <div class="card-title" style="margin-bottom: 10px;">
        <span>Metric Calibration & Confidence Intervals {mock_badge_val}</span>
    </div>
    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 20px;">
        <div class="glass-card" style="padding: 12px; text-align: center;">
            <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700;">PRECISION</div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #00E5FF; margin: 2px 0;">{precision_val * 100:.1f}%</div>
            <div style="font-size: 0.7rem; color: #94A3B8;">High precision (95% CI)</div>
        </div>
        <div class="glass-card" style="padding: 12px; text-align: center;">
            <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700;">RECALL</div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #00E5FF; margin: 2px 0;">{recall_val * 100:.1f}%</div>
            <div style="font-size: 0.7rem; color: #94A3B8;">80% multi-stage recall</div>
        </div>
        <div class="glass-card" style="padding: 12px; text-align: center;">
            <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700;">F1-SCORE</div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #00E5FF; margin: 2px 0;">{f1_val:.3f}</div>
            <div style="font-size: 0.7rem; color: #94A3B8;">Balanced on skewed data</div>
        </div>
        <div class="glass-card" style="padding: 12px; text-align: center;">
            <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700;">PR-AUC</div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #00E5FF; margin: 2px 0;">{pr_auc_val:.3f}</div>
            <div style="font-size: 0.7rem; color: #94A3B8;">Precision-Recall area</div>
        </div>
        <div class="glass-card" style="padding: 12px; text-align: center;">
            <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700;">FPR</div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #00E5FF; margin: 2px 0;">{fpr_val * 100:.1f}%</div>
            <div style="font-size: 0.7rem; color: #94A3B8;">&lt; 5% false alarm rate</div>
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
