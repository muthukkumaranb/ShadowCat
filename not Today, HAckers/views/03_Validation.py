"""
SHADOWCAT - Model Validation & Scientific Rigor
Benchmark Metrics, Horizon Stability, and Leakage Audits
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from mock_data import get_demo_data
from styles import apply_custom_css, render_sidebar, COLORS, render_html, render_footer
from components.header import render_header

st.set_page_config(
    page_title="SHADOWCAT — Validation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()
data = get_demo_data()
render_header(data)

render_html("""
<div style="margin-bottom: 18px;">
    <h2 style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF; margin: 0;">
        Model Validation & Scientific Integrity
    </h2>
    <div style="font-size: 0.84rem; color: #94A3B8; margin-top: 4px;">
        Quantitative evaluation benchmarks, multi-step horizon stability, and empirical rigor audits.
    </div>
</div>
""")

val = data["validation"]

# 1. Classification Metrics (Visible by default)
render_html("""
<div class="card-title">
    <span>Classification Benchmarks</span>
    <span class="badge">[Branch: State Classifier · Protocol: Chronological Split]</span>
</div>
""")

m_cols = st.columns(5)
metrics = [
    ("Precision", f"{val['metrics']['precision']:.1%}", "High precision (95% CI)"),
    ("Recall", f"{val['metrics']['recall']:.1%}", "80% multi-stage recall"),
    ("F1-Score", f"{val['metrics']['f1_score']:.3f}", "Balanced on skewed data"),
    ("PR-AUC", f"{val['metrics']['pr_auc']:.3f}", "Precision-Recall area"),
    ("False Pos. Rate", f"{val['metrics']['fpr']:.1%}", "< 5% false alarm rate"),
]

for col, (label, val_str, desc) in zip(m_cols, metrics):
    with col:
        render_html(f"""
        <div class="glass-card" style="text-align: center; padding: 18px 12px; min-height: 140px;">
            <div class="metric-label">{label}</div>
            <div class="metric-value-huge" style="color: #00E5FF; font-size: 1.95rem;">{val_str}</div>
            <div style="font-size: 0.72rem; color: #94A3B8;">{desc}</div>
        </div>
        """)

# 2. Horizon Stability Benchmarks (Visible by default)
render_html("""
<div class="card-title" style="margin-top: 20px; margin-bottom: 12px;">
    <span>Horizon Stability Benchmarks</span>
    <span class="badge">[Branch: Continuous Dynamics Head · Protocol: LOEO 37-Fold (H=5 Primary)]</span>
</div>
""")

horizon_df = pd.DataFrame(val["horizons"])
st.dataframe(
    horizon_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "horizon": st.column_config.TextColumn("Horizon K", width="medium"),
        "status": st.column_config.TextColumn("Status", width="small"),
        "f1": st.column_config.NumberColumn("F1 Score", format="%.2f", width="small"),
        "error_growth": st.column_config.TextColumn("Drift", width="small"),
        "verdict": st.column_config.TextColumn("Deployment Protocol", width="large"),
    }
)

render_html("""
<div style="font-size: 0.76rem; color: #64748B; margin-top: 6px; margin-bottom: 22px;">
    ⚠️ <b>Policy:</b> Horizons K=1–2 enable automated mitigation; K=3–4 trigger tiered analyst escalation. Primary benchmark bound evaluated at H=5.
</div>
""")

# 3. Deep Validation Methodology (Collapsed by default per Task 5)
with st.expander("Show validation methodology", expanded=False):
    col_loeo, col_leak = st.columns([1.2, 1.8])

    with col_loeo:
        render_html("""
        <div class="card-title">
            <span>Leave-One-Episode-Out (LOEO)</span>
            <span class="badge">[Protocol: LOEO 37-Fold Cross-Validation]</span>
        </div>
        """)

        render_html(f"""
        <div class="glass-card" style="min-height: 220px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px;">
                    {val['loeo_summary']['protocol']}
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.55; margin-bottom: 14px;">
                    Full attack episodes withheld sequentially to prevent temporal data leakage.
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 12px;">
                <div>
                    <div class="metric-label">Cross-Folds</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">{val['loeo_summary']['folds_tested']} Episodes</div>
                </div>
                <div>
                    <div class="metric-label">Mean PR-AUC</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #00E5FF; font-family: 'JetBrains Mono', monospace;">{val['loeo_summary']['mean_pr_auc']:.3f}</div>
                </div>
                <div>
                    <div class="metric-label">Variance</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #00E676; font-family: 'JetBrains Mono', monospace;">±{val['loeo_summary']['variance']:.3f}</div>
                </div>
            </div>
        </div>
        """)

    with col_leak:
        render_html("""
        <div class="card-title">
            <span>Temporal Leakage Audit</span>
            <span class="badge">Integrity Controls</span>
        </div>
        """)

        leak_df = pd.DataFrame(val["leakage_controls"])
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

    render_html("""
    <div class="card-title" style="margin-top: 22px; margin-bottom: 12px;">
        <span>Robustness Safeguards</span>
        <span class="badge">Integrity Validation</span>
    </div>
    """)

    render_html(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 10px;">
        <div class="glass-card">
            <div style="font-size: 0.98rem; font-weight: 700; color: #FFB300; margin-bottom: 8px;">
                ⏱️ Schedule Artifact Immunity
            </div>
            <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.6;">
                Timestamp ablation and jitter perturbation tests confirm the model learns traffic kinematics rather than clock-based attack scripts.
            </div>
        </div>
        <div class="glass-card">
            <div style="font-size: 0.98rem; font-weight: 700; color: #00E676; margin-bottom: 8px;">
                🔬 Multi-Stage Episode Validation
            </div>
            <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.6;">
                Complete multi-stage attack episodes are evaluated chronologically, ensuring host topology embeddings track stealth lateral movement.
            </div>
        </div>
    </div>
    """)

# Persistent Executive Footer
render_footer()

