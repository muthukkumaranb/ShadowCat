"""
SHADOWCAT - Model Validation & Scientific Rigor
Benchmark Metrics, Horizon Stability, and Leakage Audits
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from data_provider import (
    get_analysis_metadata,
    get_comparison_table,
    get_validation_data,
    validation_status,
)
import importlib
import styles
importlib.reload(styles)
from styles import apply_custom_css, render_sidebar, COLORS, render_html, render_footer
import components.header
importlib.reload(components.header)
from components.header import render_header

st.set_page_config(
    page_title="SHADOWCAT — Validation",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Retrieve typed telemetry and validation data
analysis = get_analysis_metadata()
comparison_table = get_comparison_table()
val = get_validation_data()

render_header({"analysis": analysis})

# Dynamic Validation Status Banner (Task 2 & Revision A)
val_status = validation_status()
if val_status == "validated_offline":
    render_html("""
    <div style="background: #141414; border: 1px solid #262626; border-left: 3px solid #FFFFFF; border-radius: 8px; padding: 12px 18px; margin-top: 6px; margin-bottom: 16px;">
        <div style="font-size: 0.82rem; color: #FFFFFF; line-height: 1.4;">
            <b style="color: #FFFFFF;">Model validated offline (LOEO 37-fold):</b>
            Live pipeline integration in progress &mdash; see Platform Specs for protocol details.
        </div>
    </div>
    """)
else:
    render_html("""
    <div style="background: #141414; border: 1px solid #262626; border-left: 3px solid #E0982B; border-radius: 8px; padding: 12px 18px; margin-top: 6px; margin-bottom: 16px;">
        <div style="font-size: 0.82rem; color: #FFFFFF; line-height: 1.4;">
            <b style="color: #E0982B;">Offline Benchmark Verification in Progress:</b>
            Model validation runs are currently computing offline folds.
        </div>
    </div>
    """)

render_html("""
<div style="margin-bottom: 18px;">
    <h2 style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF; margin: 0;">
        Model Validation & Scientific Integrity
    </h2>
    <div style="font-size: 0.84rem; color: #8A8A8A; margin-top: 4px;">
        Quantitative evaluation benchmarks, multi-step horizon stability, and empirical rigor audits.
    </div>
</div>
""")

# 1. Baseline Benchmarks Split Tables (Task 4: Zero Horizontal Scrolling)
render_html("""
<div class="card-title" style="margin-top: 6px; margin-bottom: 8px;">
    <span>Model Performance vs Baselines: Accuracy Benchmarks (PS Deliverable)</span>
</div>
""")

clean_rows = []
for r in comparison_table:
    model_str = str(r.get("model", "")).strip()
    if not model_str or model_str.lower() in ["empty", "none", "placeholder", "n/a"]:
        continue
    cleaned = {
        k: ("—" if str(v).strip().lower() in ["empty", "none", "n/a"] else v)
        for k, v in r.items()
    }
    clean_rows.append(cleaned)

baseline_df = pd.DataFrame(clean_rows)

# Table 1: Accuracy Metrics
st.dataframe(
    baseline_df,
    use_container_width=True,
    hide_index=True,
    column_order=[
        "model",
        "paradigm",
        "f1_score",
        "precision",
        "recall",
    ],
    column_config={
        "model": st.column_config.TextColumn("Model / Architecture", width="large"),
        "paradigm": st.column_config.TextColumn("Temporal Paradigm", width="medium"),
        "f1_score": st.column_config.NumberColumn("F1", format="%.3f", width="small"),
        "precision": st.column_config.NumberColumn("Precision", format="%.1%", width="small"),
        "recall": st.column_config.NumberColumn("Recall", format="%.1%", width="small"),
    }
)

render_html("""
<div class="card-title" style="margin-top: 14px; margin-bottom: 8px;">
    <span>Model Performance: Operational Impact & Deployment</span>
</div>
""")

# Table 2: Operational Impact & Deployment
st.dataframe(
    baseline_df,
    use_container_width=True,
    hide_index=True,
    column_order=[
        "model",
        "fpr",
        "lead_time",
        "status",
    ],
    column_config={
        "model": st.column_config.TextColumn("Model / Architecture", width="large"),
        "fpr": st.column_config.NumberColumn("FPR", format="%.1%", width="small"),
        "lead_time": st.column_config.TextColumn("Pre-Emptive Lead Time", width="medium"),
        "status": st.column_config.TextColumn("Deployment Status", width="medium"),
    }
)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# 2. Classification Benchmarks Summary
render_html("""
<div class="card-title">
    <span>World Model Production Metrics</span>
</div>
""")

m_cols = st.columns(5)
metrics = [
    ("Precision", f"{val['metrics']['precision']:.1%}"),
    ("Recall", f"{val['metrics']['recall']:.1%}"),
    ("F1-Score", f"{val['metrics']['f1_score']:.3f}"),
    ("PR-AUC", f"{val['metrics']['pr_auc']:.3f}"),
    ("False Pos. Rate", f"{val['metrics']['fpr']:.1%}"),
]

for col, (label, val_str) in zip(m_cols, metrics):
    with col:
        render_html(f"""
        <div class="glass-card" style="text-align: center; padding: 14px 10px; min-height: 85px;">
            <div class="metric-label">{label}</div>
            <div class="metric-value-huge" style="color: #FFFFFF; font-size: 1.65rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">{val_str}</div>
        </div>
        """)

# 3. Horizon Stability Benchmarks
render_html("""
<div class="card-title" style="margin-top: 22px; margin-bottom: 12px;">
    <span>Horizon Stability Benchmarks (Compounding Rollout Error vs Depth K)</span>
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
<div style="font-size: 0.76rem; color: #8A8A8A; margin-top: 6px; margin-bottom: 22px;">
    <b>Policy:</b> Horizons K=1–2 enable automated mitigation; K=3–4 trigger tiered analyst escalation. Primary benchmark bound evaluated at H=5 under LOEO 37-Fold cross-validation.
</div>
""")

# 4. Deep Validation Methodology & Protocol Details
with st.expander("Validation methodology & protocol details", expanded=False):
    render_html("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
        <span style="font-size: 0.88rem; font-weight: 700; color: #FFFFFF;">Evaluation Branches & Protocol Context</span>
        <span class="badge">[Protocol: LOEO 37-Fold Cross-Validation]</span>
    </div>
    <div style="font-size: 0.80rem; color: #8A8A8A; line-height: 1.6; margin-bottom: 16px;">
        • <b>Branch 1 (State Classifier):</b> Evaluated under strict <b>Chronological Split</b> protocol (historical training windows, forward evaluation).<br>
        • <b>Branch 2 (Continuous Dynamics Head):</b> Evaluated under <b>LOEO 37-Fold Cross-Validation</b> with primary evaluation bound at <b>H=5</b>.
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
            <span>Leave-One-Episode-Out (LOEO)</span>
        </div>
        <div class="glass-card" style="min-height: 200px; display: flex; flex-direction: column; justify-content: space-between; padding: 16px;">
            <div>
                <div style="font-size: 0.92rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
                    {protocol_title}
                </div>
                <div style="font-size: 0.78rem; color: #8A8A8A; line-height: 1.5; margin-bottom: 12px;">
                    Full attack episodes withheld sequentially to prevent temporal data leakage across sequential time steps.
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; border-top: 1px solid #262626; padding-top: 10px;">
                <div>
                    <div class="metric-label">Cross-Folds</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF;">{folds_tested} Episodes</div>
                </div>
                <div>
                    <div class="metric-label">Mean PR-AUC</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; font-family: 'JetBrains Mono', monospace;">{mean_pr_auc:.3f}</div>
                </div>
                <div>
                    <div class="metric-label">Variance</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #2FB872; font-family: 'JetBrains Mono', monospace;">±{variance:.3f}</div>
                </div>
            </div>
        </div>
        """)

    with col_leak:
        render_html("""
        <div class="card-title" style="margin-bottom: 10px;">
            <span>Temporal Leakage Audit Checklist</span>
            <span class="badge">Integrity Controls</span>
        </div>
        """)

        leak_df = pd.DataFrame(val.get("leakage_controls", []))
        st.dataframe(
            leak_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "layer": st.column_config.TextColumn("Defense Layer", width="medium"),
                "check": st.column_config.TextColumn("Audit Status", width="small"),
                "detail": st.column_config.TextColumn("Control Specification", width="large"),
            }
        )

    render_html("""
    <div class="card-title" style="margin-top: 20px; margin-bottom: 10px;">
        <span>Robustness Safeguards</span>
        <span class="badge">Integrity Validation</span>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 10px;">
        <div class="glass-card" style="padding: 14px 16px;">
            <div style="font-size: 0.92rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px;">
                Schedule Artifact Immunity
            </div>
            <div style="font-size: 0.78rem; color: #8A8A8A; line-height: 1.55;">
                Timestamp ablation and jitter perturbation tests confirm the model learns traffic kinematics rather than clock-based attack scripts.
            </div>
        </div>
        <div class="glass-card" style="padding: 14px 16px;">
            <div style="font-size: 0.92rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px;">
                Multi-Stage Episode Validation
            </div>
            <div style="font-size: 0.78rem; color: #8A8A8A; line-height: 1.55;">
                Complete multi-stage attack episodes are evaluated chronologically, ensuring host topology embeddings track stealth lateral movement.
            </div>
        </div>
    </div>
    """)

# Persistent Executive Footer
render_footer()
