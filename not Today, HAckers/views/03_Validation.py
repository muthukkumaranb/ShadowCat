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
    is_using_mock_data,
    get_mock_badge_html,
)
from styles import apply_custom_css, render_sidebar, COLORS, render_html, render_footer
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
st.caption("Protocol: LOEO 37-Fold Cross-Validation · Leave-One-Episode-Out")

# Top Mock-Mode Warning Banner (Governance & Provenance Gate)
is_val_mock = is_using_mock_data("validation_data") or is_using_mock_data("comparison_table")
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
<div style="margin-bottom: 18px;">
    <h2 style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF; margin: 0;">
        Model Validation & Scientific Integrity
    </h2>
    <div style="font-size: 0.84rem; color: #9AA7BD; margin-top: 4px;">
        Quantitative evaluation benchmarks, multi-step horizon stability, and empirical rigor audits.
    </div>
</div>
""")

mock_badge_comp = get_mock_badge_html("comparison_table")
mock_badge_val = get_mock_badge_html("validation_data")

# 1. Baseline Benchmarks (PS Evaluation Deliverable: World Model vs Logistic Regression Baseline)
render_html(f"""
<div class="card-title" style="margin-top: 6px; margin-bottom: 10px;">
    <span>Model Performance vs Baselines (PS Benchmark Deliverable) {mock_badge_comp}</span>
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
        "fpr",
        "lead_time",
        "status",
    ],
    column_config={
        "model": st.column_config.TextColumn("Model / Architecture", width="large"),
        "paradigm": st.column_config.TextColumn("Temporal Paradigm", width="medium"),
        "f1_score": st.column_config.NumberColumn("F1", format="%.3f", width="small"),
        "precision": st.column_config.NumberColumn("Precision", format="%.1%", width="small"),
        "recall": st.column_config.NumberColumn("Recall", format="%.1%", width="small"),
        "fpr": st.column_config.NumberColumn("FPR", format="%.1%", width="small"),
        "lead_time": st.column_config.TextColumn("Pre-Emptive Lead Time", width="medium"),
        "status": st.column_config.TextColumn("Deployment Status", width="medium"),
    }
)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# 2. Classification Benchmarks Summary
render_html(f"""
<div class="card-title">
    <span>World Model Production Metrics {mock_badge_val}</span>
    <span class="badge">[Protocol: LOEO 37-Fold Cross-Validation]</span>
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
            <div class="metric-value-huge" style="color: #38BDF8; font-size: 1.65rem; margin-top: 4px;">{val_str}</div>
        </div>
        """)

# 3. Horizon Stability Benchmarks (Clean header, all 5 rows preserved)
render_html(f"""
<div class="card-title" style="margin-top: 22px; margin-bottom: 12px;">
    <span>Horizon Stability Benchmarks (Compounding Rollout Error vs Depth K) {mock_badge_val}</span>
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
    <b>Policy:</b> Horizons K=1–2 enable automated mitigation; K=3–4 trigger tiered analyst escalation. Primary benchmark bound evaluated at H=5 under LOEO 37-Fold cross-validation.
</div>
""")

# Persistent Executive Footer
render_footer()
