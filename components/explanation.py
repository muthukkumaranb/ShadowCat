"""
SHADOWCAT - Explanation & MITRE Pipeline Component
"""

import streamlit as st
import plotly.graph_objects as go
from styles import COLORS, render_html


def create_attribution_chart(explanation_data):
    """Creates a sleek horizontal bar chart showing feature-group attribution contributions."""
    features = [item["feature"] for item in reversed(explanation_data)]
    contributions = [item["contribution"] for item in reversed(explanation_data)]
    deltas = [item["delta"] for item in reversed(explanation_data)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=features,
        x=contributions,
        orientation="h",
        marker=dict(
            color="#00E5FF",
            line=dict(color="rgba(255, 255, 255, 0.2)", width=1)
        ),
        text=[f"<b>{c*100:.0f}%</b>" for c in contributions],
        customdata=deltas,
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11, family="'JetBrains Mono', monospace"),
        hovertemplate="<b>%{y}</b><br>Attribution Weight: %{x:.1%}<br>Signal Deviation: %{customdata}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="<b>Feature Attribution (Deletion-Tested · Integrated Gradients)</b>",
            font=dict(size=12, color="#FFFFFF")
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        margin=dict(l=10, r=20, t=32, b=24),
        height=240,
        xaxis=dict(
            range=[0, 0.68],
            tickformat=".0%",
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.05)",
            tickfont=dict(color="#94A3B8", size=10),
            title=dict(text="Attribution Weight", font=dict(color="#94A3B8", size=10)),
        ),
        yaxis=dict(
            tickfont=dict(color="#FFFFFF", size=11, family="'Plus Jakarta Sans', sans-serif"),
            gridcolor="rgba(0,0,0,0)",
        )
    )
    return fig


def render_attack_stepper(mitre_data, current_step_idx=2):
    """
    Renders an executive horizontal MITRE ATT&CK killchain pipeline as a single continuous track.
    Compact by default with click-to-expand details per Task 10.
    """
    segments_html = []
    for i, item in enumerate(mitre_data):
        item_id = item.get("id", item.get("tactic_id", f"TA000{i+1}"))
        item_stage = item.get("stage", "Unknown Stage")
        item_desc = item.get("description", "No detailed description available.")
        border_right = "border-right: 1px solid #22304a;" if i < len(mitre_data) - 1 else ""
        if i < current_step_idx:
            segment = f"""
            <div style="flex: 1; padding: 14px 16px; {border_right} border-top: 3px solid #2fb872; background: #131b2c;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.68rem; font-weight: 700; color: #2fb872; text-transform: uppercase; letter-spacing: 0.05em;">
                        ✓ OBSERVED
                    </span>
                    <span style="font-size: 0.70rem; color: #9aa7bd; font-family: 'JetBrains Mono', monospace;">
                        {item_id}
                    </span>
                </div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #e8edf5; margin-bottom: 3px;">
                    {item_stage}
                </div>
                <div style="font-size: 0.68rem; color: #2fb872; font-weight: 600; margin-top: 2px;">
                    Historical Stage
                </div>
                <details style="margin-top: 6px; cursor: pointer;">
                    <summary style="font-size: 0.70rem; color: #2fb872; font-weight: 600; outline: none; user-select: none;">
                        Details
                    </summary>
                    <div style="font-size: 0.74rem; color: #9aa7bd; line-height: 1.35; margin-top: 4px; padding-top: 4px; border-top: 1px dashed rgba(255, 255, 255, 0.1);">
                        {item_desc}
                    </div>
                </details>
            </div>
            """
        elif i == current_step_idx:
            segment = f"""
            <div style="flex: 1.15; padding: 14px 16px; {border_right} border-top: 3px solid #e5484d; background: rgba(229, 72, 77, 0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.70rem; font-weight: 800; color: #e5484d; text-transform: uppercase; letter-spacing: 0.06em; display: inline-flex; align-items: center; gap: 5px;">
                        <span style="width: 6px; height: 6px; border-radius: 50%; background: #e5484d; display: inline-block;"></span>
                        PREDICTED THREAT
                    </span>
                    <span style="font-size: 0.72rem; color: #38bdf8; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                        {item_id} · t+3
                    </span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #ffffff; margin-bottom: 3px;">
                    {item_stage}
                </div>
                <div style="display: inline-block; background: rgba(229, 72, 77, 0.2); border: 1px solid #e5484d; border-radius: 4px; padding: 2px 6px; font-size: 0.70rem; color: #ffffff; font-weight: 700; margin-top: 2px; font-family: 'JetBrains Mono', monospace;">
                    Elevated Risk · Horizon t+3
                </div>
                <details style="margin-top: 6px; cursor: pointer;">
                    <summary style="font-size: 0.70rem; color: #38bdf8; font-weight: 600; outline: none; user-select: none;">
                        Details
                    </summary>
                    <div style="font-size: 0.74rem; color: #e8edf5; line-height: 1.35; margin-top: 4px; padding-top: 4px; border-top: 1px dashed rgba(255, 255, 255, 0.1);">
                        {item_desc}
                    </div>
                </details>
            </div>
            """
        else:
            segment = f"""
            <div style="flex: 1; padding: 14px 16px; {border_right} border-top: 3px solid #22304a; background: #0e1422; opacity: 0.75;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.68rem; font-weight: 600; color: #64708a; text-transform: uppercase; letter-spacing: 0.04em;">
                        ○ DOWNSTREAM
                    </span>
                    <span style="font-size: 0.70rem; color: #64708a; font-family: 'JetBrains Mono', monospace;">
                        {item_id}
                    </span>
                </div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #9aa7bd; margin-bottom: 3px;">
                    {item_stage}
                </div>
                <div style="font-size: 0.68rem; color: #64708a; margin-top: 2px;">
                    Horizon t+4 (Rollout)
                </div>
                <details style="margin-top: 6px; cursor: pointer;">
                    <summary style="font-size: 0.70rem; color: #64708a; font-weight: 600; outline: none; user-select: none;">
                        Details
                    </summary>
                    <div style="font-size: 0.74rem; color: #64708a; line-height: 1.35; margin-top: 4px; padding-top: 4px; border-top: 1px dashed rgba(255, 255, 255, 0.1);">
                        {item_desc}
                    </div>
                </details>
            </div>
            """
        segments_html.append(segment)

    all_segments = "".join(segments_html)
    render_html(f"""
    <div style="display: flex; border: 1px solid #22304a; border-radius: 8px; overflow: hidden; margin-top: 4px; margin-bottom: 18px;">
        {all_segments}
    </div>
    """)


def render_explanation_section(data):
    """Renders the executive MITRE pipeline, attribution chart, and two-signal manifold note."""
    explanation = data["explanation"]
    mitre = data["mitre"]
    state = data["current_state"]

    # 1. Full-Width Horizontal MITRE ATT&CK Pipeline
    render_html("""
    <div class="card-title" style="margin-top: 18px; margin-bottom: 12px;">
        <span>MITRE ATT&CK Progression</span>
    </div>
    """)
    render_attack_stepper(mitre, current_step_idx=2)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 2. Attribution & Two-Signal Analysis
    col_attrib, col_signals = st.columns([1.6, 1.4])

    with col_attrib:
        render_html("""
        <div class="card-title">
            <span>Key Feature Attribution</span>
        </div>
        """)
        fig = create_attribution_chart(explanation)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_signals:
        render_html(f"""
        <div class="card-title">
            <span>Dual-Signal Evaluation</span>
        </div>
        <div class="glass-card" style="height: 240px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="metric-label">Novelty Score</span>
                    <span style="background: rgba(0, 230, 118, 0.12); border: 1px solid #00E676; color: #00E676; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.78rem;">
                        Score: {state['novelty_score']:.2f} / 1.0
                    </span>
                </div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; line-height: 1.3;">
                    Known Attack Escalation
                </div>
                <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 8px; line-height: 1.6;">
                    • <b>Predicted Threat:</b> Lateral pivot predicted at horizon <b>t+3</b>.<br>
                    • <b>Baseline Novelty:</b> Known credential spray pattern (within normal envelope).
                </div>
            </div>
            <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 8px; font-size: 0.74rem; color: #64748B;">
                ✓ Separates familiar attack trajectories from unfamiliar baseline drift.
            </div>
        </div>
        """)
