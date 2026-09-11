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
            color="#E4E4E7",
            line=dict(color="rgba(255, 255, 255, 0.3)", width=1)
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
        paper_bgcolor="#141414",
        plot_bgcolor="#141414",
        hoverlabel=dict(
            bgcolor="#181818",
            bordercolor="#333333",
            font=dict(color="#FFFFFF", size=11, family="'JetBrains Mono', monospace")
        ),
        margin=dict(l=10, r=20, t=32, b=24),
        height=240,
        xaxis=dict(
            range=[0, 0.68],
            tickformat=".0%",
            gridcolor="#222222",
            zerolinecolor="#222222",
            tickfont=dict(color="#8A8A8A", size=10),
            title=dict(text="Attribution Weight", font=dict(color="#8A8A8A", size=10)),
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
        border_right = "border-right: 1px solid #262626;" if i < len(mitre_data) - 1 else ""
        if i < current_step_idx:
            segment = f"""
            <div style="flex: 1; padding: 14px 16px; {border_right} border-top: 3px solid #2FB872; background: #141414;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.68rem; font-weight: 700; color: #2FB872; text-transform: uppercase; letter-spacing: 0.05em;">
                        ✓ OBSERVED
                    </span>
                    <span style="font-size: 0.70rem; color: #8A8A8A; font-family: 'JetBrains Mono', monospace;">
                        {item['id']}
                    </span>
                </div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #FFFFFF; margin-bottom: 3px;">
                    {item['stage']}
                </div>
                <div style="font-size: 0.68rem; color: #2FB872; font-weight: 600; margin-top: 2px;">
                    Historical Stage
                </div>
                <details style="margin-top: 6px; cursor: pointer;">
                    <summary style="font-size: 0.70rem; color: #2FB872; font-weight: 600; outline: none; user-select: none;">
                        Details
                    </summary>
                    <div style="font-size: 0.74rem; color: #8A8A8A; line-height: 1.35; margin-top: 4px; padding-top: 4px; border-top: 1px dashed rgba(255, 255, 255, 0.1);">
                        {item['description']}
                    </div>
                </details>
            </div>
            """
        elif i == current_step_idx:
            segment = f"""
            <div style="flex: 1.15; padding: 14px 16px; {border_right} border-top: 3px solid #E5484D; background: rgba(229, 72, 77, 0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.70rem; font-weight: 800; color: #E5484D; text-transform: uppercase; letter-spacing: 0.06em; display: inline-flex; align-items: center; gap: 5px;">
                        <span style="width: 6px; height: 6px; border-radius: 50%; background: #E5484D; display: inline-block;"></span>
                        PREDICTED THREAT
                    </span>
                    <span style="font-size: 0.72rem; color: #FFFFFF; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                        {item['id']} · t+3
                    </span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #FFFFFF; margin-bottom: 3px;">
                    {item['stage']}
                </div>
                <div style="display: inline-block; background: rgba(229, 72, 77, 0.2); border: 1px solid #E5484D; border-radius: 4px; padding: 2px 6px; font-size: 0.70rem; color: #FFFFFF; font-weight: 700; margin-top: 2px; font-family: 'JetBrains Mono', monospace;">
                    Elevated Risk · Horizon t+3
                </div>
                <details style="margin-top: 6px; cursor: pointer;">
                    <summary style="font-size: 0.70rem; color: #E5484D; font-weight: 600; outline: none; user-select: none;">
                        Details
                    </summary>
                    <div style="font-size: 0.74rem; color: #FFFFFF; line-height: 1.35; margin-top: 4px; padding-top: 4px; border-top: 1px dashed rgba(255, 255, 255, 0.1);">
                        {item['description']}
                    </div>
                </details>
            </div>
            """
        else:
            segment = f"""
            <div style="flex: 1; padding: 14px 16px; {border_right} border-top: 3px solid #262626; background: #111111; opacity: 0.75;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.68rem; font-weight: 600; color: #666666; text-transform: uppercase; letter-spacing: 0.04em;">
                        ○ DOWNSTREAM
                    </span>
                    <span style="font-size: 0.70rem; color: #666666; font-family: 'JetBrains Mono', monospace;">
                        {item['id']}
                    </span>
                </div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #8A8A8A; margin-bottom: 3px;">
                    {item['stage']}
                </div>
                <div style="font-size: 0.68rem; color: #666666; margin-top: 2px;">
                    Horizon t+4 (Rollout)
                </div>
                <details style="margin-top: 6px; cursor: pointer;">
                    <summary style="font-size: 0.70rem; color: #666666; font-weight: 600; outline: none; user-select: none;">
                        Details
                    </summary>
                    <div style="font-size: 0.74rem; color: #666666; line-height: 1.35; margin-top: 4px; padding-top: 4px; border-top: 1px dashed rgba(255, 255, 255, 0.1);">
                        {item['description']}
                    </div>
                </details>
            </div>
            """
        segments_html.append(segment)

    all_segments = "".join(segments_html)
    render_html(f"""
    <div style="display: flex; border: 1px solid #262626; border-radius: 8px; overflow: hidden; margin-top: 4px; margin-bottom: 18px;">
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
                    <span style="background: rgba(47, 184, 114, 0.12); border: 1px solid #2FB872; color: #2FB872; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.78rem;">
                        Score: {state['novelty_score']:.2f} / 1.0
                    </span>
                </div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; line-height: 1.3;">
                    Known Attack Escalation
                </div>
                <div style="font-size: 0.8rem; color: #8A8A8A; margin-top: 8px; line-height: 1.6;">
                    • <b>Predicted Threat:</b> Lateral pivot predicted at horizon <b>t+3</b>.<br>
                    • <b>Baseline Novelty:</b> Known credential spray pattern (within normal envelope).
                </div>
            </div>
            <div style="border-top: 1px solid #262626; padding-top: 8px; font-size: 0.74rem; color: #8A8A8A;">
                ✓ Separates familiar attack trajectories from unfamiliar baseline drift.
            </div>
        </div>
        """)
