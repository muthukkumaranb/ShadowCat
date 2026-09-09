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
        text=[f"<b>{c*100:.0f}%</b> &nbsp;({d})" for c, d in zip(contributions, deltas)],
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=11, family="'JetBrains Mono', monospace"),
        hovertemplate="<b>%{y}</b><br>Attribution Weight: %{x:.1%}<extra></extra>"
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
    Renders an executive horizontal MITRE ATT&CK killchain pipeline using st.columns(4).
    Clearly demarcates confirmed past behavior from the predicted future threat stage.
    """
    cols = st.columns(len(mitre_data))
    for i, item in enumerate(mitre_data):
        with cols[i]:
            if i < current_step_idx:
                render_html(f"""
                <div class="mitre-card-observed">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 0.72rem; font-weight: 700; color: #00E676; text-transform: uppercase; letter-spacing: 0.06em;">
                                ✓ OBSERVED
                            </span>
                            <span style="font-size: 0.72rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">
                                {item['id']}
                            </span>
                        </div>
                        <div style="font-size: 1.08rem; font-weight: 800; color: #FFFFFF; margin-bottom: 4px;">
                            {item['stage']}
                        </div>
                        <div style="font-size: 0.78rem; color: #94A3B8; line-height: 1.4;">
                            {item['description']}
                        </div>
                    </div>
                    <div style="font-size: 0.72rem; color: #00E676; font-weight: 600; margin-top: 10px; border-top: 1px solid rgba(0, 230, 118, 0.25); padding-top: 6px;">
                        Active
                    </div>
                </div>
                """)
            elif i == current_step_idx:
                render_html(f"""
                <div class="mitre-card-predicted">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 0.75rem; font-weight: 800; color: #FF1744; text-transform: uppercase; letter-spacing: 0.08em; display: inline-flex; align-items: center; gap: 6px;">
                                <span style="width: 7px; height: 7px; border-radius: 50%; background: #FF1744; display: inline-block;"></span>
                                PREDICTED THREAT
                            </span>
                            <span style="font-size: 0.72rem; color: #FFD54F; font-weight: 700; font-family: 'JetBrains Mono', monospace;">
                                {item['id']} · t+3
                            </span>
                        </div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-bottom: 4px;">
                            {item['stage']}
                        </div>
                        <div style="font-size: 0.8rem; color: #FFFFFF; line-height: 1.4; font-weight: 500;">
                            {item['description']}
                        </div>
                    </div>
                    <div style="background: rgba(255, 23, 68, 0.25); border: 1px solid #FF1744; border-radius: 8px; padding: 5px 8px; font-size: 0.76rem; color: #FFD54F; font-weight: 800; margin-top: 10px; text-align: center; letter-spacing: 0.02em;">
                        P = 67% · Lead Time ~3 min
                    </div>
                </div>
                """)
            else:
                render_html(f"""
                <div class="mitre-card-downstream">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 0.72rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em;">
                                ○ DOWNSTREAM
                            </span>
                            <span style="font-size: 0.72rem; color: #64748B; font-family: 'JetBrains Mono', monospace;">
                                {item['id']}
                            </span>
                        </div>
                        <div style="font-size: 1.08rem; font-weight: 700; color: #94A3B8; margin-bottom: 4px;">
                            {item['stage']}
                        </div>
                        <div style="font-size: 0.78rem; color: #64748B; line-height: 1.4;">
                            {item['description']}
                        </div>
                    </div>
                    <div style="font-size: 0.72rem; color: #64748B; margin-top: 10px; border-top: 1px solid rgba(255, 255, 255, 0.06); padding-top: 6px;">
                        Horizon t+4
                    </div>
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
        <span class="badge">Killchain Pipeline</span>
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
            <span class="badge">Top Weights</span>
        </div>
        """)
        fig = create_attribution_chart(explanation)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_signals:
        render_html(f"""
        <div class="card-title">
            <span>Dual-Signal Evaluation</span>
            <span class="badge">Signature vs Novelty</span>
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
                    • <b>Risk (67%):</b> Lateral pivot predicted at horizon <b>t+3</b>.<br>
                    • <b>Novelty (0.23):</b> Known credential spray pattern (within normal envelope).
                </div>
            </div>
            <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 8px; font-size: 0.74rem; color: #64748B;">
                ✓ Separates familiar attack trajectories from unfamiliar baseline drift.
            </div>
        </div>
        """)
