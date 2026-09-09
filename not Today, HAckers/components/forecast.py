"""
SHADOWCAT - Hero Forecast Component
"""

import streamlit as st
import plotly.graph_objects as go
import textwrap
from styles import COLORS, render_html


def create_forecast_chart(forecast_data):
    """
    Creates an executive-grade Plotly chart showing the attack risk probability trajectory
    with growing epistemic uncertainty bands across forecast horizons K=1..4.
    """
    x_labels = ["Now (t)", "t+1 (1 min)", "t+2 (2 min)", "t+3 (3 min)", "t+4 (4 min)"]
    probs = [0.06] + [step["probability"] for step in forecast_data]
    lower_bounds = [0.06] + [step["lower_bound"] for step in forecast_data]
    upper_bounds = [0.06] + [step["upper_bound"] for step in forecast_data]
    uncertainties = [0.00] + [step["uncertainty"] for step in forecast_data]
    stages = ["Observation Window"] + [step["stage"] for step in forecast_data]

    fig = go.Figure()

    # 1. Lower Bound Anchor Line
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=lower_bounds,
        mode="lines",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
        name="Lower Bound"
    ))

    # 2. Upper Bound + Luminous Uncertainty Shading
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=upper_bounds,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0, 229, 255, 0.14)",
        hoverinfo="skip",
        name="Compounding Uncertainty (±σ)",
        showlegend=True
    ))

    # 3. Main Luminous Trajectory Spline
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=probs,
        mode="lines+markers+text",
        name="Forecasted Risk P(Attack)",
        line=dict(color="#00E5FF", width=4.5, shape="spline", smoothing=0.85),
        marker=dict(
            size=[12, 14, 15, 19, 17],
            color=["#00E676", "#38BDF8", "#FFB300", "#FF1744", "#F43F5E"],
            line=dict(color="#FFFFFF", width=2.5)
        ),
        text=[f"<b>{p*100:.0f}%</b>" for p in probs],
        textposition="top left",
        textfont=dict(color="#FFD54F", size=13, family="'JetBrains Mono', monospace"),
        customdata=list(zip(stages, [f"±{u*100:.0f}%" for u in uncertainties], [f"[{l*100:.0f}% – {u*100:.0f}%]" for l, u in zip(lower_bounds, upper_bounds)])),
        hovertemplate=(
            "<b>Horizon:</b> %{x}<br>"
            "<b>Predicted Stage:</b> %{customdata[0]}<br>"
            "<b>Attack Risk Probability:</b> <span style='color:#00E5FF; font-weight:700;'>%{y:.1%}</span><br>"
            "<b>Epistemic Variance:</b> %{customdata[1]} %{customdata[2]}<br>"
            "<extra></extra>"
        )
    ))

    # Reference Threshold Guidelines
    fig.add_hline(y=0.25, line_dash="dot", line_color="#64748B", line_width=1.2,
                  annotation_text="Caution (25%)", annotation_position="bottom right",
                  annotation_font=dict(color="#94A3B8", size=10))
    fig.add_hline(y=0.50, line_dash="dash", line_color="#FFB300", line_width=1.5,
                  annotation_text="Elevated (50%)", annotation_position="top right",
                  annotation_font=dict(color="#FFB300", size=10))
    fig.add_hline(y=0.75, line_dash="dot", line_color="#FF1744", line_width=1.5,
                  annotation_text="Critical (75%)", annotation_position="top right",
                  annotation_font=dict(color="#FF1744", size=10))

    # High-Visibility Pre-emptive Intervention Window Annotation
    fig.add_annotation(
        x="t+3 (3 min)",
        y=0.67,
        text="<b>⚠️ LATERAL ESCALATION: 67%</b><br>"
             "<span style='color: #FFD54F; font-weight: 700;'>Horizon t+3</span><br>"
             "<span style='color: #FFFFFF; font-size: 11px;'>Lead Time: ~3 min</span>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2.5,
        arrowcolor="#FFD54F",
        ax=-65,
        ay=-50,
        bordercolor="#FFD54F",
        borderwidth=2,
        borderpad=8,
        bgcolor="#0B111E",
        font=dict(color="#FFFFFF", size=12, family="'Plus Jakarta Sans', sans-serif")
    )

    fig.update_layout(
        title=dict(
            text="<b>Attack Risk Trajectory & Epistemic Uncertainty</b>",
            font=dict(size=15, color="#FFFFFF")
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        margin=dict(l=40, r=30, t=45, b=40),
        height=390,
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.05)",
            tickfont=dict(color="#FFFFFF", size=12, family="'Plus Jakarta Sans', sans-serif"),
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.05)",
            tickfont=dict(color="#94A3B8", size=12),
            range=[0, 1.05],
            tickformat=".0%",
            title=dict(text="Threat Risk", font=dict(color="#94A3B8", size=11)),
            showgrid=True,
        )
    )

    return fig


def render_forecast_hero(data):
    """Renders the hero trajectory chart alongside the predicted future stage card."""
    forecast = data["forecast"]
    attack = data["attack"]

    render_html("""
    <div class="card-title" style="margin-top: 6px; margin-bottom: 12px;">
        <span>Threat Trajectory Forecast</span>
        <span class="badge">4-Step Rollout</span>
    </div>
    """)

    col_chart, col_card = st.columns([2.5, 1.1])

    with col_chart:
        fig = create_forecast_chart(forecast)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_card:
        render_html(f"""
        <div class="glass-card" style="min-height: 390px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="metric-label">Predicted Attack</span>
                    <span style="background: rgba(255, 23, 68, 0.18); border: 1px solid #FF1744; color: #FF5252; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">
                        P = {attack['confidence']:.0%}
                    </span>
                </div>
                <div style="font-size: 1.22rem; font-weight: 800; color: #FFFFFF; line-height: 1.25;">
                    {attack['type']}
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">
                    Target: <code style="color: #00E5FF;">10.0.4.0/24 (Auth Cluster)</code>
                </div>

                <div style="margin: 16px 0; padding: 12px 14px; background: rgba(255, 82, 82, 0.12); border-radius: 12px; border-left: 3px solid #FF5252;">
                    <div class="metric-label" style="color: #FF5252;">Impending Stage</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #FFFFFF; margin-top: 2px;">
                        {attack['predicted_stage']}
                    </div>
                    <div style="font-size: 0.74rem; color: #CBD5E1; margin-top: 3px;">
                        Escalation at horizon <b>t+3</b> (±18% variance)
                    </div>
                </div>

                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <div>
                        <div class="metric-label">Lead Time</div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #00E5FF; font-family: 'JetBrains Mono', monospace;">
                            ~3 min
                        </div>
                    </div>
                    <div>
                        <div class="metric-label">Posture</div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #00E676; font-family: 'JetBrains Mono', monospace;">
                            Pre-Emptive
                        </div>
                    </div>
                </div>
            </div>

            <div style="font-size: 0.76rem; color: #CBD5E1; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 10px; line-height: 1.45;">
                🛡️ <b>Illustrative analyst guidance — not a system recommendation:</b> Evaluate rate-limiting port 22 & step-up MFA on subnet 10.0.4.0/24.
            </div>
        </div>
        """)
