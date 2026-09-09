"""
SHADOWCAT - Hero Forecast Component
Plotly Attack Trajectory with Epistemic Uncertainty Bands & Navy SOC Styling
"""

import streamlit as st
import plotly.graph_objects as go
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

    # 2. Upper Bound + Epistemic Uncertainty Shading
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=upper_bounds,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(56, 189, 248, 0.12)",
        hoverinfo="skip",
        name="Compounding Uncertainty (±σ)",
        showlegend=True
    ))

    # 3. Main Luminous Trajectory Spline (Sky Blue Accent)
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=probs,
        mode="lines+markers+text",
        name="Forecasted Risk P(Attack)",
        line=dict(color="#38BDF8", width=4.0, shape="spline", smoothing=0.85),
        marker=dict(
            size=[11, 13, 14, 18, 16],
            color=["#2FB872", "#38BDF8", "#E0982B", "#E5484D", "#E5484D"],
            line=dict(color="#FFFFFF", width=2.0)
        ),
        text=[f"<b>{p*100:.0f}%</b>" for p in probs],
        textposition="top left",
        textfont=dict(color="#E8EDF5", size=12, family="'JetBrains Mono', monospace"),
        customdata=list(zip(stages, [f"±{u*100:.0f}%" for u in uncertainties], [f"[{l*100:.0f}% – {u*100:.0f}%]" for l, u in zip(lower_bounds, upper_bounds)])),
        hovertemplate=(
            "<b>Horizon:</b> %{x}<br>"
            "<b>Predicted Stage:</b> %{customdata[0]}<br>"
            "<b>Attack Risk Probability:</b> <span style='color:#38BDF8; font-weight:700;'>%{y:.1%}</span><br>"
            "<b>Epistemic Variance:</b> %{customdata[1]} %{customdata[2]}<br>"
            "<extra></extra>"
        )
    ))

    # Reference Threshold Guidelines
    fig.add_hline(y=0.25, line_dash="dot", line_color="#64708A", line_width=1.0,
                  annotation_text="Caution (25%)", annotation_position="bottom right",
                  annotation_font=dict(color="#9AA7BD", size=9))
    fig.add_hline(y=0.50, line_dash="dash", line_color="#E0982B", line_width=1.2,
                  annotation_text="Elevated (50%)", annotation_position="top right",
                  annotation_font=dict(color="#E0982B", size=9))
    fig.add_hline(y=0.75, line_dash="dot", line_color="#E5484D", line_width=1.2,
                  annotation_text="Critical (75%)", annotation_position="top right",
                  annotation_font=dict(color="#E5484D", size=9))

    # High-Visibility Pre-emptive Intervention Window Annotation
    fig.add_annotation(
        x="t+3 (3 min)",
        y=0.67,
        text="<b>LATERAL ESCALATION: 67%</b><br>"
             "<span style='color: #38BDF8; font-weight: 700;'>Horizon t+3</span><br>"
             "<span style='color: #9AA7BD; font-size: 11px;'>Lead Time: ~3.5 min</span>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.0,
        arrowwidth=2.0,
        arrowcolor="#38BDF8",
        ax=-60,
        ay=-45,
        bordercolor="#22304A",
        borderwidth=1.5,
        borderpad=6,
        bgcolor="#182338",
        font=dict(color="#E8EDF5", size=11, family="'Plus Jakarta Sans', sans-serif")
    )

    fig.update_layout(
        paper_bgcolor="#131b2c",
        plot_bgcolor="#131b2c",
        margin=dict(l=35, r=25, t=30, b=35),
        height=360,
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(
            gridcolor="#22304a",
            zerolinecolor="#22304a",
            tickfont=dict(color="#E8EDF5", size=11, family="'Plus Jakarta Sans', sans-serif"),
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor="#22304a",
            zerolinecolor="#22304a",
            tickfont=dict(color="#9AA7BD", size=11),
            range=[0, 1.05],
            tickformat=".0%",
            title=dict(text="Threat Probability", font=dict(color="#9AA7BD", size=11)),
            showgrid=True,
        )
    )

    return fig
