"""
SHADOWCAT - Hero Forecast Component
Plotly Attack Trajectory with Epistemic Uncertainty Bands & Navy SOC Styling
"""

import streamlit as st
import plotly.graph_objects as go
from styles import COLORS, render_html


import math


def _clean_prob(val, default=0.0):
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f):
            return default
        return max(0.0, min(1.0, f))
    except (TypeError, ValueError):
        return default


def create_forecast_chart(forecast_data):
    """
    Creates an executive-grade Plotly chart showing the attack risk probability trajectory
    with growing epistemic uncertainty bands across forecast horizons K=1..4.
    """
    x_labels = ["Now (t)", "t+1 (1 min)", "t+2 (2 min)", "t+3 (3 min)", "t+4 (4 min)"]
    probs = [0.06] + [_clean_prob(step.get("probability")) for step in forecast_data]
    lower_bounds = [0.06] + [_clean_prob(step.get("lower_bound")) for step in forecast_data]
    upper_bounds = [0.06] + [_clean_prob(step.get("upper_bound"), default=1.0) for step in forecast_data]
    uncertainties = [0.00] + [_clean_prob(step.get("uncertainty")) for step in forecast_data]
    stages = ["Observation Window"] + [str(step.get("stage", "Unknown")) for step in forecast_data]

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
        fillcolor="rgba(255, 255, 255, 0.06)",
        hoverinfo="skip",
        name="Compounding Uncertainty (±σ)",
        showlegend=False
    ))

    lead_times = ["Baseline"] + [str(step.get("lead_time", "—")) for step in forecast_data]
    custom_tuples = list(zip(
        stages,
        [f"±{u*100:.0f}%" for u in uncertainties],
        [f"[{l*100:.0f}% – {u*100:.0f}%]" for l, u in zip(lower_bounds, upper_bounds)],
        lead_times
    ))

    # 3. Main Luminous Trajectory Spline (Clean High-Contrast White)
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=probs,
        mode="lines+markers+text",
        name="Forecasted Risk P(Attack)",
        showlegend=False,
        line=dict(color="#ffffff", width=3.5, shape="spline", smoothing=0.85),
        marker=dict(
            size=[11, 13, 14, 18, 16],
            color=["#2FB872", "#e0e0e0", "#E0982B", "#E5484D", "#E5484D"],
            line=dict(color="#0d0d0d", width=2.0)
        ),
        text=[f"<b>{p*100:.0f}%</b>" for p in probs],
        textposition="top left",
        textfont=dict(color="#ffffff", size=12, family="'JetBrains Mono', monospace"),
        customdata=custom_tuples,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "<b>Predicted Stage:</b> %{customdata[0]}<br>"
            "<b>Attack Risk Probability:</b> <span style='color:#ffffff; font-weight:700;'>%{y:.1%}</span><br>"
            "<b>Epistemic Bounds:</b> %{customdata[1]} %{customdata[2]}<br>"
            "<b>Pre-Emptive Lead Time:</b> %{customdata[3]}<extra></extra>"
        )
    ))

    # Reference Threshold Guidelines
    fig.add_hline(y=0.25, line_dash="dot", line_color="#333333", line_width=1.0,
                  annotation_text="Caution (25%)", annotation_position="bottom right",
                  annotation_font=dict(color="#8a8a8a", size=9))
    fig.add_hline(y=0.50, line_dash="dash", line_color="#E0982B", line_width=1.2,
                  annotation_text="Elevated (50%)", annotation_position="top right",
                  annotation_font=dict(color="#E0982B", size=9))
    fig.add_hline(y=0.75, line_dash="dot", line_color="#E5484D", line_width=1.2,
                  annotation_text="Critical (75%)", annotation_position="top right",
                  annotation_font=dict(color="#E5484D", size=9))

    fig.update_layout(
        paper_bgcolor="#141414",
        plot_bgcolor="#141414",
        margin=dict(l=35, r=25, t=30, b=35),
        height=360,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#181818",
            bordercolor="#333333",
            font=dict(color="#ffffff", size=11, family="'JetBrains Mono', monospace")
        ),
        showlegend=False,
        xaxis=dict(
            gridcolor="#222222",
            zerolinecolor="#222222",
            tickfont=dict(color="#ffffff", size=11, family="'Plus Jakarta Sans', sans-serif"),
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor="#222222",
            zerolinecolor="#222222",
            tickfont=dict(color="#8a8a8a", size=11),
            range=[0, 1.05],
            tickformat=".0%",
            title=dict(text="Threat Probability", font=dict(color="#8a8a8a", size=11)),
            showgrid=True,
        )
    )

    return fig
