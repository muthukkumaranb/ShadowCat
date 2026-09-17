"""
SHADOWCAT - Temporal Context & Sequence Attention (Past Windows)
Renders recent temporal windows of network state and sequence-encoder attention weights.

COMPLIANCE RULE:
Attention weights are strictly model internals, NOT validated causal attribution.
No causal phrasing ("drove", "caused", "explains prediction") is permitted.
Validated feature attribution is provided by deletion-tested methods (Integrated Gradients / Feature Ablation).
"""

import streamlit as st
from styles import render_html


def render_temporal_evidence(data: dict):
    """
    Renders chronological past network state windows alongside raw sequence encoder attention weights.
    Explicitly framed as model internals, visually separated from validated feature attribution.
    """
    history = data.get("history_windows", [
        {
            "window": "Window t-2",
            "time_range": "10:28:00 – 10:29:00",
            "behavior": "Nominal baseline density (routine HTTPS/DNS)",
            "flows": 8410,
            "packets": 51200,
            "attention_weight": 0.14,
            "weight_pct": "14%",
        },
        {
            "window": "Window t-1",
            "time_range": "10:29:00 – 10:30:00",
            "behavior": "Reconnaissance probing & sequential port scan",
            "flows": 9840,
            "packets": 64120,
            "attention_weight": 0.31,
            "weight_pct": "31%",
        },
        {
            "window": "Window t (Current)",
            "time_range": "10:30:00 – 10:31:00",
            "behavior": "Elevated SYN burst activity & SSH authentication attempts",
            "flows": 12480,
            "packets": 84216,
            "attention_weight": 0.55,
            "weight_pct": "55%",
        },
    ])

    render_html("""
    <div style="margin-top: 24px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
        <div class="card-title" style="margin: 0;">
            <span>Temporal Context & Window Attention</span>
            <span class="badge" style="background: #222222; color: #8A8A8A; border: 1px solid #333333; cursor: help;" title="Internal attention weights — exploratory model internals, not a certified explanation method. See feature attribution below for deletion-tested attribution.">
                [Model Internals · Exploratory]
            </span>
        </div>
        <span style="font-size: 0.72rem; color: #8A8A8A; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
            S(t-2) → S(t-1) → S(t)
        </span>
    </div>
    """)

    cols = st.columns(3)
    for col, win in zip(cols, history):
        is_current = "Current" in win["window"]
        border_color = "#FFFFFF" if is_current else "#262626"
        bg_card = "#181818" if is_current else "#141414"

        with col:
            render_html(f"""
            <div class="glass-card" style="border: 1px solid {border_color}; background: {bg_card}; padding: 14px 16px; min-height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 0.88rem; font-weight: 700; color: #FFFFFF;">{win['window']}</span>
                        <span style="font-size: 0.72rem; color: #8A8A8A; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">{win['time_range']}</span>
                    </div>
                    <div style="font-size: 0.78rem; color: #8A8A8A; line-height: 1.45; margin-bottom: 10px;">
                        {win['behavior']}
                    </div>
                </div>
                <div style="border-top: 1px solid #262626; padding-top: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span class="metric-label" style="margin: 0; font-size: 0.70rem;">Attention weight (internal)</span>
                        <span style="font-size: 0.82rem; font-weight: 700; color: #FFFFFF; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; letter-spacing: -0.02em;">
                            {win['weight_pct']}
                        </span>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.08); border-radius: 4px; height: 5px; width: 100%; overflow: hidden;">
                        <div style="background: #FFFFFF; height: 100%; width: {win['weight_pct']}; border-radius: 4px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 0.68rem; color: #8A8A8A;">
                        <span>Flows: {win['flows']:,}</span>
                        <span>Pkts: {win['packets']:,}</span>
                    </div>
                </div>
            </div>
            """)
