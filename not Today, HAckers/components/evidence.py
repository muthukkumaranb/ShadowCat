"""
SHADOWCAT - Evidence Component
"""

import streamlit as st
import pandas as pd
from styles import COLORS, render_html


def render_evidence_table(data, limit=None, mock_badge_html="", *args, **kwargs):
    """Renders the flagged network flows driving the forecast with clean executive formatting."""
    if not mock_badge_html and "mock_badge_html" in kwargs:
        mock_badge_html = kwargs["mock_badge_html"]
    flows = data["flagged_flows"]
    if limit:
        flows = flows[:limit]

    badge_str = f" {mock_badge_html}" if mock_badge_html else ""
    render_html(f"""
    <div class="card-title" style="margin-top: 20px; margin-bottom: 12px;">
        <span>Correlated Network Flow Evidence{badge_str}</span>
        <span class="badge">{len(flows)} High-Risk Flows</span>
    </div>
    """)

    display_rows = []
    for f in flows:
        display_rows.append({
            "Flow ID": f["id"],
            "Source Socket": f"{f['source']}:{f['sport']}",
            "Target Socket": f"{f['destination']}:{f['dport']}",
            "Proto": f["protocol"],
            "Forensic Indicator": f["reason"],
            "Risk Tier": f["risk"],
            "Packets": f"{f['pkts']:,}",
            "Volume": f"{f['bytes']:,} B",
        })

    df = pd.DataFrame(display_rows)

    def style_forensics(val):
        return "color: #FF8A80; font-weight: 700; background-color: rgba(255, 23, 68, 0.12);"

    def style_risk(val):
        if val == "High":
            return "color: #FF5252; font-weight: 800;"
        elif val == "Medium":
            return "color: #FFD54F; font-weight: 700;"
        return "color: #00E676; font-weight: 600;"

    styled_df = (
        df.style
        .map(style_forensics, subset=["Forensic Indicator"])
        .map(style_risk, subset=["Risk Tier"])
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Flow ID": st.column_config.TextColumn("Flow ID", width="small"),
            "Source Socket": st.column_config.TextColumn("Source Socket", width="medium"),
            "Target Socket": st.column_config.TextColumn("Target Socket", width="medium"),
            "Proto": st.column_config.TextColumn("Proto", width="small"),
            "Forensic Indicator": st.column_config.TextColumn("Indicator", width="large"),
            "Risk Tier": st.column_config.TextColumn("Risk", width="small"),
            "Packets": st.column_config.TextColumn("Pkts", width="small"),
            "Volume": st.column_config.TextColumn("Volume", width="small"),
        }
    )

    render_html("""
    <div style="font-size: 0.76rem; color: #64748B; margin-top: 6px; margin-bottom: 14px;">
        Flagged flows correlate with authentication spikes and connection resets on port 22.
    </div>
    """)
