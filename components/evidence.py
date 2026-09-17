"""
SHADOWCAT - Evidence Component
"""

import streamlit as st
import pandas as pd
from styles import COLORS, render_html


def render_evidence_table(data, limit=None, *args, **kwargs):
    """Renders the flagged network flows driving the forecast with clean executive formatting."""
    flows = data["flagged_flows"]
    if limit:
        flows = flows[:limit]

    render_html(f"""
    <div class="card-title" style="margin-top: 18px; margin-bottom: 8px;">
        <svg class="soc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect><rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect><line x1="6" y1="6" x2="6.01" y2="6"></line><line x1="6" y1="18" x2="6.01" y2="18"></line></svg>
        <span>Correlated Network Flow Evidence</span>
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
            "Volume": f"{f['bytes']:,} B",
        })

    df = pd.DataFrame(display_rows)

    def style_forensics(val):
        return "color: #E5484D; font-weight: 700; background-color: rgba(229, 72, 77, 0.12);"

    def style_risk(val):
        if val == "High":
            return "color: #E5484D; font-weight: 800;"
        elif val == "Medium":
            return "color: #E0982B; font-weight: 700;"
        return "color: #2FB872; font-weight: 600;"

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
            "Volume": st.column_config.TextColumn("Volume", width="small"),
        }
    )

    render_html("""
    <div class="section-caption">
        Correlated flows driving forensic attribution.
    </div>
    """)
