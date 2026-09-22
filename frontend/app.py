"""
SHADOWCAT - SOC Cockpit (Signal on Void)
Main Application Router & Shell Navigation
Exact 7-tab navigation matching Stitch export specification.
"""

import streamlit as st
from styles import apply_custom_css, TOKENS, EMBLEM_SVG, render_html

# Base page configuration
st.set_page_config(
    page_title="SHADOWCAT // SOC",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize theme
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# Apply design system tokens
apply_custom_css(theme=st.session_state.theme)
t = TOKENS[st.session_state.theme]

# Define 7 Authoritative App Pages (Overview first, then Telemetry Ingestion)
p1 = st.Page("views/02_Overview.py", title="Overview", url_path="overview", default=True)
p2 = st.Page("views/01_Telemetry_Ingestion.py", title="Telemetry Ingestion", url_path="ingestion")
p3 = st.Page("views/03_Forecast.py", title="Forecast", url_path="forecast")
p4 = st.Page("views/04_Attack_Graph.py", title="Attack Graph", url_path="attack-graph")
p5 = st.Page("views/05_Alerts.py", title="Alerts", url_path="alerts")
p6 = st.Page("views/06_Explainability.py", title="Explainability", url_path="explainability")
p7 = st.Page("views/07_Validation_Trust.py", title="Validation & Trust", url_path="validation-and-trust")

pages_list = [p1, p2, p3, p4, p5, p6, p7]
pg = st.navigation(pages_list, position="hidden")

# Render Global Persistent Top Header
header_col1, header_col2, header_col3 = st.columns([0.16, 0.67, 0.17])

with header_col1:
    render_html(f"""
    <div style="display: flex; align-items: center; gap: 0.6rem; padding-top: 4px;">
        {EMBLEM_SVG.strip()}
        <div style="display: flex; align-items: baseline; gap: 0.3rem;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.92rem; font-weight: 700; color: {t['text_high']}; letter-spacing: -0.01em;">
                SHADOWCAT
            </span>
            <span class="soc-topbar-tag" style="font-size: 0.6rem; padding: 1px 4px;">v2.4</span>
        </div>
    </div>
    """)

with header_col2:
    nav_cols = st.columns(7)
    nav_titles = [
        ("Overview", p1),
        ("Ingestion", p2),
        ("Forecast", p3),
        ("Attack Graph", p4),
        ("Alerts", p5),
        ("Explainability", p6),
        ("Validation & Trust", p7),
    ]
    for i, (title, page_obj) in enumerate(nav_titles):
        with nav_cols[i]:
            is_active = (pg == page_obj)
            if st.button(title, key=f"nav_{i}", type="primary" if is_active else "secondary", use_container_width=True):
                st.switch_page(page_obj)

with header_col3:
    col_status, col_btn = st.columns([0.72, 0.28])
    with col_status:
        render_html(f"""
        <div style="display: flex; align-items: center; justify-content: flex-end; padding-top: 5px;">
            <span class="soc-live-badge" style="font-size: 0.62rem; padding: 3px 8px; border: 1px solid {t['primary']}; background: {t['surface_card']};">
                <span class="soc-pulse-dot" style="width: 6px; height: 6px;"></span>
                AIRGAPPED // LIVE
            </span>
        </div>
        """)
    with col_btn:
        btn_label = "DARK" if st.session_state.theme == "dark" else "LIGHT"
        st.button(btn_label, on_click=toggle_theme, help="Toggle Light/Dark Theme", use_container_width=True)

render_html(f"<div style='border-bottom: 1px solid {t['border']}; margin-bottom: 1rem;'></div>")

# Run Selected Page
pg.run()
