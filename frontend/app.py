"""
SHADOWCAT - SOC Cockpit (Signal on Void)
Main Application Router & Entrypoint
"""

import streamlit as st
import importlib
import premium_styles as styles
importlib.reload(styles)
from premium_styles import apply_custom_css

# Set base page configuration
st.set_page_config(
    page_title="SHADOWCAT — SOC Cockpit",
    layout="wide",
    initial_sidebar_state="collapsed" # Using top nav primarily
)

# Initialize theme in session state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    if st.session_state.theme == "dark":
        st.session_state.theme = "light"
    else:
        st.session_state.theme = "dark"

# Apply global dark/light mode styles
apply_custom_css(theme=st.session_state.theme)

# Render Global Header and Theme Toggle
col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.markdown("<h3 style='margin: 0; padding-top: 10px;'>SHADOWCAT SOC Cockpit</h3>", unsafe_allow_html=True)
with col2:
    st.button("Toggle Light/Dark", on_click=toggle_theme, use_container_width=True)

st.markdown("<hr style='margin-top: 5px; margin-bottom: 15px;'/>", unsafe_allow_html=True)

# Configure navigation hierarchy for the 6 pages
pages = {
    "Dashboards": [
        st.Page("views/01_Overview.py", title="1. Overview", default=True, url_path="Overview"),
        st.Page("views/02_Forecast.py", title="2. Forecast Panel", url_path="Forecast"),
        st.Page("views/03_Attack_Graph.py", title="3. Attack Graph", url_path="Attack_Graph"),
        st.Page("views/04_Alerts.py", title="4. Alerts", url_path="Alerts"),
        st.Page("views/05_Explainability.py", title="5. Explainability", url_path="Explainability"),
        st.Page("views/06_Validation.py", title="6. Validation & Trust", url_path="Validation"),
    ]
}

# Run navigation
pg = st.navigation(pages)
pg.run()
