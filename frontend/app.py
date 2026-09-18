"""
SHADOWCAT - Autonomous Cyber Threat Forecasting Engine
Main Application Router & Entrypoint
"""

import streamlit as st
import importlib
import styles
importlib.reload(styles)
from styles import apply_custom_css, render_sidebar
from data_provider import get_demo_data

# Configure navigation hierarchy: Operations (Front Door) and Platform Specs & Audit
pages = {
    "Operations": [
        st.Page("views/01_Forecast.py", title="Threat Forecast", default=True, url_path="Forecast"),
        st.Page("views/01b_AttackGraph.py", title="Lateral Movement Graph", url_path="AttackGraph"),
        st.Page("views/02_Evidence.py", title="Evidence & Attribution", url_path="Evidence"),
        st.Page("views/01a_Input.py", title="Telemetry Ingestion", url_path="Input"),
        st.Page("views/03_Validation.py", title="Validation & Benchmarks", url_path="Validation"),
    ],
    "Platform Specs & Audit": [
        st.Page("views/05_About.py", title="Platform Specifications", url_path="About"),
    ],
}

# Set base page configuration
st.set_page_config(
    page_title="SHADOWCAT — SOC Cockpit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global dark mode styles
apply_custom_css()

# Load demo data
data = get_demo_data()

# Render global executive sidebar (brand, air-gapped badge, nav)
render_sidebar(data)

# Run navigation
pg = st.navigation(pages)
pg.run()
