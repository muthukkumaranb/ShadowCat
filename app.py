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

def _redirect_to_forecast():
    st.switch_page(forecast_page)

forecast_page = st.Page("views/01_Forecast.py", title="Threat Forecast", default=True, url_path="")
forecast_alias = st.Page(_redirect_to_forecast, title="Forecast Redirect", url_path="Forecast")

# Configure all 6 views with clean URL paths (and direct /Forecast deep-link support)
pages = [
    forecast_page,
    forecast_alias,
    st.Page("views/01b_AttackGraph.py", title="Lateral Movement Graph", url_path="AttackGraph"),
    st.Page("views/02_Evidence.py", title="Evidence & Attribution", url_path="Evidence"),
    st.Page("views/01a_Input.py", title="Telemetry Ingestion", url_path="Input"),
    st.Page("views/03_Validation.py", title="Validation & Benchmarks", url_path="Validation"),
    st.Page("views/05_About.py", title="Platform Specifications", url_path="About"),
]

# Set base page configuration with collapsed sidebar
st.set_page_config(
    page_title="SHADOWCAT — SOC Cockpit",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply global dark mode and full-width SOC styles
apply_custom_css()

# Load demo data
data = get_demo_data()

# Suppress Streamlit built-in sidebar navigation in favor of persistent horizontal top navigation bar
pg = st.navigation(pages, position="hidden")
pg.run()
