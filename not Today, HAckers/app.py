"""
SHADOWCAT - Autonomous Cyber Threat Forecasting Engine
Main Application Router & Entrypoint
"""

import streamlit as st
import importlib
import styles
importlib.reload(styles)
from styles import apply_custom_css, render_sidebar
from mock_data import get_demo_data

# Configure navigation hierarchy: Forecast is default landing page (Task 1 & Task 8)
pages = [
    st.Page("views/01_Forecast.py", title="Forecast", icon="🎯", default=True, url_path="Forecast"),
    st.Page("views/02_Evidence.py", title="Evidence", icon="🔍", url_path="Evidence"),
    st.Page("views/03_Validation.py", title="Validation", icon="📊", url_path="Validation"),
    st.Page("views/04_Model_Info.py", title="Model Info", icon="🧠", url_path="Model_Info"),
    st.Page("views/05_About.py", title="About", icon="ℹ️", url_path="About"),
]

# Set base page configuration
st.set_page_config(
    page_title="SHADOWCAT",
    page_icon="🛡️",
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

