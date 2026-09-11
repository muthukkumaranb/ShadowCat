"""
SHADOWCAT - Dynamic Enterprise Attack Graph & Lateral Rollout
Multi-Step Forward Adversary Simulation Across Enterprise Topology with Compounding Epistemic Uncertainty
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from data_provider import get_analysis_metadata
import importlib
import styles
importlib.reload(styles)
from styles import apply_custom_css, render_sidebar, render_html, render_footer
import components.header
importlib.reload(components.header)
from components.header import render_header
import components.attack_graph
importlib.reload(components.attack_graph)
from components.attack_graph import render_attack_graph_panel

st.set_page_config(
    page_title="SHADOWCAT — Lateral Movement Graph",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Retrieve typed telemetry data
analysis = get_analysis_metadata()

# Sticky Persistent Header
render_header({"analysis": analysis})

# Flagship Page Title & Executive Subhead
render_html("""
<div style="margin-bottom: 18px;">
    <h2 style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF; margin: 0;">
        Lateral Movement Attack Graph
    </h2>
    <div style="font-size: 0.84rem; color: #8A8A8A; margin-top: 4px;">
        Multi-step forward simulation of lateral adversary rollout across enterprise topology with compounding epistemic uncertainty.
    </div>
</div>
""")

# Render the flagship attack graph panel with expanded canvas and host leaderboard
render_attack_graph_panel()

# Persistent Executive Footer
render_footer()
