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
    initial_sidebar_state="collapsed"
)

apply_custom_css()

# Retrieve typed telemetry data
analysis = get_analysis_metadata()

# Sticky Persistent Top Navigation and Header
render_header({"analysis": analysis}, active_tab="Lateral Movement Graph")

# Flagship Page Title & Executive Subhead
render_html("""
<div style="margin-bottom: 14px;">
    <h2 class="page-title">
        Lateral Movement Attack Graph
    </h2>
    <div class="page-caption">
        Forward simulation of lateral adversary rollout across enterprise topology.
    </div>
</div>
""")

# Render the flagship attack graph panel with expanded canvas and host leaderboard
render_attack_graph_panel()

# Persistent Executive Footer
render_footer()
