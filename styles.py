"""
SHADOWCAT - Neutral High-Contrast SOC Design System & Global Styling
Overhauled to match the reference SOC Posture dashboard design language:
  - True near-black neutral base (#0d0d0d to #141414)
  - Neutral dark gray/black cards (#141414) with thin #262626 borders
  - Pure white (#ffffff) primary text and muted neutral gray (#8a8a8a) secondary text
  - Restrained palette: white/gray for structure, sparse amber/green/red for semantic state
"""

import streamlit as st
import textwrap

# Color Palette - Neutral High-Contrast SOC Theme
COLORS = {
    "bg": "#0d0d0d",               # True near-black neutral base
    "surface": "#141414",          # Flat dark gray card
    "surface_hover": "#1a1a1a",    # Card hover
    "border": "#262626",           # Thin neutral gray border
    "border_hover": "#363636",     # Border hover
    "text_primary": "#ffffff",     # Pure white high contrast
    "text_secondary": "#8a8a8a",   # Muted neutral gray
    "text_muted": "#5a5a5a",       # Subtle neutral gray
    "accent": "#ffffff",           # Restrained structural accent: pure white
    "accent_dim": "rgba(255, 255, 255, 0.08)",
    "danger": "#e5484d",           # Status: Danger / Critical
    "danger_dim": "rgba(229, 72, 77, 0.12)",
    "warning": "#e0982b",          # Status: Warning / Caution
    "warning_dim": "rgba(224, 152, 43, 0.12)",
    "success": "#2fb872",          # Status: Success / Normal
    "success_dim": "rgba(47, 184, 114, 0.12)",

    # Backward compatibility aliases for view modules
    "safe": "#2fb872",
    "caution": "#e0982b",
    "critical": "#e5484d",
}

CUSTOM_CSS = f"""
<style>
    /* Offline / Air-Gapped Base Reset - System Font Stack */
    html, body, [class*="css"] {{
        background-color: #0d0d0d !important;
        color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }}

    /* Main Container Padding */
    .block-container {{
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1340px !important;
    }}

    /* Slim Persistent Top Bar Header */
    .slim-header-bar {{
        background: #141414 !important;
        border: 1px solid #262626 !important;
        border-radius: 8px !important;
        padding: 10px 18px !important;
        margin-bottom: 18px !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        box-shadow: none !important;
    }}

    .slim-header-brand {{
        font-size: 1.15rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        letter-spacing: 0.02em !important;
    }}

    /* Neutral SOC Cards */
    .glass-card, .cyber-card, .card {{
        background: #141414 !important;
        border: 1px solid #262626 !important;
        border-radius: 8px !important;
        padding: 18px 20px !important;
        margin-bottom: 18px !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
        position: relative !important;
    }}

    .glass-card:hover, .cyber-card:hover, .card:hover {{
        background: #181818 !important;
        border-color: #333333 !important;
        box-shadow: none !important;
        transform: none !important;
    }}

    /* Section Labels - Sentence Case, Secondary Gray, No Letter Spacing */
    .card-title {{
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #8a8a8a !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        margin-bottom: 12px !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
    }}

    /* Clean Neutral Badges */
    .badge, span.badge, .card-title span.badge {{
        font-size: 0.72rem !important;
        padding: 2px 8px !important;
        border-radius: 4px !important;
        background: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        color: #8a8a8a !important;
        font-weight: 500 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
    }}

    /* High-Impact Stat Display */
    .metric-value-huge {{
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.02em !important;
        line-height: 1.1 !important;
        margin: 4px 0 !important;
        color: #ffffff !important;
    }}

    .metric-label {{
        font-size: 0.74rem !important;
        font-weight: 500 !important;
        color: #8a8a8a !important;
        text-transform: none !important;
        letter-spacing: normal !important;
    }}

    /* Status Badges */
    .badge-offline {{
        background: rgba(255, 255, 255, 0.05) !important;
        color: #d4d4d4 !important;
        border: 1px solid #2a2a2a !important;
        padding: 4px 10px !important;
        border-radius: 4px !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
    }}

    .status-dot {{
        width: 6px !important;
        height: 6px !important;
        border-radius: 50% !important;
        background-color: #2fb872 !important;
        display: inline-block !important;
    }}

    /* MITRE Pipeline Cards */
    .mitre-pipeline {{
        display: grid !important;
        grid-template-columns: repeat(4, 1fr) !important;
        gap: 12px !important;
        margin: 12px 0 16px 0 !important;
    }}

    .mitre-card-observed {{
        background: #141414 !important;
        border: 1px solid #262626 !important;
        border-left: 3px solid #2fb872 !important;
        border-radius: 6px !important;
        padding: 14px 16px !important;
        min-height: 125px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }}

    .mitre-card-predicted {{
        background: #141414 !important;
        border: 1px solid #262626 !important;
        border-left: 3px solid #e0982b !important;
        border-radius: 6px !important;
        padding: 14px 16px !important;
        min-height: 125px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }}

    .mitre-card-downstream {{
        background: #111111 !important;
        border: 1px solid #222222 !important;
        border-radius: 6px !important;
        padding: 14px 16px !important;
        min-height: 125px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }}

    /* Streamlit Native Metric Card Overrides */
    [data-testid="stMetric"] {{
        background: #141414 !important;
        border: 1px solid #262626 !important;
        border-radius: 8px !important;
        padding: 16px 18px !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }}

    [data-testid="stMetric"]:hover {{
        background: #181818 !important;
        border-color: #333333 !important;
    }}

    [data-testid="stMetricValue"] {{
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.02em !important;
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 0.74rem !important;
        font-weight: 500 !important;
        color: #8a8a8a !important;
        text-transform: none !important;
        letter-spacing: normal !important;
    }}

    [data-testid="stMetricDelta"] {{
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }}

    /* Streamlit Interactive Radio & Pill Controls */
    div[data-testid="stRadio"] > div {{
        background: #111111 !important;
        border: 1px solid #262626 !important;
        border-radius: 6px !important;
        padding: 4px !important;
        gap: 6px !important;
    }}

    div[data-testid="stRadio"] label {{
        background: transparent !important;
        border-radius: 4px !important;
        padding: 6px 12px !important;
        color: #8a8a8a !important;
        font-size: 0.82rem !important;
        transition: all 0.15s ease !important;
    }}

    div[data-testid="stRadio"] label:hover {{
        background: #1a1a1a !important;
        color: #ffffff !important;
    }}

    /* Streamlit Buttons: Secondary Default vs Primary Accent */
    div[data-testid="stButton"] button, .stButton > button {{
        background: #181818 !important;
        border: 1px solid #2a2a2a !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        letter-spacing: normal !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }}

    div[data-testid="stButton"] button:hover, .stButton > button:hover {{
        background: #222222 !important;
        border-color: #383838 !important;
        color: #ffffff !important;
        transform: none !important;
    }}

    /* Primary CTA Button: High-Contrast Crisp Clean White on Dark */
    button[kind="primary"], div[data-testid="stButton"] button[kind="primary"] {{
        background: #ffffff !important;
        border: 1px solid #ffffff !important;
        color: #0d0d0d !important;
        font-weight: 700 !important;
        font-size: 0.86rem !important;
        border-radius: 6px !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease !important;
    }}

    button[kind="primary"]:hover, div[data-testid="stButton"] button[kind="primary"]:hover {{
        background: #e5e5e5 !important;
        border-color: #e5e5e5 !important;
        color: #0d0d0d !important;
        transform: none !important;
    }}

    /* Data Table Styling */
    [data-testid="stDataFrame"] {{
        border: 1px solid #262626 !important;
        border-radius: 6px !important;
        overflow: hidden !important;
        background: #141414 !important;
        box-shadow: none !important;
    }}

    /* Clean Neutral Sidebar */
    section[data-testid="stSidebar"] {{
        background: #0f0f0f !important;
        border-right: 1px solid #222222 !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background: #141414 !important;
        border: 1px solid #262626 !important;
        border-radius: 6px !important;
        margin-bottom: 12px !important;
        box-shadow: none !important;
    }}

    /* Code and Pre typography */
    code, pre {{
        font-family: 'JetBrains Mono', Consolas, monospace !important;
        background: #111111 !important;
        border: 1px solid #222222 !important;
        color: #8a8a8a !important;
        border-radius: 4px !important;
        font-size: 0.82rem !important;
    }}

    /* Header & Sidebar Collapse / Expand Control */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        color: #ffffff !important;
        pointer-events: auto !important;
    }}

    [data-testid="stToolbar"] {{
        background: transparent !important;
        visibility: visible !important;
        display: flex !important;
    }}

    /* Sidebar reopen/expand button */
    [data-testid="stExpandSidebarButton"],
    button[data-testid="stExpandSidebarButton"] {{
        visibility: visible !important;
        display: flex !important;
        color: #8a8a8a !important;
        background: #141414 !important;
        border: 1px solid #262626 !important;
        border-radius: 6px !important;
        box-shadow: none !important;
        cursor: pointer !important;
    }}

    [data-testid="stExpandSidebarButton"]:hover {{
        background: #1a1a1a !important;
        border-color: #333333 !important;
        color: #ffffff !important;
    }}

    [data-testid="stSidebarCollapseButton"] button {{
        visibility: visible !important;
        color: #8a8a8a !important;
        background: #141414 !important;
        border: 1px solid #262626 !important;
        border-radius: 6px !important;
    }}

    [data-testid="stSidebarCollapseButton"] button:hover {{
        background: #1a1a1a !important;
        border-color: #333333 !important;
        color: #ffffff !important;
    }}

    /* Sidebar Layout */
    section[data-testid="stSidebar"] {{
        position: relative !important;
    }}

    [data-testid="stSidebarContent"] {{
        display: flex !important;
        flex-direction: column !important;
        padding-top: 0 !important;
    }}

    [data-testid="stSidebarHeader"] {{
        position: absolute !important;
        top: 14px !important;
        right: 14px !important;
        z-index: 100 !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    [data-testid="stSidebarUserContent"] {{
        order: -1 !important;
        padding-top: 16px !important;
        padding-bottom: 0px !important;
        padding-left: 16px !important;
        padding-right: 16px !important;
    }}

    [data-testid="stSidebarNav"] {{
        padding-top: 4px !important;
        padding-bottom: 8px !important;
        padding-left: 8px !important;
        padding-right: 8px !important;
        margin-top: 0 !important;
    }}

    [data-testid="stSidebarNavSeparator"] {{
        display: none !important;
    }}

    [data-testid="stSidebarNavItems"] {{
        padding-top: 0 !important;
        margin-top: 0 !important;
    }}

    /* Sidebar Navigation Section Headers */
    [data-testid="stSidebarNavSectionHeader"],
    div[data-testid="stSidebarNav"] span[data-testid="stWidgetLabel"],
    div[data-testid="stSidebarNav"] h2 {{
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #5a5a5a !important;
        padding-top: 12px !important;
        padding-bottom: 2px !important;
        margin-top: 6px !important;
        border-top: 1px solid #222222 !important;
    }}

    /* Sidebar Links: Neutral Restrained Palette */
    div[data-testid="stSidebarNav"] a {{
        font-weight: 500 !important;
        color: #8a8a8a !important;
        border-left: 3px solid transparent !important;
        border-radius: 4px !important;
        padding: 6px 12px !important;
        transition: all 0.15s ease !important;
    }}

    div[data-testid="stSidebarNav"] a:hover {{
        background: #181818 !important;
        color: #ffffff !important;
    }}

    div[data-testid="stSidebarNav"] a[aria-current="page"] {{
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.08) !important;
        border-left: 3px solid #ffffff !important;
        font-weight: 700 !important;
    }}

    /* Hide ONLY unwanted Streamlit chrome */
    [data-testid="stAppDeployButton"] {{ visibility: hidden !important; display: none !important; }}
    [data-testid="stMainMenu"] {{ visibility: hidden !important; display: none !important; }}
    [data-testid="stMainMenuButton"] {{ visibility: hidden !important; display: none !important; }}
    #MainMenu {{ visibility: hidden !important; display: none !important; }}
    footer {{ visibility: hidden !important; display: none !important; }}
    [data-testid="stDecoration"] {{ display: none !important; }}
    [data-testid="stStatusWidget"] {{ visibility: hidden !important; display: none !important; }}
</style>
"""


def safe_html(html_str: str) -> str:
    """Strips leading/trailing whitespace preventing verbatim code blocks."""
    return "\n".join(line.strip() for line in html_str.splitlines() if line.strip())


def render_html(html_str: str):
    """Safely renders HTML in Streamlit with zero code-block triggers."""
    st.markdown(safe_html(html_str), unsafe_allow_html=True)


def apply_custom_css():
    """Inject the neutral high-contrast SOC CSS into Streamlit."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar(data=None):
    """Render clean executive sidebar header."""
    with st.sidebar:
        render_html("""
        <div style="padding-bottom: 8px; border-bottom: 1px solid #222222; margin-bottom: 8px; padding-right: 36px;">
            <div style="font-size: 1.25rem; font-weight: 800; letter-spacing: 0.02em; color: #ffffff;">
                SHADOWCAT
            </div>
        </div>
        <div style="margin-bottom: 8px;">
            <span class="badge-offline">
                <span class="status-dot"></span>
                Air-Gapped System
            </span>
        </div>
        """)


def render_footer():
    """Renders the persistent executive SHADOWCAT footer with single global disclaimer."""
    render_html("""
    <div style="margin-top: 36px; padding: 14px 18px; border-top: 1px solid #262626; display: flex; justify-content: space-between; align-items: center; font-size: 0.74rem; color: #5a5a5a; flex-wrap: wrap; gap: 8px;">
        <div>
            <b style="color: #8a8a8a;">SHADOWCAT v1.0</b> &nbsp;·&nbsp; Autonomous Cyber Threat Forecasting Engine &nbsp;·&nbsp; <span style="color: #666666; font-style: italic;">Illustrative analyst guidance — not a system recommendation.</span>
        </div>
        <div>
            <span style="color: #8a8a8a; font-weight: 600;">● Active (Air-Gapped)</span>
        </div>
    </div>
    """)
