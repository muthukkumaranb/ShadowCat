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

# Color Palette - Neutral High-Contrast SOC Theme with Electric Cyan Accent
COLORS = {
    "bg": "#0d0d0d",               # True near-black neutral base
    "surface": "#171717",          # Elevated dark gray card (Depth pass)
    "surface_hover": "#1c1c1c",    # Card hover
    "border": "#282828",           # Thin neutral gray border
    "border_hover": "#383838",     # Border hover
    "text_primary": "#ffffff",     # Pure white high contrast
    "text_secondary": "#8a8a8a",   # Muted neutral gray
    "text_muted": "#5a5a5a",       # Subtle neutral gray
    "accent": "#38bdf8",           # Muted Sky-Cyan accent (70-80% saturation, restrained)
    "accent_dim": "rgba(56, 189, 248, 0.12)",
    "cyan": "#38bdf8",
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

# -- Font families (system-safe for air-gapped deployment) --
_FONT_MONO = "'JetBrains Mono', 'Cascadia Code', 'Consolas', 'Courier New', monospace"
_FONT_GEO_SANS = "'Inter', 'Segoe UI Variable', 'Segoe UI', system-ui, -apple-system, sans-serif"
_FONT_SYSTEM = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

CUSTOM_CSS = """
<style>
    /* Offline / Air-Gapped Base Reset - System Font Stack & True Near-Black Base */
    html, body, [class*="css"], [data-testid="stAppViewContainer"], .stApp, section.main, [data-testid="stHeader"] {
        background-color: #0d0d0d !important;
        background: #0d0d0d !important;
        color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }

    /* Target all Streamlit view containers to eliminate any blue tint and prevent header click interception */
    .stApp > header,
    header[data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background: transparent !important;
        pointer-events: none !important;
        z-index: 1 !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #0d0d0d !important;
    }

    .main .block-container {
        background-color: #0d0d0d !important;
    }

    /* Links: Crisp Neutral White (strictly exclude top nav items) */
    a:not(.top-nav-tab):not(.top-nav-brand),
    [data-testid="stMarkdownContainer"] a:not(.top-nav-tab):not(.top-nav-brand) {
        color: #ffffff !important;
        text-decoration: underline !important;
    }

    a:not(.top-nav-tab):not(.top-nav-brand):hover,
    [data-testid="stMarkdownContainer"] a:not(.top-nav-tab):not(.top-nav-brand):hover {
        color: #d4d4d4 !important;
    }

    /* Eliminate Streamlit default header to remove the gap at the top */
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    .stApp > header {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Main Container Full-Width Reclaim (Sidebar eliminated, clearance for fixed top header) */
    .block-container,
    .main .block-container,
    [data-testid="stMainBlockContainer"],
    div[data-testid="stMainBlockContainer"],
    section.main > div.block-container {
        padding-top: 84px !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    /* Unified Fixed Top Header Group (Navigation + Status Banner with 0px gap) */
    .shadowcat-fixed-header {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        z-index: 999999 !important;
        pointer-events: auto !important;
        margin: 0 !important;
        padding: 0 !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* Persistent Horizontal Top Navigation Bar */
    .top-nav-bar {
        position: relative !important;
        top: auto !important;
        left: auto !important;
        right: auto !important;
        width: 100% !important;
        height: 48px !important;
        pointer-events: auto !important;
        background: #181a1f !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid #282c34 !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 1.5rem !important;
        margin: 0 !important;
        box-sizing: border-box !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
    }

    /* Pinned Status Banner directly under nav bar with zero gap */
    .top-banner-bar {
        position: relative !important;
        width: 100% !important;
        min-height: 28px !important;
        height: 28px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        padding: 0 1.5rem !important;
        margin: 0 !important;
        box-sizing: border-box !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    }

    .top-banner-bar.banner-live {
        background: rgba(12, 20, 16, 0.97) !important;
        border-bottom: 1px solid rgba(47, 184, 114, 0.35) !important;
    }

    .top-banner-bar.banner-benchmark {
        background: rgba(22, 17, 12, 0.97) !important;
        border-bottom: 1px solid rgba(224, 152, 43, 0.35) !important;
    }

    .top-nav-left {
        display: flex;
        align-items: center;
        gap: 14px;
        height: 100%;
        min-width: 0;
        flex: 1;
        overflow: hidden;
    }

    .top-nav-brand {
        display: flex;
        align-items: center;
        gap: 8px;
        text-decoration: none !important;
        white-space: nowrap;
        flex-shrink: 0;
    }

    .top-nav-brand-text {
        font-size: 1.15rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        color: #FFFFFF !important;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    }

    .top-nav-brand-pill {
        font-size: 0.65rem;
        font-weight: 700;
        color: #9da3af;
        background: #2b2f38;
        border: 1px solid #3c414d;
        border-radius: 3px;
        padding: 2px 6px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .top-nav-tabs-wrapper {
        display: flex;
        align-items: center;
        height: 100%;
        position: relative;
        overflow-x: auto;
        scrollbar-width: none;
        min-width: 0;
        flex: 1;
    }
    .top-nav-tabs-wrapper::-webkit-scrollbar {
        display: none;
    }

    .top-nav-tabs {
        display: flex;
        align-items: center;
        gap: 1px;
        height: 100%;
        white-space: nowrap;
        flex-shrink: 0;
    }

    /* Top Nav Link Specificity Override (Eliminates link underlines on inactive tabs) */
    .stApp [data-testid="stMarkdownContainer"] .top-nav-bar a,
    [data-testid="stMarkdownContainer"] .top-nav-bar a,
    .top-nav-bar a,
    .stApp [data-testid="stMarkdownContainer"] .top-nav-tab,
    [data-testid="stMarkdownContainer"] .top-nav-tab,
    .top-nav-tab,
    .top-nav-brand {
        text-decoration: none !important;
        text-decoration-line: none !important;
    }

    .top-nav-tab {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        height: 48px;
        padding: 0 7px;
        font-size: 0.71rem;
        font-weight: 500;
        letter-spacing: 0.03em;
        font-family: 'Inter', 'Segoe UI Variable', 'Segoe UI', system-ui, -apple-system, sans-serif;
        line-height: 1;
        color: #9da3af !important;
        text-decoration: none !important;
        text-decoration-line: none !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.15s ease;
        white-space: nowrap;
        cursor: pointer;
    }

    .top-nav-tab svg.soc-icon {
        width: 14px;
        height: 14px;
        flex-shrink: 0;
        display: inline-block;
        vertical-align: middle;
        position: relative;
        top: -0.5px;
    }

    .stApp [data-testid="stMarkdownContainer"] .top-nav-bar a:hover,
    [data-testid="stMarkdownContainer"] .top-nav-bar a:hover,
    .top-nav-tab:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.06);
        text-decoration: none !important;
        text-decoration-line: none !important;
    }

    /* Active Tab: Box-model border-bottom in Muted Cyan (zero text-decoration underline) */
    .stApp [data-testid="stMarkdownContainer"] .top-nav-tab.active,
    [data-testid="stMarkdownContainer"] .top-nav-tab.active,
    .top-nav-tab.active {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-decoration: none !important;
        text-decoration-line: none !important;
        border-bottom: 2px solid #38bdf8 !important;
        background: transparent !important;
    }

    .top-nav-right {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-shrink: 0;
        margin-left: 10px;
    }

    .top-nav-utility-item {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        font-size: 0.70rem;
        color: #9da3af;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
    }

    .badge-air-gapped {
        display: inline-flex !important;
        align-items: center !important;
        gap: 5px !important;
        background: rgba(47, 184, 114, 0.15) !important;
        border: 1px solid rgba(47, 184, 114, 0.45) !important;
        border-radius: 4px !important;
        padding: 2px 8px !important;
        color: #2FB872 !important;
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        font-family: 'Consolas', 'Courier New', monospace !important;
    }

    .badge-air-gapped-dot {
        width: 6px !important;
        height: 6px !important;
        border-radius: 50% !important;
        background: #2FB872 !important;
        display: inline-block !important;
        flex-shrink: 0 !important;
        box-shadow: 0 0 5px rgba(47, 184, 114, 0.7) !important;
    }

    /* Clean Unboxed Status Header Bar (below top nav - zero heavy border/background) */
    .slim-header-bar {
        background: transparent !important;
        border: none !important;
        border-bottom: 1px solid #1a1a1a !important;
        border-radius: 0 !important;
        padding: 4px 0 !important;
        margin-top: 2px !important;
        margin-bottom: 8px !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        box-shadow: none !important;
    }

    /* Neutral Elevated SOC Cards (Card Depth & Contrast Pass) */
    .glass-card, .cyber-card, .card {
        background: #171717 !important;
        border: 1px solid #282828 !important;
        border-radius: 6px !important;
        padding: 16px 18px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
        transition: background-color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease !important;
        position: relative !important;
    }

    .glass-card:hover, .cyber-card:hover, .card:hover {
        background: #1c1c1c !important;
        border-color: #383838 !important;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.50) !important;
        transform: none !important;
    }

    /* Typographic Hierarchy Scale */
    .page-title {
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        letter-spacing: -0.02em !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }

    .page-caption {
        font-size: 0.78rem !important;
        color: #8a8a8a !important;
        margin-top: 3px !important;
        margin-bottom: 12px !important;
        line-height: 1.4 !important;
    }

    .section-title {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.01em !important;
        margin-top: 8px !important;
        margin-bottom: 4px !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }

    .section-caption {
        font-size: 0.74rem !important;
        color: #8a8a8a !important;
        margin-top: 2px !important;
        margin-bottom: 10px !important;
        line-height: 1.35 !important;
    }

    /* Section Labels - Sentence Case, Secondary Gray */
    .card-title {
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        color: #ededed !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        margin-bottom: 8px !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }

    .card-body-text {
        font-size: 0.76rem !important;
        color: #8a8a8a !important;
        line-height: 1.45 !important;
    }

    /* Clean Neutral Badges */
    .badge, span.badge, .card-title span.badge {
        font-size: 0.70rem !important;
        padding: 2px 7px !important;
        border-radius: 4px !important;
        background: #202020 !important;
        border: 1px solid #303030 !important;
        color: #a0a0a0 !important;
        font-weight: 600 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
    }

    /* High-Impact Stat Display */
    .metric-value-huge {
        font-size: 1.95rem !important;
        font-weight: 800 !important;
        font-family: 'JetBrains Mono', 'Cascadia Code', 'Consolas', 'Courier New', monospace !important;
        letter-spacing: -0.02em !important;
        line-height: 1.15 !important;
        margin: 4px 0 !important;
        color: #ffffff !important;
    }

    .metric-label {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: #8a8a8a !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
    }

    /* Status Badges */
    .badge-offline {
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
    }

    .status-dot {
        width: 6px !important;
        height: 6px !important;
        border-radius: 50% !important;
        background-color: #2fb872 !important;
        display: inline-block !important;
    }

    /* MITRE Pipeline Cards */
    .mitre-pipeline {
        display: grid !important;
        grid-template-columns: repeat(4, 1fr) !important;
        gap: 12px !important;
        margin: 12px 0 16px 0 !important;
    }

    .mitre-card-observed {
        background: #141414 !important;
        border: 1px solid #262626 !important;
        border-left: 3px solid #2fb872 !important;
        border-radius: 6px !important;
        padding: 14px 16px !important;
        min-height: 125px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }

    .mitre-card-predicted {
        background: #141414 !important;
        border: 1px solid #262626 !important;
        border-left: 3px solid #e0982b !important;
        border-radius: 6px !important;
        padding: 14px 16px !important;
        min-height: 125px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }

    .mitre-card-downstream {
        background: #111111 !important;
        border: 1px solid #222222 !important;
        border-radius: 6px !important;
        padding: 14px 16px !important;
        min-height: 125px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }

    /* Streamlit Native Metric Card Overrides */
    [data-testid="stMetric"] {
        background: #141414 !important;
        border: 1px solid #262626 !important;
        border-radius: 8px !important;
        padding: 16px 18px !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }

    [data-testid="stMetric"]:hover {
        background: #181818 !important;
        border-color: #333333 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        letter-spacing: -0.03em !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.74rem !important;
        font-weight: 500 !important;
        color: #8a8a8a !important;
        text-transform: none !important;
        letter-spacing: normal !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }

    /* Streamlit Interactive Radio & Pill Controls */
    div[data-testid="stRadio"] > div {
        background: #111111 !important;
        border: 1px solid #262626 !important;
        border-radius: 6px !important;
        padding: 4px !important;
        gap: 6px !important;
    }

    div[data-testid="stRadio"] label {
        background: transparent !important;
        border-radius: 4px !important;
        padding: 6px 12px !important;
        color: #8a8a8a !important;
        font-size: 0.82rem !important;
        transition: all 0.15s ease !important;
    }

    div[data-testid="stRadio"] label:hover {
        background: #1a1a1a !important;
        color: #ffffff !important;
    }

    /* Radio button active dot / selection: Pure White (No Cyan) */
    div[data-testid="stRadio"] input[type="radio"] {
        accent-color: #ffffff !important;
    }

    div[data-testid="stRadio"] [role="radiogroup"] label div[data-checked="true"],
    div[data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"] div {
        border-color: #ffffff !important;
    }

    div[data-testid="stRadio"] svg {
        fill: #ffffff !important;
    }

    /* Streamlit Slider: Crisp White Thumb & Neutral Gray Track (No Cyan) */
    div[data-testid="stSlider"] [data-baseweb="slider"] {
        accent-color: #ffffff !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] > div {
        background: #262626 !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background-color: #ffffff !important;
        border: 2px solid #141414 !important;
        box-shadow: 0 0 6px rgba(255, 255, 255, 0.4) !important;
    }

    div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
        color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        font-weight: 700 !important;
    }

    /* Streamlit Tabs: Neutral Underline (No Cyan) */
    button[data-baseweb="tab"] {
        color: #8a8a8a !important;
        background: transparent !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        border-bottom-color: #ffffff !important;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #ffffff !important;
    }

    /* Filter Toolbar Inputs & Selectboxes (Neutral SOC Styling) */
    div[data-testid="stTextInput"] input,
    div[data-baseweb="input"] input {
        background-color: #171717 !important;
        border: 1px solid #282828 !important;
        border-radius: 6px !important;
        color: #FFFFFF !important;
        font-size: 0.78rem !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        height: 36px !important;
        padding: 0 10px !important;
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-baseweb="input"] input:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.25) !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #171717 !important;
        border: 1px solid #282828 !important;
        border-radius: 6px !important;
        color: #FFFFFF !important;
        font-size: 0.78rem !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        min-height: 36px !important;
        height: 36px !important;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: #383838 !important;
    }

    div[data-baseweb="popover"] ul {
        background-color: #171717 !important;
        border: 1px solid #282828 !important;
    }

    div[data-baseweb="popover"] li {
        color: #d4d4d4 !important;
        font-size: 0.78rem !important;
    }

    div[data-baseweb="popover"] li:hover {
        background-color: #222222 !important;
        color: #FFFFFF !important;
    }

    /* Streamlit Buttons: Base & Secondary Default */
    div[data-testid="stButton"] button, .stButton > button {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        font-size: 0.84rem !important;
        letter-spacing: normal !important;
        border-radius: 6px !important;
        padding: 6px 14px !important;
        cursor: pointer !important;
        pointer-events: auto !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }

    /* Secondary Action Button (Focus / Unfocus): Muted Violet/Indigo Accent (#8B8FE8) */
    button[kind="secondary"],
    div[data-testid="stButton"] button[kind="secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background: transparent !important;
        border: 1.5px solid #8B8FE8 !important;
        color: #8B8FE8 !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        pointer-events: auto !important;
    }

    button[kind="secondary"] p,
    div[data-testid="stButton"] button[kind="secondary"] p,
    button[data-testid="stBaseButton-secondary"] p {
        color: #8B8FE8 !important;
        font-weight: 600 !important;
    }

    button[kind="secondary"]:hover,
    div[data-testid="stButton"] button[kind="secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background: rgba(139, 143, 232, 0.10) !important;
        border-color: #A5A9F8 !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 8px rgba(139, 143, 232, 0.20) !important;
        transform: none !important;
    }

    button[kind="secondary"]:hover p,
    div[data-testid="stButton"] button[kind="secondary"]:hover p,
    button[data-testid="stBaseButton-secondary"]:hover p {
        color: #FFFFFF !important;
    }

    button[kind="secondary"]:active,
    div[data-testid="stButton"] button[kind="secondary"]:active {
        background: rgba(139, 143, 232, 0.22) !important;
        border-color: #8B8FE8 !important;
        color: #FFFFFF !important;
    }

    /* Primary CTA Button (Inspect / Telemetry): Restrained Muted Cyan Treatment */
    button[kind="primary"],
    div[data-testid="stButton"] button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background: #38bdf8 !important;
        border: 1px solid #38bdf8 !important;
        color: #0b1320 !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35) !important;
        cursor: pointer !important;
        pointer-events: auto !important;
        transition: all 0.15s ease !important;
    }

    button[kind="primary"] p,
    div[data-testid="stButton"] button[kind="primary"] p,
    button[data-testid="stBaseButton-primary"] p {
        color: #0b1320 !important;
        font-weight: 700 !important;
    }

    button[kind="primary"]:hover,
    div[data-testid="stButton"] button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background: #0ea5e9 !important;
        border-color: #0ea5e9 !important;
        color: #0b1320 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.45) !important;
        transform: none !important;
    }

    /* Standardized Cross-Page Action Links */
    .soc-action-link {
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        color: #38bdf8 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-decoration: none !important;
        padding: 6px 14px !important;
        background: rgba(56, 189, 248, 0.08) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 6px !important;
        transition: all 0.15s ease !important;
    }
    .soc-action-link:hover {
        background: rgba(56, 189, 248, 0.16) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        text-decoration: none !important;
    }

    /* Custom SOC HTML Table System (Zero Truncation, Distinct Tinted Header) */
    .soc-table-wrapper {
        background: #171717;
        border: 1px solid #282828;
        border-radius: 6px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
        overflow: hidden;
        margin-bottom: 16px;
    }
    .soc-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.80rem;
        color: #ffffff;
        text-align: left;
    }
    .soc-table th {
        background: #202020;
        color: #d4d4d8;
        font-weight: 600;
        font-size: 0.72rem;
        letter-spacing: 0.02em;
        padding: 10px 14px;
        border-bottom: 1px solid #333333;
        white-space: nowrap;
    }
    .soc-table td {
        padding: 11px 14px;
        border-bottom: 1px solid #242424;
        vertical-align: middle;
        line-height: 1.4;
    }
    .soc-table tr:last-child td {
        border-bottom: none;
    }
    .soc-table tr:hover td {
        background: #1d1d1d;
    }
    .soc-num {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
        text-align: right;
    }
    .soc-mono {
        font-family: 'JetBrains Mono', Consolas, monospace;
        font-size: 0.76rem;
    }

    /* Inline Vector SVG Icons (Zero Emoji, Clean Crisp Display) */
    .soc-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        vertical-align: middle;
        width: 14px;
        height: 14px;
        flex-shrink: 0;
    }

    /* Data Table Styling - Neutral Dark Surface with Hairline Borders */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    [data-testid="stTable"],
    table,
    .dataframe {
        border: 1px solid #282828 !important;
        border-radius: 6px !important;
        overflow: hidden !important;
        background: #171717 !important;
        background-color: #171717 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45) !important;
        color: #ffffff !important;
    }

    table th, .dataframe th {
        background-color: #181818 !important;
        color: #ffffff !important;
        border-bottom: 1px solid #262626 !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        padding: 8px 12px !important;
    }

    table td, .dataframe td {
        background-color: #141414 !important;
        color: #e4e4e7 !important;
        border-bottom: 1px solid #1f1f1f !important;
        font-size: 0.82rem !important;
        padding: 8px 12px !important;
    }

    table tr:hover td, .dataframe tr:hover td {
        background-color: #1a1a1a !important;
    }

    [data-testid="stDataFrame"] canvas {
        background-color: #141414 !important;
    }

    /* Neutral Scrollbars (No Cyan / Blue) */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: #0d0d0d;
    }

    ::-webkit-scrollbar-thumb {
        background: #262626;
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #383838;
    }

    /* Clean Neutral Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f0f0f !important;
        border-right: 1px solid #222222 !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #141414 !important;
        border: 1px solid #262626 !important;
        border-radius: 6px !important;
        margin-bottom: 12px !important;
        box-shadow: none !important;
    }

    /* Code and Pre typography */
    code, pre {
        font-family: 'JetBrains Mono', Consolas, monospace !important;
        background: #111111 !important;
        border: 1px solid #222222 !important;
        color: #8a8a8a !important;
        border-radius: 4px !important;
        font-size: 0.82rem !important;
    }

    /* Header & Sidebar Collapse / Expand Control */
    header[data-testid="stHeader"] {
        background: transparent !important;
        color: #ffffff !important;
        pointer-events: auto !important;
    }

    [data-testid="stToolbar"] {
        background: transparent !important;
        visibility: visible !important;
        display: flex !important;
    }

    /* Sidebar reopen/expand button */
    [data-testid="stExpandSidebarButton"],
    button[data-testid="stExpandSidebarButton"] {
        visibility: visible !important;
        display: flex !important;
        color: #8a8a8a !important;
        background: #141414 !important;
        border: 1px solid #262626 !important;
        border-radius: 6px !important;
        box-shadow: none !important;
        cursor: pointer !important;
    }

    [data-testid="stExpandSidebarButton"]:hover {
        background: #1a1a1a !important;
        border-color: #333333 !important;
        color: #ffffff !important;
    }

    [data-testid="stSidebarCollapseButton"] button {
        visibility: visible !important;
        color: #8a8a8a !important;
        background: #141414 !important;
        border: 1px solid #262626 !important;
        border-radius: 6px !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover {
        background: #1a1a1a !important;
        border-color: #333333 !important;
        color: #ffffff !important;
    }

    /* Sidebar Layout */
    section[data-testid="stSidebar"] {
        position: relative !important;
    }

    [data-testid="stSidebarContent"] {
        display: flex !important;
        flex-direction: column !important;
        padding-top: 0 !important;
    }

    [data-testid="stSidebarHeader"] {
        position: absolute !important;
        top: 14px !important;
        right: 14px !important;
        z-index: 100 !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    [data-testid="stSidebarUserContent"] {
        order: -1 !important;
        padding-top: 16px !important;
        padding-bottom: 0px !important;
        padding-left: 16px !important;
        padding-right: 16px !important;
    }

    /* Suppress Left Sidebar Completely (Horizontal Top Navigation Layout) */
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
        visibility: hidden !important;
    }

    /* Custom Interactive SOC Tooltips */
    .info-tooltip-wrapper {
        position: relative !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        vertical-align: middle !important;
    }

    .info-icon {
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
        color: #8A8A8A !important;
        background: #1C1C1C !important;
        border: 1px solid #333333 !important;
        border-radius: 50% !important;
        width: 16px !important;
        height: 16px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.15s ease !important;
        user-select: none !important;
        cursor: pointer !important;
        line-height: 1 !important;
    }

    .info-tooltip-wrapper:hover .info-icon {
        color: #FFFFFF !important;
        background: #2E2E2E !important;
        border-color: #666666 !important;
        box-shadow: 0 0 6px rgba(255, 255, 255, 0.2) !important;
    }

    .info-tooltip-box {
        visibility: hidden !important;
        opacity: 0 !important;
        position: absolute !important;
        bottom: calc(100% + 8px) !important;
        right: -8px !important;
        width: 290px !important;
        background: #141414 !important;
        color: #CBD5E1 !important;
        border: 1px solid #333333 !important;
        border-radius: 6px !important;
        padding: 10px 13px !important;
        font-size: 0.75rem !important;
        line-height: 1.45 !important;
        font-weight: 400 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.85), 0 0 0 1px rgba(255, 255, 255, 0.07) !important;
        z-index: 99999 !important;
        pointer-events: none !important;
        transition: opacity 0.15s ease, transform 0.15s ease, visibility 0.15s ease !important;
        transform: translateY(4px) !important;
        text-align: left !important;
        white-space: normal !important;
    }

    .info-tooltip-box .tooltip-header {
        display: block !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin-bottom: 5px !important;
        font-size: 0.78rem !important;
        border-bottom: 1px solid #282828 !important;
        padding-bottom: 4px !important;
    }

    /* Bottom arrow indicator */
    .info-tooltip-box::after {
        content: '';
        position: absolute;
        top: 100%;
        right: 11px;
        border-width: 5px;
        border-style: solid;
        border-color: #141414 transparent transparent transparent;
    }
    .info-tooltip-box::before {
        content: '';
        position: absolute;
        top: 100%;
        right: 10px;
        border-width: 6px;
        border-style: solid;
        border-color: #333333 transparent transparent transparent;
    }

    .info-tooltip-wrapper:hover .info-tooltip-box {
        visibility: visible !important;
        opacity: 1 !important;
        transform: translateY(0) !important;
    }

    .info-tooltip-box.tooltip-left {
        right: auto !important;
        left: -8px !important;
    }
    .info-tooltip-box.tooltip-left::after {
        right: auto !important;
        left: 11px !important;
    }
    .info-tooltip-box.tooltip-left::before {
        right: auto !important;
        left: 10px !important;
    }

    /* Hide ONLY unwanted Streamlit chrome */
    [data-testid="stAppDeployButton"] { visibility: hidden !important; display: none !important; }
    [data-testid="stMainMenu"] { visibility: hidden !important; display: none !important; }
    [data-testid="stMainMenuButton"] { visibility: hidden !important; display: none !important; }
    #MainMenu { visibility: hidden !important; display: none !important; }
    footer { visibility: hidden !important; display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stStatusWidget"] { visibility: hidden !important; display: none !important; }
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
    """Deprecated: Sidebar navigation has been replaced by the persistent top navigation bar."""
    pass


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
