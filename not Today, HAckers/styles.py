"""
SHADOWCAT - Executive Design System & Styling
"""

import streamlit as st
import textwrap

# Color Palette - Google Cyber / DeepMind Intelligence Theme
COLORS = {
    "bg": "#080B11",
    "surface": "#0F1523",
    "surface_glass": "rgba(16, 23, 38, 0.75)",
    "surface_hover": "#162035",
    "border": "rgba(255, 255, 255, 0.08)",
    "border_highlight": "rgba(255, 255, 255, 0.18)",
    "text_primary": "#FFFFFF",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    "accent": "#00E5FF",           # Luminous Electric Cyan
    "accent_dim": "rgba(0, 229, 255, 0.12)",
    "accent_purple": "#7C4DFF",    # DeepMind Quantum Purple
    "safe": "#00E676",             # Neon Emerald
    "safe_dim": "rgba(0, 230, 118, 0.12)",
    "caution": "#FFB300",          # Luminous Amber
    "caution_dim": "rgba(255, 179, 0, 0.14)",
    "warning": "#FF5252",          # Vibrant Crimson
    "warning_dim": "rgba(255, 82, 82, 0.14)",
    "critical": "#FF1744",         # Laser Coral Red
    "critical_dim": "rgba(255, 23, 68, 0.18)",
}

CUSTOM_CSS = f"""
<style>
    /* Google Fonts local fallback */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    /* Global Base Reset with Ambient Atmospheric Glow */
    html, body, [class*="css"] {{
        background-color: #080B11 !important;
        background-image: 
            radial-gradient(circle at 10% 8%, rgba(0, 229, 255, 0.07) 0%, transparent 45%),
            radial-gradient(circle at 90% 80%, rgba(124, 77, 255, 0.06) 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, rgba(255, 23, 68, 0.03) 0%, transparent 60%) !important;
        background-attachment: fixed !important;
        color: #F8FAFC !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }}

    /* Main Container Padding */
    .block-container {{
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1340px !important;
    }}

    /* Micro Animations */
    @keyframes pulse-beacon {{
        0% {{
            box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.7);
            transform: scale(0.98);
        }}
        70% {{
            box-shadow: 0 0 0 8px rgba(0, 230, 118, 0);
            transform: scale(1.02);
        }}
        100% {{
            box-shadow: 0 0 0 0 rgba(0, 230, 118, 0);
            transform: scale(0.98);
        }}
    }}

    @keyframes threat-radar {{
        0% {{
            box-shadow: 0 0 12px rgba(255, 23, 68, 0.4), inset 0 0 8px rgba(255, 23, 68, 0.2);
            border-color: #FF5252;
        }}
        50% {{
            box-shadow: 0 0 28px rgba(255, 23, 68, 0.75), inset 0 0 16px rgba(255, 23, 68, 0.35);
            border-color: #FF1744;
        }}
        100% {{
            box-shadow: 0 0 12px rgba(255, 23, 68, 0.4), inset 0 0 8px rgba(255, 23, 68, 0.2);
            border-color: #FF5252;
        }}
    }}

    @keyframes border-shimmer {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* Slim Persistent Top Bar Header */
    .slim-header-bar {{
        background: linear-gradient(135deg, rgba(16, 24, 40, 0.75) 0%, rgba(10, 15, 26, 0.85) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 10px 18px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.45);
    }}

    .slim-header-brand {{
        font-size: 1.18rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        background: linear-gradient(135deg, #FFFFFF 30%, #00E5FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    /* Executive Top Banner (Fallback) */
    .header-banner {{
        background: linear-gradient(135deg, rgba(16, 24, 40, 0.85) 0%, rgba(10, 15, 26, 0.9) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 18px 26px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.6), 0 1px 3px rgba(255, 255, 255, 0.05);
    }}

    .header-title {{
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        background: linear-gradient(135deg, #FFFFFF 30%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }}

    .header-subtitle {{
        font-size: 0.84rem;
        color: #94A3B8;
        margin-top: 4px;
        margin-bottom: 0;
    }}

    /* Glassmorphic Precision Cards */
    .glass-card, .cyber-card, .card {{
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.85) 0%, rgba(11, 16, 26, 0.95) 100%) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 20px !important;
        padding: 22px 24px !important;
        margin-bottom: 22px !important;
        box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.55), 0 0 1px rgba(255, 255, 255, 0.1) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }}

    .glass-card:hover, .cyber-card:hover, .card:hover {{
        border-color: rgba(0, 229, 255, 0.35) !important;
        border-top-color: rgba(0, 229, 255, 0.6) !important;
        box-shadow: 0 24px 48px -10px rgba(0, 0, 0, 0.75), 0 0 24px rgba(0, 229, 255, 0.15) !important;
        transform: translateY(-2px) !important;
    }}

    .card-title {{
        font-size: 0.88rem;
        font-weight: 700;
        color: #E2E8F0;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    .card-title span.badge {{
        font-size: 0.72rem;
        padding: 3px 10px;
        border-radius: 20px;
        background: rgba(0, 229, 255, 0.12);
        border: 1px solid rgba(0, 229, 255, 0.28);
        color: #00E5FF;
        font-weight: 600;
        text-transform: none;
        letter-spacing: 0.02em;
    }}

    /* High-Impact Stat Display */
    .metric-value-huge {{
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin: 6px 0;
        background: linear-gradient(135deg, #FFFFFF 60%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .metric-label {{
        font-size: 0.72rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    /* Live Status Badges */
    .badge-offline {{
        background: rgba(0, 230, 118, 0.12);
        color: #00E676;
        border: 1px solid rgba(0, 230, 118, 0.35);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }}

    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #00E676;
        display: inline-block;
        animation: pulse-beacon 2s infinite ease-in-out;
    }}

    /* MITRE Pipeline Modernized Pipeline */
    .mitre-pipeline {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 14px 0 20px 0;
    }}

    .mitre-card-observed {{
        background: linear-gradient(135deg, rgba(16, 28, 22, 0.8) 0%, rgba(10, 18, 14, 0.9) 100%) !important;
        border: 1px solid rgba(0, 230, 118, 0.25) !important;
        border-top: 1px solid rgba(0, 230, 118, 0.5) !important;
        border-radius: 18px !important;
        padding: 18px 20px !important;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 10px 24px -6px rgba(0, 0, 0, 0.5), 0 0 16px rgba(0, 230, 118, 0.08) !important;
        transition: all 0.25s ease !important;
    }}

    .mitre-card-observed:hover {{
        border-color: rgba(0, 230, 118, 0.6) !important;
        transform: translateY(-2px) !important;
    }}

    .mitre-card-predicted {{
        background: linear-gradient(135deg, rgba(38, 14, 20, 0.85) 0%, rgba(22, 8, 12, 0.95) 100%) !important;
        border: 2px solid #FF1744 !important;
        border-radius: 18px !important;
        padding: 18px 20px !important;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        animation: threat-radar 2.4s infinite ease-in-out !important;
        transform: scale(1.02);
        z-index: 2;
    }}

    .mitre-card-downstream {{
        background: linear-gradient(135deg, rgba(15, 20, 30, 0.6) 0%, rgba(10, 14, 22, 0.8) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 18px !important;
        padding: 18px 20px !important;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 10px 20px -6px rgba(0, 0, 0, 0.4) !important;
    }}

    /* Streamlit Native Metric Card Overrides */
    [data-testid="stMetric"] {{
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.85) 0%, rgba(11, 16, 26, 0.95) 100%) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 20px !important;
        padding: 20px 24px !important;
        box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.55), 0 0 1px rgba(255, 255, 255, 0.1) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}

    [data-testid="stMetric"]:hover {{
        border-color: rgba(0, 229, 255, 0.35) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 22px 44px -8px rgba(0, 0, 0, 0.7), 0 0 20px rgba(0, 229, 255, 0.12) !important;
    }}

    [data-testid="stMetricValue"] {{
        font-size: 2.25rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.03em !important;
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        color: #94A3B8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }}

    [data-testid="stMetricDelta"] {{
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }}

    /* Streamlit Interactive Radio & Pill Controls */
    div[data-testid="stRadio"] > div {{
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 6px !important;
        gap: 8px !important;
    }}

    div[data-testid="stRadio"] label {{
        background: transparent !important;
        border-radius: 12px !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
    }}

    div[data-testid="stRadio"] label:hover {{
        background: rgba(255, 255, 255, 0.05) !important;
    }}

    /* Buttons with Animated Sheen */
    button[kind="primary"], .stButton > button {{
        background: linear-gradient(135deg, #00E5FF 0%, #00B0FF 100%) !important;
        color: #080B11 !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 10px 22px !important;
        box-shadow: 0 8px 24px -4px rgba(0, 229, 255, 0.45) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}

    button[kind="primary"]:hover, .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px -4px rgba(0, 229, 255, 0.65) !important;
    }}

    /* Data Table Styling */
    [data-testid="stDataFrame"] {{
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 18px !important;
        overflow: hidden !important;
        background: rgba(15, 23, 42, 0.7) !important;
        box-shadow: 0 14px 30px -8px rgba(0, 0, 0, 0.5) !important;
    }}

    /* Clean Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0B0F19 0%, #080B11 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        margin-bottom: 12px !important;
    }}

    /* Code and Pre typography */
    code, pre {{
        font-family: 'JetBrains Mono', Consolas, monospace !important;
        background: rgba(0, 0, 0, 0.3) !important;
        border-radius: 6px !important;
    }}

    /* Header & Sidebar Collapse / Expand Control */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        color: #FFFFFF !important;
        pointer-events: auto !important;
    }}

    /* Keep Streamlit toolbar container visible so the expand button displays properly */
    [data-testid="stToolbar"] {{
        background: transparent !important;
        visibility: visible !important;
        display: flex !important;
    }}

    /* Always style and display the sidebar reopen/expand button (chevron >>) */
    [data-testid="stExpandSidebarButton"],
    button[data-testid="stExpandSidebarButton"] {{
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
        color: #00E5FF !important;
        background: rgba(15, 23, 42, 0.95) !important;
        border: 1.5px solid rgba(0, 229, 255, 0.45) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6), 0 0 14px rgba(0, 229, 255, 0.3) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer !important;
        z-index: 999999 !important;
    }}

    [data-testid="stExpandSidebarButton"]:hover {{
        background: rgba(0, 229, 255, 0.25) !important;
        border-color: #00E5FF !important;
        box-shadow: 0 6px 24px rgba(0, 229, 255, 0.6) !important;
        transform: scale(1.1) !important;
    }}

    /* Style the collapse button (<<) inside the sidebar header */
    [data-testid="stSidebarCollapseButton"] button {{
        visibility: visible !important;
        color: #00E5FF !important;
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }}

    [data-testid="stSidebarCollapseButton"] button:hover {{
        background: rgba(0, 229, 255, 0.2) !important;
        border-color: #00E5FF !important;
        transform: scale(1.08) !important;
    }}

    /* Sidebar Layout: Clean Top-Aligned Architecture */
    section[data-testid="stSidebar"] {{
        position: relative !important;
    }}

    [data-testid="stSidebarContent"] {{
        display: flex !important;
        flex-direction: column !important;
        padding-top: 0 !important;
    }}

    /* Pin collapse button (<<) neatly to top-right corner */
    [data-testid="stSidebarHeader"] {{
        position: absolute !important;
        top: 14px !important;
        right: 14px !important;
        z-index: 100 !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    /* Brand & Status Badge at the very top */
    [data-testid="stSidebarUserContent"] {{
        order: -1 !important;
        padding-top: 16px !important;
        padding-bottom: 0px !important;
        padding-left: 16px !important;
        padding-right: 16px !important;
    }}

    /* Page Navigation links start immediately below the brand without vertical dead space */
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

    /* Hide ONLY unwanted Streamlit chrome: Deploy button, 3-dots Menu, Decoration bar, and Footer */
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
    """
    Strips all leading and trailing whitespace from each line,
    completely preventing Streamlit's Markdown parser from treating
    4-space indented HTML tags as verbatim code blocks (<pre><code>).
    """
    return "\n".join(line.strip() for line in html_str.splitlines() if line.strip())


def render_html(html_str: str):
    """Safely renders HTML in Streamlit with zero code-block triggers."""
    st.markdown(safe_html(html_str), unsafe_allow_html=True)


def apply_custom_css():
    """Inject the executive dark mode CSS into Streamlit."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar(data=None):
    """
    Render clean, executive sidebar containing only:
    brand, status badge, and page nav links. (Task 6).
    """
    with st.sidebar:
        render_html("""
        <div style="padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 10px; padding-right: 36px;">
            <div style="font-size: 1.35rem; font-weight: 900; letter-spacing: 0.08em; color: #FFFFFF;">
                SHADOWCAT
            </div>
        </div>
        <div style="margin-bottom: 10px;">
            <span class="badge-offline">
                <span class="status-dot"></span>
                AIR-GAPPED SYSTEM
            </span>
        </div>
        """)


def render_footer():
    """Renders the persistent executive SHADOWCAT footer."""
    render_html("""
    <div style="margin-top: 36px; padding: 16px 20px; border-top: 1px solid rgba(255, 255, 255, 0.08); display: flex; justify-content: space-between; align-items: center; font-size: 0.78rem; color: #64748B;">
        <div>
            <b style="color: #FFFFFF; letter-spacing: 0.06em;">SHADOWCAT v1.0</b> &nbsp;·&nbsp; Autonomous Cyber Threat Forecasting Engine
        </div>
        <div>
            <span style="color: #00E676; font-weight: 700;">● Active (Air-Gapped)</span>
        </div>
    </div>
    """)

