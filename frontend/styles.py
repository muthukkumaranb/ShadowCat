"""
SHADOWCAT SOC Cockpit - Signal on Void Design System
Direct translation of signal_on_void/DESIGN.md design tokens and rules into pure CSS
with zero runtime dependencies on cdn.tailwindcss.com.
"""

from typing import Literal

# =============================================================================
# 1. DESIGN TOKENS (Source of Truth: signal_on_void/DESIGN.md)
# =============================================================================

TOKENS = {
    "dark": {
        # Surface Foundations (Signal on Void)
        "void": "#0A0D12",
        "surface": "#111319",
        "surface_dim": "#111319",
        "surface_bright": "#36393f",
        "surface_lowest": "#0b0e13",
        "surface_low": "#191c21",
        "surface_card": "#12161D",
        "surface_container": "#1d2025",
        "surface_high": "#272a30",
        "surface_highest": "#32353b",
        "overlay": "#18202A",

        # Telemetry Emitter Signals
        "primary": "#39FF88",           # Nominal Signal
        "on_primary": "#003918",
        "primary_container": "#39FF88",
        "primary_fixed_dim": "#00e473",
        "primary_subtle": "rgba(57, 255, 136, 0.12)",

        "secondary": "#FF3B5C",         # Critical Threat / Anomaly
        "on_secondary": "#68001a",
        "secondary_container": "#c7003a",
        "secondary_subtle": "rgba(255, 59, 92, 0.14)",

        "tertiary": "#FFB84D",          # Caution / Suspicious / Dispersion
        "on_tertiary": "#452b00",
        "tertiary_container": "#ffdaab",
        "tertiary_fixed_dim": "#ffb951",
        "tertiary_subtle": "rgba(255, 184, 77, 0.14)",

        # Borders & Grid
        "border": "#1E2633",
        "border_subtle": "#283344",
        "outline": "#849584",
        "outline_variant": "#3b4a3d",

        # Text & Telemetry Layers
        "text_high": "#F3F4F6",
        "text_secondary": "#A0AEC0",
        "text_muted": "#7E8B9B",
        "text_variant": "#bacbb9",
    },
    "light": {
        # Light mode derived using tonal relationships (same emitter signals)
        "void": "#f5f6f8",
        "surface": "#ffffff",
        "surface_dim": "#eef1f5",
        "surface_bright": "#ffffff",
        "surface_lowest": "#f0f2f6",
        "surface_low": "#ffffff",
        "surface_card": "#ffffff",
        "surface_container": "#e8ecf2",
        "surface_high": "#dfe4eb",
        "surface_highest": "#d3d9e2",
        "overlay": "#e8edf5",

        "primary": "#00A84D",
        "on_primary": "#ffffff",
        "primary_container": "#00A84D",
        "primary_fixed_dim": "#00873d",
        "primary_subtle": "rgba(0, 168, 77, 0.10)",

        "secondary": "#D61B3C",
        "on_secondary": "#ffffff",
        "secondary_container": "#b01330",
        "secondary_subtle": "rgba(214, 27, 60, 0.12)",

        "tertiary": "#D97706",
        "on_tertiary": "#ffffff",
        "tertiary_container": "#fbbf24",
        "tertiary_fixed_dim": "#b45309",
        "tertiary_subtle": "rgba(217, 119, 6, 0.12)",

        "border": "#cbd5e1",
        "border_subtle": "#94a3b8",
        "outline": "#64748b",
        "outline_variant": "#94a3b8",

        "text_high": "#0f172a",
        "text_secondary": "#334155",
        "text_muted": "#64748b",
        "text_variant": "#475569",
    },
}

# =============================================================================
# BACKWARD COMPATIBILITY PALETTE (For components importing COLORS)
# =============================================================================
COLORS = {
    "bg": TOKENS["dark"]["void"],
    "surface": TOKENS["dark"]["surface_card"],
    "surface_hover": TOKENS["dark"]["surface_container"],
    "border": TOKENS["dark"]["border"],
    "border_hover": TOKENS["dark"]["border_subtle"],
    "text_primary": TOKENS["dark"]["text_high"],
    "text_secondary": TOKENS["dark"]["text_secondary"],
    "text_muted": TOKENS["dark"]["text_muted"],
    "accent": "#ffffff",
    "accent_dim": "rgba(255, 255, 255, 0.08)",
    "danger": TOKENS["dark"]["secondary"],
    "danger_dim": TOKENS["dark"]["secondary_subtle"],
    "warning": TOKENS["dark"]["tertiary"],
    "warning_dim": TOKENS["dark"]["tertiary_subtle"],
    "success": TOKENS["dark"]["primary"],
    "success_dim": TOKENS["dark"]["primary_subtle"],
    "safe": TOKENS["dark"]["primary"],
    "caution": TOKENS["dark"]["tertiary"],
    "critical": TOKENS["dark"]["secondary"],
    "primary": TOKENS["dark"]["primary"],
    "secondary": TOKENS["dark"]["secondary"],
    "neutral": TOKENS["dark"]["text_secondary"],
}

# =============================================================================
# 2. BRAND EMBLEM SVG ASSET (From shadowcat_soc_emblem)
# =============================================================================

EMBLEM_SVG = """
<svg viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg" style="height: 32px; width: 32px; vertical-align: middle;">
  <rect width="120" height="120" rx="16" fill="#0A0D12"/>
  <rect x="1" y="1" width="118" height="118" rx="15" stroke="#1E2633" stroke-width="2"/>
  <!-- Stylized radar/threat cat feline geometric nodes & vector lines -->
  <circle cx="60" cy="64" r="28" stroke="#39FF88" stroke-width="2.5" stroke-dasharray="4 3" opacity="0.6"/>
  <circle cx="60" cy="64" r="14" stroke="#39FF88" stroke-width="2" opacity="0.9"/>
  <circle cx="60" cy="64" r="3.5" fill="#39FF88"/>
  <!-- Pointed sleek geometric ears connected by signal vectors -->
  <path d="M38 48 L46 22 L60 38 L74 22 L82 48" stroke="#39FF88" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
  <!-- Eye nodes with pulse dots -->
  <circle cx="48" cy="58" r="3" fill="#39FF88"/>
  <circle cx="72" cy="58" r="3" fill="#39FF88"/>
  <!-- Threat scan crosshairs -->
  <line x1="60" y1="36" x2="60" y2="44" stroke="#39FF88" stroke-width="2"/>
  <line x1="60" y1="84" x2="60" y2="92" stroke="#39FF88" stroke-width="2"/>
  <line x1="32" y1="64" x2="40" y2="64" stroke="#39FF88" stroke-width="2"/>
  <line x1="80" y1="64" x2="88" y2="64" stroke="#39FF88" stroke-width="2"/>
</svg>
"""

# =============================================================================
# 3. GLOBAL CSS GENERATOR
# =============================================================================

def get_custom_css(theme: Literal["dark", "light"] = "dark") -> str:
    """
    Returns full CSS stylesheet implementing the Signal on Void design system.
    Strict 4px radii, 1px tonal borders, no drop shadows, dual-typeface font rules.
    """
    t = TOKENS.get(theme, TOKENS["dark"])

    return f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">

    <style>
    /* Reset & Base Canvas */
    html, body, [data-testid="stAppViewContainer"], .stApp {{
        background-color: {t["void"]} !important;
        color: {t["text_high"]} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-size: 14px !important;
        line-height: 1.4 !important;
        letter-spacing: 0em !important;
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
        display: none !important;
    }}

    .block-container {{
        padding-top: 1.25rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }}

    /* Global Typography Protocols */
    .font-mono, .telemetry, code, pre, .font-code-telemetry {{
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums !important;
    }}
    .font-prose, p, .font-body {{
        font-family: 'Inter', sans-serif !important;
    }}

    /* Top Shell Navigation Bar */
    .soc-topbar {{
        background-color: {t["surface_low"]};
        border-bottom: 1px solid {t["border"]};
        padding: 0.6rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.25rem;
        border-radius: 4px;
    }}
    .soc-topbar-brand {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.05em;
        color: {t["text_high"]};
    }}
    .soc-topbar-tag {{
        font-size: 0.6875rem;
        font-weight: 600;
        padding: 0.15rem 0.45rem;
        background: {t["surface_highest"]};
        color: {t["outline"]};
        border-radius: 4px;
        text-transform: uppercase;
    }}
    .soc-live-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6875rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        background: {t["surface_highest"]};
        color: {t["primary"]};
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .soc-pulse-dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: {t["primary"]};
        animation: socPulse 1.5s infinite;
    }}
    @keyframes socPulse {{
        0% {{ opacity: 0.4; transform: scale(0.9); }}
        50% {{ opacity: 1; transform: scale(1.15); }}
        100% {{ opacity: 0.4; transform: scale(0.9); }}
    }}

    /* Tonal Stacking Cards (Elevation without drop shadow) */
    .soc-card {{
        background-color: {t["surface_card"]};
        border: 1px solid {t["border"]};
        border-radius: 4px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        position: relative;
    }}
    .soc-card-nested {{
        background-color: {t["surface_lowest"]};
        border: 1px solid {t["border"]};
        border-radius: 4px;
        padding: 0.75rem 1rem;
    }}
    .soc-card-interactive:hover {{
        background-color: {t["overlay"]};
        border-color: {t["border_subtle"]};
        transition: background 0.15s ease, border-color 0.15s ease;
    }}

    /* Stat Cards */
    .soc-stat-card {{
        background-color: {t["surface_card"]};
        border: 1px solid {t["border"]};
        border-radius: 4px;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 105px;
    }}
    .soc-stat-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6875rem;
        font-weight: 500;
        text-transform: uppercase;
        color: {t["text_muted"]};
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }}
    .soc-stat-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 2.25rem;
        color: {t["text_high"]};
        letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums;
    }}
    .soc-stat-delta {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }}
    .delta-nominal {{ color: {t["primary"]}; }}
    .delta-threat {{ color: {t["secondary"]}; }}
    .delta-caution {{ color: {t["tertiary"]}; }}

    /* Status Badges & Pills */
    .soc-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .badge-critical {{
        background: {t["secondary_subtle"]};
        border: 1px solid {t["secondary"]};
        color: {t["secondary"]};
    }}
    .badge-caution {{
        background: {t["tertiary_subtle"]};
        border: 1px solid {t["tertiary"]};
        color: {t["tertiary"]};
    }}
    .badge-nominal {{
        background: {t["primary_subtle"]};
        border: 1px solid {t["primary"]};
        color: {t["primary"]};
    }}
    .badge-neutral {{
        background: {t["surface_highest"]};
        border: 1px solid {t["border"]};
        color: {t["text_secondary"]};
    }}

    /* Buttons */
    .stButton > button {{
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8125rem !important;
        font-weight: 600 !important;
        border-radius: 4px !important;
        border: 1px solid {t["border"]} !important;
        background: {t["surface_high"]} !important;
        color: {t["text_high"]} !important;
        padding: 0.45rem 1rem !important;
        transition: all 0.15s ease !important;
        box-shadow: none !important;
    }}
    .stButton > button:hover {{
        background: {t["surface_highest"]} !important;
        border-color: {t["border_subtle"]} !important;
        color: {t["text_high"]} !important;
    }}
    .stButton > button:active {{
        transform: scale(0.98);
    }}
    /* Primary button — covers ALL Streamlit versions */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"],
    .stButton > button[data-testid*="primary"],
    button[kind="primary"],
    [data-testid="stBaseButton-primary"] {{
        background: {t["primary"]} !important;
        color: #000000 !important;
        border: 2px solid {t["primary"]} !important;
        font-weight: 700 !important;
        font-size: 0.875rem !important;
        min-height: 48px !important;
        letter-spacing: 0.02em !important;
        text-shadow: none !important;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover,
    .stButton > button[data-testid*="primary"]:hover,
    button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {{
        background: {t["primary_fixed_dim"]} !important;
        border-color: {t["primary_fixed_dim"]} !important;
        color: #000000 !important;
    }}
    /* Also force p inside primary buttons to be dark */
    .stButton > button[kind="primary"] p,
    .stButton > button[data-testid="stBaseButton-primary"] p,
    .stButton > button[data-testid*="primary"] p,
    button[kind="primary"] p,
    [data-testid="stBaseButton-primary"] p {{
        color: #000000 !important;
    }}

    /* Header Nav Button Overrides - Prevent Text Truncation and Dots */
    div[data-testid="stHorizontalBlock"] .stButton > button {{
        padding: 0.35rem 0.3rem !important;
        font-size: 0.76rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        white-space: nowrap !important;
        letter-spacing: -0.01em !important;
    }}
    div[data-testid="stHorizontalBlock"] .stButton > button p {{
        font-size: 0.76rem !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        margin: 0 !important;
    }}

    /* Streamlit Metric Overrides */
    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: {t["text_high"]} !important;
        font-variant-numeric: tabular-nums !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.6875rem !important;
        text-transform: uppercase !important;
        color: {t["text_muted"]} !important;
        letter-spacing: 0.04em !important;
    }}

    /* Input & Text Fields */
    input, textarea, select, .stTextInput > div > div > input {{
        background-color: {t["surface_lowest"]} !important;
        border: 1px solid {t["border"]} !important;
        color: {t["text_high"]} !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8125rem !important;
    }}
    input:focus, .stTextInput > div > div > input:focus {{
        border-color: {t["primary"]} !important;
        box-shadow: none !important;
    }}

    /* Streamlit Tabs (Pill Tabs) */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.35rem !important;
        background: {t["surface_lowest"]} !important;
        border: 1px solid {t["border"]} !important;
        padding: 3px !important;
        border-radius: 9999px !important;
        margin-bottom: 1.25rem !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 9999px !important;
        padding: 0.35rem 0.95rem !important;
        color: {t["text_muted"]} !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        background: transparent !important;
        border: none !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {t["surface_high"]} !important;
        color: {t["primary"]} !important;
        border: 1px solid {t["border_subtle"]} !important;
        font-weight: 600 !important;
    }}

    /* Tables */
    table, [data-testid="stDataFrame"] {{
        border-collapse: collapse !important;
        width: 100% !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8125rem !important;
    }}
    th {{
        background: {t["overlay"]} !important;
        color: {t["text_muted"]} !important;
        font-size: 0.6875rem !important;
        text-transform: uppercase !important;
        padding: 0.5rem 0.75rem !important;
        border-bottom: 1px solid {t["border"]} !important;
        text-align: left !important;
    }}
    td {{
        background: {t["surface_card"]} !important;
        color: {t["text_secondary"]} !important;
        padding: 0.5rem 0.75rem !important;
        border-bottom: 1px solid {t["border"]} !important;
    }}
    tr:hover td {{
        background: {t["overlay"]} !important;
        color: {t["text_high"]} !important;
    }}

    /* Progress & Sliders */
    .stProgress > div > div > div > div {{
        background-color: {t["primary"]} !important;
    }}
    .stSlider [data-baseweb="slider"] {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Custom Terminal Log */
    .soc-terminal {{
        background-color: {t["surface_lowest"]};
        border: 1px solid {t["border"]};
        border-radius: 4px;
        padding: 0.75rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        line-height: 1.6;
        color: {t["text_secondary"]};
        max-height: 260px;
        overflow-y: auto;
    }}
    .soc-terminal-time {{ color: {t["text_muted"]}; margin-right: 0.5rem; }}
    .soc-terminal-info {{ color: {t["text_high"]}; font-weight: 600; margin-right: 0.5rem; }}
    .soc-terminal-warn {{ color: {t["tertiary"]}; font-weight: 600; margin-right: 0.5rem; }}
    .soc-terminal-error {{ color: {t["secondary"]}; font-weight: 600; margin-right: 0.5rem; }}

    /* Caution Banner */
    .soc-caution-banner {{
        background: {t["tertiary_subtle"]};
        border: 1px solid {t["tertiary"]};
        border-radius: 4px;
        padding: 0.65rem 1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }}
    .soc-caution-title {{
        font-weight: 700;
        color: {t["tertiary"]};
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Section Headers */
    .soc-section-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
        padding-bottom: 0.25rem;
    }}
    .soc-section-title {{
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.125rem;
        color: {t["text_high"]};
        letter-spacing: 0em;
        text-transform: uppercase;
    }}
    .soc-subsystem-tag {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6875rem;
        color: {t["primary"]};
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}
    </style>
    """

def apply_custom_css(theme: Literal["dark", "light"] = "dark") -> None:
    """Injects the custom CSS into the Streamlit session."""
    import streamlit as st
    st.markdown(get_custom_css(theme), unsafe_allow_html=True)


def render_html(html_str: str) -> None:
    """Safely renders HTML without markdown indentation block formatting.
    Strips leading whitespace from every line so CommonMark never treats 4-space indented lines
    as preformatted code blocks.
    """
    import streamlit as st
    cleaned = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)
