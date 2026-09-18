"""
SHADOWCAT - SOC Cockpit (Signal on Void Design System)
Premium glassmorphism and neon aesthetics.
"""

import streamlit as st
import streamlit.components.v1 as components

CUSTOM_CSS = """
<style>
/* CSS Variables for Light/Dark Mode */
:root {
    --bg-base: #0A0D12;
    --surface-card: rgba(18, 22, 29, 0.65);
    --surface-hover: rgba(28, 34, 43, 0.85);
    --border-color: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(57, 255, 136, 0.3);
    --text-primary: #E8ECF1;
    --text-secondary: #8B95A5;
    --accent-primary: #39FF88;
    --accent-alert: #FF3B5C;
    --accent-warning: #FFB84D;
    
    --font-mono: 'JetBrains Mono', 'IBM Plex Mono', 'Cascadia Code', monospace;
    --font-sans: 'Inter', system-ui, sans-serif;
    
    --glow-primary: 0 0 15px rgba(57, 255, 136, 0.2);
    --glow-alert: 0 0 20px rgba(255, 59, 92, 0.3);
}

.light-mode {
    --bg-base: #F4F6F8;
    --surface-card: rgba(255, 255, 255, 0.85);
    --surface-hover: rgba(255, 255, 255, 1);
    --border-color: rgba(0, 0, 0, 0.1);
    --border-hover: rgba(45, 187, 99, 0.4);
    --text-primary: #12161D;
    --text-secondary: #5A6372;
    --accent-primary: #2DBB63;
    --accent-alert: #D32F2F;
    --accent-warning: #F57C00;
    
    --glow-primary: 0 4px 15px rgba(45, 187, 99, 0.15);
    --glow-alert: 0 4px 20px rgba(211, 47, 47, 0.2);
}

/* Global Reset */
html, body, [class*="css"], [data-testid="stAppViewContainer"], .stApp, section.main {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    transition: background-color 0.4s ease, color 0.4s ease;
}

/* Hide Default Streamlit Header */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Typography Enhancements */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-mono) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
}

/* Glassmorphism Cards */
div[data-testid="stMetric"], 
div[data-testid="stExpander"], 
div.st-emotion-cache-12w0qpk { /* standard container */
    background: var(--surface-card) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}

div[data-testid="stMetric"]:hover, 
div[data-testid="stExpander"]:hover {
    background: var(--surface-hover) !important;
    border-color: var(--border-hover) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--glow-primary) !important;
}

/* Metrics Typography */
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, var(--accent-primary), #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.85rem !important;
}

/* Buttons - Premium Neon treatment */
div[data-testid="stButton"] button {
    background: rgba(18, 22, 29, 0.4) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 8px 24px !important;
    transition: all 0.3s ease !important;
    backdrop-filter: blur(8px) !important;
}

div[data-testid="stButton"] button:hover {
    border-color: var(--accent-primary) !important;
    color: var(--accent-primary) !important;
    box-shadow: var(--glow-primary) !important;
    background: rgba(57, 255, 136, 0.05) !important;
}

div[data-testid="stButton"] button[kind="primary"] {
    background: var(--accent-primary) !important;
    border: none !important;
    color: #000000 !important;
    box-shadow: var(--glow-primary) !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #ffffff !important;
    box-shadow: 0 0 25px rgba(255, 255, 255, 0.6) !important;
}

/* Blockquotes (Used for Plain-Language Summaries) */
blockquote {
    border-left: 4px solid var(--accent-primary) !important;
    background: rgba(57, 255, 136, 0.05) !important;
    padding: 15px 20px !important;
    border-radius: 0 8px 8px 0 !important;
    margin: 20px 0 !important;
    color: var(--text-primary) !important;
    font-size: 1.05rem !important;
    line-height: 1.6 !important;
}

/* Clean up Streamlit Chrome */
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], footer { 
    display: none !important; 
}

/* Custom Alert Badges */
.badge-critical {
    background: rgba(255, 59, 92, 0.15);
    border: 1px solid var(--accent-alert);
    color: var(--accent-alert);
    padding: 4px 10px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-weight: bold;
    box-shadow: var(--glow-alert);
}
.badge-warning {
    background: rgba(255, 184, 77, 0.15);
    border: 1px solid var(--accent-warning);
    color: var(--accent-warning);
    padding: 4px 10px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-weight: bold;
}
.badge-nominal {
    background: rgba(57, 255, 136, 0.15);
    border: 1px solid var(--accent-primary);
    color: var(--accent-primary);
    padding: 4px 10px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-weight: bold;
    box-shadow: var(--glow-primary);
}
</style>
"""

def apply_custom_css(theme='dark'):
    """Injects the premium glassmorphism SOC CSS into Streamlit."""
    # 1. Inject the pure CSS via standard markdown
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
