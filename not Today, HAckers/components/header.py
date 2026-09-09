"""
SHADOWCAT - Executive Persistent Header Component
"""

from styles import COLORS, render_html


def render_header(data):
    """Renders the slim, single-row persistent top bar with brand, air-gapped beacon, and stream timestamp."""
    analysis = data["analysis"]
    render_html(f"""
    <div class="slim-header-bar">
        <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
            <span class="slim-header-brand">SHADOWCAT</span>
            <span class="badge-offline">
                <span class="status-dot"></span>
                AIR-GAPPED
            </span>
            <span style="color: #475569; font-size: 0.85rem;">|</span>
            <span style="font-size: 0.82rem; color: #94A3B8; font-weight: 500;">
                Autonomous Cyber Threat Forecasting Engine
            </span>
        </div>
        <div style="display: flex; align-items: center; gap: 16px; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace;">
            <span style="color: #94A3B8;">STREAM: <b style="color: #FFFFFF;">{analysis['source']}</b></span>
            <span style="color: #00E5FF;">WINDOW: {analysis['window']}</span>
        </div>
    </div>
    """)

