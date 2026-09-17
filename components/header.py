"""
SHADOWCAT - Executive Persistent Header Component
"""

import streamlit as st
from datetime import datetime, timezone
from styles import render_html
from data_provider import inference_status


def render_header(data, active_tab: str = "Threat Forecast"):
    """Renders the persistent horizontal top navigation bar, sticky status banner, and stream info bar."""
    analysis = data["analysis"]
    telemetry_source = st.session_state.get("telemetry_source", "BENCHMARK: CIC-IDS2018 (Infiltration)")
    window_str = st.session_state.get("window_str", analysis.get("window", "t+1 → t+4 (Active)"))

    short_feed = telemetry_source.replace("BENCHMARK: ", "").replace(" (Infiltration)", "")
    inf_status = inference_status()

    # 1. Top Navigation Bar Configuration
    nav_tabs = [
        {
            "label": "Threat Forecast",
            "url": "Forecast",
            "icon": '<svg class="soc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"></circle><path d="M12 3v18M3 12h18"></path></svg>'
        },
        {
            "label": "Lateral Movement Graph",
            "url": "AttackGraph",
            "icon": '<svg class="soc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><path d="M8.59 13.51l6.83 3.98M15.41 6.51l-6.82 3.98"></path></svg>'
        },
        {
            "label": "Evidence & Attribution",
            "url": "Evidence",
            "icon": '<svg class="soc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"></circle><path d="M21 21l-4.35-4.35"></path></svg>'
        },
        {
            "label": "Telemetry Ingestion",
            "url": "Input",
            "icon": '<svg class="soc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242M12 12v9M8 17l4 4 4-4"></path></svg>'
        },
        {
            "label": "Validation & Benchmarks",
            "url": "Validation",
            "icon": '<svg class="soc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M2 7l10-5 10 5M4 17l-2-6h8l-2 6a4 4 0 0 1-4 0zM18 17l-2-6h8l-2 6a4 4 0 0 1-4 0z"></path></svg>'
        },
        {
            "label": "Platform Specs",
            "url": "About",
            "icon": '<svg class="soc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
        },
    ]

    tabs_html_parts = []
    for tab in nav_tabs:
        is_active = (tab["label"].lower() == active_tab.lower()) or (
            tab["url"] == "About" and active_tab.lower() in ["platform specifications", "platform specs", "about"]
        )
        active_cls = " active" if is_active else ""
        active_border = "border-bottom: 2px solid #38bdf8 !important;" if is_active else "border-bottom: 2px solid transparent !important;"
        tabs_html_parts.append(
            f'<a class="top-nav-tab{active_cls}" href="{tab["url"]}" target="_self" style="text-decoration: none !important; {active_border}">{tab["icon"]}<span>{tab["label"]}</span></a>'
        )
    tabs_html = "".join(tabs_html_parts)

    utc_now = datetime.now(timezone.utc).strftime("%H:%M:%S")

    top_nav_html = f"""
    <div class="top-nav-bar">
        <div class="top-nav-left">
            <a href="Forecast" target="_self" class="top-nav-brand" style="text-decoration: none !important;">
                <span class="top-nav-brand-text">SHADOWCAT</span>
                <span class="top-nav-brand-pill">v1.0</span>
            </a>
            <div class="top-nav-tabs-wrapper">
                <div class="top-nav-tabs">
                    {tabs_html}
                </div>
            </div>
        </div>
        <div class="top-nav-right">
            <span class="badge-air-gapped" style="display:inline-flex;align-items:center;gap:5px;background:rgba(47,184,114,0.15) !important;border:1px solid rgba(47,184,114,0.45) !important;border-radius:4px;padding:2px 8px;color:#2FB872 !important;">
                <span class="badge-air-gapped-dot" style="width:6px;height:6px;border-radius:50%;background:#2FB872 !important;display:inline-block;flex-shrink:0;"></span>
                <span style="font-size:0.65rem;font-weight:700;color:#2FB872 !important;letter-spacing:0.05em;font-family:'Consolas','Courier New',monospace;">AIR-GAPPED</span>
            </span>
            <span class="top-nav-utility-item" style="font-family:'Consolas','Courier New',monospace;font-size:0.68rem;color:#9da3af;letter-spacing:0.02em;">
                UTC {utc_now}
            </span>
        </div>
    </div>
    """

    # 2. Pinned sticky top banner directly beneath top nav (0px gap, unified fixed header)
    banner_html: str = ""
    if inf_status == "live":
        banner_html = """
        <div class="top-banner-bar banner-live">
            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.74rem; color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
                <span style="background: rgba(47, 184, 114, 0.15); border: 1px solid rgba(47, 184, 114, 0.45); color: #2FB872; font-size: 0.65rem; font-weight: 800; font-family: 'Consolas', 'Courier New', monospace; padding: 1px 8px; border-radius: 4px; letter-spacing: 0.05em; display: inline-flex; align-items: center;">
                    ● LIVE
                </span>
                <span style="letter-spacing: 0.01em; color: #c9d1d9;">Live inference pipeline connected</span>
            </div>
            <div style="font-size: 0.68rem; color: #2FB872; font-family: 'Consolas', 'Courier New', monospace; font-weight: 700; letter-spacing: 0.06em; display: flex; align-items: center; gap: 6px;">
                ACTIVE STREAM
            </div>
        </div>
        """
    else:
        banner_html = """
        <div class="top-banner-bar banner-benchmark">
            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.74rem; color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
                <span style="background: rgba(224, 152, 43, 0.15); border: 1px solid rgba(224, 152, 43, 0.45); color: #E0982B; font-size: 0.64rem; font-weight: 800; font-family: 'Consolas', 'Courier New', monospace; padding: 1px 7px; border-radius: 4px; letter-spacing: 0.05em;">
                    BENCHMARK MODE
                </span>
                <span style="letter-spacing: 0.01em; color: #c9d1d9;">Running on benchmark data (CIC-IDS2018) &mdash; live inference not yet connected</span>
            </div>
            <div style="font-size: 0.68rem; color: #8a8a8a; font-family: 'Consolas', 'Courier New', monospace; letter-spacing: 0.05em; font-weight: 600;">
                LOEO 37-FOLD VALIDATED
            </div>
        </div>
        """

    info_bar_html = f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 2px 0 6px 0; margin-bottom: 4px; font-size: 0.73rem; color: #8a8a8a; font-family: 'Inter', 'Segoe UI Variable', 'Segoe UI', system-ui, sans-serif; flex-wrap: wrap; gap: 12px;">
        <div style="display: flex; align-items: center; gap: 20px;">
            <span>Feed: <b style="color: #ffffff; font-family: 'JetBrains Mono','Cascadia Code','Consolas','Courier New',monospace; font-size: 0.70rem;">{short_feed}</b></span>
            <span>Window: <b style="color: #ffffff; font-family: 'JetBrains Mono','Cascadia Code','Consolas','Courier New',monospace; font-size: 0.70rem;">{window_str}</b></span>
        </div>
        <div style="display: flex; align-items: center; gap: 16px; font-size: 0.70rem;">
            <span>Sensor: <span style="color: #ffffff; font-family: 'JetBrains Mono','Cascadia Code','Consolas','Courier New',monospace;">TAP-DMZ-01</span></span>
            <span style="color: #5a5a5a; letter-spacing: 0.03em;">LOEO 37-FOLD VALIDATED</span>
        </div>
    </div>
    """

    render_html(f"""
    <div class="shadowcat-fixed-header">
        {top_nav_html}
        {banner_html}
    </div>
    {info_bar_html}
    """)


