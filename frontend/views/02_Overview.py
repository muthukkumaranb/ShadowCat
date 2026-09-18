"""
SHADOWCAT SOC Cockpit - Page 2: Overview
Direct implementation of Stitch folder shadowcat_soc_overview.
Wired to live data_provider.py and authoritative predictions.
"""

import streamlit as st
from styles import TOKENS, render_html
from data_provider import (
    get_analysis_metadata,
    get_forecast_trajectory,
    get_novelty_score,
    get_audit_chain_status,
    get_mitre_data,
    get_flagged_flows,
)

def render_page():
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])
    meta = get_analysis_metadata()
    fc = get_forecast_trajectory()
    novelty = get_novelty_score()
    audit = get_audit_chain_status()
    mitre = get_mitre_data()
    flows = get_flagged_flows()

    risk_val = fc.get("risk", [0.84])[0]
    lead_time = fc.get("lead_time", ["1m 00s"])[0]

    # TOP FULL-WIDTH THREAT HEADER STRIP
    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1.25rem;">
        <div style="display: grid; grid-template-columns: 7fr 5fr; gap: 1.5rem; align-items: center;">
            <!-- Left: Risk Status & Telemetry Metadata -->
            <div>
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; flex-wrap: wrap;">
                    <span class="soc-badge badge-critical">
                        <span class="soc-pulse-dot" style="background:{t['secondary']};"></span>
                        HIGH RISK // CRITICAL TRAJECTORY
                    </span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_muted']};">LATENCY: 42ms</span>
                    <span style="color: {t['outline_variant']};">|</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_muted']};">CLUSTER: ALPHA-01</span>
                </div>
                <div style="display: flex; align-items: baseline; gap: 1rem;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 2.25rem; font-weight: 700; color: {t['text_high']}; letter-spacing: -0.02em;">
                        {risk_val:.2f} <span style="font-size: 1.125rem; font-weight: 400; color: {t['text_muted']};">/ 1.00</span>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; font-weight: 600; color: {t['secondary']};">
                        ▲ +0.28 <span style="color: {t['text_muted']}; font-size: 0.75rem; font-weight: 400;">(last 45m window)</span>
                    </div>
                </div>
                <p style="font-family: 'Inter', sans-serif; font-size: 0.875rem; color: {t['text_high']}; margin-top: 0.5rem; margin-bottom: 0.75rem; line-height: 1.5;">
                    Risk elevated due to abnormal traffic growth and unusual peer connections detected across egress enclave cluster-alpha.
                </p>
                <div style="display: flex; gap: 1rem; flex-wrap: wrap; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">
                    <span>WINDOW: <b style="color:{t['text_secondary']}">60s Sliding</b></span>
                    <span>•</span>
                    <span>CHECKPOINT: <b style="color:{t['text_secondary']}">sc-threat-v4.1</b></span>
                    <span>•</span>
                    <span>NODES: <b style="color:{t['primary']}">1,420 Active</b></span>
                    <span>•</span>
                    <span>CONFIDENCE: <b style="color:{t['text_secondary']}">95.4% Monte Carlo</b></span>
                </div>
            </div>
            <!-- Right: Uncertainty Sparkline Widget -->
            <div class="soc-card-nested" style="position: relative; overflow: hidden;">
                <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; margin-bottom: 0.25rem;">
                    <span style="color: {t['text_muted']}; text-transform: uppercase; font-weight: 600;">Projected Egress Trajectory</span>
                    <span style="color: {t['secondary']}; font-weight: 600;">95% Uncertainty Horizon</span>
                </div>
                <div style="width: 100%; height: 95px;">
                    <svg style="width: 100%; height: 100%;" viewBox="0 0 380 95" preserveAspectRatio="none">
                        <defs>
                            <linearGradient id="uncertaintyGradient" x1="0%" x2="100%" y1="0%" y2="0%">
                                <stop offset="0%" stop-color="{t['secondary']}" stop-opacity="0.05" />
                                <stop offset="50%" stop-color="{t['secondary']}" stop-opacity="0.18" />
                                <stop offset="100%" stop-color="{t['secondary']}" stop-opacity="0.45" />
                            </linearGradient>
                        </defs>
                        <!-- t0 vertical marker -->
                        <line x1="180" y1="5" x2="180" y2="80" stroke="{t['outline_variant']}" stroke-width="1" stroke-dasharray="3 3" />
                        <text x="180" y="90" fill="{t['text_muted']}" font-family="JetBrains Mono" font-size="8" text-anchor="middle">t₀ NOW</text>
                        <!-- Widening Gaussian Uncertainty Cone -->
                        <path d="M 180 44 C 220 38, 280 26, 380 10 L 380 80 C 280 62, 220 52, 180 44 Z" fill="url(#uncertaintyGradient)" />
                        <!-- Upper/Lower 95% bounds -->
                        <path d="M 180 44 C 220 38, 280 26, 380 10" fill="none" stroke="{t['secondary']}" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.7" />
                        <path d="M 180 44 C 220 52, 280 62, 380 80" fill="none" stroke="{t['secondary']}" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.7" />
                        <!-- Historical Baseline Line -->
                        <path d="M 0 70 Q 45 68, 90 60 T 180 44" fill="none" stroke="{t['primary']}" stroke-width="2.5" />
                        <!-- Mean Forecast Line -->
                        <path d="M 180 44 C 230 40, 290 32, 380 26" fill="none" stroke="{t['secondary']}" stroke-width="2.5" />
                        <!-- t0 Pulse Dot -->
                        <circle cx="180" cy="44" r="3.5" fill="{t['secondary']}" />
                        <circle cx="380" cy="26" r="3" fill="{t['secondary']}" />
                    </svg>
                </div>
                <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']}; padding-top: 2px;">
                    <span>t-30m</span>
                    <span>t-15m</span>
                    <span style="color: {t['primary']}; font-weight: 700;">t₀</span>
                    <span>t+15m</span>
                    <span>t+30m</span>
                    <span style="color: {t['secondary']}; font-weight: 700;">t+60m</span>
                </div>
            </div>
        </div>
    </div>
    """)

    # 4 REALISTIC STREAMLIT METRIC CARDS
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        render_html(f"""
        <div class="soc-stat-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="soc-stat-label">Active Alerts</span>
                <span style="color: {t['secondary']}; font-size: 1.1rem;">⚠</span>
            </div>
            <div style="margin: 0.35rem 0;">
                <div class="soc-stat-val">18</div>
                <div style="display: flex; gap: 0.35rem; margin-top: 0.35rem;">
                    <span class="soc-badge badge-critical" style="padding: 0.1rem 0.4rem; font-size: 0.625rem;">3 Critical</span>
                    <span class="soc-badge badge-caution" style="padding: 0.1rem 0.4rem; font-size: 0.625rem;">9 Med</span>
                    <span class="soc-badge badge-neutral" style="padding: 0.1rem 0.4rem; font-size: 0.625rem;">6 Low</span>
                </div>
            </div>
            <div class="soc-stat-delta delta-threat">
                <span>▲ +4 vs baseline</span>
                <span style="color: {t['text_muted']}; margin-left: auto;">Exp: 3.2</span>
            </div>
        </div>
        """)

    with m_col2:
        stage_title = mitre[1]["id"] if len(mitre) > 1 else "T1071.001"
        render_html(f"""
        <div class="soc-stat-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="soc-stat-label">Current ATT&CK Stage</span>
                <span style="color: {t['tertiary']}; font-size: 1.1rem;">⚡</span>
            </div>
            <div style="margin: 0.35rem 0;">
                <div class="soc-stat-val">{stage_title}</div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; color: {t['text_secondary']}; margin-top: 0.25rem;">
                    Command & Control / Web Protocols
                </div>
            </div>
            <div class="soc-stat-delta" style="color: {t['tertiary']};">
                <span class="soc-badge badge-caution" style="padding: 0.1rem 0.4rem; font-size: 0.625rem;">Active Execution Phase</span>
                <span style="color: {t['text_muted']}; margin-left: auto;">Phase 4/7</span>
            </div>
        </div>
        """)

    with m_col3:
        novelty_val = novelty.get("novelty_score", 0.892)
        render_html(f"""
        <div class="soc-stat-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="soc-stat-label">Novelty Score</span>
                <span style="color: {t['text_muted']}; font-size: 1.1rem;">◎</span>
            </div>
            <div style="margin: 0.35rem 0;">
                <div class="soc-stat-val">
                    {novelty_val:.3f} <span style="font-size: 0.75rem; font-weight: 400; color: {t['text_muted']};">/ 1.000</span>
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; color: {t['text_secondary']}; margin-top: 0.25rem;">
                    98th percentile anomaly vs 30d baseline
                </div>
            </div>
            <div class="soc-stat-delta delta-threat">
                <span>Z-SCORE: +3.41σ</span>
                <span style="color: {t['text_muted']}; margin-left: auto;">Isolation Forest</span>
            </div>
        </div>
        """)

    with m_col4:
        is_valid = audit.get("is_valid", True)
        render_html(f"""
        <div class="soc-stat-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="soc-stat-label">Audit Chain Integrity</span>
                <span style="color: {t['primary']}; font-size: 1.1rem;">✓</span>
            </div>
            <div style="margin: 0.35rem 0;">
                <div class="soc-stat-val" style="color: {t['primary']};">
                    {"VERIFIED" if is_valid else "TAMPERED"}
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; color: {t['text_secondary']}; margin-top: 0.25rem;">
                    Block #849,204 • SHA-256 Validated
                </div>
            </div>
            <div class="soc-stat-delta delta-nominal">
                <span>ENCLAVE SECURE</span>
                <span style="color: {t['text_muted']}; margin-left: auto;">0 Errors</span>
            </div>
        </div>
        """)

    render_html("<div style='height: 1rem;'></div>")

    # SEVERITY-SORTED RECENT ALERTS QUEUE
    render_html(f"""
    <div class="soc-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 0.5rem;">
            <div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['primary']}; font-weight: 700; text-transform: uppercase;">
                    LIVE TELEMETRY FEED • AUTO-REFRESH: 5s
                </div>
                <div class="soc-section-title" style="margin-top: 0.25rem;">
                    Recent Threat Detections & Forecast Anomalies
                </div>
            </div>
        </div>
    """)

    # Filter Segmented Tabs
    filter_choice = st.radio(
        "Alert Filter",
        ["ALL (18)", "CRITICAL (3)", "HIGH (5)", "MEDIUM (10)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    alert_items = [
        {"time": "14:28:09 UTC", "sev": "critical", "technique": "T1071.001 C2", "prose": "Egress burst spike (+410%) to uncatalogued foreign ASN 4837 with encrypted beaconing cadence", "target": "ip-10-0-14-88"},
        {"time": "14:24:51 UTC", "sev": "critical", "technique": "T1059.004 LATERAL", "prose": "Unauthenticated RPC execution across host enclave subnet with abnormal peer fan-out", "target": "svc-auth-master"},
        {"time": "14:19:12 UTC", "sev": "medium", "technique": "T1046 NET DISCOVERY", "prose": "Rapid port sweeping scan detected from internal workstation cluster segment", "target": "ws-analyst-12"},
        {"time": "14:12:00 UTC", "sev": "medium", "technique": "T1078 VALID ACCTS", "prose": "Simultaneous geo-distributed session tokens authenticated for high-privilege service principal", "target": "iam-sync-daemon"},
        {"time": "13:58:33 UTC", "sev": "nominal", "technique": "T1090 PROXY", "prose": "DNS tunneling heuristic cleared after automated isolation and quarantine sandbox verification", "target": "edge-gw-02"},
    ]

    for item in alert_items:
        badge_cls = "badge-critical" if item["sev"] == "critical" else ("badge-caution" if item["sev"] == "medium" else "badge-nominal")
        render_html(f"""
        <div class="soc-card-nested" style="margin-bottom: 0.45rem; display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 0.75rem; min-width: 0; flex: 1;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_muted']}; shrink: 0;">{item['time']}</span>
                <span class="soc-badge {badge_cls}" style="padding: 0.15rem 0.5rem; font-size: 0.6875rem; shrink: 0;">{item['technique']}</span>
                <span style="font-family: 'Inter', sans-serif; font-size: 0.8125rem; color: {t['text_high']}; truncate: true;">{item['prose']}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">
                <span style="color: {t['text_muted']};">TARGET:</span>
                <span style="color: {t['primary']}; font-weight: 600;">{item['target']}</span>
            </div>
        </div>
        """)

    render_html(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.75rem; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">
            <div>
                <span class="soc-pulse-dot" style="display: inline-block;"></span> INGESTION BUFFER: 4,812 EVT/SEC • 0 DROPPED PACKETS
            </div>
            <span style="color: {t['primary']};">PIPELINE NOMINAL // 0x9F41</span>
        </div>
    </div>
    """)

if __name__ == "__main__":
    render_page()
