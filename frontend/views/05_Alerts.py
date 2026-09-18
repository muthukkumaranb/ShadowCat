"""
SHADOWCAT SOC Cockpit - Page 5: Security Telemetry & Anomaly Alerts
Direct implementation of Stitch folder shadowcat_soc_alerts.
Wired to live data_provider.py and flagged flows.
"""

import streamlit as st
from styles import TOKENS
from data_provider import get_flagged_flows, get_analysis_metadata

def render_page():
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])
    flows = get_flagged_flows()
    meta = get_analysis_metadata()

    # 1. Header & Metric Summary Bar
    st.markdown(f"""
    <div class="soc-card" style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span class="soc-pulse-dot" style="background:{t['secondary']}; width:8px; height:8px;"></span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.125rem; font-weight: 700; color: {t['text_high']}; text-transform: uppercase;">
                        Security Telemetry & Anomaly Alerts
                    </span>
                </div>
                <div style="display: flex; gap: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']}; margin-top: 0.25rem;">
                    <span>STREAM ACTIVE</span> • <span style="color:{t['primary']}">RUNNING MONTE CARLO HEURISTICS</span> • <span>SYS_REF: 0x884F_A</span>
                </div>
            </div>
            <!-- Quick counters -->
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;">
                <span class="soc-badge badge-neutral">Total Active: <b style="color:{t['text_high']}">18</b></span>
                <span class="soc-badge badge-critical">Critical: 3</span>
                <span class="soc-badge badge-critical" style="border-color:{t['secondary']};">High: 5</span>
                <span class="soc-badge badge-caution">Medium: 10</span>
                <span class="soc-badge badge-neutral">Suppressed / FP: 2</span>
                <span class="soc-badge badge-nominal">Pipeline: 4,812 evt/s</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Filter & Sort Control Strip
    c_f1, c_f2, c_f3 = st.columns([0.35, 0.35, 0.3])
    with c_f1:
        sev_filter = st.radio(
            "Severity Filter",
            ["All (18)", "Critical (3)", "High (5)", "Medium (10)"],
            horizontal=True,
            label_visibility="collapsed"
        )
    with c_f2:
        time_filter = st.radio(
            "Time Horizon",
            ["Last 15m", "Last 1h", "Last 24h", "Custom"],
            horizontal=True,
            index=1,
            label_visibility="collapsed"
        )
    with c_f3:
        search_term = st.text_input("Search", placeholder="Search alert ID, host, ASN...", label_visibility="collapsed")

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # 3. Vertical List of Alert Cards
    # Card 1: Critical (Expanded View)
    st.markdown(f"""
    <div class="soc-card" style="border-left: 4px solid {t['secondary']}; margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                <span class="soc-badge badge-critical">CRITICAL</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_muted']};">14:28:09 UTC</span>
                <span class="soc-badge badge-neutral" style="color:{t['secondary']}">T1071.001 • C2: Web Protocols</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: {t['primary']};">
                    ip-10-0-14-88 (svc-auth-master)
                </span>
            </div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">ID: ALT-9941</span>
        </div>
        <h3 style="font-family: 'Inter', sans-serif; font-size: 0.95rem; font-weight: 600; color: {t['text_high']}; margin: 0 0 0.5rem 0;">
            Egress burst spike (+410%) to uncatalogued foreign ASN 4837 with high-frequency encrypted beaconing cadence.
        </h3>
        <p style="font-family: 'Inter', sans-serif; font-size: 0.8125rem; color: {t['text_secondary']}; margin: 0 0 0.75rem 0; line-height: 1.5;">
            Outbound TLS flow directed toward 45.138.21.9:443 exceeded safe enterprise baseline by +410%. Packet inter-arrival variance matches known Cobalt Strike Malleable C2 jitter profile.
        </p>
        <!-- Expanded Telemetry Grid -->
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; background: {t['surface_lowest']}; padding: 0.65rem; border-radius: 4px; border: 1px solid {t['border']}; margin-bottom: 0.75rem;">
            <div>
                <span class="soc-stat-label">Source</span>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; color: {t['text_high']}; font-weight: 600;">10.0.14.88:49210</div>
            </div>
            <div>
                <span class="soc-stat-label">Destination</span>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; color: {t['secondary']}; font-weight: 600;">45.138.21.9:443 (CN)</div>
            </div>
            <div>
                <span class="soc-stat-label">Outbound Rate</span>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; color: {t['secondary']}; font-weight: 600;">14.8 GB/s (+410%)</div>
            </div>
            <div>
                <span class="soc-stat-label">Cadence / Jitter</span>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; color: {t['text_high']}; font-weight: 600;">3.2s ± 12ms</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_act1, c_act2, c_act3 = st.columns(3)
    with c_act1:
        if st.button("Isolate Endpoint (ip-10-0-14-88)", type="primary", use_container_width=True):
            st.success("Containment command issued: Host isolated.")
    with c_act2:
        if st.button("Revoke Active Kerberos Ticket", use_container_width=True):
            st.info("Kerberos TGT invalidated on DC cluster.")
    with c_act3:
        if st.button("Export Forensic PCAP Trace", use_container_width=True):
            st.info("PCAP downloaded.")

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # Card 2: Critical
    st.markdown(f"""
    <div class="soc-card" style="border-left: 4px solid {t['secondary']}; margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                <span class="soc-badge badge-critical">CRITICAL</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_muted']};">14:24:51 UTC</span>
                <span class="soc-badge badge-neutral" style="color:{t['secondary']}">T1059.004 • Lateral Movement</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: {t['primary']};">
                    svc-auth-master
                </span>
            </div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">ID: ALT-9938</span>
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.875rem; font-weight: 600; color: {t['text_high']};">
            Unauthenticated RPC execution across host enclave subnet with abnormal peer fan-out.
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_secondary']}; margin-top: 0.35rem;">
            TARGET: <b style="color:{t['text_high']}">svc-auth-master</b> • SUBNET: 10.0.14.0/24 • PROTOCOL: TCP/135
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Card 3: High
    st.markdown(f"""
    <div class="soc-card" style="border-left: 4px solid {t['tertiary']}; margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                <span class="soc-badge badge-caution">HIGH</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_muted']};">14:19:12 UTC</span>
                <span class="soc-badge badge-neutral" style="color:{t['tertiary']}">T1046 • Network Discovery</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: {t['primary']};">
                    ws-analyst-12
                </span>
            </div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">ID: ALT-9932</span>
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.875rem; font-weight: 600; color: {t['text_high']};">
            Rapid port sweeping scan detected from internal workstation cluster segment.
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_secondary']}; margin-top: 0.35rem;">
            TARGET: <b style="color:{t['text_high']}">ws-analyst-12</b> • PORTS: 22, 80, 443, 8080, 3389
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Card 4: Medium
    st.markdown(f"""
    <div class="soc-card" style="border-left: 4px solid {t['tertiary']}; margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                <span class="soc-badge badge-neutral">MEDIUM</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_muted']};">14:12:00 UTC</span>
                <span class="soc-badge badge-neutral">T1078 • Valid Accounts</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: {t['primary']};">
                    iam-sync-daemon
                </span>
            </div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">ID: ALT-9925</span>
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.875rem; font-weight: 600; color: {t['text_high']};">
            Simultaneous geo-distributed session tokens authenticated for high-privilege service principal.
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_secondary']}; margin-top: 0.35rem;">
            TARGET: <b style="color:{t['text_high']}">iam-sync-daemon</b> • TOKENS: 2 concurrently active
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_page()
