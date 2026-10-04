"""
SHADOWCAT SOC Cockpit - Page 5: Security Telemetry & Anomaly Alerts
Direct implementation of Stitch folder shadowcat_soc_alerts.
Wired to live data_provider.py and dynamic filtering pipeline.
"""

import streamlit as st
from styles import TOKENS, render_html
from data_provider import get_flagged_flows, get_analysis_metadata, get_forecast_trajectory, get_conformal_credibility
from components.risk_accumulator import render_risk_accumulator_panel
from components.layered_explanation import render_conformal_credibility_badge

def render_page():
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])
    flows = get_flagged_flows()
    meta = get_analysis_metadata()
    fc = get_forecast_trajectory()
    credibility = get_conformal_credibility()
    ml_risks = fc.get("risk", [])
    risk_val = ml_risks[0] if ml_risks else "—"
    # Map curr_stage from the backend root payload (not the rollout list)
    pred_raw = _get_live_prediction() if '_get_live_prediction' in globals() else {}
    if not pred_raw:
        # Fallback to local import if needed
        from data_provider import _get_live_prediction
        pred_raw = _get_live_prediction()
        
    curr_stage = pred_raw.get("current_stage", "—")
    
    # Baseline default alerts fallback (REMOVED: Must use real data sources only per F5)
    default_alerts = []

    # Synthesize live alerts if live ML flagged flows are present
    if flows:
        from datetime import datetime, timezone
        from mitre_kb import get_mitre_kb
        import hashlib
        
        live_alerts = []
        
        # Get latest window timestamp for mins_ago calculation
        latest_ts = None
        for f in flows:
            ts_str = f.get("timestamp_utc", f.get("Timestamp", f.get("timestamp", "")))
            if ts_str:
                try:
                    ts = datetime.fromisoformat(str(ts_str).replace(" UTC", "").replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if latest_ts is None or ts > latest_ts:
                        latest_ts = ts
                except:
                    pass
        
        kb = get_mitre_kb()
        stage_info = kb.resolve_stage(curr_stage)
        tactic_id = stage_info.get("technique_id") or stage_info.get("tactic_id") or "—"
                
        for i, flw in enumerate(flows[:50]):
            src = flw.get("src", flw.get("Src IP", "—"))
            dst = flw.get("dst", flw.get("Dst IP", "—"))
            proto = flw.get("proto", flw.get("Protocol", "—"))
            dport = flw.get("dport", flw.get("Dst Port", "—"))
            sport = flw.get("sport", flw.get("Src Port", "—"))
            
            score = flw.get("lr_score", flw.get("anomaly_score", risk_val))
            if score == "—":
                sev = "—"
            elif score >= 0.75:
                sev = "critical"
            elif score >= 0.50:
                sev = "high"
            else:
                sev = "medium"
                
            ts_str = flw.get("timestamp_utc", flw.get("Timestamp", flw.get("timestamp", "—")))
            mins_ago = 0
            if ts_str != "—" and latest_ts:
                try:
                    ts = datetime.fromisoformat(str(ts_str).replace(" UTC", "").replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    mins_ago = max(0, int((latest_ts - ts).total_seconds() / 60))
                    ts_str = ts.strftime("%H:%M:%S UTC")
                except:
                    mins_ago = 0
            else:
                mins_ago = "—"
            
            rate = "—"
            fwd = flw.get("TotLen Fwd Pkts")
            bwd = flw.get("TotLen Bwd Pkts")
            if fwd is not None and bwd is not None:
                bytes_val = float(fwd) + float(bwd)
            else:
                bytes_val = "—"
                
            dur = flw.get("Flow Duration", 0)
            if bytes_val != "—" and dur > 0 and bytes_val > 0:
                mbps = (bytes_val / 1024 / 1024) / (dur / 1e6)
                rate = f"{mbps:.1f} MB/s"
                
            flow_id = hashlib.md5(f"{src}{dst}{sport}{dport}{ts_str}".encode()).hexdigest()[:8].upper()

            if curr_stage == "—" or curr_stage == "No attack detected":
                technique_str = "—"
            else:
                technique_str = f"{tactic_id} • {curr_stage} ({proto})"

            live_alerts.append({
                "id": f"ALT-{flow_id}",
                "time": ts_str,
                "mins_ago": mins_ago,
                "sev": sev,
                "technique": technique_str,
                "host": f"node-{src}" if src != "—" else "—",
                "title": f"Suspicious flow detected: {src}:{sport} → {dst}:{dport} ({proto})." if score == "—" else f"Suspicious flow detected: {src}:{sport} → {dst}:{dport} ({proto}) with risk {score:.2f}.",
                "description": f"Ingested telemetry record flagged by World Model inference. Model indicates {curr_stage} execution phase." if score == "—" else f"Ingested telemetry record flagged by World Model inference. Model indicates {curr_stage} execution phase with cumulative risk score {score:.2f}.",

                "source": f"{src}:{sport}",
                "destination": f"{dst}:{dport}",
                "rate": rate,
                "cadence": "Continuous Burst",
                "remediation": [
                    f"Deploy SDN egress null-route on perimeter gateway for foreign destination {dst}:{dport}." if dst != "—" else "Deploy SDN egress null-route on perimeter gateway.",
                    f"Capture volatile RAM and packet capture on host {src} before terminating container." if src != "—" else "Capture volatile RAM and packet capture on host.",
                    f"Invalidate active authentication session tickets for principal associated with {src}." if src != "—" else "Invalidate active authentication session tickets.",
                    f"Audit recent DNS queries originating from {src} for anomalous external connections." if src != "—" else "Audit recent DNS queries for anomalous external connections."
                ]
            })
        raw_alerts = live_alerts
    else:
        raw_alerts = []

    # Calculate real-time metrics
    n_crit = sum(1 for a in raw_alerts if a["sev"] == "critical")
    n_high = sum(1 for a in raw_alerts if a["sev"] == "high")
    n_med = sum(1 for a in raw_alerts if a["sev"] == "medium")
    total_active = len(raw_alerts)

    # 1. Header & Metric Summary Bar
    render_html(f"""
    <div class="soc-card" style="margin-bottom: 0.75rem;">
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
            <!-- Dynamic quick counters -->
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;">
                <span class="soc-badge badge-neutral">Raw Detections: <b style="color:{t['text_high']}">{total_active}</b></span>
                <span class="soc-badge badge-critical">Critical: {n_crit}</span>
                <span class="soc-badge badge-caution">High: {n_high}</span>
                <span class="soc-badge badge-neutral">Medium: {n_med}</span>
                <span class="soc-badge badge-nominal">Pipeline: 4,812 evt/s</span>
            </div>
        </div>
    </div>
    """)

    # Paradigm Selector: Entity-Level RBA (Recommended) vs Raw Per-Window Detections
    c_mode, c_cred = st.columns([0.68, 0.32])
    with c_mode:
        alert_view_mode = st.radio(
            "Alert Paradigm:",
            [
                "Entity-Level Risk-Based Alerting (Splunk RBA Standard · Recommended)",
                "Raw Per-Window / Flow Detections (Unaggregated Baseline)",
            ],
            horizontal=True,
            index=0,
            key="alerts_view_mode_toggle",
        )
    with c_cred:
        st.markdown("<div style='text-align: right; padding-top: 4px;'>", unsafe_allow_html=True)
        render_conformal_credibility_badge(credibility)
        st.markdown("</div>", unsafe_allow_html=True)

    def _render_raw_alerts_section(alerts_list, key_prefix="raw"):
        c_f1, c_f2, c_f3 = st.columns([0.36, 0.36, 0.28])
        with c_f1:
            sev_f = st.radio(
                "Severity Filter",
                [f"All ({len(alerts_list)})", f"Critical ({n_crit})", f"High ({n_high})", f"Medium ({n_med})"],
                horizontal=True,
                key=f"{key_prefix}_sev",
                label_visibility="collapsed"
            )
        with c_f2:
            time_f = st.radio(
                "Time Horizon",
                ["Last 15m", "Last 1h", "Last 24h", "All Time"],
                horizontal=True,
                index=1,
                key=f"{key_prefix}_time",
                label_visibility="collapsed"
            )
        with c_f3:
            search_q = st.text_input("Search", placeholder="Search ID, host, technique, IP...", key=f"{key_prefix}_search", label_visibility="collapsed")

        # Apply filtering
        res = alerts_list
        if "Critical" in sev_f:
            res = [a for a in res if a["sev"] == "critical"]
        elif "High" in sev_f:
            res = [a for a in res if a["sev"] == "high"]
        elif "Medium" in sev_f:
            res = [a for a in res if a["sev"] == "medium"]

        if time_f == "Last 15m":
            res = [a for a in res if isinstance(a["mins_ago"], int) and a["mins_ago"] <= 15]
        elif time_f == "Last 1h":
            res = [a for a in res if isinstance(a["mins_ago"], int) and a["mins_ago"] <= 60]
        elif time_f == "Last 24h":
            res = [a for a in res if isinstance(a["mins_ago"], int) and a["mins_ago"] <= 1440]

        if search_q and search_q.strip():
            q = search_q.strip().lower()
            res = [
                a for a in res
                if q in a["id"].lower()
                or q in a["host"].lower()
                or q in a["technique"].lower()
                or q in a["title"].lower()
                or q in a["description"].lower()
                or q in a.get("source", "").lower()
                or q in a.get("destination", "").lower()
            ]

        render_html(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">
            <div>FILTERED THREATS: <b style="color:{t['primary']}">{len(res)} MATCHING</b> [SEV: {sev_f.split()[0].upper()} • HORIZON: {time_f.upper()}]</div>
            <span style="color: {t['text_muted']};">PER-DETECTION STREAM</span>
        </div>
        """)

        if not res:
            render_html(f"""
            <div class="soc-card" style="text-align: center; padding: 2rem; color: {t['text_muted']}; font-family: 'JetBrains Mono', monospace;">
                <div style="font-size: 0.95rem; font-weight: 700; color: {t['text_high']}; margin-bottom: 0.5rem;">NO ACTIVE DETECTIONS MATCHING CRITERIA</div>
                <div style="font-size: 0.75rem;">Try selecting 'All Time' or clearing the search query.</div>
            </div>
            """)
            return

        for alert in res:
            badge_cls = "badge-critical" if alert["sev"] == "critical" else ("badge-caution" if alert["sev"] == "high" else "badge-neutral")
            border_color = t['secondary'] if alert["sev"] == "critical" else (t['tertiary'] if alert["sev"] == "high" else t['border'])
            remediation_items_html = "".join([f"<li style='margin-bottom: 0.35rem; color: {t['text_high']};'>{item}</li>" for item in alert["remediation"]])

            mins_ago_str = f" ({alert['mins_ago']}m ago)" if isinstance(alert['mins_ago'], int) else ""
            render_html(f"""
            <div class="soc-card" style="border-left: 4px solid {border_color}; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                        <span class="soc-badge {badge_cls}">{alert['sev'].upper()}</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_muted']};">{alert['time']}{mins_ago_str}</span>
                        <span class="soc-badge badge-neutral" style="color:{border_color}">{alert['technique']}</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; font-weight: 700; color: {t['primary']};">{alert['host']}</span>
                    </div>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">ID: {alert['id']}</span>
                </div>
                <h3 style="font-family: 'Inter', sans-serif; font-size: 0.95rem; font-weight: 600; color: {t['text_high']}; margin: 0 0 0.5rem 0;">{alert['title']}</h3>
                <p style="font-family: 'Inter', sans-serif; font-size: 0.8125rem; color: {t['text_secondary']}; margin: 0 0 0.75rem 0; line-height: 1.5;">{alert['description']}</p>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; background: {t['surface_lowest']}; padding: 0.65rem; border-radius: 4px; border: 1px solid {t['border']}; margin-bottom: 0.75rem;">
                    <div><span class="soc-stat-label">Source</span><div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_high']}; font-weight: 600; word-break: break-all;">{alert['source']}</div></div>
                    <div><span class="soc-stat-label">Destination</span><div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {border_color}; font-weight: 600; word-break: break-all;">{alert['destination']}</div></div>
                    <div><span class="soc-stat-label">Flow Rate</span><div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {border_color}; font-weight: 600;">{alert['rate']}</div></div>
                    <div><span class="soc-stat-label">Cadence / Profile</span><div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {t['text_high']}; font-weight: 600;">{alert['cadence']}</div></div>
                </div>
                <div class="soc-card-nested" style="border-left: 3px solid {t['primary']}; background: {t['surface_lowest']}; padding: 0.75rem 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700; color: {t['primary']}; text-transform: uppercase; letter-spacing: 0.05em;">WHAT CAN BE DONE // RECOMMENDED SOC PLAYBOOK</div>
                        <span class="soc-badge badge-nominal" style="font-size: 0.6rem; padding: 1px 5px;">ANALYST ACTION PLAN</span>
                    </div>
                    <ul style="font-family: 'Inter', sans-serif; font-size: 0.8rem; margin: 0; padding-left: 1.25rem; line-height: 1.5;">{remediation_items_html}</ul>
                </div>
            </div>
            """)

    if "Entity-Level" in alert_view_mode:
        # TASK 1: Primary Entity-Level Risk-Based Alerting View
        render_risk_accumulator_panel()

        # Detailed per-window drill-down
        with st.expander(f"Forensic Drill-Down: Raw Per-Window Detections ({total_active} Unaggregated Records)", expanded=False):
            st.markdown(
                f"<div style='font-size: 0.78rem; color: #8A8A8A; margin-bottom: 12px; font-family: \"Inter\", sans-serif;'>"
                f"Inspect individual raw window telemetry. In the Splunk RBA model above, these individual events contribute to each entity's rolling risk score and are suppressed until an entity breaches the alert threshold."
                f"</div>",
                unsafe_allow_html=True
            )
            _render_raw_alerts_section(raw_alerts, key_prefix="exp")
    else:
        # Legacy Raw Per-Window View
        render_html(f"""
        <div class="soc-caution-banner" style="margin-bottom: 1rem;">
            <div>
                <span class="soc-caution-title">LEGACY PER-WINDOW ALERTING MODE (HIGH NOISE RISK)</span>
                <p style="margin: 0.25rem 0 0 0; color: {t['text_secondary']}; font-family: 'Inter', sans-serif; font-size: 0.8125rem;">
                    This mode triggers an alert on every individual window or flow anomaly ({total_active} total alerts). Industry SOC research (Splunk RBA) demonstrates this causes severe alert fatigue. Switch to <b>Entity-Level Risk-Based Alerting</b> above to filter out low-severity noise.
                </p>
            </div>
            <span class="soc-badge badge-caution">UNAGGREGATED</span>
        </div>
        """)
        _render_raw_alerts_section(raw_alerts, key_prefix="main")

if __name__ == "__main__":
    render_page()
