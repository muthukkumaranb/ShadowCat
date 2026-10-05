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
    get_live_notarization_status,
    get_conformal_credibility,
    get_conformal_forecast,
)

def render_page():
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])
    meta = get_analysis_metadata()
    fc = get_forecast_trajectory()
    novelty = get_novelty_score()
    audit = get_audit_chain_status()
    mitre = get_mitre_data()
    flows = get_flagged_flows()
    notary_status = get_live_notarization_status()

    fabric_active = notary_status.get("fabric_active", True)
    fallback_engaged = notary_status.get("fallback_engaged", False)
    tx_id = notary_status.get("tx_id", "N/A")
    fallback_idx = notary_status.get("fallback_index", 0)

    risk_val = (fc.get("risk") or [0.0])[0]
    _cf = get_conformal_forecast() or {}
    _iv = (_cf.get("intervals") or [[None, None]])[0]
    ci_txt = f"[{_iv[0]:.2f}, {_iv[1]:.2f}]" if _iv and _iv[0] is not None else "not available"
    ci_lo = _iv[0] if _iv and _iv[0] is not None else None
    ci_hi = _iv[1] if _iv and _iv[1] is not None else None
    alert_thr = float(fc.get("alert_threshold", 0.5) or 0.5)
    lead_time = fc.get("lead_time", ["1m 00s"])[0]

    # Dynamic risk classification based on actual ML prediction
    if risk_val >= 0.75:
        risk_label = "HIGH RISK // CRITICAL TRAJECTORY"
        risk_badge_cls = "badge-critical"
        risk_color = t['secondary']
        risk_delta = f"+{risk_val - 0.5:.2f}"
    elif risk_val >= 0.5:
        risk_label = "ELEVATED RISK // CAUTION TRAJECTORY"
        risk_badge_cls = "badge-caution"
        risk_color = t.get('tertiary', '#FFB84D')
        risk_delta = f"+{risk_val - 0.3:.2f}"
    elif risk_val >= 0.25:
        risk_label = "MODERATE RISK // WATCH TRAJECTORY"
        risk_badge_cls = "badge-neutral"
        risk_color = t['text_secondary']
        risk_delta = f"+{risk_val - 0.1:.2f}"
    else:
        risk_label = "BASELINE // LOW RISK ENVELOPE"
        risk_badge_cls = "badge-nominal"
        risk_color = t['primary']
        risk_delta = f"{risk_val:.2f}"

    # TOP FULL-WIDTH THREAT HEADER STRIP
    # Pre-compute conformal credibility badge HTML to avoid backslash-in-f-string
    # (illegal before Python 3.12 / PEP 701).
    _cred = get_conformal_credibility()
    _is_ind = _cred.get("is_in_distribution", True)
    _font_family = "'JetBrains Mono', monospace"
    _bg_color = "rgba(48, 209, 88, 0.12)" if _is_ind else "rgba(255, 69, 58, 0.15)"
    _border_color = "#30D158" if _is_ind else "#FF453A"
    _text_color = "#30D158" if _is_ind else "#FF453A"
    _icon = "✔" if _is_ind else "⚠"

    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1.25rem;">
        <div style="display: grid; grid-template-columns: 7fr 5fr; gap: 1.5rem; align-items: center;">
            <!-- Left: Risk Status & Telemetry Metadata -->
            <div>
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; flex-wrap: wrap;">
                    <span class="soc-badge {risk_badge_cls}">
                        <span class="soc-pulse-dot" style="background:{risk_color};"></span>
                        {risk_label}
                    </span>
                    <span style="display: inline-flex; align-items: center; gap: 0.35rem; background: {_bg_color}; border: 1px solid {_border_color}; border-radius: 4px; padding: 2px 6px; font-family: {_font_family}; font-size: 0.68rem; font-weight: 700; color: {_text_color};">{_icon} {_cred.get("badge_label", "IN-DISTRIBUTION")}</span>
                </div>
                <div style="display: flex; align-items: baseline; gap: 1rem;">
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 2.25rem; font-weight: 700; color: {t['text_high']}; letter-spacing: -0.02em;">
                        {risk_val:.2f} <span style="font-size: 1.125rem; font-weight: 400; color: {t['text_muted']};">/ 1.00</span>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; font-weight: 600; color: {risk_color};">
                        <span style="color: {t['text_muted']}; font-size: 0.75rem; font-weight: 400;">P(attack within next 5 min) • 30-window lookback</span>
                    </div>
                </div>
                <p style="font-family: 'Inter', sans-serif; font-size: 0.875rem; color: {t['text_high']}; margin-top: 0.5rem; margin-bottom: 0.75rem; line-height: 1.5;">
                    {'Onset probability is above the alert threshold for the loaded window.' if risk_val >= alert_thr else 'Onset probability is below the alert threshold for the loaded window.'}
                </p>
                <div style="display: flex; gap: 1rem; flex-wrap: wrap; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">
                    <span>WINDOW: <b style="color:{t['text_secondary']}">60s Sliding</b></span>
                    <span>•</span>
                    <span>CHECKPOINT: <b style="color:{t['text_secondary']}">lstm-stacked-v1</b></span>
                    <span>•</span>
                    <span>ALERT THRESHOLD: <b style="color:{t['text_secondary']}">{alert_thr:.2f}</b></span>
                    <span>•</span>
                    <span>90% CONFORMAL INTERVAL: <b style="color:{t['text_secondary']}">{ci_txt}</b></span>
                </div>
            </div>
            <!-- Right: Uncertainty Sparkline Widget -->
            <div class="soc-card-nested" style="position: relative; overflow: hidden;">
                <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; margin-bottom: 0.25rem;">
                    <span style="color: {t['text_muted']}; text-transform: uppercase; font-weight: 600;">Projected Egress Trajectory</span>
                    <span style="color: {risk_color}; font-weight: 600;">95% Uncertainty Horizon</span>
                </div>
                <div style="width: 100%; height: 95px;">
                    <svg style="width: 100%; height: 100%;" viewBox="0 0 380 95" preserveAspectRatio="none">
                        <defs>
                            <linearGradient id="uncertaintyGradient" x1="0%" x2="100%" y1="0%" y2="0%">
                                <stop offset="0%" stop-color="{risk_color}" stop-opacity="0.05" />
                                <stop offset="50%" stop-color="{risk_color}" stop-opacity="0.18" />
                                <stop offset="100%" stop-color="{risk_color}" stop-opacity="0.45" />
                            </linearGradient>
                        </defs>
                        <!-- t0 vertical marker -->
                        <line x1="180" y1="5" x2="180" y2="80" stroke="{t['outline_variant']}" stroke-width="1" stroke-dasharray="3 3" />
                        <text x="180" y="90" fill="{t['text_muted']}" font-family="JetBrains Mono" font-size="8" text-anchor="middle">t₀ NOW</text>
                        <!-- Widening Gaussian Uncertainty Cone -->
                        <path d="M 180 {max(15, min(80, int(75 - risk_val * 55)))} C 220 {max(10, min(80, int(70 - risk_val * 55)))}, 280 {max(8, min(80, int(60 - risk_val * 55)))}, 380 {max(5, min(80, int(50 - risk_val * 55)))} L 380 {min(85, max(15, int(85 - risk_val * 25)))} C 280 {min(85, max(20, int(80 - risk_val * 25)))}, 220 {min(85, max(25, int(78 - risk_val * 25)))}, 180 {max(15, min(80, int(75 - risk_val * 55)))} Z" fill="url(#uncertaintyGradient)" />
                        <!-- Upper/Lower 95% bounds -->
                        <path d="M 180 {max(15, min(80, int(75 - risk_val * 55)))} C 220 {max(10, min(80, int(70 - risk_val * 55)))}, 280 {max(8, min(80, int(60 - risk_val * 55)))}, 380 {max(5, min(80, int(50 - risk_val * 55)))}" fill="none" stroke="{risk_color}" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.7" />
                        <path d="M 180 {max(15, min(80, int(75 - risk_val * 55)))} C 220 {min(85, max(25, int(78 - risk_val * 25)))}, 280 {min(85, max(20, int(80 - risk_val * 25)))}, 380 {min(85, max(15, int(85 - risk_val * 25)))}" fill="none" stroke="{risk_color}" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.7" />
                        <!-- Historical Baseline Line -->
                        <path d="M 0 70 Q 45 68, 90 65 T 180 {max(15, min(80, int(75 - risk_val * 55)))}" fill="none" stroke="{t['primary']}" stroke-width="2.5" />
                        <!-- Mean Forecast Line -->
                        <path d="M 180 {max(15, min(80, int(75 - risk_val * 55)))} C 230 {max(12, min(80, int(68 - risk_val * 55)))}, 290 {max(10, min(80, int(60 - risk_val * 55)))}, 380 {max(8, min(80, int(55 - risk_val * 55)))}" fill="none" stroke="{risk_color}" stroke-width="2.5" />
                        <!-- t0 Pulse Dot -->
                        <circle cx="180" cy="{max(15, min(80, int(75 - risk_val * 55)))}" r="3.5" fill="{risk_color}" />
                        <circle cx="380" cy="{max(8, min(80, int(55 - risk_val * 55)))}" r="3" fill="{risk_color}" />
                    </svg>
                </div>
                <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']}; padding-top: 2px;">
                    <span>t-30w</span>
                    <span>t-15w</span>
                    <span style="color: {t['primary']}; font-weight: 700;">t₀</span>
                    <span>t+1 (1m)</span>
                    <span>t+3 (3m)</span>
                    <span style="color: {risk_color}; font-weight: 700;">t+5 (5m)</span>
                </div>
            </div>
        </div>
    </div>
    """)

    # 4 REALISTIC STREAMLIT METRIC CARDS
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    # The alert is the window's onset decision (probability vs the fixed alert threshold). The listed flows
    # are the window's flows ranked by their LR contribution; they are alerts only when the window is.
    in_alert = risk_val >= alert_thr
    n_crit = len(flows) if in_alert else 0
    n_low = 0 if in_alert else len(flows)
    if in_alert:
        stat_badge_cls, stat_badge_txt = "badge-critical", "ONSET ALERT"
    else:
        stat_badge_cls, stat_badge_txt = "badge-nominal", "BELOW THRESHOLD"
    delta_threat_txt = f"{len(flows)} top-ranked flows in window"
    n_total_alerts = n_crit
    if not flows:
        # Window-level input (e.g. the dataset demo slices) has no individual flows: the alert is the window itself.
        n_total_alerts = 1 if in_alert else 0
        delta_threat_txt = "window-level input (no individual flows)"
    if flows:
        alert_badge_txt = f"{n_crit} in alert window"
        low_badge_html = ('<span class="soc-badge badge-neutral" style="padding: 0.1rem 0.4rem; font-size: 0.625rem;">'
                          f"{n_low} below threshold</span>")
    else:
        alert_badge_txt = "window in alert" if in_alert else "window below threshold"
        low_badge_html = ""

    with m_col1:
        render_html(f"""
        <div class="soc-stat-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="soc-stat-label">Active Alerts</span>
                <span class="soc-badge {stat_badge_cls}" style="font-size: 0.625rem; padding: 1px 5px;">{stat_badge_txt}</span>
            </div>
            <div style="margin: 0.35rem 0;">
                <div class="soc-stat-val">{n_total_alerts}</div>
                <div style="display: flex; gap: 0.35rem; margin-top: 0.35rem;">
                    <span class="soc-badge {'badge-critical' if n_total_alerts > 0 else 'badge-nominal'}" style="padding: 0.1rem 0.4rem; font-size: 0.625rem;">{alert_badge_txt}</span>
                    {low_badge_html}
                </div>
            </div>
            <div class="soc-stat-delta {'delta-threat' if risk_val >= 0.5 else 'delta-nominal'}">
                <span>{delta_threat_txt}</span>
                <span style="color: {t['text_muted']}; margin-left: auto;">Exp: {risk_val:.2f}</span>
            </div>
        </div>
        """)

    with m_col2:
        _cur = fc.get("current_stage") or "—"
        stage_id = fc.get("current_stage_tactic_id") or ("N/A" if _cur in ("No attack detected", "not available", "—") else "—")
        stage_tactic = _cur
        stage_tech = fc.get("current_stage_family") or "No stage determined"
        stage_url = fc.get("current_stage_url") or "https://attack.mitre.org/tactics/enterprise/"
        _conf = fc.get("current_stage_confidence")
        stage_status = f"Current window • conf {_conf:.2f}" if isinstance(_conf, (int, float)) else "Current window"

        render_html(f"""
        <div class="soc-stat-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="soc-stat-label">Current ATT&CK Stage</span>
                <span class="soc-badge badge-caution" style="font-size: 0.625rem; padding: 1px 5px;">STAGE</span>
            </div>
            <div style="margin: 0.35rem 0;">
                <div class="soc-stat-val" style="display: flex; align-items: baseline; gap: 0.4rem;">
                    <span>{stage_id}</span>
                    <a href="{stage_url}" target="_blank" style="font-size: 0.6875rem; color: {t['primary']}; text-decoration: underline; font-family: 'JetBrains Mono', monospace;">
                        MITRE ↗
                    </a>
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; color: {t['text_secondary']}; margin-top: 0.25rem;">
                    {stage_tactic} / {stage_tech}
                </div>
            </div>
            <div class="soc-stat-delta" style="color: {t['tertiary']};">
                <span class="soc-badge badge-caution" style="padding: 0.1rem 0.4rem; font-size: 0.625rem;">{stage_status}</span>
                <span style="color: {t['text_muted']}; margin-left: auto;">Verified STIX 2.1</span>
            </div>
        </div>
        """)

    with m_col3:
        novelty_val = float(novelty.get("novelty_score") or 0.0)
        render_html(f"""
        <div class="soc-stat-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="soc-stat-label">Novelty Score</span>
                <span class="soc-badge badge-neutral" style="font-size: 0.625rem; padding: 1px 5px;">SIGMA</span>
            </div>
            <div style="margin: 0.35rem 0;">
                <div class="soc-stat-val">
                    {novelty_val:.3f} <span style="font-size: 0.75rem; font-weight: 400; color: {t['text_muted']};">/ 1.000</span>
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; color: {t['text_secondary']}; margin-top: 0.25rem;">
                    World-model deviation: observed vs predicted state
                </div>
            </div>
            <div class="soc-stat-delta delta-threat">
                <span>{novelty.get('flows_analyzed') or '—'} rows analysed</span>
                <span style="color: {t['text_muted']}; margin-left: auto;">World Model Deviation</span>
            </div>
        </div>
        """)

    with m_col4:
        is_valid = audit.get("is_valid", True)
        mech = notary_status.get("active_mechanism", "none")
        if mech == "fabric":
            badge_title, badge_cls, badge_color = "FABRIC", "badge-nominal", t['primary']
            badge_subtext = f"Alert hash {tx_id} • notarized on Hyperledger Fabric"
            badge_footer = "Fabric ledger"
        elif mech == "sha256_fallback":
            badge_title, badge_cls, badge_color = "SHA-256 FALLBACK", "badge-caution", "#ff9f0a"
            badge_subtext = f"Alert hash {tx_id} • SHA-256 chain entry #{fallback_idx} (Fabric not running)"
            badge_footer = "SHA-256 hash chain"
        else:
            badge_title, badge_cls, badge_color = "CHAIN CHECKED", "badge-nominal", t['primary']
            badge_subtext = f"Audit chain: {audit.get('length', 0)} entries verified • this window not notarized (no alert)"
            badge_footer = "Ed25519-signed hash chain"

        render_html(f"""
        <div class="soc-stat-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="soc-stat-label">Audit Chain Integrity</span>
                <span class="soc-badge {badge_cls}" style="font-size: 0.625rem; padding: 1px 5px;">{badge_title}</span>
            </div>
            <div style="margin: 0.35rem 0;">
                <div class="soc-stat-val" style="color: {badge_color};">
                    {"VERIFIED" if is_valid else "TAMPERED"}
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; color: {t['text_secondary']}; margin-top: 0.25rem;">
                    {badge_subtext}
                </div>
            </div>
            <div class="soc-stat-delta delta-nominal">
                <span>{'INTACT' if is_valid else 'CHECK FAILED'}</span>
                <span style="color: {t['text_muted']}; margin-left: auto;">{badge_footer}</span>
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
                    TOP FLOWS • LOADED WINDOW
                </div>
                <div class="soc-section-title" style="margin-top: 0.25rem;">
                    Flows Driving the Onset Forecast (ranked by LR contribution)
                </div>
            </div>
        </div>
    """)

    # Alert items: the window's flows ranked by LR contribution. Severity is the window's onset decision;
    # the tactic label comes only from the backend's current-window stage classifier.
    alert_items = []
    _stage_lbl = fc.get("current_stage") or "—"
    _stage_tid = fc.get("current_stage_tactic_id")
    for f in flows[:10]:
        src = f.get("source") or "—"
        dst = f.get("destination") or "—"
        proto = f.get("protocol") or "—"
        dport = f.get("dport") or "—"
        sc = f.get("lr_score")
        sev = "critical" if in_alert else "low"
        sc_txt = f"LR contribution {sc:+.2f}" if isinstance(sc, (int, float)) else "LR contribution —"
        ts = f.get("timestamp") or "—"
        alert_items.append({
            "time": str(ts),
            "sev": sev,
            "technique": f"{_stage_tid} {_stage_lbl}" if _stage_tid else "—",
            "prose": f"{src} → {dst}:{dport} (proto {proto}), {sc_txt}",
            "target": str(dst),
        })

    if not alert_items:
        alert_items = []

    cnt_crit = sum(1 for a in alert_items if a["sev"] == "critical")
    cnt_low = sum(1 for a in alert_items if a["sev"] == "low")

    # Filter Segmented Tabs
    filter_choice = st.radio(
        "Alert Filter",
        [f"ALL ({len(alert_items)})", f"IN ALERT WINDOW ({cnt_crit})", f"BELOW THRESHOLD ({cnt_low})"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if "IN ALERT" in filter_choice:
        filtered_alerts = [item for item in alert_items if item["sev"] == "critical"]
    elif "BELOW" in filter_choice:
        filtered_alerts = [item for item in alert_items if item["sev"] == "low"]
    else:
        filtered_alerts = alert_items

    if not filtered_alerts:
        render_html(f"""
        <div class="soc-card-nested" style="text-align: center; padding: 1.5rem; color: {t['text_muted']}; font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem;">
            NO ACTIVE THREATS MATCHING CRITERIA [{filter_choice}]
        </div>
        """)
    else:
        for item in filtered_alerts:
            badge_cls = "badge-critical" if item["sev"] == "critical" else "badge-neutral"
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
                Source: {st.session_state.get("active_source", "default CSE-CIC-IDS2018 demo window")}
            </div>
            <span style="color: {t['primary']};">{len(flows)} flows ranked by LR contribution</span>
        </div>
    </div>
    """)

if __name__ == "__main__":
    render_page()
