"""
NOT USED BY THE DASHBOARD. Design sketch of entity-level risk-based alerting: the per-host window history
below is simulated (there is no multi-window per-host history in a single upload), so it is not rendered
on any page. Kept for future work on rolling, per-entity alert aggregation.

SHADOWCAT SOC Cockpit - Entity-Level Risk-Based Alerting (RBA) Component
Implements the Splunk Risk-Based Alerting (RBA) paradigm:
Accumulates per-window hazard and stage scores per entity (Host or Subnet)
over a rolling time window (e.g. 24 hours), only surfacing high-fidelity
alerts when an entity's cumulative risk crosses a configurable threshold.
Suppresses raw per-window noise to eliminate SOC alert fatigue.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

from styles import TOKENS, render_html
from data_provider import (
    get_forecast_trajectory,
    get_host_risk_graph,
    get_mitre_data,
    get_flagged_flows,
    get_analysis_metadata,
)

# Standard stage risk weights based on MITRE ATT&CK killchain severity
STAGE_WEIGHTS: Dict[str, float] = {
    "Reconnaissance": 0.8,
    "Initial Access": 1.0,
    "Discovery": 1.0,
    "Credential Access": 1.3,
    "Lateral Movement": 1.5,
    "Command and Control": 1.6,
    "Impact": 2.0,
    "Unknown/Other": 0.9,
}

# Asset criticality risk multipliers
CRITICALITY_MULTIPLIERS: Dict[str, float] = {
    "Tier 1 (Critical Asset)": 1.8,
    "Tier 1 Crown Jewel": 1.8,
    "Tier 2 (Management Gateway)": 1.4,
    "Tier 2 (Auth Infrastructure)": 1.5,
    "Tier 3 (User Endpoint)": 1.0,
    "Tier 3 (Enterprise Storage)": 1.1,
    "Standard": 1.0,
}


def _extract_subnet(ip: str) -> str:
    """Extracts /24 subnet CIDR from an IPv4 address."""
    parts = ip.strip().split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
    return "10.0.0.0/24"


def calculate_entity_risk(
    threshold: float = 100.0,
    lookback_hours: int = 24,
    group_by: str = "host",  # "host" or "subnet"
) -> Dict[str, Any]:
    """
    Computes rolling entity-level risk accumulation from model forecast,
    host graph, flagged flows, and canonical MITRE mappings.
    Returns structured entity profiles, threshold crossing alerts, and noise reduction metrics.
    """
    host_graph = get_host_risk_graph()
    node_roles = host_graph.get("node_roles", {})
    host_telemetry = host_graph.get("host_telemetry", {})
    rollout_steps = host_graph.get("rollout_steps", {})
    forecast = get_forecast_trajectory()
    mitre_stages = get_mitre_data()
    flows = get_flagged_flows()

    # Build canonical stage lookup
    stage_lookup: Dict[str, Dict[str, Any]] = {}
    for item in mitre_stages:
        s_name = item.get("stage", "Unknown")
        stage_lookup[s_name] = item

    # Base ML hazard scores from forecast
    raw_risks = forecast.get("risk", [0.84, 0.72, 0.58, 0.45, 0.38])
    pred_stages = forecast.get("stage", ["Credential Access", "Lateral Movement", "Impact"])
    primary_stage = pred_stages[0] if pred_stages else "Credential Access"

    # Entities to track
    known_hosts = list(node_roles.keys()) if node_roles else [
        "172.31.69.21", "172.31.69.1"
    ]

    # Map each host to its simulated window contributions over lookback
    total_raw_window_events = 0
    entity_contributions: Dict[str, List[Dict[str, Any]]] = {}

    for host_ip in known_hosts:
        t_info = host_telemetry.get(host_ip, {})
        crit_tier = t_info.get("criticality_tier", "Tier 3 (User Endpoint)")
        crit_mult = CRITICALITY_MULTIPLIERS.get(crit_tier, 1.0)
        subnet = t_info.get("subnet", _extract_subnet(host_ip)).split()[0]
        entity_key = host_ip if group_by == "host" else subnet

        if entity_key not in entity_contributions:
            entity_contributions[entity_key] = []

        # Pull host specific risk from rollout
        step0 = rollout_steps.get(0, {}).get("host_risks", {}).get(host_ip, {"risk": 0.20})
        base_h = step0.get("risk", 0.20)

        # Generate realistic historical window contributions over the rolling window
        # Reflecting periodic baseline activity + escalating attack bursts
        n_windows = min(24, max(4, lookback_hours * 2))
        
        for w_idx in range(n_windows):
            total_raw_window_events += 1
            # Age of window: w_idx 0 is oldest, n_windows - 1 is most recent
            recency_factor = (w_idx + 1) / n_windows
            
            # Host-specific behavior simulation matching CSE-CIC-IDS2018 infiltration
            if base_h >= 0.5:  # Elevated host risk
                assigned_stage = "Reconnaissance" if w_idx < n_windows * 0.4 else "Initial Access"
                win_hazard = min(0.98, max(0.15, base_h * (0.6 + recency_factor * 0.7)))
                reason = "TCP SYN port sweep & HTTP boundary probe" if assigned_stage == "Reconnaissance" else "Brute-force SSH credential spray"
            elif False:  # Jump Host (Breached)
                assigned_stage = "Credential Access" if w_idx > n_windows * 0.3 else "Reconnaissance"
                win_hazard = min(0.96, max(0.12, base_h * (0.5 + recency_factor * 0.85)))
                reason = "SSH PAM authentication failure burst (18 req/min)" if assigned_stage == "Credential Access" else "Inbound port scan response"
            elif base_h >= 0.25:
                assigned_stage = "Lateral Movement" if w_idx > n_windows * 0.6 else "Credential Access"
                win_hazard = min(0.92, max(0.08, base_h * (0.3 + recency_factor * 0.9)))
                reason = "Anomalous Kerberos TGS request spike (RC4 encryption)" if assigned_stage == "Lateral Movement" else "LDAP authentication query sweep"
            elif False:  # Domain Controller
                assigned_stage = "Impact" if w_idx > n_windows * 0.8 else "Lateral Movement"
                win_hazard = min(0.88, max(0.05, base_h * (0.2 + recency_factor * 0.8)))
                reason = "Privileged directory replication request (DCSync pattern)" if assigned_stage == "Impact" else "Active Directory query burst"
            else:  # Storage share (10.0.3.50)
                assigned_stage = "Discovery"
                win_hazard = min(0.40, max(0.04, base_h * (0.4 + recency_factor * 0.3)))
                reason = "Routine SMB file metadata queries"

            s_weight = STAGE_WEIGHTS.get(assigned_stage, 1.0)
            stage_info = stage_lookup.get(assigned_stage, {})
            tech_id = stage_info.get("id", "T1046")

            # Splunk RBA Score Formula: Raw Hazard * Base Weight * Stage Weight * Criticality Multiplier
            raw_pts = win_hazard * 30.0 * s_weight * crit_mult
            score_pts = round(raw_pts, 1)

            # Calculate relative timestamp
            mins_ago = int((n_windows - w_idx) * (lookback_hours * 60 / n_windows))
            h_ago = mins_ago // 60
            m_ago = mins_ago % 60
            time_str = f"-{h_ago}h {m_ago}m" if h_ago > 0 else f"-{m_ago}m"

            entity_contributions[entity_key].append({
                "window_idx": w_idx,
                "time_ago": time_str,
                "mins_ago": mins_ago,
                "host_ip": host_ip,
                "raw_hazard": round(win_hazard, 3),
                "stage": assigned_stage,
                "technique_id": tech_id,
                "stage_weight": s_weight,
                "criticality_mult": crit_mult,
                "score": score_pts,
                "reason": reason,
            })

    # Aggregate entity profiles
    entity_profiles: List[Dict[str, Any]] = []

    for ent_key, contribs in entity_contributions.items():
        total_risk = round(sum(c["score"] for c in contribs), 1)
        is_alerting = total_risk >= threshold

        # Stage breakdown calculation
        stage_breakdown: Dict[str, float] = {}
        for c in contribs:
            stg = c["stage"]
            stage_breakdown[stg] = round(stage_breakdown.get(stg, 0.0) + c["score"], 1)

        # Entity metadata
        if group_by == "host":
            t_info = host_telemetry.get(ent_key, {})
            role = node_roles.get(ent_key, t_info.get("role", "Monitored Host"))
            crit = t_info.get("criticality_tier", "Tier 3 (User Endpoint)")
            subnet_label = t_info.get("subnet", _extract_subnet(ent_key))
        else:
            role = f"Subnet Segment ({len([h for h in known_hosts if _extract_subnet(h).startswith(ent_key.split('/')[0][:7])])} Hosts)"
            crit = "Enterprise Enclave"
            subnet_label = ent_key

        # Risk severity
        if total_risk >= threshold * (3 / 2):
            sev = "CRITICAL"
            sev_color = "#FF453A"
        elif total_risk >= threshold:
            sev = "HIGH"
            sev_color = "#FF9F0A"
        elif total_risk >= threshold * 0.7:
            sev = "MEDIUM (WATCH)"
            sev_color = "#E0982B"
        else:
            sev = "LOW"
            sev_color = "#30D158"

        sorted_contribs = sorted(contribs, key=lambda x: x["mins_ago"])

        entity_profiles.append({
            "entity_id": ent_key,
            "entity_type": group_by,
            "role": role,
            "criticality": crit,
            "subnet": subnet_label,
            "accumulated_risk": total_risk,
            "threshold": threshold,
            "is_alerting": is_alerting,
            "severity": sev,
            "severity_color": sev_color,
            "stage_breakdown": stage_breakdown,
            "contributions": sorted_contribs,
            "total_windows": len(contribs),
            "top_stage": max(stage_breakdown.items(), key=lambda x: x[1])[0] if stage_breakdown else "Unknown",
        })

    # Sort so alerting entities appear first, ranked by highest risk
    entity_profiles.sort(key=lambda x: (not x["is_alerting"], -x["accumulated_risk"]))

    alerting_entities = [e for e in entity_profiles if e["is_alerting"]]
    sub_threshold_entities = [e for e in entity_profiles if not e["is_alerting"]]

    # Noise suppression statistics
    suppressed_events = total_raw_window_events - len(alerting_entities)
    noise_reduction_pct = (
        round((suppressed_events / total_raw_window_events) * 100, 1)
        if total_raw_window_events > 0 else 0.0
    )

    return {
        "threshold": threshold,
        "lookback_hours": lookback_hours,
        "group_by": group_by,
        "total_tracked_entities": len(entity_profiles),
        "alerting_count": len(alerting_entities),
        "sub_threshold_count": len(sub_threshold_entities),
        "total_raw_events": total_raw_window_events,
        "suppressed_events": suppressed_events,
        "noise_reduction_pct": noise_reduction_pct,
        "alerting_entities": alerting_entities,
        "sub_threshold_entities": sub_threshold_entities,
        "all_entities": entity_profiles,
    }


def render_risk_accumulator_panel():
    """
    Renders the interactive Entity-Level Risk-Based Alerting dashboard component.
    """
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])

    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1rem; border-left: 4px solid {t['primary']};">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.35rem;">
                    <span class="soc-badge badge-nominal">SPLUNK RBA ENGINE</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">
                        ISO/IEC 27035 • ENTITY ATTRIBUTION
                    </span>
                </div>
                <h2 style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: {t['text_high']}; text-transform: uppercase; margin: 0;">
                    Entity-Level Risk-Based Alerting (RBA)
                </h2>
                <p style="font-family: 'Inter', sans-serif; font-size: 0.8125rem; color: {t['text_secondary']}; margin: 0.35rem 0 0 0; line-height: 1.4;">
                    Replaces alert-fatiguing per-window triggers by accumulating hazard scores per <b>Host</b> or <b>Subnet</b> over rolling time windows. Alerts fire <i>only</i> when an entity's cumulative risk breaches the threshold.
                </p>
            </div>
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;">
                <span class="soc-badge badge-neutral">Rolling Window: 24h</span>
                <span class="soc-badge badge-nominal">MITRE-Weighted</span>
            </div>
        </div>
    </div>
    """)

    # Interactive Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([0.38, 0.32, 0.30])
    with ctrl_col1:
        threshold_val = st.slider(
            "RBA Alert Threshold (Cumulative Risk Points):",
            min_value=50,
            max_value=250,
            value=100,
            step=10,
            help="Entities whose accumulated risk points cross this threshold generate a priority SOC alert.",
            key="rba_threshold_slider"
        )
    with ctrl_col2:
        entity_mode = st.radio(
            "Entity Aggregation Scope:",
            ["Per Host (Endpoint/Server)", "Per Subnet (/24 Enclave)"],
            horizontal=True,
            key="rba_scope_radio"
        )
        group_key = "host" if "Host" in entity_mode else "subnet"
    with ctrl_col3:
        window_mode = st.selectbox(
            "Rolling Time Horizon:",
            ["24 Hours (Splunk RBA Default)", "12 Hours", "4 Hours", "1 Hour"],
            index=0,
            key="rba_window_select"
        )
        hours_map = {
            "24 Hours (Splunk RBA Default)": 24,
            "12 Hours": 12,
            "4 Hours": 4,
            "1 Hour": 1,
        }
        lookback_hrs = hours_map.get(window_mode, 24)

    # Compute RBA state
    rba_data = calculate_entity_risk(
        threshold=float(threshold_val),
        lookback_hours=lookback_hrs,
        group_by=group_key,
    )

    # High-impact KPI Strip highlighting SOC fatigue reduction
    render_html(f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin: 1rem 0;">
        <div class="soc-card-nested" style="border-left: 3px solid #FF453A;">
            <div class="soc-stat-label">Alerting Entities (Crossed Threshold)</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 800; color: #FF453A; margin: 0.2rem 0;">
                {rba_data['alerting_count']} <span style="font-size: 0.85rem; color: {t['text_muted']};">/ {rba_data['total_tracked_entities']} tracked</span>
            </div>
            <div style="font-size: 0.70rem; color: {t['text_secondary']};">Cumulative risk &ge; {threshold_val} pts</div>
        </div>
        <div class="soc-card-nested" style="border-left: 3px solid {t['primary']};">
            <div class="soc-stat-label">Raw Window Detections Analyzed</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 800; color: {t['text_high']}; margin: 0.2rem 0;">
                {rba_data['total_raw_events']}
            </div>
            <div style="font-size: 0.70rem; color: {t['text_muted']};">Across {lookback_hrs}h rolling horizon</div>
        </div>
        <div class="soc-card-nested" style="border-left: 3px solid #30D158;">
            <div class="soc-stat-label">Sub-Threshold Noisy Alerts Suppressed</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 800; color: #30D158; margin: 0.2rem 0;">
                {rba_data['suppressed_events']}
            </div>
            <div style="font-size: 0.70rem; color: {t['text_secondary']};">Low-severity noise filtered out</div>
        </div>
        <div class="soc-card-nested" style="border-left: 3px solid #00F5FF;">
            <div class="soc-stat-label">SOC Fatigue Reduction Ratio</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 800; color: #00F5FF; margin: 0.2rem 0;">
                {rba_data['noise_reduction_pct']}%
            </div>
            <div style="font-size: 0.70rem; color: {t['text_secondary']};">Alert volume reduction factor</div>
        </div>
    </div>
    """)

    # Section 1: Active Threshold-Crossing Entity Alerts
    render_html(f"""
    <div class="soc-section-header" style="margin-top: 1.25rem;">
        <div class="soc-section-title">
            Active Entity-Level Risk Alerts (Threshold &ge; {threshold_val} pts)
        </div>
        <span class="soc-subsystem-tag">SURFACED PRIORITY QUEUE</span>
    </div>
    """)

    alerting = rba_data["alerting_entities"]
    if not alerting:
        render_html(f"""
        <div class="soc-card" style="text-align: center; padding: 2rem; color: {t['text_muted']};">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 700; color: #30D158; margin-bottom: 0.4rem;">
                ✔ NO ENTITIES CURRENTLY EXCEED THRESHOLD ({threshold_val} pts)
            </div>
            <div style="font-size: 0.78rem;">
                All monitored network entities remain below alerting threshold. Raw noise is safely suppressed.
            </div>
        </div>
        """)
    else:
        for ent in alerting:
            # Gauge percentage
            pct = min(200, int((ent["accumulated_risk"] / threshold_val) * 100))
            bar_pct = min(100, pct)

            # Stage badges
            stage_chips = ""
            for stg, pts in ent["stage_breakdown"].items():
                stage_chips += f"""
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.70rem; background: {t['surface_card']}; border: 1px solid {t['border']}; padding: 2px 7px; border-radius: 4px; color: {t['text_high']};">
                    <b>{stg}</b>: {pts:.1f} pts
                </span>
                """

            render_html(f"""
            <div class="soc-card" style="border-left: 4px solid {ent['severity_color']}; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.6rem;">
                    <div style="display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;">
                        <span class="soc-badge" style="background: rgba(255, 69, 58, 0.15); border: 1px solid {ent['severity_color']}; color: {ent['severity_color']}; font-weight: 700;">
                            {ent['severity']} THRESHOLD BREACH
                        </span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 800; color: {t['text_high']};">
                            {ent['entity_id']}
                        </span>
                        <span class="soc-badge badge-neutral" style="font-size: 0.72rem;">
                            {ent['role']}
                        </span>
                        <span class="soc-badge badge-neutral" style="font-size: 0.68rem; color: {t['text_muted']};">
                            {ent['criticality']}
                        </span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 800; color: {ent['severity_color']};">
                            {ent['accumulated_risk']} <span style="font-size: 0.8rem; color: {t['text_muted']}; font-weight: 400;">/ {threshold_val} pts</span>
                        </span>
                        <div style="font-size: 0.65rem; color: {ent['severity_color']}; font-family: 'JetBrains Mono', monospace; font-weight: 600;">
                            +{ent['accumulated_risk'] - threshold_val:.1f} pts over threshold ({pct}%)
                        </div>
                    </div>
                </div>

                <!-- Progress Bar -->
                <div style="width: 100%; height: 8px; background: {t['surface_highest']}; border-radius: 4px; overflow: hidden; margin-bottom: 0.75rem;">
                    <div style="width: {bar_pct}%; height: 100%; background: {ent['severity_color']}; border-radius: 4px; transition: width 0.3s ease;"></div>
                </div>

                <!-- MITRE ATT&CK Stage Attribution Breakdown -->
                <div style="margin-bottom: 0.6rem;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']}; text-transform: uppercase;">
                        ATTRIBUTED MITRE ATT&CK STAGES:
                    </span>
                    <div style="display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.3rem;">
                        {stage_chips}
                    </div>
                </div>
            </div>
            """)

            # Detailed Expandable Audit Trail
            with st.expander(f"Audit Trail: Inspect {len(ent['contributions'])} Individual Window Contributions for {ent['entity_id']}", expanded=False):
                st.markdown(
                    f"<div style='font-size: 0.75rem; color: #8A8A8A; margin-bottom: 8px; font-family: \"Inter\", sans-serif;'>"
                    f"Chronological per-window model hazard scores and points contributing to {ent['entity_id']}'s risk accumulation:"
                    f"</div>",
                    unsafe_allow_html=True
                )
                rows_html = ""
                for c in ent["contributions"]:
                    stg_color = "#FF453A" if c["raw_hazard"] >= 0.70 else ("#FF9F0A" if c["raw_hazard"] >= 0.35 else "#30D158")
                    rows_html += f"""
                    <tr>
                        <td style="font-family: 'JetBrains Mono', monospace; color: {t['text_muted']};">{c['time_ago']}</td>
                        <td style="font-family: 'JetBrains Mono', monospace; color: {t['text_high']}; font-weight: 600;">{c['host_ip']}</td>
                        <td><span class="soc-badge badge-neutral">{c['stage']} ({c['technique_id']})</span></td>
                        <td style="font-family: 'JetBrains Mono', monospace; color: {stg_color}; font-weight: 700;">{c['raw_hazard']:.2f}</td>
                        <td style="font-family: 'JetBrains Mono', monospace; color: {t['primary']}; font-weight: 700;">+{c['score']:.1f} pts</td>
                        <td style="font-size: 0.75rem; color: {t['text_secondary']};">{c['reason']}</td>
                    </tr>
                    """

                render_html(f"""
                <table class="soc-card" style="width: 100%; font-size: 0.75rem; border-collapse: collapse;">
                    <thead>
                        <tr style="text-align: left; border-bottom: 1px solid {t['border']};">
                            <th style="padding: 6px;">Window Age</th>
                            <th style="padding: 6px;">Target Host</th>
                            <th style="padding: 6px;">MITRE Stage</th>
                            <th style="padding: 6px;">Model Hazard</th>
                            <th style="padding: 6px;">Risk Points</th>
                            <th style="padding: 6px;">Driving Indicator</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
                """)

    # Section 2: Sub-Threshold Monitored Entities (Noise Suppressed)
    sub_thresh = rba_data["sub_threshold_entities"]
    if sub_thresh:
        with st.expander(f"View {len(sub_thresh)} Sub-Threshold Monitored Entities (Suppressed from Main SOC Alert Queue)", expanded=False):
            st.markdown(
                f"<div style='font-size: 0.75rem; color: #8A8A8A; margin-bottom: 8px; font-family: \"Inter\", sans-serif;'>"
                f"These entities have recorded low or moderate telemetry activity, but their cumulative risk has not breached the {threshold_val} pt threshold. In traditional SOCs, each of these generates a separate false-alarm alert."
                f"</div>",
                unsafe_allow_html=True
            )
            sub_rows = ""
            for se in sub_thresh:
                sub_rows += f"""
                <tr>
                    <td style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: {t['text_high']};">{se['entity_id']}</td>
                    <td>{se['role']}</td>
                    <td><span class="soc-badge badge-neutral">{se['criticality']}</span></td>
                    <td style="font-family: 'JetBrains Mono', monospace; color: {se['severity_color']}; font-weight: 700;">{se['accumulated_risk']} / {threshold_val}</td>
                    <td><span class="soc-badge badge-nominal">SUPPRESSED ({se['severity']})</span></td>
                    <td><span class="soc-badge badge-neutral">{se['top_stage']}</span></td>
                </tr>
                """
            render_html(f"""
            <table class="soc-card" style="width: 100%; font-size: 0.75rem; border-collapse: collapse;">
                <thead>
                    <tr style="text-align: left; border-bottom: 1px solid {t['border']};">
                        <th style="padding: 6px;">Entity Identifier</th>
                        <th style="padding: 6px;">Entity Role</th>
                        <th style="padding: 6px;">Asset Tier</th>
                        <th style="padding: 6px;">Accumulated Score</th>
                        <th style="padding: 6px;">SOC Status</th>
                        <th style="padding: 6px;">Primary Observed Stage</th>
                    </tr>
                </thead>
                <tbody>
                    {sub_rows}
                </tbody>
            </table>
            """)
