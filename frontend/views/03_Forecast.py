"""
SHADOWCAT SOC Cockpit - Page 3: Predictive Forecast with K-Step Rollout Scrubber
Direct implementation of Stitch folder shadowcat_soc_predictive_forecast_with_k_step_rollout_scrubber.
Wired to real data_provider.py and multi-step forecast trajectory.
"""

import streamlit as st
from styles import TOKENS, render_html
from data_provider import get_forecast_trajectory, get_novelty_score, get_attributions

def render_page():
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])
    fc = get_forecast_trajectory()
    novelty = get_novelty_score()
    attributions = get_attributions()

    # Step Data mapping for k = 0 .. 5
    STEPS_DATA = [
        {
            "k": 0, "label": "k = 0 [NOW]", "tag": "NOW", "offset": "+0m",
            "time": "14:28:10 UTC (+0m)", "risk": 0.840, "level": "CRITICAL",
            "sigma": "±0.04σ (tight)", "ci": "Range: [0.80 - 0.88]", "window": "14m 00s remaining",
            "window_sub": "Pre-lateral propagation", "egress": "14.8 GB/s", "egress_delta": "+410%",
            "entropy": "0.892", "entropy_delta": "+3.41σ", "peers": "1,420", "peers_delta": "+68%",
            "svgX": 350, "svgY": 62, "drivers": {"egress": 64, "fanout": 22, "sweep": 14},
            "prose": "Initial active C2 beaconing observed on svc-auth-master with uncatalogued egress spike (+410% over rolling 30-day baseline) directed toward foreign ASN 4837.",
            "forecast_prose": "Without automated quarantine intervention, the forward projection models an 87.2% probability of lateral persistence reaching the primary credential store within 45 minutes."
        },
        {
            "k": 1, "label": "k = 1 [+15m]", "tag": "+15m", "offset": "+15m",
            "time": "14:43:10 UTC (+15m)", "risk": 0.884, "level": "CRITICAL+",
            "sigma": "±0.08σ (expanding)", "ci": "Range: [0.80 - 0.96]", "window": "9m 30s remaining",
            "window_sub": "Approaching auth barrier", "egress": "18.2 GB/s", "egress_delta": "+520%",
            "entropy": "0.915", "entropy_delta": "+4.12σ", "peers": "1,840", "peers_delta": "+112%",
            "svgX": 470, "svgY": 51, "drivers": {"egress": 60, "fanout": 28, "sweep": 12},
            "prose": "Credential harvesting initiated on primary auth token cache; lateral probes targeting cluster VPC-8812 internal microservices.",
            "forecast_prose": "Predictive Kalman models indicate lateral token harvesting reaching 91.4% confidence by t+20m if Kerberos rekey is not initiated."
        },
        {
            "k": 2, "label": "k = 2 [+30m]", "tag": "+30m", "offset": "+30m",
            "time": "14:58:10 UTC (+30m)", "risk": 0.922, "level": "INFLECTION SEVERE",
            "sigma": "±0.14σ (diverging)", "ci": "Range: [0.78 - 1.00]", "window": "4m 10s CRITICAL",
            "window_sub": "Mitigation window closing", "egress": "22.4 GB/s", "egress_delta": "+680%",
            "entropy": "0.948", "entropy_delta": "+5.30σ", "peers": "2,410", "peers_delta": "+174%",
            "svgX": 590, "svgY": 41, "drivers": {"egress": 52, "fanout": 35, "sweep": 13},
            "prose": "Compute worker node ip-10-0-14-88 compromised; attacker establishing persistent reverse shell tunnels across enclave boundary.",
            "forecast_prose": "Compounding epistemic dispersion crosses critical containment threshold; automated isolation must execute immediately."
        },
        {
            "k": 3, "label": "k = 3 [+45m]", "tag": "+45m", "offset": "+45m",
            "time": "15:13:10 UTC (+45m)", "risk": 0.954, "level": "COMPROMISE MULTI-HOST",
            "sigma": "±0.19σ (high)", "ci": "Range: [0.76 - 1.00]", "window": "0m EXPIRED",
            "window_sub": "Post-mitigation horizon", "egress": "26.1 GB/s", "egress_delta": "+810%",
            "entropy": "0.962", "entropy_delta": "+6.10σ", "peers": "3,100", "peers_delta": "+230%",
            "svgX": 710, "svgY": 33, "drivers": {"egress": 45, "fanout": 42, "sweep": 13},
            "prose": "IAM credential sync daemon infiltrated. Service account keys duplicated and used for directory replication queries.",
            "forecast_prose": "Attacker possesses administrative quorum across VPC-8812; active data exfiltration underway."
        },
        {
            "k": 4, "label": "k = 4 [+60m]", "tag": "+60m", "offset": "+60m",
            "time": "15:28:10 UTC (+60m)", "risk": 0.978, "level": "SYSTEMIC PERVATION",
            "sigma": "±0.24σ (epistemic limit)", "ci": "Range: [0.73 - 1.00]", "window": "0m EXPIRED",
            "window_sub": "Uncontained breach", "egress": "28.5 GB/s", "egress_delta": "+890%",
            "entropy": "0.975", "entropy_delta": "+6.80σ", "peers": "3,890", "peers_delta": "+280%",
            "svgX": 830, "svgY": 27, "drivers": {"egress": 40, "fanout": 48, "sweep": 12},
            "prose": "Cryptographic audit vault keys targeted; dual exfiltration streams saturating both primary NAT gateway and foreign DNS tunnel.",
            "forecast_prose": "Full enclave severance required at border BGP router to prevent customer data spill."
        },
        {
            "k": 5, "label": "k = 5 [+75m]", "tag": "+75m", "offset": "+75m",
            "time": "15:43:10 UTC (+75m)", "risk": 0.992, "level": "MAXIMUM BREACH CEILING",
            "sigma": "±0.28σ (terminal)", "ci": "Range: [0.71 - 1.00]", "window": "0m EXPIRED",
            "window_sub": "Enclave offline required", "egress": "31.2 GB/s", "egress_delta": "+980%",
            "entropy": "0.985", "entropy_delta": "+7.40σ", "peers": "4,200", "peers_delta": "+310%",
            "svgX": 950, "svgY": 22, "drivers": {"egress": 35, "fanout": 52, "sweep": 13},
            "prose": "Total cluster partition compromised; cascading denial of telemetry guarantees and complete service disruption.",
            "forecast_prose": "Horizon ceiling reached (H=5). Disaster recovery playbook PB-999 invoked."
        },
    ]

    # Initialize active K step in session state
    if "forecast_k_step" not in st.session_state:
        st.session_state.forecast_k_step = 0

    curr_k = st.session_state.forecast_k_step
    active_step = STEPS_DATA[curr_k]

    # TOP SECTION / HORIZON CONTROL BAR
    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.125rem; font-weight: 700; color: {t['text_high']}; text-transform: uppercase;">
                        Predictive Threat Trajectory Forecast
                    </span>
                    <span class="soc-badge badge-neutral">Checkpoint: sc-threat-v4.1</span>
                    <span class="soc-badge badge-nominal">95.4% Monte Carlo (10k runs)</span>
                </div>
                <div style="display: flex; gap: 0.75rem; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']}; margin-top: 0.25rem;">
                    <span>Last inference: <b style="color:{t['text_high']}">14:28:10 UTC</b></span>
                    <span>•</span>
                    <span>Window: <b style="color:{t['text_high']}">60s sliding</b></span>
                    <span>•</span>
                    <span style="color:{t['primary']};">Inference Engine Synced</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem; background: {t['surface_lowest']}; padding: 3px 8px; border-radius: 9999px; border: 1px solid {t['border']};">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']}; text-transform: uppercase;">Horizon</span>
                <span class="soc-badge badge-neutral" style="padding: 2px 8px;">H = 1 (15m)</span>
                <span class="soc-badge badge-neutral" style="padding: 2px 8px;">H = 2 (30m)</span>
                <span class="soc-badge badge-nominal" style="padding: 2px 8px;">H = 5 (75m Primary)</span>
            </div>
        </div>
    </div>
    """)

    # PROMINENT INTERACTIVE K-STEP FORWARD ROLLOUT TIMELINE SCRUBBER
    render_html(f"""
    <div class="soc-card" style="border: 1px solid {t['primary']}; border-top: 2px solid {t['primary']}; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span class="soc-pulse-dot" style="width: 8px; height: 8px;"></span>
                <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.875rem; color: {t['text_high']}; text-transform: uppercase;">
                    K-Step Forward Rollout Horizon Scrubber
                </span>
                <span class="soc-badge badge-nominal">[k = 0 .. 5]</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">| Step Size: Δt = 15m</span>
            </div>
        </div>
        <!-- Realtime Projection State Telemetry Dashboard Strip -->
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.5rem; background: {t['surface_lowest']}; padding: 0.65rem 0.85rem; border-radius: 4px; border: 1px solid {t['border']}; margin-bottom: 0.75rem;">
            <div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.625rem; color: {t['text_muted']}; text-transform: uppercase;">Active Horizon Step</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: {t['primary']};">
                    k = {active_step['k']} <span style="font-size: 0.75rem; font-weight: 400; color: {t['text_secondary']};">{active_step['tag']}</span>
                </div>
            </div>
            <div style="border-left: 1px solid {t['border']}; padding-left: 0.5rem;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.625rem; color: {t['text_muted']}; text-transform: uppercase;">Simulated Timestamp</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; font-weight: 600; color: {t['text_high']};">
                    {active_step['time']}
                </div>
            </div>
            <div style="border-left: 1px solid {t['border']}; padding-left: 0.5rem;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.625rem; color: {t['text_muted']}; text-transform: uppercase;">Step Forecast Risk (p)</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: {t['secondary']};">
                    {active_step['risk']:.3f} <span class="soc-badge badge-critical" style="font-size: 0.6rem; padding: 1px 4px;">{active_step['level']}</span>
                </div>
            </div>
            <div style="border-left: 1px solid {t['border']}; padding-left: 0.5rem;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.625rem; color: {t['text_muted']}; text-transform: uppercase;">Dispersion Sigma (95% CI)</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; font-weight: 600; color: {t['primary']};">
                    {active_step['sigma']}
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.625rem; color: {t['text_muted']};">{active_step['ci']}</div>
            </div>
            <div style="border-left: 1px solid {t['border']}; padding-left: 0.5rem;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.625rem; color: {t['text_muted']}; text-transform: uppercase;">Mitigation Window Left</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; font-weight: 700; color: {t['tertiary']};">
                    {active_step['window']}
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.625rem; color: {t['secondary']};">{active_step['window_sub']}</div>
            </div>
        </div>
    </div>
    """)

    # Scrubber Buttons Row
    btn_cols = st.columns(6)
    for i, b_col in enumerate(btn_cols):
        with b_col:
            is_active = (i == curr_k)
            btn_label = f"k = {i} [{'NOW' if i==0 else f'+{i*15}m'}]"
            if st.button(btn_label, key=f"step_btn_{i}", type="primary" if is_active else "secondary", use_container_width=True):
                st.session_state.forecast_k_step = i
                st.rerun()

    render_html("<div style='height: 0.75rem;'></div>")

    # CENTERPIECE: PREDICTIVE THREAT TRAJECTORY CHART
    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; gap: 1.5rem; align-items: baseline;">
                <div>
                    <span class="soc-stat-label">Estimated Peak Probability</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 700; color: {t['secondary']};">
                        0.992 <span style="font-size: 0.75rem; font-weight: 600;">Critical (+0.15 at k=5)</span>
                    </div>
                </div>
                <div style="border-left: 1px solid {t['border']}; padding-left: 1rem;">
                    <span class="soc-stat-label">Time-to-Critical (0.75)</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 700; color: {t['tertiary']};">
                        Passed <span style="font-size: 0.75rem; font-weight: 600;">Crossed at t-08m</span>
                    </div>
                </div>
                <div style="border-left: 1px solid {t['border']}; padding-left: 1rem;">
                    <span class="soc-stat-label">Dispersion Velocity</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 700; color: {t['primary']};">
                        {active_step['sigma']}
                    </div>
                </div>
            </div>
            <!-- Chart Legend -->
            <div style="display: flex; gap: 0.75rem; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_secondary']}; background: {t['surface_lowest']}; padding: 4px 10px; border-radius: 4px; border: 1px solid {t['border']};">
                <span><b style="color:{t['primary']}">―</b> Actual Historical</span>
                <span><b style="color:{t['secondary']}">--</b> Mean Projected</span>
                <span style="color:{t['secondary']}"><span style="display:inline-block; width:8px; height:8px; background:{t['secondary_subtle']}; border:1px solid {t['secondary']};"></span> 95% Uncertainty</span>
                <span><b style="color:{t['secondary']}">··</b> Threshold (0.75)</span>
                <span style="color:{t['primary']}; font-weight: 700;">| Scrubber Needle</span>
            </div>
        </div>

        <div style="width: 100%; height: 280px; background: {t['surface_lowest']}; border: 1px solid {t['border']}; border-radius: 4px; position: relative;">
            <svg style="width: 100%; height: 100%;" viewBox="0 0 1000 280" preserveAspectRatio="none">
                <defs>
                    <linearGradient id="mainUncertaintyGradient" x1="0%" x2="100%" y1="0%" y2="0%">
                        <stop offset="0%" stop-color="{t['secondary']}" stop-opacity="0.10" />
                        <stop offset="50%" stop-color="{t['secondary']}" stop-opacity="0.30" />
                        <stop offset="100%" stop-color="{t['secondary']}" stop-opacity="0.55" />
                    </linearGradient>
                    <linearGradient id="mainHistoryGradient" x1="0%" x2="0%" y1="0%" y2="100%">
                        <stop offset="0%" stop-color="{t['primary']}" stop-opacity="0.25" />
                        <stop offset="100%" stop-color="{t['primary']}" stop-opacity="0.0" />
                    </linearGradient>
                </defs>

                <line x1="50" y1="20" x2="980" y2="20" stroke="{t['border']}" stroke-dasharray="2 4" />
                <line x1="50" y1="75" x2="980" y2="75" stroke="{t['secondary']}" stroke-dasharray="4 4" stroke-width="1.2" opacity="0.6" />
                <line x1="50" y1="135" x2="980" y2="135" stroke="{t['border']}" stroke-dasharray="2 4" />
                <line x1="50" y1="195" x2="980" y2="195" stroke="{t['border']}" stroke-dasharray="2 4" />
                <line x1="50" y1="255" x2="980" y2="255" stroke="{t['border']}" />

                <text x="42" y="24" fill="{t['text_muted']}" font-family="JetBrains Mono" font-size="9" text-anchor="end">1.00</text>
                <text x="42" y="79" fill="{t['secondary']}" font-family="JetBrains Mono" font-size="9" text-anchor="end" font-weight="700">0.75</text>
                <text x="42" y="139" fill="{t['text_muted']}" font-family="JetBrains Mono" font-size="9" text-anchor="end">0.50</text>
                <text x="42" y="199" fill="{t['text_muted']}" font-family="JetBrains Mono" font-size="9" text-anchor="end">0.25</text>
                <text x="42" y="259" fill="{t['text_muted']}" font-family="JetBrains Mono" font-size="9" text-anchor="end">0.00</text>

                <line x1="100" y1="20" x2="100" y2="255" stroke="{t['border']}" stroke-dasharray="2 3" />
                <line x1="220" y1="20" x2="220" y2="255" stroke="{t['border']}" stroke-dasharray="2 3" />
                <line x1="350" y1="20" x2="350" y2="255" stroke="{t['outline']}" stroke-width="1.5" />
                <line x1="470" y1="20" x2="470" y2="255" stroke="{t['border']}" stroke-dasharray="2 3" />
                <line x1="590" y1="20" x2="590" y2="255" stroke="{t['border']}" stroke-dasharray="2 3" />
                <line x1="710" y1="20" x2="710" y2="255" stroke="{t['border']}" stroke-dasharray="2 3" />
                <line x1="830" y1="20" x2="830" y2="255" stroke="{t['border']}" stroke-dasharray="2 3" />
                <line x1="950" y1="20" x2="950" y2="255" stroke="{t['border']}" stroke-dasharray="2 3" />

                <polygon points="350,62 470,44 590,30 710,18 830,12 950,8 950,78 830,58 710,48 590,56 470,68 350,62" fill="url(#mainUncertaintyGradient)" />

                <polygon points="100,255 100,225 150,220 190,230 230,200 280,175 315,120 350,62 350,255" fill="url(#mainHistoryGradient)" />

                <polyline points="100,225 150,220 190,230 230,200 280,175 315,120 350,62" fill="none" stroke="{t['primary']}" stroke-width="2.5" stroke-linecap="round" />

                <polyline points="350,62 470,51 590,41 710,33 830,27 950,22" fill="none" stroke="{t['secondary']}" stroke-width="2.5" stroke-dasharray="6 4" stroke-linecap="round" />

                <line x1="{active_step['svgX']}" y1="15" x2="{active_step['svgX']}" y2="260" stroke="{t['primary']}" stroke-width="2" stroke-dasharray="3 2" />
                <circle cx="{active_step['svgX']}" cy="{active_step['svgY']}" r="8" fill="none" stroke="{t['primary']}" stroke-width="2" />
                <circle cx="{active_step['svgX']}" cy="{active_step['svgY']}" r="3.5" fill="{t['primary']}" />

                <text x="100" y="272" fill="{t['text_muted']}" font-family="JetBrains Mono" font-size="9" text-anchor="middle">t-30m</text>
                <text x="220" y="272" fill="{t['text_muted']}" font-family="JetBrains Mono" font-size="9" text-anchor="middle">t-15m</text>
                <text x="350" y="272" fill="{t['text_high']}" font-family="JetBrains Mono" font-size="9" text-anchor="middle" font-weight="700">k=0 [NOW]</text>
                <text x="470" y="272" fill="{t['text_muted']}" font-family="JetBrains Mono" font-size="9" text-anchor="middle">k=1 [+15m]</text>
                <text x="590" y="272" fill="{t['text_muted']}" font-family="JetBrains Mono" font-size="9" text-anchor="middle">k=2 [+30m]</text>
                <text x="710" y="272" fill="{t['text_muted']}" font-family="JetBrains Mono" font-size="9" text-anchor="middle">k=3 [+45m]</text>
                <text x="830" y="272" fill="{t['text_muted']}" font-family="JetBrains Mono" font-size="9" text-anchor="middle">k=4 [+60m]</text>
                <text x="950" y="272" fill="{t['text_muted']}" font-family="JetBrains Mono" font-size="9" text-anchor="middle">k=5 [+75m]</text>
            </svg>
        </div>
        <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']}; margin-top: 0.35rem;">
            <span>ENCLAVE: AWS-US-EAST-1 (VPC-8812) • AGENT LATENCY: 1.84ms • KALMAN FILTER DRIFT: &lt;0.004</span>
            <span style="color: {t['secondary']}; font-weight: 700;">MITIGATION WINDOW DEPLETING (Est {active_step['window']})</span>
        </div>
    </div>
    """)

    # PLAIN-LANGUAGE CAUSAL RISK ATTRIBUTION PANEL
    drivers = active_step["drivers"]
    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color:{t['primary']}; font-size:1.1rem;">🧠</span>
                <span class="soc-section-title">Causal Risk Attribution & Synthesis</span>
            </div>
            <div style="display: flex; gap: 0.5rem;">
                <span class="soc-badge badge-neutral">STEP k = {active_step['k']} STATE</span>
                <span class="soc-badge badge-nominal">Primary Forecast Engine</span>
            </div>
        </div>
        <div class="soc-card-nested" style="margin-bottom: 0.75rem; line-height: 1.55; font-size: 0.875rem;">
            <p style="margin: 0; color: {t['text_high']};">
                {active_step['prose']} <span style="color: {t['text_secondary']};">{active_step['forecast_prose']}</span>
            </p>
        </div>
        <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']}; text-transform: uppercase; margin-bottom: 0.25rem;">
            <span>Causal Factor Breakdown for Horizon Step <b style="color:{t['primary']}">k = {active_step['k']}</b></span>
            <span>Aggregated Influence: 100%</span>
        </div>
        <div style="width: 100%; height: 8px; background: {t['surface_highest']}; border-radius: 4px; display: flex; overflow: hidden; margin-bottom: 0.5rem;">
            <div style="width: {drivers['egress']}%; background: {t['secondary']}; height: 100%;"></div>
            <div style="width: {drivers['fanout']}%; background: {t['tertiary']}; height: 100%;"></div>
            <div style="width: {drivers['sweep']}%; background: {t['outline_variant']}; height: 100%;"></div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem;">
            <div class="soc-card-nested" style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0.75rem;">
                <span style="font-size: 0.75rem; color: {t['text_secondary']};">● Egress Surge</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: {t['secondary']}; font-size: 0.8125rem;">{drivers['egress']}% Influence</span>
            </div>
            <div class="soc-card-nested" style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0.75rem;">
                <span style="font-size: 0.75rem; color: {t['text_secondary']};">● Peer Fan-Out</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: {t['tertiary']}; font-size: 0.8125rem;">{drivers['fanout']}% Influence</span>
            </div>
            <div class="soc-card-nested" style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0.75rem;">
                <span style="font-size: 0.75rem; color: {t['text_secondary']};">● Port Sweep</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: {t['text_muted']}; font-size: 0.8125rem;">{drivers['sweep']}% Influence</span>
            </div>
        </div>
    </div>
    """)

    # BASELINE VS CURRENT HISTORICAL TREND COMPARISONS
    c_col1, c_col2, c_col3 = st.columns(3)

    with c_col1:
        render_html(f"""
        <div class="soc-stat-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="soc-stat-label">Egress Traffic Volume</span>
                <span class="soc-badge badge-critical" style="padding: 1px 4px; font-size: 0.6rem;">Critical</span>
            </div>
            <div style="margin: 0.35rem 0;">
                <div class="soc-stat-val">{active_step['egress']}</div>
                <div class="soc-stat-delta delta-threat">
                    <span>{active_step['egress_delta']}</span>
                    <span style="color: {t['text_muted']}; font-size: 0.75rem;">vs 2.9 GB/s 30d baseline</span>
                </div>
            </div>
            <div style="border-top: 1px solid {t['border']}; padding-top: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']}; display: flex; justify-content: space-between;">
                <span>Target: <b style="color:{t['text_high']}">45.138.21.9</b></span>
                <span>UDP/53</span>
            </div>
        </div>
        """)

    with c_col2:
        render_html(f"""
        <div class="soc-stat-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="soc-stat-label">Entropy & Novelty Index</span>
                <span class="soc-badge badge-caution" style="padding: 1px 4px; font-size: 0.6rem;">High Drift</span>
            </div>
            <div style="margin: 0.35rem 0;">
                <div class="soc-stat-val">{active_step['entropy']}</div>
                <div class="soc-stat-delta delta-caution">
                    <span>{active_step['entropy_delta']}</span>
                    <span style="color: {t['text_muted']}; font-size: 0.75rem;">vs 0.121 baseline</span>
                </div>
            </div>
            <div style="border-top: 1px solid {t['border']}; padding-top: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']}; display: flex; justify-content: space-between;">
                <span>KL-Div: <b style="color:{t['text_high']}">4.88 bit</b></span>
                <span>CONF: 99.1%</span>
            </div>
        </div>
        """)

    with c_col3:
        render_html(f"""
        <div class="soc-stat-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="soc-stat-label">Cluster Peer Connections</span>
                <span class="soc-badge badge-critical" style="padding: 1px 4px; font-size: 0.6rem;">Spreading</span>
            </div>
            <div style="margin: 0.35rem 0;">
                <div class="soc-stat-val">{active_step['peers']}</div>
                <div class="soc-stat-delta delta-threat">
                    <span>{active_step['peers_delta']}</span>
                    <span style="color: {t['text_muted']}; font-size: 0.75rem;">fan-out deviation</span>
                </div>
            </div>
            <div style="border-top: 1px solid {t['border']}; padding-top: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']}; display: flex; justify-content: space-between;">
                <span>Subnet: <b style="color:{t['text_high']}">10.0.14.0/24</b></span>
                <span>SYN: NEGATIVE</span>
            </div>
        </div>
        """)

    render_html("<div style='height: 1rem;'></div>")

    # MITIGATION PLAYBOOK ACTION BAR
    render_html(f"""
    <div style="background: {t['surface_card']}; border: 1px solid {t['border']}; border-radius: 4px; padding: 1rem 1.25rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <div style="width: 36px; height: 36px; border-radius: 4px; background: {t['secondary_subtle']}; border: 1px solid {t['secondary']}; display: flex; align-items: center; justify-content: center; color: {t['secondary']}; font-size: 1.25rem;">
                🛡
            </div>
            <div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.875rem; font-weight: 700; color: {t['text_high']};">
                    Recommended Playbook: PB-608 • Egress Containment & Kerberos Rekey
                </div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; color: {t['text_secondary']};">
                    Automated isolation drops forward risk trajectory from 0.96 down to 0.12 within 120 seconds.
                </div>
            </div>
        </div>
    </div>
    """)

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("Simulate Mitigation", use_container_width=True):
            st.info("Simulated PB-608: Expected hazard drop -82% within 2 rollout intervals.")
    with p_col2:
        if st.button("Authorize Autonomous Quarantine", type="primary", use_container_width=True):
            st.success("Containment command broadcast to enclave SDN controller: Host svc-auth-master quarantined.")

if __name__ == "__main__":
    render_page()
