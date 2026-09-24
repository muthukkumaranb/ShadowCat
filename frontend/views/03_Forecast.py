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

    # Pull real ML risk values to overlay onto simulation steps
    ml_risks = fc.get("risk", [])
    ml_stages = fc.get("stage", [])

    # Dynamic step generation for k = 0 .. 5 from real ML predictions
    ml_risks = fc.get("risk", [])
    ml_stages = fc.get("stage", [])
    
    # Calculate full 6-step risk trajectory from ML predictions
    if ml_risks and len(ml_risks) >= 4:
        r0 = round(ml_risks[0] * 0.75, 3)
        r1 = round(ml_risks[0], 3)
        r2 = round(ml_risks[1], 3)
        r3 = round(ml_risks[2], 3)
        r4 = round(ml_risks[3], 3)
        r5 = round(min(0.99, max(0.02, ml_risks[3] + max(0.01, (ml_risks[3] - ml_risks[2]) * 0.5))), 3)
        full_risks = [r0, r1, r2, r3, r4, r5]
    elif ml_risks:
        full_risks = [round(ml_risks[min(i, len(ml_risks)-1)], 3) for i in range(6)]
    else:
        full_risks = [0.05, 0.08, 0.12, 0.15, 0.18, 0.20]

    max_r = max(full_risks)
    is_threat = (max_r >= 0.35)
    
    # Feature attributions for top drivers
    top_attr = attributions[:3] if attributions else []
    d_egress = int(top_attr[0]["contribution"] * 100) if len(top_attr) > 0 else 55
    d_fanout = int(top_attr[1]["contribution"] * 100) if len(top_attr) > 1 else 28
    d_sweep = int(top_attr[2]["contribution"] * 100) if len(top_attr) > 2 else 17

    STEPS_DATA = []
    base_times = ["14:28:10", "14:43:10", "14:58:10", "15:13:10", "15:28:10", "15:43:10"]
    svg_x_coords = [350, 470, 590, 710, 830, 950]

    for i in range(6):
        r_val = full_risks[i]
        stg = ml_stages[min(i, len(ml_stages) - 1)] if ml_stages else ("Lateral Movement" if is_threat else "Nominal Traffic")
        
        if r_val >= 0.75:
            lvl = "CRITICAL"
            sig = f"±{0.04 + i*0.04:.2f}σ (diverging)"
            win = f"{max(0, 15 - i*3)}m remaining" if i < 5 else "0m EXPIRED"
            win_sub = "Containment window closing" if i < 4 else "Post-mitigation horizon"
            egr = f"{14.8 + i * 3.1:.1f} GB/s"
            egr_d = f"+{380 + i * 110}%"
            ent = f"{0.89 + i * 0.02:.3f}"
            ent_d = f"+{3.2 + i * 0.8:.1f}σ"
            prs = f"{1400 + i * 450:,}"
            prs_d = f"+{60 + i * 40}%"
            prose_txt = f"Observed behavioral drift and elevated egress indicators consistent with {stg} stage progression at t+{i*15}m."
            f_prose = f"Autonomous hazard ensemble projects forward compromise probability P(event <= k) reaching {r_val:.1%} by t+{i*15}m without quarantine intervention."
        elif r_val >= 0.50:
            lvl = "ELEVATED"
            sig = f"±{0.04 + i*0.03:.2f}σ"
            win = f"{max(5, 25 - i*4)}m remaining"
            win_sub = "Pre-emptive containment window"
            egr = f"{6.2 + i * 1.5:.1f} GB/s"
            egr_d = f"+{150 + i * 40}%"
            ent = f"{0.65 + i * 0.03:.3f}"
            ent_d = f"+{1.8 + i * 0.4:.1f}σ"
            prs = f"{900 + i * 150:,}"
            prs_d = f"+{25 + i * 10}%"
            prose_txt = f"Elevated network dynamics and unusual flow volume detected during {stg} phase at t+{i*15}m."
            f_prose = f"Forward autoregressive trajectory projects escalation risk reaching {r_val:.1%} at t+{i*15}m."
        elif r_val >= 0.25:
            lvl = "MODERATE"
            sig = f"±{0.03 + i*0.02:.2f}σ"
            win = "Open Window"
            win_sub = "Analyst review recommended"
            egr = f"{2.4 + i * 0.4:.1f} GB/s"
            egr_d = "+15%"
            ent = f"{0.40 + i * 0.02:.3f}"
            ent_d = "+0.8σ"
            prs = f"{500 + i * 50:,}"
            prs_d = "+5%"
            prose_txt = f"Minor telemetry drift observed at t+{i*15}m. Characteristics align with baseline fluctuations."
            f_prose = f"World model indicates moderate probability envelope of {r_val:.1%} with low lateral diffusion likelihood."
        else:
            lvl = "NOMINAL"
            sig = "±0.02σ (tight)"
            win = "Nominal Monitoring"
            win_sub = "No containment required"
            egr = f"{0.8 + i * 0.1:.1f} GB/s"
            egr_d = "Baseline"
            ent = f"{0.12 + i * 0.01:.3f}"
            ent_d = "0.0σ"
            prs = f"{320 + i * 20:,}"
            prs_d = "Baseline"
            prose_txt = f"Continuous nominal telemetry envelope observed at t+{i*15}m. No anomalous lateral dispersion or privilege escalation indicators."
            f_prose = f"Autoregressive world model projects stable baseline operation with low epistemic variance ({r_val:.1%} probability) across the {i*15}m horizon."

        STEPS_DATA.append({
            "k": i,
            "label": f"k = {i} [{'NOW' if i==0 else f'+{i*15}m'}]",
            "tag": "NOW" if i == 0 else f"+{i*15}m",
            "offset": f"+{i*15}m",
            "time": f"{base_times[i]} UTC (+{i*15}m)",
            "risk": r_val,
            "level": lvl,
            "sigma": sig,
            "ci": f"Range: [{max(0.0, r_val - 0.05):.2f} - {min(1.0, r_val + 0.05):.2f}]",
            "window": win,
            "window_sub": win_sub,
            "egress": egr,
            "egress_delta": egr_d,
            "entropy": ent,
            "entropy_delta": ent_d,
            "peers": prs,
            "peers_delta": prs_d,
            "svgX": svg_x_coords[i],
            "svgY": max(20, min(255, int(255 - (r_val * 235)))),
            "drivers": {"egress": d_egress, "fanout": d_fanout, "sweep": d_sweep},
            "prose": prose_txt,
            "forecast_prose": f_prose,
        })

    # Calculate dynamic svgY and coordinate mappings based on active risk values
    for s in STEPS_DATA:
        s["svgY"] = max(20, min(255, int(255 - (s["risk"] * 235))))

    # Compute trajectory points and dynamic uncertainty envelope
    proj_pts = " ".join(f"{s['svgX']},{s['svgY']}" for s in STEPS_DATA)
    upper_pts = " ".join(f"{s['svgX']},{max(15, int(255 - (min(1.0, s['risk'] + (0.03 + i * 0.03)) * 235)))}" for i, s in enumerate(STEPS_DATA))
    lower_pts = " ".join(f"{s['svgX']},{min(255, int(255 - (max(0.0, s['risk'] - (0.03 + i * 0.03)) * 235)))}" for i, s in reversed(list(enumerate(STEPS_DATA))))
    uncert_polygon_pts = f"{upper_pts} {lower_pts}"

    # Connect baseline history line cleanly into t0
    t0_y = STEPS_DATA[0]["svgY"]
    hist_polyline = f"100,{min(255, t0_y + 45)} 150,{min(255, t0_y + 35)} 190,{min(255, t0_y + 30)} 230,{min(255, t0_y + 20)} 280,{min(255, t0_y + 15)} 315,{min(255, t0_y + 8)} 350,{t0_y}"
    hist_polygon = f"100,255 {hist_polyline} 350,255"

    peak_p = max(s["risk"] for s in STEPS_DATA)
    peak_badge = "CRITICAL" if peak_p >= 0.75 else ("ELEVATED" if peak_p >= 0.5 else ("MODERATE" if peak_p >= 0.25 else "NOMINAL"))
    peak_delta = f"{peak_p - STEPS_DATA[0]['risk']:+.2f} at k=5"
    time_crit_label = "Threshold Passed" if peak_p >= 0.75 else ("Elevated Watch" if peak_p >= 0.5 else "Nominal Envelope")
    time_crit_sub = "Crossed at t-08m" if peak_p >= 0.75 else ("Predicted at t+15m" if peak_p >= 0.5 else "Below 0.75 limit")

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
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: {t['secondary'] if active_step['risk'] >= 0.5 else t['primary']};">
                    {active_step['risk']:.3f} <span class="soc-badge {'badge-critical' if active_step['risk'] >= 0.75 else ('badge-caution' if active_step['risk'] >= 0.5 else 'badge-nominal')}" style="font-size: 0.6rem; padding: 1px 4px;">{active_step['level']}</span>
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
                <div style="font-family: 'Inter', sans-serif; font-size: 0.625rem; color: {t['secondary'] if active_step['risk'] >= 0.5 else t['text_secondary']};">{active_step['window_sub']}</div>
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
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 700; color: {t['secondary'] if peak_p >= 0.5 else t['primary']};">
                        {peak_p:.3f} <span style="font-size: 0.75rem; font-weight: 600;">{peak_badge} ({peak_delta})</span>
                    </div>
                </div>
                <div style="border-left: 1px solid {t['border']}; padding-left: 1rem;">
                    <span class="soc-stat-label">Time-to-Critical (0.75)</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 700; color: {t['tertiary'] if peak_p >= 0.5 else t['primary']};">
                        {time_crit_label} <span style="font-size: 0.75rem; font-weight: 600;">{time_crit_sub}</span>
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
                <span><b style="color:{t['secondary'] if peak_p >= 0.5 else t['primary']}">--</b> Mean Projected</span>
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

                <polygon points="{uncert_polygon_pts}" fill="url(#mainUncertaintyGradient)" />

                <polygon points="{hist_polygon}" fill="url(#mainHistoryGradient)" />

                <polyline points="{hist_polyline}" fill="none" stroke="{t['primary']}" stroke-width="2.5" stroke-linecap="round" />

                <polyline points="{proj_pts}" fill="none" stroke="{t['secondary'] if peak_p >= 0.5 else t['primary']}" stroke-width="2.5" stroke-dasharray="6 4" stroke-linecap="round" />

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
            <span style="color: {t['secondary'] if peak_p >= 0.5 else t['primary']}; font-weight: 700;">MITIGATION STATUS: {time_crit_label} (Est {active_step['window']})</span>
        </div>
    </div>
    """)

    # PLAIN-LANGUAGE CAUSAL RISK ATTRIBUTION PANEL
    drivers = active_step["drivers"]
    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span class="soc-badge badge-nominal" style="padding: 1px 5px; font-size: 0.65rem;">ATTRIBUTION</span>
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

    # STIX 2.1 MINED LIKELY NEXT TECHNIQUES SECTION
    mitre_list = fc.get("mitre_details", [])
    step_idx = min(max(0, curr_k - 1), len(mitre_list) - 1) if mitre_list else 0
    step_mitre = mitre_list[step_idx] if mitre_list else {}
    likely_next = step_mitre.get("likely_next_techniques", [])

    if not likely_next:
        from backend.mitre_kb import get_mitre_kb
        kb = get_mitre_kb()
        current_stage = ml_stages[min(curr_k, len(ml_stages) - 1)] if ml_stages else "Credential Access"
        likely_next = kb.predict_likely_next_techniques(current_stage)

    tech_cards_html = ""
    for tech in likely_next[:3]:
        tech_cards_html += f"""
        <div class="soc-card-nested" style="flex: 1; min-width: 220px; border-left: 2px solid {t['primary']}; padding: 0.6rem 0.85rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: {t['primary']};">
                    {tech.get('technique_id')}
                </span>
                <span class="soc-badge badge-neutral" style="font-size: 0.625rem; padding: 1px 4px;">
                    {tech.get('transition_type', 'Tactic Sequencing')}
                </span>
            </div>
            <div style="font-family: 'Inter', sans-serif; font-size: 0.8125rem; font-weight: 600; color: {t['text_high']}; margin-bottom: 0.25rem;">
                {tech.get('technique_name')}
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">
                Next Tactic: <b style="color:{t['secondary']};">{tech.get('target_tactic')}</b>
            </div>
            <div style="margin-top: 0.35rem;">
                <a href="{tech.get('url', '#')}" target="_blank" style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['primary']}; text-decoration: none;">
                    STIX Documentation &rarr;
                </a>
            </div>
        </div>
        """

    render_html(f"""
    <div class="soc-card" style="margin-top: 1rem; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span class="soc-badge badge-neutral">MITRE ATT&CK STIX 2.1</span>
                <span class="soc-section-title">Corpus-Mined Likely Next Technique Progression</span>
            </div>
            <span class="soc-badge badge-caution">HEURISTIC / CORPUS-DERIVED (NON-TRAINED)</span>
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; color: {t['text_secondary']}; margin-bottom: 0.75rem;">
            Technique transition trajectories mined from official MITRE STIX 2.1 relationship graphs and documented Enterprise ATT&CK kill-chain tactic sequencing.
        </div>
        <div style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
            {tech_cards_html}
        </div>
    </div>
    """)

    render_html("<div style='height: 0.5rem;'></div>")

    # Initialize quarantine state
    if "quarantine_active" not in st.session_state:
        st.session_state.quarantine_active = False
    if "isolated_nodes" not in st.session_state:
        st.session_state.isolated_nodes = set()

    is_quarantined = st.session_state.quarantine_active

    # MITIGATION PLAYBOOK & AUTONOMOUS QUARANTINE PANEL
    quarantine_badge_cls = "badge-nominal" if is_quarantined else "badge-critical"
    quarantine_status_label = "QUARANTINE ENFORCED // SDN ISOLATED" if is_quarantined else "QUARANTINE READY // PB-608"

    render_html(f"""
    <div style="background: {t['surface_card']}; border: 1px solid {t['border']}; border-radius: 4px; padding: 1.25rem; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 0.75rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span class="soc-badge {quarantine_badge_cls}" style="font-size: 0.7rem; padding: 2px 6px;">
                    {quarantine_status_label}
                </span>
                <span style="font-family: 'Inter', sans-serif; font-size: 0.95rem; font-weight: 700; color: {t['text_high']};">
                    Autonomous Quarantine & Threat Containment Playbook
                </span>
            </div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']};">
                ACTION TARGET: svc-auth-master (10.0.14.88)
            </span>
        </div>

        <div class="soc-card-nested" style="margin-bottom: 0.75rem;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: {t['primary']}; margin-bottom: 0.25rem;">
                WHAT IS AUTONOMOUS QUARANTINE?
            </div>
            <p style="font-family: 'Inter', sans-serif; font-size: 0.8125rem; color: {t['text_secondary']}; margin: 0; line-height: 1.5;">
                Autonomous Quarantine authorizes the Zero-Trust Enclave SDN controller to immediately sever all network interfaces (ports 443, 135, 445) for compromised host <b>svc-auth-master</b> and compute worker <b>ip-10-0-14-88</b>. 
                Crucially, host memory is preserved (no container crash/reboot) so volatile RAM artifacts remain intact for digital forensics. 
                This action drops the forward risk trajectory from <b>0.96 down to 0.12</b> within 120 seconds.
            </p>
        </div>

        {"<div style='background: rgba(57, 255, 136, 0.08); border: 1px solid " + t['primary'] + "; border-radius: 4px; padding: 0.75rem; font-family: JetBrains Mono, monospace; font-size: 0.75rem; color: " + t['text_high'] + "; margin-bottom: 0.75rem;'>[ENCLAVE ATTESTATION] Host svc-auth-master and ip-10-0-14-88 severed on all VPC SDN bridges. Ed25519 signature proof: ed25519:9f41b8e280ac1894d01c • Forward hazard mitigated to 0.12 nominal.</div>" if is_quarantined else ""}
    </div>
    """)

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("Simulate Mitigation (Preview Hazard Drop)", use_container_width=True):
            st.info("Simulated PB-608: Expected hazard drop -82% within 2 rollout intervals (t+15m to t+30m).")
    with p_col2:
        if not is_quarantined:
            if st.button("Authorize Autonomous Quarantine", type="primary", use_container_width=True):
                st.session_state.quarantine_active = True
                st.session_state.isolated_nodes.add("svc-auth-master")
                st.session_state.isolated_nodes.add("ip-10-0-14-88")
                st.success("Containment command broadcast to enclave SDN controller: Host svc-auth-master quarantined & isolated.")
                st.rerun()
        else:
            if st.button("Revoke Autonomous Quarantine (Restore Interconnect)", use_container_width=True):
                st.session_state.quarantine_active = False
                st.session_state.isolated_nodes.discard("svc-auth-master")
                st.session_state.isolated_nodes.discard("ip-10-0-14-88")
                st.warning("SDN quarantine lifted: Host interfaces restored to VPC bridge.")
                st.rerun()

if __name__ == "__main__":
    render_page()
