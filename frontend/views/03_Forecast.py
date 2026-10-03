"""
SHADOWCAT SOC Cockpit - Page 3: Onset Forecast
Shows only model outputs from backend.predict for the active inference result:
the stacked onset probability P(attack within the next 5 minutes), the stacked
detection probability for the current window, and the split-conformal interval.
"""

import html

import streamlit as st
from styles import TOKENS, render_html
from data_provider import get_forecast_trajectory, get_detection_probability


def _fmt(p):
    return "—" if p is None else f"{p:.3f}"


def render_page():
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])
    fc = get_forecast_trajectory()
    p_onset = fc.get("onset_probability")

    if p_onset is None:
        render_html(f"""
        <div class="soc-card">
            <span class="soc-section-title">Onset Forecast</span>
            <p style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: {t['text_secondary']};">
                No inference result is loaded. Open the Telemetry Ingestion page, load a telemetry file and run inference.
            </p>
        </div>
        """)
        return

    label = fc.get("onset_probability_label", "P(attack within the next 5 minutes)")
    threshold = fc.get("alert_threshold")
    alert = bool(fc.get("onset_alert"))
    p_det = get_detection_probability()
    ci = fc.get("conformal_interval") or [None, None]
    coverage = fc.get("conformal_coverage")
    n_cal = fc.get("conformal_sample_size")
    window_id = html.escape(str(fc.get("window_id", "")))
    model_desc = html.escape(str(fc.get("onset_model", "")))
    alert_color = t["secondary"] if alert else t["primary"]
    alert_text = "ALERT" if alert else "NO ALERT"

    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.125rem; font-weight: 700; color: {t['text_high']}; text-transform: uppercase;">
                Onset Forecast
            </span>
            <span class="soc-badge badge-neutral">Window: {window_id}</span>
            <span class="soc-badge badge-neutral">1-minute windows, 30-window lookback</span>
        </div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; color: {t['text_muted']}; margin-top: 0.35rem;">
            Model: {model_desc}
        </div>
    </div>
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        render_html(f"""
        <div class="soc-stat-card" style="border-top: 2px solid {alert_color};">
            <span class="soc-stat-label">{html.escape(label)}</span>
            <div class="soc-stat-val" style="color: {alert_color};">{_fmt(p_onset)}</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: {t['text_muted']};">
                {alert_text} (threshold {_fmt(threshold)})
            </div>
        </div>
        """)
    with c2:
        render_html(f"""
        <div class="soc-stat-card">
            <span class="soc-stat-label">P(attack in current window)</span>
            <div class="soc-stat-val">{_fmt(p_det)}</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: {t['text_muted']};">
                Stacked detection ensemble (target: label_binary)
            </div>
        </div>
        """)
    with c3:
        cov_txt = f"{int(round(coverage * 100))}%" if coverage is not None else "—"
        render_html(f"""
        <div class="soc-stat-card">
            <span class="soc-stat-label">{cov_txt} split-conformal interval (onset)</span>
            <div class="soc-stat-val">[{_fmt(ci[0])}, {_fmt(ci[1])}]</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: {t['text_muted']};">
                Calibrated on {n_cal if n_cal is not None else '—'} pooled validation predictions
            </div>
        </div>
        """)

    render_html(f"""
    <div class="soc-card" style="margin-top: 1rem;">
        <span class="soc-section-title">What this number is</span>
        <ul style="font-family: 'Inter', sans-serif; font-size: 0.8rem; color: {t['text_secondary']}; line-height: 1.6; margin: 0.4rem 0 0 1rem;">
            <li>The onset model is trained to predict whether any of the next five 1-minute windows (t+1 .. t+5) contains attack traffic. It gives one probability, not a separate value per minute.</li>
            <li>Most onset-positive windows in CSE-CIC-IDS2018 are already inside an attack; few are true precursors. A high value usually means an attack is in progress, not that one is about to start.</li>
            <li>No per-minute horizon forecast is shown. The world model's rollout does not beat a persistence baseline at any K (H* = 0).</li>
        </ul>
    </div>
    """)


if __name__ == "__main__":
    render_page()
