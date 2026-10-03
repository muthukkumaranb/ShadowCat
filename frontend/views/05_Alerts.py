"""
SHADOWCAT SOC Cockpit - Page 5: Alerts
Alerts are only those produced by backend.predict: the onset alert (onset probability >= 0.5)
and the detection alert (detection probability >= 0.5) of the active prediction, plus onset
alerts previously recorded in the local reports database. No default or scripted alerts.
"""

import html

import pandas as pd
import streamlit as st
from styles import TOKENS, render_html
from data_provider import (
    get_active_source,
    get_detection_probability,
    get_flagged_flows,
    get_forecast_trajectory,
    get_report_history,
)


def render_page():
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])
    fc = get_forecast_trajectory()
    p_onset = fc.get("onset_probability")
    p_det = get_detection_probability()
    thr = fc.get("alert_threshold", 0.5)

    current = []
    if p_onset is not None and fc.get("onset_alert"):
        current.append(("Onset alert", f"{fc.get('onset_probability_label')} = {p_onset:.3f} >= {thr}"))
    if p_det is not None and p_det >= 0.5:
        current.append(("Detection alert", f"P(attack in current window) = {p_det:.3f} >= 0.5"))

    rows = "".join(f"<tr><td><b>{a}</b></td><td>{html.escape(b)}</td></tr>" for a, b in current)
    none_txt = (f"No alerts for this window (onset {'—' if p_onset is None else f'{p_onset:.3f}'}, "
                f"detection {'—' if p_det is None else f'{p_det:.3f}'}; threshold {thr}).")
    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1rem;">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: {t['text_high']}; text-transform: uppercase;">Alerts</span>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem; color: {t['text_secondary']}; margin-top: 0.3rem;">
            Input: <b>{html.escape(get_active_source())}</b> &bull; window <code>{html.escape(str(fc.get('window_id', '—')))}</code>
        </div>
        <table style="margin-top: 0.6rem;">
            {rows or f"<tr><td>{none_txt}</td></tr>"}
        </table>
    </div>
    """)

    flows = get_flagged_flows()
    if flows:
        render_html("<div class='soc-section-title'>Flows of the last window ranked by the onset LR stage</div>")
        st.caption("Score = sum over the flow's own '<column>_mean' features of coefficient x scaled value, averaged over "
                   "the 37 onset folds (logit units; used only for ranking). dataset_label is the raw CSV label, not a model output.")
        st.dataframe(pd.DataFrame(flows), hide_index=True, width="stretch")
    else:
        st.caption("No flow ranking: the active input is UCS windows (no individual flows).")

    history = get_report_history(record_type="alert", limit=50)
    render_html("<div class='soc-section-title' style='margin-top: 1rem;'>Recorded onset alerts (local reports database)</div>")
    if history:
        st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")
    else:
        st.caption("No alerts recorded.")


if __name__ == "__main__":
    render_page()
