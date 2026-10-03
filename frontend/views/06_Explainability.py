"""
SHADOWCAT SOC Cockpit - Page 6: Explainability
Integrated Gradients on the stacked ONSET logit (the risk number shown on the Forecast page),
computed by backend/temporal_attribution.py inside backend.predict for the active prediction,
plus the bounded-perturbation counterfactual. Every value comes from the prediction payload.
"""

import html

import pandas as pd
import streamlit as st
from styles import TOKENS, render_html
from data_provider import (
    get_attributions,
    get_counterfactual,
    get_forecast_trajectory,
    get_temporal_attributions,
)


def render_page():
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])
    fc = get_forecast_trajectory()
    attributions = get_attributions()
    ig = get_temporal_attributions()
    p = fc.get("onset_probability")

    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1rem;">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: {t['text_high']}; text-transform: uppercase;">
            Explainability: Integrated Gradients
        </span>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem; color: {t['text_secondary']}; margin-top: 0.3rem; line-height: 1.5;">
            Attributes the {html.escape(str(ig.get('target', 'stacked onset logit')))} to the 406 inputs in each of the 30 lookback windows.
            Baseline: {html.escape(str(ig.get('baseline', '—')))}. Current {html.escape(fc.get('onset_probability_label', 'onset probability'))}:
            <b>{'—' if p is None else f'{p:.3f}'}</b>.
        </div>
    </div>
    """)

    if ig.get("error"):
        st.error(f"Integrated Gradients failed for this input: {ig['error']}")

    # 1. Top input features
    rows = "".join(
        f"<tr><td><code>{html.escape(a['feature'])}</code></td><td>{html.escape(str(a.get('category')))}</td>"
        f"<td>{a['contribution']:.1%}</td><td>{a.get('signed_attribution', 0):+.4f}</td></tr>"
        for a in attributions
    )
    render_html(f"""
    <div class="soc-card">
        <div class="soc-section-title">Top input features (presence masks excluded)</div>
        <table style="margin-top: 0.4rem;">
            <thead><tr><th>Feature</th><th>Group</th><th>Share of total |attribution|</th><th>Signed attribution (logit units, summed over 30 windows)</th></tr></thead>
            <tbody>{rows or '<tr><td colspan="4">No attribution computed</td></tr>'}</tbody>
        </table>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.72rem; color: {t['text_muted']}; margin-top: 0.4rem;">
            Positive signed attribution raises the onset logit relative to the baseline; negative lowers it.
            Sum of all attributions: {ig.get('attribution_sum', '—')}.
        </div>
    </div>
    """)

    # 2. Attribution over the lookback windows
    steps = ig.get("timestep_attributions") or []
    if steps:
        df = pd.DataFrame(steps)[["step_offset", "importance"]].rename(
            columns={"step_offset": "window offset (0 = current)", "importance": "share of |attribution|"})
        render_html("<div class='soc-section-title' style='margin-top: 1rem;'>Share of |attribution| per lookback window</div>")
        st.bar_chart(df, x="window offset (0 = current)", y="share of |attribution|")
    events = ig.get("top_temporal_events") or []
    if events:
        render_html("<div class='soc-section-title' style='margin-top: 0.5rem;'>Largest single (window, feature) attributions</div>")
        st.dataframe(pd.DataFrame(events), hide_index=True, width="stretch")

    # 3. Counterfactual (bounded perturbation search on the onset probability)
    cf = get_counterfactual()
    rows = "".join(
        f"<tr><td><code>{html.escape(str(pf.get('feature')))}</code></td>"
        f"<td>{pf.get('original_value_raw', '')} {html.escape(str(pf.get('unit', '')))}</td>"
        f"<td>{pf.get('perturbed_value_raw', '')} {html.escape(str(pf.get('unit', '')))}</td>"
        f"<td>{pf.get('percent_change', 0):+.1f}%</td></tr>"
        for pf in cf.get("perturbed_features", [])
    )
    init_p, cf_p, thr = cf.get("initial_hazard"), cf.get("counterfactual_hazard"), cf.get("calibrated_threshold")
    probs = ""
    if init_p is not None:
        probs = f"Onset probability {init_p:.3f}" + (f" &rarr; {cf_p:.3f}" if cf_p is not None else "") + \
                (f" (alert threshold {thr:.2f})" if thr is not None else "")
    render_html(f"""
    <div class="soc-card" style="margin-top: 1rem;">
        <div class="soc-section-title">Counterfactual: smallest change that brings the onset probability below the alert threshold</div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem; color: {t['text_secondary']}; margin-top: 0.4rem;">
            Status: <b>{html.escape(str(cf.get('status', 'none')))}</b>. {probs}<br>{html.escape(str(cf.get('summary', '')))}
        </div>
        {f'<table style="margin-top: 0.5rem;"><thead><tr><th>Feature</th><th>Observed</th><th>Counterfactual</th><th>Change</th></tr></thead><tbody>{rows}</tbody></table>' if rows else ''}
        <div style="font-size: 0.7rem; color: {t['text_muted']}; font-style: italic; margin-top: 0.4rem;">{html.escape(str(cf.get('honesty_label', '')))}</div>
    </div>
    """)


if __name__ == "__main__":
    render_page()
