"""
SHADOWCAT SOC Cockpit - Layered Explanation UI Component
Shows, for the active prediction only:
- Layer 1: onset probability P(attack within the next 5 minutes) and whether it alerts.
- Layer 2: its split-conformal interval.
- Layer 3: top input features by Integrated Gradients on the stacked onset logit.
- Layer 4: the bounded-perturbation counterfactual, if one was computed.
No value is shown that the backend did not produce.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

import streamlit as st

from styles import TOKENS, render_html
from data_provider import (
    get_forecast_trajectory,
    get_attributions,
    get_conformal_credibility,
    get_counterfactual,
)


def render_conformal_credibility_badge(credibility_data: Optional[Dict[str, Any]] = None):
    """Renders the out-of-distribution badge only when a real (non-stub) check produced it."""
    if credibility_data is None:
        credibility_data = get_conformal_credibility()
    if not credibility_data or credibility_data.get("is_stub", True):
        return
    ok = credibility_data.get("is_in_distribution", True)
    color = "#30D158" if ok else "#FF453A"
    render_html(f"""
    <div style="display: inline-flex; border: 1px solid {color}; border-radius: 4px; padding: 3px 8px;
         font-family: 'JetBrains Mono', monospace; font-size: 0.70rem; font-weight: 700; color: {color};">
        {html.escape(str(credibility_data.get('badge_label', '')))}
    </div>
    """)


def render_layered_explanation(
    step_k: int = 1,
    forecast_data: Optional[Dict[str, Any]] = None,
    attributions_data: Optional[List[Dict[str, Any]]] = None,
    conformal_data: Optional[Dict[str, Any]] = None,
    credibility_data: Optional[Dict[str, Any]] = None,
    counterfactual_data: Optional[Dict[str, Any]] = None,
    title: str = "Explanation of the onset probability",
):
    """Renders the explanation of the active prediction (step_k is accepted for compatibility)."""
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])
    fc = forecast_data or get_forecast_trajectory()
    attributions = attributions_data if attributions_data is not None else get_attributions()
    counterfactual = counterfactual_data or get_counterfactual()

    p = fc.get("onset_probability")
    if p is None:
        render_html(f"<div class='soc-card'>No inference result loaded.</div>")
        return
    label = html.escape(fc.get("onset_probability_label", "Onset probability"))
    ci = fc.get("conformal_interval") or [None, None]
    cov = fc.get("conformal_coverage")
    alert = "ALERT" if fc.get("onset_alert") else "NO ALERT"

    rows = "".join(
        f"<tr><td><code>{html.escape(a['feature'])}</code></td><td>{html.escape(str(a.get('category')))}</td>"
        f"<td>{a['contribution']:.1%}</td><td>{a.get('signed_attribution', 0):+.4f}</td></tr>"
        for a in attributions[:5]
    )
    cf_summary = counterfactual.get("summary") if counterfactual else None
    cf_label = counterfactual.get("honesty_label", "") if counterfactual else ""

    render_html(f"""
    <div class="soc-card" style="margin-top: 1rem;">
        <div class="soc-section-title">{html.escape(title)}</div>
        <table style="margin-top: 0.4rem;">
            <tr><td>1. {label}</td><td><b>{p:.3f}</b> ({alert}, threshold {fc.get('alert_threshold')})</td></tr>
            <tr><td>2. {'' if cov is None else f'{int(round(cov * 100))}% '}split-conformal interval</td>
                <td>{'—' if ci[0] is None else f'[{ci[0]:.3f}, {ci[1]:.3f}]'}</td></tr>
        </table>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.78rem; color: {t['text_secondary']}; margin-top: 0.6rem;">
            3. Integrated Gradients on the stacked onset logit: top input features
        </div>
        <table>
            <thead><tr><th>Feature</th><th>Group</th><th>Share of |attribution|</th><th>Signed attribution (logit)</th></tr></thead>
            <tbody>{rows or '<tr><td colspan="4">No attribution computed</td></tr>'}</tbody>
        </table>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.78rem; color: {t['text_secondary']}; margin-top: 0.6rem;">
            4. Counterfactual: {html.escape(cf_summary) if cf_summary else 'none computed'}
        </div>
        <div style="font-size: 0.68rem; color: {t['text_muted']}; font-style: italic;">{html.escape(cf_label)}</div>
    </div>
    """)
