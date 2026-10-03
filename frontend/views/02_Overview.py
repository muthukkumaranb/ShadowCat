"""
SHADOWCAT SOC Cockpit - Page 2: Overview
Summary of the active prediction. Every value comes from backend.predict's payload,
backend/audit_chain.json or the committed demo-slice metadata.
"""

import html

import pandas as pd
import streamlit as st
from styles import TOKENS, render_html
from data_provider import (
    get_active_source,
    get_audit_chain_status,
    get_detection_probability,
    get_flagged_flows,
    get_forecast_trajectory,
    get_live_notarization_status,
    get_mitre_data,
    get_novelty_score,
)


def _fmt(p):
    return "—" if p is None else f"{p:.3f}"


def render_page():
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])
    fc = get_forecast_trajectory()
    p_onset = fc.get("onset_probability")
    p_det = get_detection_probability()
    alert = bool(fc.get("onset_alert"))
    color = t["secondary"] if alert else t["primary"]

    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1rem;">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: {t['text_high']}; text-transform: uppercase;">Overview</span>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem; color: {t['text_secondary']}; margin-top: 0.3rem;">
            Input: <b>{html.escape(get_active_source())}</b> &bull; last window <code>{html.escape(str(fc.get('window_id', '—')))}</code>
        </div>
    </div>
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        render_html(f"""
        <div class="soc-stat-card" style="border-top: 2px solid {color};">
            <span class="soc-stat-label">{html.escape(fc.get('onset_probability_label', 'Onset probability'))}</span>
            <div class="soc-stat-val" style="color: {color};">{_fmt(p_onset)}</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: {t['text_muted']};">
                {'ALERT' if alert else 'NO ALERT'} (threshold {fc.get('alert_threshold', '—')})
            </div>
        </div>
        """)
    with c2:
        render_html(f"""
        <div class="soc-stat-card">
            <span class="soc-stat-label">P(attack in current window)</span>
            <div class="soc-stat-val">{_fmt(p_det)}</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: {t['text_muted']};">Stacked detection ensemble</div>
        </div>
        """)
    with c3:
        mitre = get_mitre_data()
        s1 = mitre[0] if mitre else {}
        if s1.get("technique_id"):
            stage = f"{html.escape(s1['tactic_name'])} / {s1['technique_id']}"
            sub = f"stage head p = {s1['confidence']:.2f} on S_hat(t+1)"
        else:
            stage = "No stage determined"
            sub = (f"top class {html.escape(str(s1.get('stage_head_top_class')))}, p = {_fmt(s1.get('confidence'))}; "
                   "shown only if p &gt; 0.5 and not Unknown/Other") if s1 else ""
        render_html(f"""
        <div class="soc-stat-card">
            <span class="soc-stat-label">ATT&amp;CK stage</span>
            <div class="soc-stat-val" style="font-size: 1.05rem;">{stage}</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: {t['text_muted']};">{sub}</div>
        </div>
        """)

    nv = get_novelty_score()
    rows = [
        ("World-model deviation score", _fmt(nv.get("novelty_score")),
         "sigmoid of mean |S(t) - S_hat(t)| / sigma over the 406 features"),
        ("Rows in the input", nv.get("flows_analyzed", "—"), "flows (CSV) or UCS windows (parquet)"),
        ("Distinct IP endpoints in the input", nv.get("active_endpoints") if nv.get("active_endpoints") is not None else "—",
         "only when the input has Src/Dst IP columns"),
    ]
    body = "".join(f"<tr><td>{a}</td><td><b>{b}</b></td><td>{c}</td></tr>" for a, b, c in rows)
    render_html(f"""
    <div class="soc-card" style="margin-top: 1rem;">
        <div class="soc-section-title">Input and world-model state</div>
        <table style="margin-top: 0.4rem;">{body}</table>
    </div>
    """)

    flows = get_flagged_flows(limit=5)
    if flows:
        render_html("<div class='soc-section-title' style='margin-top: 1rem;'>Highest-scoring flows</div>")
        st.dataframe(pd.DataFrame(flows), hide_index=True, width="stretch")

    audit = get_audit_chain_status()
    notary = get_live_notarization_status()
    render_html(f"""
    <div class="soc-card" style="margin-top: 1rem;">
        <div class="soc-section-title">Audit trail</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: {t['text_secondary']}; margin-top: 0.3rem;">
            backend audit chain: {audit.get('length', 0)} entries, verifies: {'yes' if audit.get('is_valid') else 'no'} &bull;
            notarization used by the last inference: {html.escape(str(notary.get('active_mechanism', 'none')))}
        </div>
    </div>
    """)


if __name__ == "__main__":
    render_page()
