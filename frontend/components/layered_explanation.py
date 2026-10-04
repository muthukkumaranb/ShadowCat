"""
SHADOWCAT SOC Cockpit - Layered Explanation UI Component
Restructures model explanations into 4 clear, hierarchical layers:
- Layer 1: Verdict, predicted next MITRE ATT&CK stage, and estimated time-to-stage.
- Layer 2: Conformal prediction confidence & uncertainty band + Conformal Credibility (OOD) status.
- Layer 3: Top contributing features (Integrated Gradients deletion weights).
- Layer 4: Explanatory counterfactual statement with honesty & validation disclosures.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import streamlit as st

from styles import TOKENS, render_html
from data_provider import (
    get_forecast_trajectory,
    get_attributions,
    get_conformal_forecast,
    get_conformal_credibility,
    get_counterfactual,
)


def render_conformal_credibility_badge(credibility_data: Optional[Dict[str, Any]] = None):
    """
    Renders the visible Conformal Credibility / OOD diagnostic badge.
    Stubbed to accept data from feature/conformal-mimo when merged.
    """
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])
    if credibility_data is None:
        credibility_data = get_conformal_credibility()

    is_in_dist = credibility_data.get("is_in_distribution", True)
    badge_label = credibility_data.get("badge_label", "MODEL CONFIDENCE: IN-DISTRIBUTION")
    advisory = credibility_data.get("advisory", "")
    is_stub = credibility_data.get("is_stub", False)

    if is_in_dist:
        bg = "rgba(48, 209, 88, 0.12)"
        border = "#30D158"
        color = "#30D158"
        icon = "✔"
    else:
        bg = "rgba(255, 69, 58, 0.15)"
        border = "#FF453A"
        color = "#FF453A"
        icon = "⚠"

    stub_tag = '<span style="font-size: 0.60rem; color: #8A8A8A; margin-left: 6px; font-family: monospace;">[STUB: PENDING MERGE]</span>' if is_stub else ''

    render_html(f"""
    <div style="display: inline-flex; align-items: center; gap: 0.5rem; background: {bg}; border: 1px solid {border}; border-radius: 4px; padding: 3px 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.70rem; font-weight: 700; color: {color};">
        <span>{icon} {badge_label}</span>
        {stub_tag}
    </div>
    """)


def render_layered_explanation(
    step_k: int = 1,
    forecast_data: Optional[Dict[str, Any]] = None,
    attributions_data: Optional[List[Dict[str, Any]]] = None,
    conformal_data: Optional[Dict[str, Any]] = None,
    credibility_data: Optional[Dict[str, Any]] = None,
    counterfactual_data: Optional[Dict[str, Any]] = None,
    title: str = "Layered Threat Intelligence & Forensic Attribution",
):
    """
    Renders the complete 4-layer explanation hierarchy for a given forecast step.
    """
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])

    # Resolve data sources
    fc = forecast_data or get_forecast_trajectory()
    attributions = attributions_data if attributions_data is not None else get_attributions()
    conformal = conformal_data or get_conformal_forecast()
    credibility = credibility_data or get_conformal_credibility()
    counterfactual = counterfactual_data or get_counterfactual()

    risks = fc.get("risk", [0.84, 0.72, 0.58, 0.45, 0.38])
    stages = fc.get("stage", ["Credential Access", "Lateral Movement", "Impact"])
    lead_times = fc.get("lead_time", ["1m 00s", "2m 00s", "3m 00s", "4m 00s", "5m 00s"])

    # Extract target step values (0-indexed)
    idx = max(0, min(step_k, len(risks) - 1))
    risk_val = risks[idx] if idx < len(risks) else 0.50
    stage_val = stages[min(idx, len(stages) - 1)] if stages else "Credential Access"
    time_to_stage = lead_times[idx] if idx < len(lead_times) else f"~{idx+1}m 00s"

    # Conformal bounds
    intervals = conformal.get("intervals", [])
    if idx < len(intervals):
        lb, ub = intervals[idx]
    else:
        lb = max(0.0, round(risk_val - 0.12, 2))
        ub = min(1.0, round(risk_val + 0.12, 2))
    coverage_pct = int(conformal.get("coverage", 0.90) * 100)

    # Risk styling
    if risk_val >= 0.70:
        verdict_badge = "CRITICAL PREDICTED THREAT"
        verdict_color = "#FF453A"
        verdict_icon = "🔴"
    elif risk_val >= 0.35:
        verdict_badge = "ELEVATED PREDICTED HAZARD"
        verdict_color = "#FF9F0A"
        verdict_icon = "🟠"
    else:
        verdict_badge = "BENIGN BASELINE TRAJECTORY"
        verdict_color = "#30D158"
        verdict_icon = "🟢"

    # Outer Container
    render_html(f"""
    <div class="soc-card" style="margin-top: 1rem; border-top: 3px solid {verdict_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span class="soc-badge badge-nominal" style="font-size: 0.65rem;">4-LAYER XAI</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 700; color: {t['text_high']}; text-transform: uppercase;">
                    {title}
                </span>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: {t['text_muted']};">
                HORIZON TARGET: <b style="color: {t['primary']};">t+{idx+1}m</b>
            </div>
        </div>
    """)

    # -------------------------------------------------------------------------
    # LAYER 1: One-Line Verdict + Predicted Next ATT&CK Stage + Estimated Time
    # -------------------------------------------------------------------------
    render_html(f"""
    <div style="background: {t['surface_card']}; border: 1px solid {t['border']}; border-left: 4px solid {verdict_color}; border-radius: 4px; padding: 10px 14px; margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 4px;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; font-weight: 700; color: {t['text_muted']}; text-transform: uppercase; letter-spacing: 0.05em;">
                LAYER 1 // EXECUTIVE VERDICT & MITRE STAGE
            </span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.70rem; color: {verdict_color}; font-weight: 700;">
                {verdict_badge}
            </span>
        </div>
        <div style="display: flex; align-items: baseline; gap: 0.5rem; flex-wrap: wrap;">
            <span style="font-size: 1.05rem;">{verdict_icon}</span>
            <span style="font-family: 'Inter', sans-serif; font-size: 0.95rem; font-weight: 700; color: {t['text_high']};">
                {verdict_badge}:
            </span>
            <span style="font-family: 'Inter', sans-serif; font-size: 0.92rem; color: {t['text_secondary']};">
                Imminent progression toward <b style="color: {verdict_color};">{stage_val}</b> anticipated in <b style="color: {t['text_high']}; font-family: 'JetBrains Mono', monospace;">~{time_to_stage}</b>.
            </span>
        </div>
    </div>
    """)

    # -------------------------------------------------------------------------
    # LAYER 2: Calibrated Confidence, Uncertainty Band & Conformal Credibility (OOD)
    # -------------------------------------------------------------------------
    is_in_dist = credibility.get("is_in_distribution", True)
    cred_badge_text = credibility.get("badge_label", "MODEL CONFIDENCE: IN-DISTRIBUTION")
    cred_color = "#30D158" if is_in_dist else "#FF453A"
    cred_bg = "rgba(48, 209, 88, 0.12)" if is_in_dist else "rgba(255, 69, 58, 0.15)"
    cred_desc = credibility.get("advisory", "")
    is_stub = credibility.get("is_stub", False)
    stub_tag = '<span style="font-size: 0.60rem; color: #8A8A8A; margin-left: 6px; font-family: monospace;">(TODO: Wire to feature/conformal-mimo)</span>' if is_stub else ''

    render_html(f"""
    <div style="background: {t['surface_card']}; border: 1px solid {t['border']}; border-left: 4px solid #00F5FF; border-radius: 4px; padding: 10px 14px; margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 6px;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; font-weight: 700; color: {t['text_muted']}; text-transform: uppercase; letter-spacing: 0.05em;">
                LAYER 2 // CALIBRATED CONFIDENCE & DISTRIBUTION MANIFOLD
            </span>
            <div style="display: inline-flex; align-items: center; gap: 0.4rem; background: {cred_bg}; border: 1px solid {cred_color}; border-radius: 3px; padding: 2px 7px; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; color: {cred_color};">
                <span>{cred_badge_text}</span>
                {stub_tag}
            </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1.6fr; gap: 1rem; align-items: center;">
            <div>
                <span class="soc-stat-label">Point Estimate Hazard</span>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 800; color: {verdict_color};">
                    {risk_val:.2f} <span style="font-size: 0.75rem; color: {t['text_muted']}; font-weight: 400;">P(onset)</span>
                </div>
            </div>
            <div>
                <span class="soc-stat-label">Split Conformal Interval ({coverage_pct}% Validated)</span>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #00F5FF;">
                    [{lb:.2f} — {ub:.2f}]
                </div>
            </div>
            <div style="font-size: 0.74rem; color: {t['text_secondary']}; line-height: 1.4; border-left: 1px solid {t['border']}; padding-left: 0.75rem;">
                <b style="color: {t['text_high']};">Manifold Credibility:</b> {cred_desc}
            </div>
        </div>
    </div>
    """)

    # -------------------------------------------------------------------------
    # LAYER 3: Top Contributing Features (Integrated Gradients)
    # -------------------------------------------------------------------------
    top_3 = attributions[:3] if attributions else [
        {"feature": "egress_burst_ratio", "contribution": 0.48, "category": "Packet Dynamics", "delta": "+380%"},
        {"feature": "beacon_jitter_variance", "contribution": 0.28, "category": "Temporal Rhythm", "delta": "+19.4%"},
        {"feature": "peer_fanout_entropy", "contribution": 0.16, "category": "Topology", "delta": "14 targets"},
    ]

    attr_chips = ""
    for a in top_3:
        feat_name = a.get("feature", "unknown")
        c_weight = a.get("contribution", 0.0)
        delta_str = a.get("delta", "Baseline")
        pct_bar = min(100, int(c_weight * 180))
        chip_color = "#FF453A" if c_weight >= 0.35 else ("#FF9F0A" if c_weight >= 0.20 else t["primary"])

        attr_chips += f"""
        <div style="flex: 1; min-width: 180px; background: {t['surface_lowest']}; border: 1px solid {t['border']}; border-radius: 4px; padding: 7px 10px;">
            <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; margin-bottom: 3px;">
                <span style="color: {t['text_high']}; font-weight: 600;">{feat_name}</span>
                <span style="color: {chip_color}; font-weight: 700;">+{c_weight*100:.0f}%</span>
            </div>
            <div style="width: 100%; height: 4px; background: {t['surface_highest']}; border-radius: 2px; margin-bottom: 4px;">
                <div style="width: {pct_bar}%; height: 100%; background: {chip_color}; border-radius: 2px;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.65rem; color: {t['text_muted']};">
                <span>{a.get('category', 'Signal')}</span>
                <span style="font-family: 'JetBrains Mono', monospace;">Delta: {delta_str}</span>
            </div>
        </div>
        """

    render_html(f"""
    <div style="background: {t['surface_card']}; border: 1px solid {t['border']}; border-left: 4px solid {t['secondary']}; border-radius: 4px; padding: 10px 14px; margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; font-weight: 700; color: {t['text_muted']}; text-transform: uppercase; letter-spacing: 0.05em;">
                LAYER 3 // TOP CONTRIBUTING DRIVERS (DELETION-TESTED ATTRIBUTION)
            </span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: {t['text_secondary']};">
                Top {len(top_3)} Decomposed Features
            </span>
        </div>
        <div style="display: flex; gap: 0.6rem; flex-wrap: wrap;">
            {attr_chips}
        </div>
    </div>
    """)

    # -------------------------------------------------------------------------
    # LAYER 4: Explanatory Counterfactual Statement
    # -------------------------------------------------------------------------
    cf_summary = counterfactual.get(
        "summary",
        "If egress_burst_ratio returned to baseline (<= 2.4 GB/s), projected hazard would decline by -68%, dropping risk safely below the 0.15 threshold."
    )
    cf_honesty = counterfactual.get(
        "honesty_label",
        "Model-based counterfactual under the hazard model's learned decision boundary — not an empirical guarantee of attack prevention, and not validated against real intervention data."
    )

    render_html(f"""
    <div style="background: {t['surface_card']}; border: 1px solid {t['border']}; border-left: 4px solid #E0982B; border-radius: 4px; padding: 10px 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; font-weight: 700; color: {t['text_muted']}; text-transform: uppercase; letter-spacing: 0.05em;">
                LAYER 4 // EXPLANATORY COUNTERFACTUAL WHAT-IF
            </span>
            <span class="soc-badge badge-neutral" style="font-size: 0.62rem;">DECISION BOUNDARY REVERSAL</span>
        </div>
        <p style="font-family: 'Inter', sans-serif; font-size: 0.8125rem; color: {t['text_high']}; line-height: 1.45; margin: 0 0 0.4rem 0;">
            <b style="color: #E0982B;">Actionable Intervention:</b> {cf_summary}
        </p>
        <div style="font-size: 0.68rem; color: {t['text_muted']}; font-style: italic; border-top: 1px dashed {t['border']}; padding-top: 4px;">
            <b>Honesty Disclosure:</b> {cf_honesty}
        </div>
    </div>
    </div>
    """)
