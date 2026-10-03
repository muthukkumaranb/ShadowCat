"""
SHADOWCAT SOC Cockpit - Page 8: Reports & History
Persistent, restart-surviving view of all prediction alerts and lineage records
from the SQLite reports database.

Follows existing view conventions (theme tokens, render_html, session-state theme).
"""

import html

import streamlit as st
from styles import TOKENS, render_html
from data_provider import get_report_history


def render_page():
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])

    # ── Page Header ──────────────────────────────────────────────────────
    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['primary']}; font-weight: 700; text-transform: uppercase;">
                        PERSISTENT STORAGE // SQLITE-BACKED REPORT LEDGER
                    </span>
                    <span style="color: {t['outline_variant']};">•</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.6875rem; color: {t['text_muted']}; text-transform: uppercase;">
                        SURVIVES APP RESTART &amp; BROWSER REFRESH
                    </span>
                </div>
                <h1 style="font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: {t['text_high']}; text-transform: uppercase; margin: 0;">
                    Reports &amp; History
                </h1>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.78rem; color: {t['text_secondary']}; margin-top: 0.2rem;">
                    Durable record of every prediction alert and lineage record captured by the SHADOWCAT pipeline.
                    Unlike in-memory session data, these records persist across full application restarts.
                </div>
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
                <span class="soc-badge badge-nominal" style="padding: 0.35rem 0.75rem;">
                    <span class="soc-pulse-dot" style="background:{t['primary']};"></span>
                    DB-BACKED // DURABLE
                </span>
            </div>
        </div>
    </div>
    """)

    # ── Filters ──────────────────────────────────────────────────────────
    filter_cols = st.columns([1, 1, 1, 2])

    with filter_cols[0]:
        sev_filter = st.selectbox(
            "Severity",
            options=["All", "HIGH", "MEDIUM", "LOW"],
            index=0,
            key="rh_severity_filter",
        )

    with filter_cols[1]:
        type_filter = st.selectbox(
            "Record Type",
            options=["All", "alert", "lineage"],
            index=0,
            key="rh_type_filter",
        )

    with filter_cols[2]:
        limit = st.selectbox(
            "Show Latest",
            options=[25, 50, 100, 250, 500],
            index=2,
            key="rh_limit_filter",
        )

    with filter_cols[3]:
        st.markdown("")  # spacer
        st.markdown("")
        refresh = st.button("⟳  Refresh", key="rh_refresh", width='stretch')

    # ── Query ────────────────────────────────────────────────────────────
    severity_arg = None if sev_filter == "All" else sev_filter
    type_arg = None if type_filter == "All" else type_filter
    reports = get_report_history(severity=severity_arg, record_type=type_arg, limit=limit)

    # ── Summary Stats ────────────────────────────────────────────────────
    total_count = len(reports)
    alert_count = sum(1 for r in reports if r.get("record_type") == "alert")
    lineage_count = sum(1 for r in reports if r.get("record_type") == "lineage")
    high_count = sum(1 for r in reports if r.get("severity") == "HIGH")

    stat_cols = st.columns(4)
    stat_data = [
        ("TOTAL RECORDS", str(total_count), t["text_high"]),
        ("ALERTS", str(alert_count), t["secondary"]),
        ("LINEAGE", str(lineage_count), t["primary"]),
        ("HIGH SEVERITY", str(high_count), t["tertiary"]),
    ]
    for i, (label, value, color) in enumerate(stat_data):
        with stat_cols[i]:
            render_html(f"""
            <div class="soc-card" style="text-align: center; padding: 0.75rem;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: {t['text_muted']}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.25rem;">
                    {label}
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.75rem; font-weight: 700; color: {color};">
                    {value}
                </div>
            </div>
            """)

    # ── Empty state ──────────────────────────────────────────────────────
    if total_count == 0:
        render_html(f"""
        <div class="soc-card" style="text-align: center; padding: 3rem 2rem;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; color: {t['text_muted']}; margin-bottom: 0.75rem;">
                ◇ NO RECORDS
            </div>
            <div style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: {t['text_secondary']}; max-width: 480px; margin: 0 auto; line-height: 1.6;">
                No prediction reports have been captured yet. Run a prediction
                via the <strong>Telemetry Ingestion</strong> page to populate this view.
                Records persist in the SQLite database and survive app restarts.
            </div>
        </div>
        """)
        return

    # ── Report Rows ──────────────────────────────────────────────────────
    for idx, report in enumerate(reports):
        rec_type = report.get("record_type", "unknown")
        rec_id = report.get("record_id", "—")
        severity = report.get("severity", "—")
        notarized_via = report.get("notarized_via", "—")
        created_at = report.get("created_at", "—")
        window_id = report.get("window_id", "—")
        window_start = report.get("window_start", "—")
        max_risk = report.get("max_risk")
        target_node = report.get("target_node", "—")
        model_id = report.get("model_id", "—")
        raw_data_hash = report.get("raw_data_hash", "—")
        prediction_hash = report.get("prediction_hash", "—")
        # Records can carry input-derived strings (window ids, host names): escape before HTML rendering
        rec_type, rec_id, severity, notarized_via, created_at, window_id, window_start, target_node, model_id, \
            raw_data_hash, prediction_hash = (
                html.escape(str(v)) for v in (rec_type, rec_id, severity, notarized_via, created_at, window_id,
                                              window_start, target_node, model_id, raw_data_hash, prediction_hash))

        # Severity styling
        if severity == "HIGH":
            sev_badge_class = "badge-critical"
            sev_color = t["secondary"]
        elif severity == "MEDIUM":
            sev_badge_class = "badge-caution"
            sev_color = t["tertiary"]
        else:
            sev_badge_class = "badge-nominal"
            sev_color = t["primary"]

        # Record type styling
        if rec_type == "alert":
            type_icon = "⚠"
            type_color = t["secondary"]
            border_color = t["secondary"]
        else:
            type_icon = "◈"
            type_color = t["primary"]
            border_color = t["primary"]

        # Notarization badge
        if notarized_via == "fabric":
            notary_badge = f'<span class="soc-badge badge-nominal" style="font-size: 0.6rem; padding: 1px 6px;">FABRIC</span>'
        elif notarized_via == "sha256_fallback":
            notary_badge = f'<span class="soc-badge badge-caution" style="font-size: 0.6rem; padding: 1px 6px;">SHA-256</span>'
        elif notarized_via == "failed":
            notary_badge = f'<span class="soc-badge badge-critical" style="font-size: 0.6rem; padding: 1px 6px;">FAILED</span>'
        else:
            notary_badge = f'<span class="soc-badge badge-neutral" style="font-size: 0.6rem; padding: 1px 6px;">{notarized_via}</span>'

        # Risk display
        risk_display = f"{max_risk:.4f}" if max_risk is not None else "—"

        render_html(f"""
        <div class="soc-card" style="margin-bottom: 0.5rem; border-left: 3px solid {border_color};">
            <!-- Row Header -->
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1rem;">{type_icon}</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 700; color: {type_color}; text-transform: uppercase;">
                        {rec_type}
                    </span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: {t['text_muted']};">
                        {rec_id}
                    </span>
                </div>
                <div style="display: flex; gap: 0.4rem; align-items: center;">
                    <span class="soc-badge {sev_badge_class}" style="font-size: 0.6rem; padding: 1px 6px;">{severity}</span>
                    {notary_badge}
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.625rem; color: {t['text_muted']};">
                        {created_at}
                    </span>
                </div>
            </div>

            <!-- Detail Grid -->
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.4rem; background: {t['surface_lowest']}; padding: 0.5rem; border-radius: 4px; border: 1px solid {t['border']};">
                <div>
                    <span class="soc-stat-label">Window ID</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: {t['text_high']}; font-weight: 600; word-break: break-all;">{window_id}</div>
                </div>
                <div>
                    <span class="soc-stat-label">Window Start</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: {t['text_high']}; font-weight: 600;">{window_start}</div>
                </div>
                <div>
                    <span class="soc-stat-label">Max Risk</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: {sev_color}; font-weight: 600;">{risk_display}</div>
                </div>
                <div>
                    <span class="soc-stat-label">Target Node</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: {t['text_high']}; font-weight: 600; word-break: break-all;">{target_node}</div>
                </div>
                <div>
                    <span class="soc-stat-label">Model ID</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: {t['text_high']}; font-weight: 600;">{model_id}</div>
                </div>
                <div>
                    <span class="soc-stat-label">Notarized Via</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: {t['text_high']}; font-weight: 600;">{notarized_via}</div>
                </div>
            </div>

            <!-- Hashes (collapsed detail) -->
            <div style="margin-top: 0.35rem; font-family: 'JetBrains Mono', monospace; font-size: 0.575rem; color: {t['text_muted']}; display: flex; flex-wrap: wrap; gap: 0.75rem;">
                <span>RAW: {raw_data_hash[:16] if raw_data_hash and raw_data_hash != '—' else '—'}…</span>
                <span>PRED: {prediction_hash[:16] if prediction_hash and prediction_hash != '—' else '—'}…</span>
            </div>
        </div>
        """)

    # ── Footer ───────────────────────────────────────────────────────────
    render_html(f"""
    <div style="text-align: center; padding: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: {t['text_muted']}; text-transform: uppercase; letter-spacing: 0.05em;">
        SHOWING {total_count} RECORD{'S' if total_count != 1 else ''} // SQLITE PERSISTENT STORAGE // BACKEND/DATA/SHADOWCAT_REPORTS.DB
    </div>
    """)


if __name__ == "__main__":
    render_page()
