"""
SHADOWCAT SOC Cockpit - Page 1: Telemetry Ingestion
Loads one of three committed real demo slices (frontend/demo_data/, built by
frontend/scripts/build_demo_slices.py) or an uploaded file, and runs backend.predict on it.
"""

import html
import time

import pandas as pd
import streamlit as st
from styles import TOKENS, render_html
from data_provider import (
    REPO_ROOT,
    get_active_source,
    get_demo_slices,
    load_demo_slice,
    run_core_ml_inference,
)

SLICE_TITLES = {
    "benign": "Benign traffic",
    "botnet": "Botnet episode",
    "ssh": "SSH-Bruteforce episode",
}


def _slice_label(key, meta):
    start, end = meta["time_range_utc"]
    onset = f", attack onset {meta['episode_onset_utc'][11:16]}" if meta.get("episode_onset_utc") else ""
    return f"{SLICE_TITLES.get(key, key)}: {start[:10]} {start[11:16]}-{end[11:16]} UTC{onset}"


def _run(df, source_type, source_desc):
    with st.spinner("Running feature extraction and the stacked ensemble..."):
        pred = run_core_ml_inference(df, source_type=source_type)
    st.session_state["ml_prediction_result"] = pred
    st.session_state["ml_prediction_timestamp"] = time.strftime("%H:%M:%S")
    st.session_state["active_source"] = source_desc
    st.session_state["ingested_df"] = df


def render_page():
    t = TOKENS.get(st.session_state.get("theme", "dark"), TOKENS["dark"])
    slices = get_demo_slices()

    render_html(f"""
    <div class="soc-card" style="margin-bottom: 1rem;">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: {t['text_high']}; text-transform: uppercase;">
            Telemetry Ingestion
        </span>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem; color: {t['text_secondary']}; margin-top: 0.3rem;">
            Active input: <b>{html.escape(get_active_source())}</b>
        </div>
    </div>
    """)

    # 1. Real demo slices
    render_html(f"""
    <div class="soc-card">
        <div class="soc-section-title">Demo slices (CSE-CIC-IDS2018)</div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.78rem; color: {t['text_secondary']}; line-height: 1.5; margin-top: 0.3rem;">
            Each slice is 32 consecutive 1-minute UCS windows from <code>ucs_windows_models_v1.parquet</code>, the dataset the
            37 LOEO fold models were trained and validated on, plus the raw CICFlowMeter rows for the same minutes.
            Every demo window was in the training split of most fold models, so this shows the pipeline running on real data;
            it is not a held-out evaluation (see Validation &amp; Trust for that).
        </div>
    </div>
    """)
    if not slices:
        st.error("frontend/demo_data/slices.json is missing. Run frontend/scripts/build_demo_slices.py.")
    cols = st.columns(max(len(slices), 1))
    for col, (key, meta) in zip(cols, slices.items()):
        with col:
            if st.button(_slice_label(key, meta), key=f"demo_{key}", width="stretch"):
                _run(load_demo_slice(key), "windows", f"Demo slice: {_slice_label(key, meta)}")
                st.rerun()

    if slices:
        rows = []
        for key, m in slices.items():
            rows.append({
                "slice": key,
                "UTC range": f"{m['time_range_utc'][0]} - {m['time_range_utc'][1][11:]}",
                "raw source file": m["source_file"],
                "raw rows": m["rows"],
                "window labels (dataset)": ", ".join(f"{k}: {v}" for k, v in m["window_label_counts"].items()),
                "last window label": m["last_window"]["label_attack_type"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with st.expander("Raw CICFlowMeter rows for each slice"):
            for key, m in slices.items():
                path = REPO_ROOT / m["raw_csv"]
                if path.exists():
                    st.download_button(f"{path.name} ({m['rows']:,} rows)", path.read_bytes(),
                                       file_name=path.name, key=f"dl_{key}")

    # 2. Upload
    render_html(f"""
    <div class="soc-card" style="margin-top: 1rem;">
        <div class="soc-section-title">Upload</div>
        <div style="font-family: 'Inter', sans-serif; font-size: 0.78rem; color: {t['text_secondary']}; line-height: 1.5; margin-top: 0.3rem;">
            Parquet: UCS windows (same columns as <code>ucs_windows_models_v1.parquet</code>), used as-is.
            CSV: CICFlowMeter flows, windowed by the live extractor. A CICFlowMeter CSV carries no packet-level
            features, so they are marked absent (<code>mask_has_packet_level_features = 0</code>); the models were trained
            with them present. On the three demo slices the CSV route gives near-zero probabilities whatever the label
            (<code>model_output.csv</code> in <code>frontend/demo_data/slices.json</code>), so CSV results are not comparable
            to the validated numbers.
        </div>
    </div>
    """)
    uploaded = st.file_uploader("CICFlowMeter CSV (.csv, .csv.gz) or UCS windows (.parquet)",
                                type=["csv", "gz", "parquet"], key="telemetry_uploader")
    if uploaded is not None and st.session_state.get("_last_uploaded_name") != uploaded.name:
        fname = uploaded.name
        try:
            if fname.endswith(".parquet"):
                df, s_type = pd.read_parquet(uploaded), "windows"
            else:
                df, s_type = pd.read_csv(uploaded, compression="gzip" if fname.endswith(".gz") else None), "csv"
            st.session_state["_last_uploaded_name"] = fname
            _run(df, s_type, f"Uploaded file: {fname} ({len(df):,} rows, read as {s_type})")
            st.rerun()
        except Exception as ex:
            st.error(f"Could not read {fname}: {ex}")

    # 3. Result of the last run
    pred = st.session_state.get("ml_prediction_result")
    if pred:
        if pred.get("_inference_error"):
            st.error(f"Inference failed: {pred['_inference_error']}")
            with st.expander("Traceback"):
                st.code(pred.get("_inference_traceback", ""), language="python")
            return
        fc = pred.get("forecast_trajectory", {})
        p_on = fc.get("onset_probability")
        p_det = pred.get("detection_probability")
        render_html(f"""
        <div class="soc-card" style="margin-top: 1rem;">
            <div class="soc-section-title">Last inference ({html.escape(st.session_state.get('ml_prediction_timestamp', ''))})</div>
            <table>
                <tr><td>Input</td><td>{html.escape(st.session_state.get('active_source', ''))}</td></tr>
                <tr><td>Last window</td><td>{html.escape(str(fc.get('window_id', '')))}</td></tr>
                <tr><td>{html.escape(fc.get('onset_probability_label', 'Onset probability'))}</td><td><b>{'—' if p_on is None else f'{p_on:.3f}'}</b>{' (alert)' if fc.get('onset_alert') else ''}</td></tr>
                <tr><td>P(attack in current window)</td><td><b>{'—' if p_det is None else f'{p_det:.3f}'}</b></td></tr>
                <tr><td>Fewer than 30 windows (history padded)</td><td>{'yes' if pred.get('is_warmup') else 'no'}</td></tr>
            </table>
        </div>
        """)
        df = st.session_state.get("ingested_df")
        if df is not None:
            st.caption(f"Input preview: first 10 of {len(df):,} rows")
            st.dataframe(df.head(10), width="stretch")


if __name__ == "__main__":
    render_page()
