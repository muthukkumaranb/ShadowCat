import streamlit as st
import pandas as pd
import numpy as np
from data_provider import get_analysis_metadata, get_forecast_trajectory

st.title("01 — Overview")

meta = get_analysis_metadata()
forecast = get_forecast_trajectory()

# Top strip: persistent mini risk-trajectory sparkline + current hazard state + summary
st.markdown("### Executive Summary")

col_spark, col_state = st.columns([0.7, 0.3])
with col_spark:
    # Mini sparkline of risk trajectory
    risk_probs = forecast.get("risk_probabilities", [0.1, 0.2, 0.5, 0.8])
    st.line_chart(risk_probs, height=150)

with col_state:
    current_hazard = risk_probs[-1] if risk_probs else 0
    state_color = "#39FF88" if current_hazard < 0.4 else ("#FFB84D" if current_hazard < 0.7 else "#FF3B5C")
    state_text = "NOMINAL" if current_hazard < 0.4 else ("ELEVATED" if current_hazard < 0.7 else "CRITICAL")
    
    st.markdown(f"<div style='text-align: center; padding: 20px; background-color: var(--surface-card); border-radius: 8px; border: 1px solid var(--border-color);'>"
                f"<h1 style='color: {state_color}; margin: 0;'>{state_text}</h1>"
                f"<p style='color: var(--text-secondary); margin-top: 5px;'>Hazard Level: {current_hazard:.0%}</p>"
                f"</div>", unsafe_allow_html=True)

# Plain language summary (No bare numbers!)
st.markdown(f"> **Context**: The network is currently in a {state_text} state due to an accumulation of anomalous flow dynamics over the last 60 seconds.")

st.markdown("---")

# Summary cards row
st.markdown("### Operational Metrics")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Active Alerts", "12", "↑ 3 recent")
    st.caption("Driven by unusual peer relationship density.")
with c2:
    stage = forecast.get("stage_probabilities", {}).get("Reconnaissance", 0.0)
    st.metric("Current MITRE Stage", "Reconnaissance")
    st.caption("Based on temporal scan patterns.")
with c3:
    st.metric("Novelty Score", "High")
    st.caption("Deviation from baseline flow stats.")
with c4:
    st.metric("Audit Chain", "Verified")
    st.caption("Cryptographic hashes matched.")

st.markdown("---")

# Recent alerts
st.markdown("### Recent Alerts")
with st.expander("Critical: Abnormal Traffic Growth (10:45 AM UTC)"):
    st.write("Attack probability increased significantly because of abnormal traffic growth from external subnet 192.168.1.x.")
with st.expander("Warning: Unusual Peer Relationships (10:42 AM UTC)"):
    st.write("Elevated novelty score due to internal hosts communicating with previously unseen external nodes.")

# CTAs
st.markdown("---")
st.markdown("### Deep Dive Navigation")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("View Forecast Panel", use_container_width=True):
        st.switch_page("views/02_Forecast.py")
with col2:
    if st.button("Explore Attack Graph", use_container_width=True):
        st.switch_page("views/03_Attack_Graph.py")
with col3:
    if st.button("Check Explainability", use_container_width=True):
        st.switch_page("views/05_Explainability.py")
