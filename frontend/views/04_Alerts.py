import streamlit as st
import pandas as pd
from data_provider import get_analysis_metadata

st.title("04 — Alerts")

# Filter / Sort Controls
col1, col2, col3 = st.columns(3)
with col1:
    severity_filter = st.selectbox("Severity", ["All", "Critical", "Warning", "Info"])
with col2:
    stage_filter = st.selectbox("MITRE Stage", ["All", "Reconnaissance", "Lateral Movement", "Exfiltration"])
with col3:
    time_filter = st.selectbox("Time Range", ["Last 1 Hour", "Last 24 Hours", "Last 7 Days"])

st.markdown("---")

# Mock Alert Data (normally fetched from pipeline)
alerts = [
    {
        "severity": "Critical",
        "stage": "Lateral Movement",
        "time": "10:45 AM UTC",
        "cause": "Attack probability increased because of abnormal traffic growth from external subnet 192.168.1.x.",
        "details": "Attribution scores show a heavy spike in Flow Dynamics (Tot Fwd Pkts) associated with this host, deviating 4.2x from baseline."
    },
    {
        "severity": "Warning",
        "stage": "Reconnaissance",
        "time": "10:42 AM UTC",
        "cause": "Elevated novelty score due to internal hosts communicating with previously unseen external nodes.",
        "details": "Temporal Rhythm features flagged anomalous connection frequencies across multiple distinct destination ports."
    }
]

# Apply filters
filtered_alerts = [a for a in alerts if 
                   (severity_filter == "All" or a["severity"] == severity_filter) and 
                   (stage_filter == "All" or a["stage"] == stage_filter)]

if not filtered_alerts:
    st.info("No alerts match the current filter criteria.")

for alert in filtered_alerts:
    color = "#FF3B5C" if alert["severity"] == "Critical" else ("#FFB84D" if alert["severity"] == "Warning" else "#39FF88")
    
    st.markdown(f"""
    <div style='background-color: var(--surface-card); border-left: 4px solid {color}; padding: 15px; margin-bottom: 10px; border-radius: 4px; border-top: 1px solid var(--border-color); border-right: 1px solid var(--border-color); border-bottom: 1px solid var(--border-color);'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
            <span style='font-weight: bold; color: {color};'>{alert["severity"].upper()}</span>
            <span style='color: var(--text-secondary); font-size: 0.8em;'>{alert["time"]}</span>
        </div>
        <div style='margin-bottom: 8px;'>
            <span style='background-color: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; font-size: 0.8em;'>{alert["stage"]}</span>
        </div>
        <div style='color: var(--text-primary); font-size: 1.1em;'>
            {alert["cause"]}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("View Attribution Details"):
        st.write(alert["details"])
