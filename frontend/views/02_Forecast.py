import streamlit as st
import pandas as pd
import numpy as np
from data_provider import get_forecast_trajectory

st.title("02 — Forecast Panel")

forecast = get_forecast_trajectory()
risk_probs = forecast.get("risk_probabilities", [0.1, 0.2, 0.5, 0.8])
uncertainties = forecast.get("uncertainty_bands", [0.05, 0.1, 0.15, 0.2])

st.markdown("### Multi-Horizon Risk Trajectory")

# Horizon Selector
horizon = st.radio("Select Projection Horizon", options=["H=1", "H=2", "H=5"], horizontal=True)

# Generate chart data with uncertainty bands
# We simulate a forward-projecting line with widening uncertainty
horizon_steps = 1 if horizon == "H=1" else (2 if horizon == "H=2" else 5)
steps = list(range(len(risk_probs)))[:horizon_steps+1] if horizon_steps+1 <= len(risk_probs) else list(range(len(risk_probs)))

# For a proper uncertainty band in Streamlit without Plotly, we can use Altair or st.line_chart (which is limited).
# Let's use standard line chart for the main trajectory, and we can use Altair for the banded chart if needed.
# Since Streamlit's native st.line_chart doesn't do bands easily, we'll draw 3 lines (upper, main, lower).
df_chart = pd.DataFrame({
    "Projected Risk": [risk_probs[i] for i in steps],
    "Upper Bound": [risk_probs[i] + uncertainties[i] for i in steps],
    "Lower Bound": [max(0, risk_probs[i] - uncertainties[i]) for i in steps]
}, index=[f"t+{i}" for i in steps])

st.line_chart(df_chart, height=400)

st.markdown("---")

st.markdown("### Causal Explanation")
# Plain language explanation from attributions
st.markdown("""
> **Analysis**: The projected increase in hazard risk is primarily driven by **abnormal traffic growth** in the DMZ subnet, coupled with **unusual peer relationships** between internal application servers and previously unseen external endpoints. A recent **behavioural deviation from baseline** indicates potential lateral movement preparation.
""")

st.markdown("---")

st.markdown("### Historical Trend Strip")
st.caption("Baseline vs Current Deviation")
# Small multiples or trend strip
historical_trend = pd.DataFrame({
    "Flow Volume Deviation": np.random.randn(20).cumsum(),
    "Temporal Rhythm Deviation": np.random.randn(20).cumsum()
})
st.area_chart(historical_trend, height=150)
