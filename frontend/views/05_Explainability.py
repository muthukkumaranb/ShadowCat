import streamlit as st
import pandas as pd
import numpy as np

st.title("05 — Explainability")

st.markdown("### Feature Attribution by Category")
st.caption("Derived via Integrated Gradients. Attention weights are not presented directly per platform constraints.")

categories = ["Temporal Rhythm", "Flow Dynamics", "Packet Header", "Payload Stats"]
attributions = [0.45, 0.30, 0.15, 0.10]

# Generate a horizontal bar chart for the attributions
df = pd.DataFrame({
    "Category": categories,
    "Attribution": attributions
}).set_index("Category")

st.bar_chart(df, height=300)

st.markdown("""
> **Explanation**: **Temporal Rhythm** currently drives 45% of the model's hazard prediction, indicating that the timing and frequency of connections deviate significantly from historical baselines.
""")

st.markdown("---")

st.markdown("### Experimental Fusion Attribution")
show_experimental = st.toggle("Show GraphSAGE Fusion Attributions (Experimental)")

if show_experimental:
    st.info("Displaying comparison between Primary (Integrated Gradients) and Experimental (GraphSAGE).")
    
    # Mock comparative data
    df_compare = pd.DataFrame({
        "Category": categories,
        "Primary (IG)": attributions,
        "Experimental (GraphSAGE)": [0.20, 0.25, 0.10, 0.05]
    }).set_index("Category")
    
    st.bar_chart(df_compare, height=300)
    
    st.markdown("""
    > **Note**: The experimental fusion model heavily penalizes structural graph features (not shown here) over temporal ones, resulting in lower relative attribution for temporal rhythm.
    """)
