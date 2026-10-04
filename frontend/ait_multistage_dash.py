import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def run_dashboard():
    st.set_page_config(page_title="AIT-LDS Multistage Dashboard", layout="wide")
    st.title("🛡️ AIT-LDS Multistage Forecasting Dashboard")
    st.markdown("Visualizing Zero-Shot vs Fine-Tuned Model Performance and Conformal Uncertainty Bounds for the `russellmitchell` scenario.")
    
    # 1. Metrics Overview
    st.header("1. Model Performance (G-t4 & G-t5)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Zero-Shot MSE", value="0.1175")
    with col2:
        st.metric(label="Fine-Tuned MSE (1 Epoch)", value="0.0825", delta="-0.0350", delta_color="inverse")
    with col3:
        st.metric(label="Empirical Conformal Coverage", value="99.25%", delta="+9.25% (Target 90%)")
        
    st.divider()
    
    # 2. Conformal Prediction Bounds Visualization
    st.header("2. State Trajectory with Conformal Uncertainty Bounds")
    st.markdown("This chart illustrates the true state (synthetic for demonstration) versus the predicted state, enveloped by the $1-\\alpha$ conformal prediction intervals derived during calibration.")
    
    # Generate synthetic sequence data for visualization
    steps = 100
    x = np.arange(steps)
    true_state = np.sin(x / 5.0) + np.random.normal(0, 0.1, steps)
    pred_state = np.sin(x / 5.0)
    
    # Bound width = 0.3012 from G-t5
    upper_bound = pred_state + 0.3012
    lower_bound = pred_state - 0.3012
    
    fig = go.Figure()
    
    # Conformal Bounds Area
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% Conformal Interval'
    ))
    
    # True State
    fig.add_trace(go.Scatter(
        x=x, y=true_state, mode='lines', name='True Observed State', line=dict(color='red')
    ))
    
    # Predicted State
    fig.add_trace(go.Scatter(
        x=x, y=pred_state, mode='lines', name='Predicted State (Fine-Tuned)', line=dict(color='blue', dash='dash')
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Transition Logic
    st.header("3. Multi-Stage Transitions Detected")
    st.markdown("The sequential nature of the AIT-LDS dataset allows the world model to trace stage-by-stage transitions.")
    
    # From transition_stats.py
    transitions = pd.DataFrame({
        "Transition Type": ["benign -> attack", "attackA -> attackB", "attack -> benign", "benign -> benign"],
        "Count": [1, 23, 1, 745]
    })
    st.table(transitions)

if __name__ == "__main__":
    run_dashboard()
