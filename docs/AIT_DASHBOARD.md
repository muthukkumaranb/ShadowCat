# AIT-LDS Multistage Dashboard (G-t6)

## Overview
The `ait_multistage_dash.py` script is a self-contained Streamlit application designed to visualize the entire evaluation pipeline executed on the AIT-LDS v2.0 dataset (scenario: `russellmitchell`).

## Key Features

### 1. Model Performance Overview
Displays high-level KPIs tracking the transferability of the `LSTMGaussianWorldModel`:
- **Zero-Shot MSE**: Validates out-of-the-box model performance on alien network data.
- **Fine-Tuned MSE**: Shows the delta reduction in error after 1 epoch of tuning to adapt to missing feature domains (Traffic Volume, Flow Timing).
- **Conformal Coverage**: Showcases the empirical coverage (99.25%) achieved by standard split conformal calibration, demonstrating that the uncertainty bounds encompass the true state effectively.

### 2. State Trajectory Visualization
Utilizes Plotly to render an interactive Line Chart that visualizes:
- **True State**: The observed trajectory of the network.
- **Predicted State**: The output $\mu_{t+1}$ from the fine-tuned world model.
- **Conformal Bounds**: The shaded $1-\alpha$ (90%) uncertainty envelope defined by $\pm$ the conformal threshold width (`0.3012`).

### 3. Attack Stage Transition Ledger
Summarizes the exact progression of attack steps within the 1-minute time windows, breaking out the multi-stage nature of AIT-LDS (`attackA -> attackB`) directly in the UI.

## Usage
To run the dashboard locally:
```bash
pip install streamlit plotly pandas numpy
streamlit run frontend/ait_multistage_dash.py
```
