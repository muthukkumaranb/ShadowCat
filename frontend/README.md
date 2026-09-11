# SHADOWCAT — Frontend Dashboard & Analyst Interface

The frontend track delivers the executive and operational user interface for the SHADOWCAT platform. Built with **Streamlit**, **Plotly**, and a custom dark-mode design system, it renders pre-emptive attack forecasts, multi-step horizon trajectories, forensic attributions, and scientific validation benchmarks.

---

## 🧭 Dashboard Views & Navigation

```
frontend/
├── app.py                   # Master entrypoint & page router
├── data_provider.py         # Decoupled data access layer (Live backend vs mock fallback)
├── styles.py                # CSS design system (Dark glassmorphism, responsive tokens)
├── views/
│   ├── 01_Forecast.py       # Live risk forecasting & pre-emptive trajectory (t+1..t+4)
│   ├── 01a_Input.py         # Network flow ingestion & real-time telemetry analyzer
│   ├── 02_Evidence.py       # Telemetry evidence, dual-signal evaluation & attributions
│   ├── 03_Validation.py     # Scientific rigor, LOEO 37-fold cross-validation & audit chain
│   └── 05_About.py          # Platform specs, threat models & architecture handbook
├── components/              # Modular UI components (header, charts, cards, tables)
├── models/                  # Pre-computed benchmark outputs & metadata manifests
└── requirements.txt         # Frontend dependencies
```

---

## 🎨 Key Features & Capabilities

1. **Pre-Emptive Lead Time Visualization**:
   - Visualizes multi-step hazard trajectories across forecasting horizons ($t+1$ to $t+4$).
   - Displays estimated pre-emptive lead time (up to 3.5 minutes prior to reactive signature detection).
2. **Dual-Signal Telemetry Evaluation**:
   - Separates **Known Attack Risk** (hazard trajectory) from **Novelty Drift** (Gaussian NLL anomaly score).
3. **Forensic Feature Attribution**:
   - Renders deletion-tested feature rankings (e.g. `Duration Sec Max`, `Fwd Iat Tot Max`) to explain model decisions without black-box opacity.
4. **Scientific Validation & Baseline Comparisons**:
   - Direct comparison tables against Lagged Logistic Regression and Reactive Signature IDS.
   - Comprehensive LOEO (Leave-One-Episode-Out) 37-fold cross-validation benchmark metrics.
5. **Tamper-Evident Hash Chain Panel**:
   - Live visual status of the blockchain-inspired SHA-256 audit ledger verifying model weights and evaluation reports.

---

## 🚀 Running the Dashboard

### Standalone Launch:
```bash
streamlit run frontend/app.py
```
Default URL: `http://localhost:8501`

### Data Decoupling Architecture (`data_provider.py`):
`data_provider.py` acts as a clean boundary between the UI and backend logic:
- Auto-detects live model checkpoints and executes `backend.predict()` on demand.
- Seamlessly falls back to pre-computed benchmark artifacts when operating in offline demo mode.
