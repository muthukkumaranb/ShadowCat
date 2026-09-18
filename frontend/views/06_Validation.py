import streamlit as st
import subprocess
import os

st.title("06 — Validation & Trust")

# --- Audit Chain Viewer ---
st.markdown("### Cryptographic Audit Chain")
st.caption("Verifies the integrity of the predictive pipeline via SHA-256 hash chaining.")

if st.button("Verify Chain Integrity", type="primary"):
    with st.spinner("Running verification..."):
        try:
            # Assuming verify_audit_chain.py is in the backend directory
            backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "backend", "verify_audit_chain.py")
            result = subprocess.run(["python", backend_path], capture_output=True, text=True, timeout=10)
            
            st.code(result.stdout, language="text")
            
            if result.returncode == 0:
                st.success("Chain Verified Successfully")
            else:
                st.error("Chain Verification Failed")
        except Exception as e:
            st.error(f"Failed to execute verification script: {e}")

st.markdown("---")

# --- Model Provenance ---
st.markdown("### Model Provenance")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Core Models Loaded**")
    st.write("- LSTM v4 (Sequence Forecaster)")
    st.write("- Hazard Head v4")
    st.write("- Stage Classifier Head v2")
with col2:
    st.markdown("**Fallback Status**")
    st.warning("pca_available: False")
    st.markdown("> **Note**: The hazard head is operating without full PCA calibration. Probabilities may exhibit higher variance in extreme out-of-distribution scenarios.")

st.markdown("---")

# --- TRL & Limitations ---
st.markdown("### Platform Status & Limitations")
st.info("**TRL 4 (MVP)**: Technology validated in lab environment.")

st.markdown("""
**Known Limitations & Honest Disclosures (Extracted from README)**:
- **Dataset Constraints**: Trained primarily on CSE-CIC-IDS2018. May not generalize to entirely novel topologies without fine-tuning.
- **Latency**: GraphSAGE fusion currently adds ~450ms overhead per window, which is why it is held back from the primary operational forecast.
- **Explainability**: Integrated Gradients provide local feature attribution, but causal directionality in highly cyclical flows remains an approximation.
""")
