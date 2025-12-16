# ai_predictive_maintenance_copilot/streamlit_app/pages/1_Overview.py

import streamlit as st

st.title("📌 Overview")

dataset = st.session_state.get("selected_dataset", "FD001")

st.write(f"Current dataset: **{dataset}**")

st.markdown(
"""
### What this system does
This dashboard connects to a FastAPI backend that serves trained RUL (Remaining Useful Life) models for NASA CMAPSS datasets.

### What you will be able to do in Phase 8
- Upload engine sensor data (CSV/JSON)
- Get predicted RUL for FD001–FD004
- Visualize degradation trends
- Use an AI Copilot (RAG) to explain maintenance recommendations

"""
)

st.success("✅ Phase 8 Step 1 complete: Sidebar + Overview page is working.")
