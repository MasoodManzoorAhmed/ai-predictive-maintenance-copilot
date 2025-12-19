# ai_predictive_maintenance_copilot/streamlit_app/pages/1_Overview.py
from __future__ import annotations

import streamlit as st

# NOTE:
# Page config is set once in streamlit_app/app.py.
# Avoid calling st.set_page_config() in individual pages to prevent warnings.

dataset = st.session_state.get("selected_dataset", "FD001")

st.title("AI Predictive Maintenance Copilot")
st.caption(
    "Production-style demo for Remaining Useful Life (RUL) prediction on NASA CMAPSS turbofan datasets "
    "with an API-first architecture (FastAPI) and a stakeholder-friendly UI (Streamlit)."
)

# ---- Quick status cards ----
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Selected Dataset", dataset)
with c2:
    st.metric("UI", "Streamlit (Cloud Run)")
with c3:
    st.metric("API", "FastAPI (Cloud Run)")
with c4:
    st.metric("Use Case", "Predictive Maintenance")

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("What this system does")
    st.write(
        """
This application predicts **Remaining Useful Life (RUL)** for aircraft engine units using the **NASA CMAPSS** benchmark datasets (FD001–FD004).

It is built as a **real deployment-style product demo**:
- A **FastAPI backend** loads trained models + preprocessing artifacts and serves low-latency inference endpoints.
- A **Streamlit UI** provides a clean interface for non-technical users (operations/maintenance/management).
- The design supports local development, Docker, and cloud deployment with environment-based configuration.
"""
    )

    st.subheader("Key capabilities")
    st.markdown(
        """
- **RUL Prediction (FD001–FD004)** via a unified API interface
- **Input validation** (request schema) and predictable error handling
- **Health endpoint** for service readiness checks
- **Dashboard UX** for quick testing and demonstrations
- **Cloud deployment** on Google Cloud Run (API + UI as separate services)
"""
    )

with right:
    st.subheader("Architecture")
    st.markdown(
        """
**UI (Streamlit)** → calls → **API (FastAPI)** → loads → **Models + Scalers + Configs** → returns → **RUL prediction**

This split proves:
- MLOps deployment readiness
- Service boundaries (frontend/backend)
- Clean configuration management using environment variables
"""
    )

    st.subheader("How to use this demo")
    st.markdown(
        """
1) Use the left sidebar to choose dataset (FD001–FD004)  
2) Run **Health Check** to confirm the API is reachable  
3) Go to **Predict RUL** to run an end-to-end prediction  
4) Use **Analytics** to view model/behavior insights  
5) Use **Copilot** (RAG) to ask maintenance-oriented questions (Phase 10/11)
"""
    )

st.divider()

st.subheader("Skills demonstrated")
st.markdown(
    """
- **Machine Learning Engineering:** model packaging, artifact management, inference consistency  
- **Backend Engineering:** FastAPI, request/response validation, health checks, service design  
- **MLOps / Deployment:** Docker images, Cloud Run deployment, env-based configs, reproducibility  
- **Product Thinking:** clear UX for demos, stakeholder-friendly layout, real-world delivery story
"""
)

st.info(
    f"✅ You are viewing the system in **{dataset}** mode. Next: open **Predict RUL** and run one inference end-to-end."
)
