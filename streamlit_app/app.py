# ai_predictive_maintenance_copilot/streamlit_app/app.py
from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

# ============================================================
# ENV LOADING (SAFE FOR LOCAL + DOCKER + CLOUD)
# - Loads streamlit_app/.env.local ONLY if it exists (local dev)
# - In Docker/Cloud, pass env vars via runtime (Compose / Cloud Run)
# ============================================================
LOCAL_ENV = Path(__file__).resolve().parent / ".env.local"  # streamlit_app/.env.local
if LOCAL_ENV.exists():
    load_dotenv(LOCAL_ENV)


def get_api_base_url() -> str:
    """
    Priority:
    1) API_BASE_URL from environment (Docker/Cloud Run/CI should set this)
    2) fallback to local default (127.0.0.1) ONLY for local development

    NOTE (Cloud Run):
    - You MUST set API_BASE_URL in the Cloud Run Streamlit service.
      Example: https://cmaps-api-xxxxx-<region>.a.run.app
    """
    url = os.getenv("API_BASE_URL", "").strip()

    # Local dev fallback only
    if not url:
        url = "http://127.0.0.1:8000"

    return url.rstrip("/")


def get_api_timeout() -> int:
    raw = os.getenv("API_TIMEOUT", "30").strip()
    try:
        return int(raw)
    except ValueError:
        return 30


API_BASE_URL = get_api_base_url()
API_TIMEOUT = get_api_timeout()

# Only set defaults once (Streamlit reruns the script often)
if "API_BASE_URL" not in st.session_state:
    st.session_state["API_BASE_URL"] = API_BASE_URL
if "API_TIMEOUT" not in st.session_state:
    st.session_state["API_TIMEOUT"] = API_TIMEOUT

st.set_page_config(page_title="AI Predictive Maintenance Copilot", layout="wide")


# ----------------------------
# Sidebar (global app controls)
# ----------------------------
st.sidebar.title("🛠️ Maintenance Copilot")
st.sidebar.caption("Streamlit Dashboard (Phase 8/9)")

dataset = st.sidebar.selectbox(
    "Select dataset",
    ["FD001", "FD002", "FD003", "FD004"],
    index=0,
)
st.session_state["selected_dataset"] = dataset

st.sidebar.divider()
st.sidebar.markdown("### 🔌 Backend")

st.sidebar.write("API Base URL:")
st.sidebar.code(st.session_state["API_BASE_URL"])

# Extra: show a warning if you're clearly in cloud but still on localhost
if "127.0.0.1" in st.session_state["API_BASE_URL"] or "localhost" in st.session_state["API_BASE_URL"]:
    st.sidebar.warning(
        "You are pointing to localhost. For Cloud Run, set API_BASE_URL to the Cloud Run API URL."
    )

if st.sidebar.button("Health Check"):
    try:
        r = requests.get(
            f"{st.session_state['API_BASE_URL']}/health",
            timeout=st.session_state["API_TIMEOUT"],
        )
        if r.status_code == 200:
            st.sidebar.success(f"✅ OK: {r.json()}")
        else:
            st.sidebar.warning(f"⚠️ Status {r.status_code}: {r.text[:200]}")
    except Exception as e:
        st.sidebar.error(f"❌ Cannot reach API: {e}")


# ----------------------------
# Main landing page
# ----------------------------
st.title("AI Predictive Maintenance Copilot")
st.write(
    f"Selected dataset: **{st.session_state['selected_dataset']}**  \n"
    "Use the pages on the left (Streamlit Pages) to navigate."
)

st.info("Next: open **Pages → 1_Overview** to see the project overview.")
