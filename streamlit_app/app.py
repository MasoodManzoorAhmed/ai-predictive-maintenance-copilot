# ai_predictive_maintenance_copilot/streamlit_app/app.py
from __future__ import annotations

import os

import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()


def get_api_base_url() -> str:
    url = os.getenv("API_BASE_URL", "http://localhost:8000").strip()
    return url.rstrip("/")


def get_api_timeout() -> int:
    raw = os.getenv("API_TIMEOUT", "30").strip()
    try:
        return int(raw)
    except ValueError:
        return 30


API_BASE_URL = get_api_base_url()
API_TIMEOUT = get_api_timeout()

# Only set defaults once (Streamlit reruns the file often)
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
# Main landing page (simple)
# ----------------------------
st.title("AI Predictive Maintenance Copilot")
st.write(
    f"Selected dataset: **{st.session_state['selected_dataset']}**  \n"
    "Use the pages on the left (Streamlit Pages) to navigate."
)

st.info("Next: open **Pages → 1_Overview** to see the project overview.")
