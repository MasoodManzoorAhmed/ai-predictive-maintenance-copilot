# ai_predictive_maintenance_copilot/streamlit_app/app.py

from dotenv import load_dotenv
load_dotenv()

import os
import requests
import streamlit as st


def get_api_base_url() -> str:
    url = os.getenv("API_BASE_URL", "http://localhost:8000").strip()
    return url.rstrip("/")


API_BASE_URL = get_api_base_url()
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

st.session_state["API_BASE_URL"] = API_BASE_URL
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
    index=0
)
st.session_state["selected_dataset"] = dataset

st.sidebar.divider()
st.sidebar.markdown("### 🔌 Backend")

st.sidebar.write("API Base URL:")
st.sidebar.code(API_BASE_URL)

if st.sidebar.button("Health Check"):
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=API_TIMEOUT)
        if r.status_code == 200:
            st.sidebar.success(f"✅ OK: {r.json()}")
        else:
            st.sidebar.warning(f"⚠️ Status {r.status_code}: {r.text}")
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
