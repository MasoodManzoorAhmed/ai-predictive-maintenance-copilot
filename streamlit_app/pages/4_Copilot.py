# ai_predictive_maintenance_copilot/streamlit_app/pages/4_Copilot.py
from __future__ import annotations

import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================
# OPTIONAL PDF export (ReportLab) — Option B (SAFE)
# =========================
REPORTLAB_AVAILABLE = True
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
except Exception:
    REPORTLAB_AVAILABLE = False


# ============================================================
# Rules (must match your other pages)
# ============================================================
FD_RULES = {
    "FD001": {"seq_len": 30, "min_cycles_batch": 30},
    "FD002": {"seq_len": 100, "min_cycles_batch": 100},
    "FD003": {"seq_len": 30, "min_cycles_batch": 30},
    "FD004": {"seq_len": 100, "min_cycles_batch": 100},
}

# Risk bands (cycles)
RISK_BANDS = [
    {"name": "Critical", "icon": "🔴", "min": 0, "max": 20},
    {"name": "Warning", "icon": "🟠", "min": 21, "max": 60},
    {"name": "Watch", "icon": "🟡", "min": 61, "max": 100},
    {"name": "Healthy", "icon": "🟢", "min": 101, "max": 10_000},
]


# ============================================================
# Helpers
# ============================================================
def get_api_base_url() -> str:
    url = st.session_state.get("API_BASE_URL") or os.getenv("API_BASE_URL", "http://localhost:8000")
    return str(url).strip().rstrip("/")


def get_api_timeout() -> int:
    t = st.session_state.get("API_TIMEOUT") or os.getenv("API_TIMEOUT", "30")
    try:
        return int(t)
    except Exception:
        return 30


def get_batch_endpoint(fd_name: str) -> str:
    return f"/predict/{fd_name.strip().lower()}"


def get_single_endpoint() -> str:
    return "/single/predict"


def get_copilot_endpoint() -> str:
    # Matches FastAPI include_router(prefix="/copilot") + router.post("/query")
    return "/copilot/query"


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    return df2


def safe_sort_by_cycle(df: pd.DataFrame, cycle_col: str) -> pd.DataFrame:
    df2 = df.copy()
    df2[cycle_col] = pd.to_numeric(df2[cycle_col], errors="coerce")
    df2 = df2.dropna(subset=[cycle_col]).sort_values(cycle_col).reset_index(drop=True)
    return df2


def detect_window_from_df(df: pd.DataFrame, cycle_col: Optional[str]) -> Tuple[Optional[int], Optional[int], str]:
    df2 = _clean_columns(df)
    if cycle_col and cycle_col in df2.columns:
        s = pd.to_numeric(df2[cycle_col], errors="coerce")
        if s.notna().any():
            return int(s.min()), int(s.max()), "cycle column found"
        return None, None, "cycle column present but not numeric"
    if len(df2) == 0:
        return None, None, "no rows"
    return 1, len(df2), "cycle column missing; using row order (1..N)"


def dataframe_to_batch_payload(df: pd.DataFrame, unit_col: Optional[str], cycle_col: Optional[str]) -> Dict[str, Any]:
    df2 = _clean_columns(df)

    if unit_col and unit_col in df2.columns:
        if unit_col != "unit":
            df2 = df2.rename(columns={unit_col: "unit"})
    else:
        df2["unit"] = 1

    if cycle_col and cycle_col in df2.columns:
        if cycle_col != "cycle":
            df2 = df2.rename(columns={cycle_col: "cycle"})
    else:
        df2["cycle"] = df2.groupby("unit").cumcount() + 1

    drop_targets = [c for c in ["RUL", "rul", "target", "label"] if c in df2.columns]
    if drop_targets:
        df2 = df2.drop(columns=drop_targets)

    return {"records": df2.to_dict(orient="records")}


def dataframe_to_single_payload(
    df: pd.DataFrame,
    fd_name: str,
    unit_col: Optional[str],
    cycle_col: Optional[str],
) -> Dict[str, Any]:
    df2 = _clean_columns(df)

    if unit_col and unit_col in df2.columns:
        try:
            unit_id = int(pd.to_numeric(df2[unit_col], errors="coerce").dropna().iloc[0])
        except Exception:
            unit_id = 1
    else:
        unit_id = 1

    if cycle_col and cycle_col in df2.columns:
        if cycle_col != "cycle":
            df2 = df2.rename(columns={cycle_col: "cycle"})
    else:
        df2["cycle"] = range(1, len(df2) + 1)

    df2 = df2.drop(columns=[unit_col] if (unit_col and unit_col in df2.columns) else [], errors="ignore")
    df2 = df2.drop(columns=["unit"], errors="ignore")

    drop_targets = [c for c in ["RUL", "rul", "target", "label"] if c in df2.columns]
    if drop_targets:
        df2 = df2.drop(columns=drop_targets)

    return {"fd_name": fd_name.upper(), "unit": int(unit_id), "rows": df2.to_dict(orient="records")}


def extract_rul_any(out: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    raw = None
    cal = None

    if isinstance(out, dict) and "predictions" in out and isinstance(out["predictions"], list) and out["predictions"]:
        p0 = out["predictions"][0]
        if isinstance(p0, dict):
            raw = p0.get("pred_rul_raw", p0.get("predicted_rul", p0.get("rul")))
            cal = p0.get("pred_rul_calibrated", p0.get("predicted_rul_calibrated"))
    else:
        raw = out.get("pred_rul_raw", out.get("predicted_rul", out.get("rul")))
        cal = out.get("pred_rul_calibrated", out.get("predicted_rul_calibrated"))

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    raw_f = _to_float(raw)
    cal_f = _to_float(cal)
    display = cal_f if cal_f is not None else raw_f
    return raw_f, cal_f, display


def risk_band_from_rul(rul_value: Optional[float]) -> Dict[str, Any]:
    if rul_value is None or (isinstance(rul_value, float) and np.isnan(rul_value)):
        return {"name": "Unknown", "icon": "⚪", "min": None, "max": None}

    rv = float(rul_value)
    for b in RISK_BANDS:
        if b["min"] <= rv <= b["max"]:
            return b
    return {"name": "Unknown", "icon": "⚪", "min": None, "max": None}


def risk_score_0_100(rul_value: Optional[float]) -> Optional[float]:
    """
    Simple deterministic risk score:
      - 0 risk at RUL>=200
      - 100 risk at RUL<=0
      - linear in between
    """
    if rul_value is None or (isinstance(rul_value, float) and np.isnan(rul_value)):
        return None
    rv = float(rul_value)
    rv = max(0.0, min(200.0, rv))
    return float(round(100.0 * (1.0 - (rv / 200.0)), 2))


def pick_numeric_sensors(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def compute_trend_slopes(df_unit: pd.DataFrame, cycle_col: str, sensor_cols: List[str]) -> pd.DataFrame:
    """
    Deterministic 'degradation hints' using slope over cycle.
    Returns a table with slope, abs_slope, corr, last_z
    """
    x = pd.to_numeric(df_unit[cycle_col], errors="coerce").values.astype(float)
    out_rows = []

    if len(x) < 5 or np.isnan(x).all():
        return pd.DataFrame(columns=["feature", "slope", "abs_slope", "corr", "last_z"])

    x = np.nan_to_num(x, nan=np.nanmedian(x))
    x_centered = x - np.mean(x)
    denom = np.sum(x_centered**2) if np.sum(x_centered**2) != 0 else 1.0

    for c in sensor_cols:
        y = pd.to_numeric(df_unit[c], errors="coerce").values.astype(float)
        if np.isnan(y).all():
            continue
        y = np.nan_to_num(y, nan=np.nanmedian(y))

        y_centered = y - np.mean(y)
        slope = float(np.sum(x_centered * y_centered) / denom)

        try:
            corr = float(np.corrcoef(x, y)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0

        mu = float(np.mean(y))
        sd = float(np.std(y))
        if sd == 0:
            sd = 1.0
        last_z = float((y[-1] - mu) / sd)

        out_rows.append({"feature": c, "slope": slope, "abs_slope": abs(slope), "corr": corr, "last_z": last_z})

    res = pd.DataFrame(out_rows)
    if len(res) == 0:
        return res
    res = res.sort_values("abs_slope", ascending=False).reset_index(drop=True)
    return res


def copilot_actions(risk_name: str) -> List[str]:
    if risk_name == "Critical":
        return [
            "Stop/run-to-fail decision NOW: schedule immediate inspection.",
            "Prioritize safety checks + critical subsystems first.",
            "Prepare spares + maintenance crew (same shift if possible).",
            "Increase monitoring frequency (shorter sampling window).",
        ]
    if risk_name == "Warning":
        return [
            "Schedule maintenance soon (next available slot).",
            "Run focused diagnostics on top drifting sensors.",
            "Compare against fleet units with similar cycle range.",
            "Monitor trend daily; watch for sudden drops in RUL.",
        ]
    if risk_name == "Watch":
        return [
            "Continue operation but monitor trend weekly.",
            "Plan preventive maintenance parts ordering (no urgency).",
            "Track sensor drift; investigate if drift accelerates.",
        ]
    if risk_name == "Healthy":
        return [
            "No maintenance needed now; keep standard monitoring.",
            "Use this unit as a reference baseline for fleet.",
            "Re-check at next scheduled reporting interval.",
        ]
    return [
        "Upload valid data and run prediction first (RUL missing).",
        "Ensure you have ≥ seq_len rows for the selected FD dataset.",
    ]


def build_backend_copilot_payload(
    question: str,
    fd_name: str,
    unit: Optional[int],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Matches backend CopilotQueryRequest:
      {
        "question": "...",
        "fd_name": "FD001",
        "unit": 1,
        "extra": {...}
      }
    """
    payload: Dict[str, Any] = {
        "question": str(question).strip(),
        "fd_name": str(fd_name).strip().upper(),
        "extra": extra or {},
    }
    if unit is not None:
        payload["unit"] = int(unit)
    return payload


# =========================
# Export helpers
# =========================
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ✅ FIX B (PDF): make all text safe for ReportLab Helvetica (removes emoji/unicode)
def pdf_safe_text(x: Any) -> str:
    s = "" if x is None else str(x)
    return s.encode("ascii", errors="ignore").decode("ascii")


def build_fleet_pdf_report(
    dataset: str,
    seq_len: int,
    fleet_df: pd.DataFrame,
    worst_df: pd.DataFrame,
    summary: Dict[str, Any],
) -> bytes:
    """
    Minimal, professional PDF summary for managers (no images).
    Requires reportlab. Caller MUST check REPORTLAB_AVAILABLE first.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab not available")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    _, height = A4

    y = height - 0.8 * inch
    c.setFont("Helvetica-Bold", 14)
    c.drawString(0.8 * inch, y, "Predictive Maintenance Fleet Report")
    y -= 0.35 * inch

    c.setFont("Helvetica", 10)
    c.drawString(0.8 * inch, y, pdf_safe_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    y -= 0.2 * inch
    c.drawString(0.8 * inch, y, pdf_safe_text(f"Dataset: {dataset} | seq_len: {seq_len}"))
    y -= 0.35 * inch

    c.setFont("Helvetica-Bold", 11)
    c.drawString(0.8 * inch, y, "Summary")
    y -= 0.22 * inch
    c.setFont("Helvetica", 10)
    for k, v in summary.items():
        c.drawString(0.95 * inch, y, pdf_safe_text(f"- {k}: {v}"))
        y -= 0.18 * inch

    y -= 0.2 * inch

    def draw_table(title: str, df: pd.DataFrame, max_rows: int = 15):
        nonlocal y
        if y < 2.2 * inch:
            c.showPage()
            y = height - 0.8 * inch

        c.setFont("Helvetica-Bold", 11)
        c.drawString(0.8 * inch, y, pdf_safe_text(title))
        y -= 0.25 * inch

        c.setFont("Helvetica", 9)
        cols = [pdf_safe_text(col) for col in df.columns]
        c.drawString(0.8 * inch, y, pdf_safe_text(" | ".join(cols)[:110]))
        y -= 0.18 * inch

        n = min(len(df), max_rows)
        for i in range(n):
            line = " | ".join([pdf_safe_text(v) for v in df.iloc[i].tolist()])
            c.drawString(0.8 * inch, y, pdf_safe_text(line[:110]))
            y -= 0.16 * inch

        if len(df) > max_rows:
            c.drawString(0.8 * inch, y, pdf_safe_text(f"... truncated ({len(df)} rows total)"))
            y -= 0.18 * inch

        y -= 0.15 * inch

    draw_table("Fleet results (top rows)", fleet_df, max_rows=15)
    draw_table("Top priority units (lowest RUL)", worst_df, max_rows=10)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()


# =========================
# Heatmap plot helper
# =========================
def plot_fleet_risk_heatmap(ok_df: pd.DataFrame, value_col: str = "risk_score", max_units: int = 30) -> None:
    dfh = ok_df.copy()
    dfh = dfh.dropna(subset=[value_col]).copy()
    if len(dfh) == 0:
        st.info("Heatmap unavailable (no valid values).")
        return

    dfh = dfh.sort_values(value_col, ascending=False).head(max_units)
    units = dfh["unit"].astype(int).tolist()
    vals = dfh[value_col].astype(float).tolist()
    data = np.array(vals).reshape(-1, 1)

    fig = plt.figure(figsize=(6, max(2.5, 0.35 * len(units))))
    ax = plt.gca()
    im = ax.imshow(data, aspect="auto")

    ax.set_yticks(range(len(units)))
    ax.set_yticklabels([str(u) for u in units])
    ax.set_xticks([0])
    ax.set_xticklabels([value_col])

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.title("Fleet Risk Heatmap (higher = more risk)")
    plt.xlabel("Metric")
    plt.ylabel("Unit")

    for i, v in enumerate(vals):
        ax.text(0, i, f"{v:.1f}", ha="center", va="center")

    st.pyplot(fig, clear_figure=True)


# =========================
# Enhancements: Days / SLA / Scenario
# =========================
def rul_to_days(rul_cycles: Optional[float], cycles_per_day: float) -> Optional[float]:
    if rul_cycles is None or (isinstance(rul_cycles, float) and np.isnan(rul_cycles)):
        return None
    cpd = float(max(0.0001, cycles_per_day))
    return float(round(float(rul_cycles) / cpd, 2))


def sla_alert(days_to_failure: Optional[float], sla_days: Dict[str, float]) -> str:
    """
    sla_days = {"critical":2, "warning":7, "watch":30, "healthy":90}
    """
    if days_to_failure is None:
        return "⚪ SLA: N/A (days-to-failure unknown)."
    d = float(days_to_failure)
    if d <= sla_days["critical"]:
        return f"🔴 SLA BREACH: predicted failure in ~{d} days (≤ {sla_days['critical']}d)."
    if d <= sla_days["warning"]:
        return f"🟠 SLA warning: predicted failure in ~{d} days (≤ {sla_days['warning']}d)."
    if d <= sla_days["watch"]:
        return f"🟡 SLA attention: predicted failure in ~{d} days (≤ {sla_days['watch']}d)."
    if d <= sla_days["healthy"]:
        return f"🟢 SLA attention: predicted failure in ~{d} days (≤ {sla_days['healthy']}d)."
    return f"🟢 SLA OK: predicted failure in ~{d} days (> {sla_days['healthy']}d)."


def apply_delay_to_rul(rul_cycles: Optional[float], delay_days: float, cycles_per_day: float) -> Optional[float]:
    if rul_cycles is None or (isinstance(rul_cycles, float) and np.isnan(rul_cycles)):
        return None
    new_rul = float(rul_cycles) - float(delay_days) * float(cycles_per_day)
    return float(max(0.0, round(new_rul, 2)))


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Copilot", layout="wide")

API_BASE_URL = get_api_base_url()
API_TIMEOUT = get_api_timeout()

dataset = st.session_state.get("selected_dataset", "FD001").upper()
seq_len = FD_RULES.get(dataset, {}).get("seq_len", 30)

batch_url = API_BASE_URL + get_batch_endpoint(dataset)
single_url = API_BASE_URL + get_single_endpoint()
copilot_url = API_BASE_URL + get_copilot_endpoint()

st.title("🧠  Maintenance Copilot (Local Rules + Backend RAG/LLM) ")
st.caption(f"Dataset: **{dataset}** | seq_len required: **{seq_len}**")
st.caption(f"Batch endpoint: `{batch_url}` | Single endpoint: `{single_url}` | Copilot endpoint: `{copilot_url}`")

# -------------------------------------------------------------------
# ✅ FIX A (UI): REMOVE the top Enhancements UI section completely
# (No other logic changed; the underlying functions still exist and are used later.)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# ✅ Persist fleet results + enhancement values across reruns
# -------------------------------------------------------------------
if "fleet_cache_key" not in st.session_state:
    st.session_state["fleet_cache_key"] = ""
if "fleet_res" not in st.session_state:
    st.session_state["fleet_res"] = None

# Enhancement defaults (persisted) — kept because later parts rely on these values
if "cycles_per_day" not in st.session_state:
    st.session_state["cycles_per_day"] = 29.0
if "sla_days" not in st.session_state:
    st.session_state["sla_days"] = {"critical": 2.0, "warning": 7.0, "watch": 30.0, "healthy": 90.0}
if "delay_days_fleet" not in st.session_state:
    st.session_state["delay_days_fleet"] = 3.0

st.divider()

with st.expander("📌 Which file should I upload here? (simple)", expanded=False):
    st.markdown(
        f"""
**For single engine Copilot:** upload a CSV with **≥ {seq_len} rows** (1 unit is fine).  
- If you have `unit` and `cycle` columns → perfect  
- If not, Copilot will assume `unit=1` and infer cycles by row order.

**For fleet Copilot:** upload a **multi-engine CSV** containing a `unit` column.  
Each unit should have **≥ {seq_len} rows** for a clean comparison.
"""
    )

st.subheader("1) Upload engine data (CSV)")
uploaded = st.file_uploader("Upload CSV for Copilot", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to start Copilot recommendations.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

df = _clean_columns(df)

col_options = ["(not present)"] + list(df.columns)
default_unit = "unit" if "unit" in df.columns else "(not present)"
default_cycle = "cycle" if "cycle" in df.columns else "(not present)"

m1, m2 = st.columns(2)
with m1:
    unit_choice = st.selectbox("Unit/Engine column (optional)", col_options, index=col_options.index(default_unit))
with m2:
    cycle_choice = st.selectbox("Cycle/Time column (optional)", col_options, index=col_options.index(default_cycle))

unit_col = None if unit_choice == "(not present)" else unit_choice
cycle_col = None if cycle_choice == "(not present)" else cycle_choice

st.write("Preview:")
st.dataframe(df.head(15), use_container_width=True)

w_start, w_end, w_reason = detect_window_from_df(df, cycle_col)
if w_start is not None and w_end is not None:
    st.success(f"Window detected: cycles **{w_start}–{w_end}**")
else:
    st.warning(f"Window detection issue: {w_reason}")

st.divider()


# Backend Copilot  (RAG — LLM optional)
# ------------------------------------------------------------
st.subheader("🤖 Backend Copilot ")
st.caption("Calls FastAPI Copilot endpoint. Payload uses fd_name/unit/extra")

# Persist debug info safely
if "copilot_debug_payload" not in st.session_state:
    st.session_state["copilot_debug_payload"] = None
if "copilot_debug_response" not in st.session_state:
    st.session_state["copilot_debug_response"] = None

q_default = "Give maintenance recommendation based on risk band and what I should do next."
question = st.text_area("Copilot question", value=q_default, height=80)

bc1, bc2, bc3 = st.columns([1, 1, 2])
with bc1:
    unit_str = st.text_input("Optional unit (integer)", value="")
with bc2:
    style = st.selectbox("Response style", ["Checklist", "Concise", "Detailed"], index=0)
with bc3:
    role = st.selectbox("Role", ["Maintenance Manager", "Technician", "Reliability Engineer"], index=0)

extra = {"style": style, "role": role, "phase": "10_rag_ui"}

unit_val: Optional[int] = None
if unit_str.strip():
    if unit_str.strip().isdigit():
        unit_val = int(unit_str.strip())
    else:
        st.warning("Unit must be an integer. Leaving unit empty is fine.")

call_backend = st.button("Ask Backend Copilot", type="primary")

if call_backend:
    payload = build_backend_copilot_payload(
        question=question,
        fd_name=dataset,
        unit=unit_val,
        extra=extra,
    )

    st.session_state["copilot_debug_payload"] = payload

    try:
        with st.spinner("Calling /copilot/query ..."):
            r = requests.post(copilot_url, json=payload, timeout=API_TIMEOUT)

        if r.status_code != 200:
            st.error(f"Copilot API error {r.status_code}")
            st.code(r.text[:1500])
            st.session_state["copilot_debug_response"] = r.text
        else:
            out = r.json()
            st.session_state["copilot_debug_response"] = out

            answer = out.get("answer") if isinstance(out, dict) else ""

            st.success("Copilot response received ✅")
            st.markdown("### Answer")
            st.write(answer if answer else "(No `answer` field in response)")

            if isinstance(out, dict) and out.get("sources"):
                st.markdown("### Sources / Evidence")
                st.json(out.get("sources"))

    except Exception as e:
        st.error(f"Request failed: {e}")
        st.info("Check API_BASE_URL, FastAPI running, and /copilot/query route exists.")

# ------------------------------------------------------------
# Debug sections (TOP-LEVEL — Streamlit safe)
# ------------------------------------------------------------
if st.session_state.get("copilot_debug_payload") is not None:
    with st.expander("🛠 Debug payload"):
        st.code(
            json.dumps(st.session_state["copilot_debug_payload"], indent=2),
            language="json",
        )

if st.session_state.get("copilot_debug_response") is not None:
    with st.expander("🛠 Debug response JSON"):
        st.code(
            json.dumps(st.session_state["copilot_debug_response"], indent=2),
            language="json",
        )


st.divider()

# ------------------------------------------------------------
# Mode (local deterministic copilot)
# ------------------------------------------------------------
st.subheader("2) Local Copilot (deterministic, uses prediction endpoints)")
mode = st.radio(
    "Copilot mode",
    [
        "Single unit: predict latest-window RUL + recommendations",
        "Fleet: predict last-window RUL for multiple units + fleet analytics",
        "Offline: I already know RUL (enter manually) + recommendations",
    ],
    index=0,
)

# ------------------------------------------------------------
# Offline manual RUL
# ------------------------------------------------------------
if mode.startswith("Offline"):
    c1, c2 = st.columns(2)
    with c1:
        manual_rul = st.number_input("Enter Predicted RUL (cycles)", min_value=0.0, max_value=10_000.0, value=80.0, step=1.0)
    with c2:
        _ = st.number_input("Unit ID (label only)", min_value=1, value=1, step=1)

    band = risk_band_from_rul(manual_rul)
    score = risk_score_0_100(manual_rul)

    days = rul_to_days(manual_rul, float(st.session_state["cycles_per_day"]))
    sla_msg = sla_alert(days, st.session_state["sla_days"])

    st.markdown(
        f"### Risk Band: {band['icon']} **{band['name']}**  \n"
        f"Risk score (0–100): **{score if score is not None else 'N/A'}**  \n"
        f"Estimated days-to-failure: **{days if days is not None else 'N/A'}** days  \n"
        f"{sla_msg}"
    )

    st.markdown("### Recommended Actions")
    for a in copilot_actions(band["name"]):
        st.write(f"- {a}")

    st.stop()

# ------------------------------------------------------------
# Single unit
# ------------------------------------------------------------
if mode.startswith("Single unit"):
    st.caption("Uses your prediction API once (latest window = last seq_len rows).")

    use_endpoint = st.radio(
        "Inference endpoint",
        ["Auto (Batch preferred)", "Batch only", "Single only"],
        index=0,
        horizontal=True,
    )

    run = st.button("Run Single-Unit Copilot (local)", type="primary")
    if not run:
        st.stop()

    df_work = df.copy()

    if cycle_col and cycle_col in df_work.columns:
        df_work = safe_sort_by_cycle(df_work, cycle_col)
    else:
        df_work["cycle"] = range(1, len(df_work) + 1)
        cycle_col = "cycle"

    if unit_col and unit_col in df_work.columns:
        units_ = sorted(pd.to_numeric(df_work[unit_col], errors="coerce").dropna().unique().astype(int).tolist())
        if len(units_) == 0:
            units_ = [1]
        unit_pick = st.selectbox("Pick unit", units_, index=0)
        df_u = df_work[df_work[unit_col].astype(str) == str(unit_pick)].copy()
    else:
        unit_pick = 1
        df_u = df_work.copy()
        df_u["unit"] = 1
        unit_col = "unit"

    if len(df_u) < seq_len:
        st.error(f"Not enough rows for single-unit copilot. Need ≥ {seq_len}, got {len(df_u)}.")
        st.stop()

    win = df_u.tail(seq_len).copy()

    if use_endpoint == "Single only":
        use_batch = False
    elif use_endpoint == "Batch only":
        use_batch = True
    else:
        use_batch = True

    batch_payload = dataframe_to_batch_payload(win, unit_col=unit_col, cycle_col=cycle_col)
    single_payload = dataframe_to_single_payload(win, fd_name=dataset, unit_col=unit_col, cycle_col=cycle_col)

    with st.spinner("Calling backend..."):
        try:
            if use_batch:
                r = requests.post(batch_url, json=batch_payload, timeout=API_TIMEOUT)
                used_ep = "batch"
            else:
                r = requests.post(single_url, json=single_payload, timeout=API_TIMEOUT)
                used_ep = "single"

            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {r.text}")
                st.stop()

            out = r.json()
            _, _, rul = extract_rul_any(out)
            model_name = out.get("model_name") or out.get("model") or out.get("best_model_name") or "unknown"

        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    band = risk_band_from_rul(rul)
    score = risk_score_0_100(rul)

    days = rul_to_days(rul, float(st.session_state["cycles_per_day"]))
    sla_msg = sla_alert(days, st.session_state["sla_days"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unit", int(unit_pick))
    c2.metric("Endpoint", used_ep)
    c3.metric("Model", model_name)
    c4.metric("Predicted RUL", "N/A" if rul is None else int(rul))

    st.markdown(
        f"### Risk Band: {band['icon']} **{band['name']}**  \n"
        f"Risk score (0–100): **{score if score is not None else 'N/A'}**  \n"
        f"Estimated days-to-failure: **{days if days is not None else 'N/A'}** days (using {st.session_state['cycles_per_day']} cycles/day)  \n"
        f"{sla_msg}"
    )

    st.markdown("### Recommended Actions (deterministic)")
    for a in copilot_actions(band["name"]):
        st.write(f"- {a}")

    st.divider()
    st.subheader("3) Degradation hints (top drifting sensors, deterministic)")

    exclude = [cycle_col]
    if unit_col and unit_col in win.columns:
        exclude.append(unit_col)
    sensor_cols = pick_numeric_sensors(win, exclude=exclude)

    if len(sensor_cols) == 0:
        st.info("No numeric sensor columns found for drift analysis.")
        st.stop()

    max_features = min(15, len(sensor_cols))  # fixed, no UI control
    trends = compute_trend_slopes(win, cycle_col=cycle_col, sensor_cols=sensor_cols).head(max_features)

    if len(trends) == 0:
        st.info("Could not compute trends.")
        st.stop()

    st.dataframe(trends, use_container_width=True)

    top3 = trends["feature"].head(3).tolist()
    fig = plt.figure()
    for c in top3:
        y = pd.to_numeric(win[c], errors="coerce")
        y = y.fillna(y.median())
        mu = y.mean()
        sd = y.std()
        if sd == 0 or pd.isna(sd):
            sd = 1.0
        y = (y - mu) / sd
        plt.plot(win[cycle_col], y, label=c)

    plt.xlabel("Cycle")
    plt.ylabel("Normalized value")
    plt.title(f"{dataset} — Top drift signals (Unit {unit_pick})")
    plt.grid(True)
    plt.legend()
    st.pyplot(fig, clear_figure=True)

    with st.expander("Debug (raw API JSON)"):
        st.code(json.dumps(out, indent=2), language="json")

    st.stop()

# ------------------------------------------------------------
# Fleet mode (CACHED so sliders won't wipe results)
# ------------------------------------------------------------
st.caption("Fleet mode runs multiple API calls (one per unit). Keep max units low for demos.")

if not (unit_col and unit_col in df.columns):
    st.error("Fleet mode requires a `unit` column in your uploaded CSV.")
    st.stop()

df_work = df.copy()
if cycle_col and cycle_col in df_work.columns:
    df_work = safe_sort_by_cycle(df_work, cycle_col)
else:
    df_work["cycle"] = df_work.groupby(unit_col).cumcount() + 1
    cycle_col = "cycle"

units = sorted(pd.to_numeric(df_work[unit_col], errors="coerce").dropna().unique().astype(int).tolist())
if len(units) == 0:
    st.error("No valid units found.")
    st.stop()

n_units = len(units)
if n_units <= 1:
    st.info("Only 1 unit detected in the uploaded CSV. Fleet mode needs multiple units. I’ll treat this as single-unit fleet analysis.")
    max_units = 1
else:
    max_units = st.slider("Max units to evaluate (limits API calls)", min_value=1, max_value=min(50, n_units), value=min(10, n_units), step=1)

endpoint_mode = st.radio("Fleet inference endpoint", ["Batch (recommended)", "Single (fallback)"], index=0, horizontal=True)

cache_key = f"{dataset}|{endpoint_mode}|{max_units}|{unit_col}|{cycle_col}"
run_fleet = st.button("Run Fleet Copilot (local)", type="primary")

if run_fleet:
    rows = []
    with st.spinner("Scoring fleet..."):
        for u in units[:max_units]:
            df_u = df_work[df_work[unit_col].astype(str) == str(u)].copy()
            if len(df_u) < seq_len:
                rows.append({"unit": int(u), "rul": None, "band": "Too short", "risk_score": None})
                continue

            win = df_u.tail(seq_len).copy()
            batch_payload = dataframe_to_batch_payload(win, unit_col=unit_col, cycle_col=cycle_col)
            single_payload = dataframe_to_single_payload(win, fd_name=dataset, unit_col=unit_col, cycle_col=cycle_col)

            try:
                if endpoint_mode.startswith("Batch"):
                    r = requests.post(batch_url, json=batch_payload, timeout=API_TIMEOUT)
                else:
                    r = requests.post(single_url, json=single_payload, timeout=API_TIMEOUT)

                if r.status_code != 200:
                    rows.append({"unit": int(u), "rul": None, "band": f"API {r.status_code}", "risk_score": None})
                    continue

                out = r.json()
                _, _, rul = extract_rul_any(out)
                band_ = risk_band_from_rul(rul)
                score = risk_score_0_100(rul)

                # keep emoji in UI (fine), PDF will sanitize
                rows.append({"unit": int(u), "rul": rul, "band": f"{band_['icon']} {band_['name']}", "risk_score": score})

            except Exception as e:
                rows.append({"unit": int(u), "rul": None, "band": f"Error: {e}", "risk_score": None})

    res = pd.DataFrame(rows)
    st.session_state["fleet_cache_key"] = cache_key
    st.session_state["fleet_res"] = res

if st.session_state.get("fleet_cache_key") != cache_key and st.session_state.get("fleet_res") is not None:
    st.warning("Fleet settings changed since last run. Click **Run Fleet Copilot** to recompute.")

if st.session_state.get("fleet_res") is None:
    st.info("Click **Run Fleet Copilot (local)** to compute fleet outputs.")
    st.stop()

res = st.session_state["fleet_res"].copy()

st.subheader("Fleet results")
st.dataframe(res, use_container_width=True)

export_col1, export_col2, export_col3 = st.columns([1, 1, 2])
with export_col1:
    st.download_button(
        "⬇️ Download Fleet Results (CSV)",
        data=dataframe_to_csv_bytes(res),
        file_name=f"fleet_results_{dataset.lower()}.csv",
        mime="text/csv",
        use_container_width=True,
    )

ok_for_export = res.dropna(subset=["rul"]).copy()
if len(ok_for_export) > 0:
    ok_for_export["rul"] = ok_for_export["rul"].astype(float)
    ok_for_export = ok_for_export.sort_values("rul", ascending=True).reset_index(drop=True)
    worst_export = ok_for_export.head(min(10, len(ok_for_export))).copy()
else:
    worst_export = pd.DataFrame(columns=res.columns)

with export_col2:
    st.download_button(
        "⬇️ Download Worst Units (CSV)",
        data=dataframe_to_csv_bytes(worst_export),
        file_name=f"fleet_worst_units_{dataset.lower()}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with export_col3:
    if REPORTLAB_AVAILABLE:
        st.caption("PDF export enabled ✅ (summary + fleet rows + worst units).")
    else:
        st.caption("PDF export disabled (install `reportlab` if you want PDF).")

st.divider()

ok = res.dropna(subset=["rul"]).copy()
st.subheader("Fleet analytics (deterministic)")
if len(ok) == 0:
    st.warning("No valid fleet RUL values to summarize.")
    st.stop()

ok["rul"] = ok["rul"].astype(float)
ok = ok.sort_values("rul", ascending=True).reset_index(drop=True)

ok["days_to_failure"] = ok["rul"].apply(lambda x: rul_to_days(x, float(st.session_state["cycles_per_day"])))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Units scored", len(ok))
c2.metric("Min RUL", round(float(ok["rul"].min()), 2))
c3.metric("Median RUL", round(float(ok["rul"].median()), 2))
c4.metric("Max RUL", round(float(ok["rul"].max()), 2))
c5.metric("Min Days-to-failure", round(float(ok["days_to_failure"].min()), 2))

band_counts = ok["band"].value_counts()
st.write("Risk band distribution:")
st.dataframe(band_counts.rename_axis("band").reset_index(name="count"), use_container_width=True)

st.subheader("Fleet Risk Heatmap")
heat_metric = st.selectbox("Heatmap metric", ["risk_score", "rul"], index=0)

heat_df = ok.copy()
if heat_metric == "risk_score":
    plot_fleet_risk_heatmap(heat_df, value_col="risk_score", max_units=min(30, len(heat_df)))
else:
    heat_df["risk_from_rul"] = heat_df["rul"].apply(risk_score_0_100)
    plot_fleet_risk_heatmap(heat_df, value_col="risk_from_rul", max_units=min(30, len(heat_df)))

st.caption("Interpretation: higher numbers = higher risk (priority for maintenance).")

st.divider()

st.subheader("Top priority units (lowest RUL)")
worst_n = min(10, len(ok))
worst = ok.head(worst_n).copy()

fig = plt.figure()
plt.bar(worst["unit"].astype(str), worst["rul"])
plt.xlabel("Unit")
plt.ylabel("Predicted RUL")
plt.title(f"{dataset} — Worst {worst_n} units by RUL")
plt.grid(True, axis="y")
st.pyplot(fig, clear_figure=True)

st.subheader("Fleet Copilot recommendation (deterministic)")
lowest_rul = float(ok["rul"].iloc[0])
band = risk_band_from_rul(lowest_rul)

worst_days = rul_to_days(lowest_rul, float(st.session_state["cycles_per_day"]))
sla_msg = sla_alert(worst_days, st.session_state["sla_days"])

st.markdown(
    f"### Fleet status: {band['icon']} **{band['name']}** (worst unit RUL ≈ {int(lowest_rul)})\n\n"
    f"Estimated days-to-failure (worst unit): **{worst_days if worst_days is not None else 'N/A'}** days (using {st.session_state['cycles_per_day']} cycles/day)\n\n"
    f"{sla_msg}"
)

st.write("Do this next (based on the WORST unit):")
for a in copilot_actions(band["name"]):
    st.write(f"- {a}")

st.caption("Tip: Fleet recommendation is conservative: driven by the lowest-RUL unit.")

st.subheader("Scenario simulation (fleet)")
st.caption("If we delay maintenance by X days (fleet-wide decision)...")

st.slider(
    "Delay maintenance by (days)",
    min_value=0,
    max_value=30,
    value=int(st.session_state["delay_days_fleet"]),
    step=1,
    key="delay_days_fleet_int",
)

st.session_state["delay_days_fleet"] = float(st.session_state["delay_days_fleet_int"])

delay_days = float(st.session_state["delay_days_fleet"])
cpd = float(st.session_state["cycles_per_day"])

sim = ok.copy()
sim["rul_after_delay"] = sim["rul"].apply(lambda r: apply_delay_to_rul(r, delay_days, cpd))
sim["days_after_delay"] = sim["rul_after_delay"].apply(lambda r: rul_to_days(r, cpd))
sim["band_after_delay"] = sim["rul_after_delay"].apply(lambda r: f"{risk_band_from_rul(r)['icon']} {risk_band_from_rul(r)['name']}")

worst_after = float(sim["rul_after_delay"].min())
worst_after_days = rul_to_days(worst_after, cpd)
band_after = risk_band_from_rul(worst_after)
sla_after = sla_alert(worst_after_days, st.session_state["sla_days"])

st.write(
    f"Worst unit after delaying **{delay_days:.0f}** days: "
    f"RUL ≈ **{int(worst_after)}**, Days-to-failure ≈ **{worst_after_days if worst_after_days is not None else 'N/A'}**"
)
st.write(f"New band: {band_after['icon']} **{band_after['name']}** | {sla_after}")

st.write("Fleet impact (band counts after delay):")
impact = sim["band_after_delay"].value_counts().rename_axis("band_after_delay").reset_index(name="count")
st.dataframe(impact, use_container_width=True)

if REPORTLAB_AVAILABLE:
    # ✅ FIX B (PDF): avoid emoji in summary field (optional extra safety)
    summary = {
        "Units scored": str(len(ok)),
        "Min RUL": str(round(float(ok["rul"].min()), 2)),
        "Median RUL": str(round(float(ok["rul"].median()), 2)),
        "Max RUL": str(round(float(ok["rul"].max()), 2)),
        "Worst unit risk band": f"{band['name']}",  # was f"{band['icon']} {band['name']}"
        "Cycles/day assumption": str(st.session_state["cycles_per_day"]),
        "Worst days-to-failure": str(worst_days),
        "Delay scenario (days)": str(int(delay_days)),
        "Worst days after delay": str(worst_after_days),
    }

    pdf_bytes = build_fleet_pdf_report(
        dataset=dataset,
        seq_len=seq_len,
        fleet_df=ok[["unit", "rul", "band", "risk_score", "days_to_failure"]].copy(),
        worst_df=worst[["unit", "rul", "band", "risk_score", "days_to_failure"]].copy(),
        summary=summary,
    )

    st.download_button(
        "⬇️ Download Fleet Report (PDF)",
        data=pdf_bytes,
        file_name=f"fleet_report_{dataset.lower()}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
else:
    st.warning("PDF export disabled: `reportlab` not installed. Install with `pip install reportlab` to enable.")
