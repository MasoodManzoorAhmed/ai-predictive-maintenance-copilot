# ai_predictive_maintenance_copilot/streamlit_app/pages/3_Analytics.py

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

# ============================================================
# FD rules (match backend expectations)
# ============================================================
FD_RULES = {
    "FD001": {"seq_len": 30, "min_cycles_batch": 30},
    "FD002": {"seq_len": 100, "min_cycles_batch": 100},
    "FD003": {"seq_len": 30, "min_cycles_batch": 30},
    "FD004": {"seq_len": 100, "min_cycles_batch": 100},
}

# Existing thresholds (kept)
STATUS_THRESHOLDS = {"critical": 20, "warning": 60}

# Added: extended banding thresholds (Phase 8 completion)
# You can tune these later but recruiters LOVE this being explicit.
RISK_BANDS = {
    "critical": 20,          # <= 20
    "warning": 60,           # 21..60
    "plan_maintenance": 100, # 61..100
    # healthy > 100
}


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


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
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
    """
    Batch schema:
      { "records": [ {unit, cycle, setting1.., sensor1..}, ... ] }
    """
    df2 = _clean_columns(df)

    # unit
    if unit_col and unit_col in df2.columns:
        if unit_col != "unit":
            df2 = df2.rename(columns={unit_col: "unit"})
    else:
        df2["unit"] = 1

    # cycle
    if cycle_col and cycle_col in df2.columns:
        if cycle_col != "cycle":
            df2 = df2.rename(columns={cycle_col: "cycle"})
    else:
        df2["cycle"] = df2.groupby("unit").cumcount() + 1

    # drop targets
    drop_targets = [c for c in ["RUL", "rul", "target", "label"] if c in df2.columns]
    if drop_targets:
        df2 = df2.drop(columns=drop_targets)

    return {"records": df2.to_dict(orient="records")}


def dataframe_to_single_payload(
    df: pd.DataFrame, fd_name: str, unit_col: Optional[str], cycle_col: Optional[str]
) -> Dict[str, Any]:
    """
    Single schema:
      { "fd_name": "FD00X", "unit": 1, "rows": [ {cycle, setting.. sensors..}, ... ] }
    """
    df2 = _clean_columns(df)

    # unit id
    if unit_col and unit_col in df2.columns:
        try:
            unit_id = int(pd.to_numeric(df2[unit_col], errors="coerce").dropna().iloc[0])
        except Exception:
            unit_id = 1
    else:
        unit_id = 1

    # cycle
    if cycle_col and cycle_col in df2.columns:
        if cycle_col != "cycle":
            df2 = df2.rename(columns={cycle_col: "cycle"})
    else:
        df2["cycle"] = range(1, len(df2) + 1)

    # drop unit from rows
    df2 = df2.drop(columns=[unit_col] if (unit_col and unit_col in df2.columns) else [], errors="ignore")
    df2 = df2.drop(columns=["unit"], errors="ignore")

    # drop targets
    drop_targets = [c for c in ["RUL", "rul", "target", "label"] if c in df2.columns]
    if drop_targets:
        df2 = df2.drop(columns=drop_targets)

    return {"fd_name": fd_name.upper(), "unit": int(unit_id), "rows": df2.to_dict(orient="records")}


def extract_rul_any(out: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Return (raw, calibrated, display_used)
    Supports BOTH schemas:
    1) predictions[0].pred_rul_raw / pred_rul_calibrated
    2) top-level pred_rul_raw / pred_rul_calibrated
    """
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


def compute_status(rul_value: Optional[float]) -> Tuple[str, str]:
    """
    Existing 3-state status (kept for backward compatibility in UI)
    """
    if rul_value is None:
        return "Unknown", "⚪"
    if rul_value <= STATUS_THRESHOLDS["critical"]:
        return "Critical", "🔴"
    if rul_value <= STATUS_THRESHOLDS["warning"]:
        return "Warning", "🟠"
    return "Healthy", "🟢"


def risk_band(rul_value: Optional[float]) -> Tuple[str, str]:
    """
    Added: 4-state operational banding (Phase 8 completion)
    """
    if rul_value is None or pd.isna(rul_value):
        return "Unknown", "⚪"
    r = float(rul_value)
    if r <= RISK_BANDS["critical"]:
        return "Critical", "🔴"
    if r <= RISK_BANDS["warning"]:
        return "Warning", "🟠"
    if r <= RISK_BANDS["plan_maintenance"]:
        return "Plan Maintenance", "🟡"
    return "Healthy", "🟢"


def safe_sort_by_cycle(df: pd.DataFrame, cycle_col: str) -> pd.DataFrame:
    df2 = df.copy()
    df2[cycle_col] = pd.to_numeric(df2[cycle_col], errors="coerce")
    df2 = df2.dropna(subset=[cycle_col]).sort_values(cycle_col).reset_index(drop=True)
    return df2


def pick_numeric_sensors(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ============================================================
# Page UI
# ============================================================
st.set_page_config(page_title="Analytics", layout="wide")

API_BASE_URL = get_api_base_url()
API_TIMEOUT = get_api_timeout()

dataset = st.session_state.get("selected_dataset", "FD001").upper()
seq_len = FD_RULES.get(dataset, {}).get("seq_len", 30)

batch_url = API_BASE_URL + get_batch_endpoint(dataset)
single_url = API_BASE_URL + get_single_endpoint()

st.title("📊 Analytics")
st.caption(f"Dataset: **{dataset}** | Default sequence length: **{seq_len}**")
st.caption(f"Batch endpoint: `{batch_url}` | Single endpoint: `{single_url}`")

st.divider()

# ------------------------------------------------------------
# Input: upload CSV
# ------------------------------------------------------------
st.subheader("1) Upload data for analytics")
uploaded = st.file_uploader("Upload a CSV (head/tail sample or multi-engine CSV)", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to begin analytics. Use your generated `sample_FD00X_...csv` files.")
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

left_map, right_map = st.columns(2)
with left_map:
    unit_choice = st.selectbox("Unit/Engine column (optional)", col_options, index=col_options.index(default_unit))
with right_map:
    cycle_choice = st.selectbox("Cycle/Time column (optional)", col_options, index=col_options.index(default_cycle))

unit_col = None if unit_choice == "(not present)" else unit_choice
cycle_col = None if cycle_choice == "(not present)" else cycle_choice

st.write("Preview:")
st.dataframe(df.head(15), use_container_width=True)

# Window display (derived)
w_start, w_end, w_reason = detect_window_from_df(df, cycle_col)
if w_start is not None and w_end is not None:
    st.success(f"Window used: cycles **{w_start}–{w_end}** (derived from CSV)")
else:
    st.warning(f"Window used: (unable to detect) — {w_reason}")

# ------------------------------------------------------------
# 8.1: Data sanity panel
# ------------------------------------------------------------
with st.expander("✅ Data sanity check (recommended)", expanded=True):
    n_rows = len(df)
    if unit_col and unit_col in df.columns:
        units_detected = (
            pd.to_numeric(df[unit_col], errors="coerce").dropna().unique().astype(int).tolist()
            if n_rows > 0
            else []
        )
        n_units = len(units_detected)
    else:
        units_detected = [1]
        n_units = 1

    if cycle_col and cycle_col in df.columns:
        cyc = pd.to_numeric(df[cycle_col], errors="coerce")
        cyc_min = _safe_int(cyc.min(), 0) if cyc.notna().any() else None
        cyc_max = _safe_int(cyc.max(), 0) if cyc.notna().any() else None
        cyc_note = "cycle column present"
    else:
        cyc_min, cyc_max = 1, n_rows
        cyc_note = "cycle missing → using row order"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", n_rows)
    c2.metric("Units detected", n_units)
    c3.metric("Cycle min", "N/A" if cyc_min is None else cyc_min)
    c4.metric("Cycle max", "N/A" if cyc_max is None else cyc_max)
    c5.metric("seq_len required", seq_len)

    st.caption(f"Notes: {cyc_note}. Upload should have at least **{seq_len} rows per unit** for predictions.")

st.divider()

# ------------------------------------------------------------
# Tabs: RUL trend | Degradation | Multi-engine
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📉 RUL Trend (rolling windows)", "🧪 Degradation Curve", "🏭 Multi-engine comparison"])


# ============================================================
# TAB 1 — RUL Trend
# ============================================================
with tab1:
    st.subheader("RUL Trend Plot (rolling inference over the window)")
    st.caption("This calls your backend multiple times using rolling windows.")

    max_windows = st.slider("Max rolling windows (limit API calls)", min_value=3, max_value=30, value=10, step=1)

    use_mode = st.radio(
        "Inference mode for trend",
        ["Auto (recommended)", "Batch only", "Single only"],
        index=0,
        horizontal=True,
    )

    if cycle_col and cycle_col in df.columns:
        df_sorted = safe_sort_by_cycle(df, cycle_col)
    else:
        df_sorted = df.copy().reset_index(drop=True)
        df_sorted["cycle"] = range(1, len(df_sorted) + 1)
        cycle_col = "cycle"

    # Determine unit list
    if unit_col and unit_col in df_sorted.columns:
        units = sorted(pd.to_numeric(df_sorted[unit_col], errors="coerce").dropna().unique().astype(int).tolist())
        if len(units) == 0:
            units = [1]
        selected_unit = st.selectbox("Select unit for trend", units, index=0)
        df_u = df_sorted[df_sorted[unit_col].astype(str) == str(selected_unit)].copy()
    else:
        selected_unit = 1
        df_u = df_sorted.copy()
        df_u["unit"] = 1
        unit_col = "unit"

    if len(df_u) < seq_len:
        st.error(f"Not enough rows for trend. Need at least {seq_len} rows, got {len(df_u)}.")
        st.stop()

    total_possible = len(df_u) - seq_len + 1
    if total_possible <= 0:
        st.error("Cannot compute rolling windows (seq_len too large).")
        st.stop()

    if total_possible <= max_windows:
        starts = list(range(0, total_possible))
    else:
        step = max(1, total_possible // max_windows)
        starts = list(range(0, total_possible, step))[:max_windows]

    run_btn = st.button("Run rolling inference & plot trend", type="primary")

    if run_btn:
        xs: List[int] = []
        ys: List[float] = []
        details: List[Dict[str, Any]] = []
        endpoints_used: List[str] = []

        with st.spinner("Calling backend for rolling predictions..."):
            for sidx in starts:
                win = df_u.iloc[sidx: sidx + seq_len].copy()

                cmin = _safe_int(pd.to_numeric(win[cycle_col], errors="coerce").min(), 0)
                cmax = _safe_int(pd.to_numeric(win[cycle_col], errors="coerce").max(), 0)

                batch_payload = dataframe_to_batch_payload(win, unit_col=unit_col, cycle_col=cycle_col)
                single_payload = dataframe_to_single_payload(win, fd_name=dataset, unit_col=unit_col, cycle_col=cycle_col)

                if use_mode == "Single only":
                    use_batch = False
                elif use_mode == "Batch only":
                    use_batch = True
                else:
                    use_batch = True  # auto default

                used_ep = "batch" if use_batch else "single"

                try:
                    if use_batch:
                        r = requests.post(batch_url, json=batch_payload, timeout=API_TIMEOUT)
                    else:
                        r = requests.post(single_url, json=single_payload, timeout=API_TIMEOUT)

                    if r.status_code != 200:
                        details.append(
                            {"cycles": f"{cmin}-{cmax}", "endpoint": used_ep, "error": f"{r.status_code}: {r.text[:200]}"}
                        )
                        continue

                    out = r.json()
                    raw, cal, display = extract_rul_any(out)

                    xs.append(cmax)
                    ys.append(display if display is not None else float("nan"))
                    endpoints_used.append(used_ep)
                    details.append(
                        {
                            "cycles": f"{cmin}-{cmax}",
                            "endpoint": used_ep,
                            "pred_rul_raw": raw,
                            "pred_rul_calibrated": cal,
                            "display_rul_used": display,
                            "model": out.get("model_name") or out.get("model") or out.get("best_model_name"),
                        }
                    )

                except Exception as e:
                    details.append({"cycles": f"{cmin}-{cmax}", "endpoint": used_ep, "error": str(e)})

        if len(xs) == 0:
            st.error("No successful predictions. Open debug below.")
        else:
            fig = plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.xlabel("Cycle (end of window)")
            plt.ylabel("Predicted RUL (display used)")
            plt.title(f"{dataset} — RUL Trend (Unit {selected_unit}, seq_len={seq_len})")
            plt.grid(True)
            st.pyplot(fig, clear_figure=True)

            # Status summary based on last point
            last_rul = None
            try:
                last_rul = float(ys[-1])
            except Exception:
                last_rul = None

            s_txt, s_icon = compute_status(last_rul)
            band_txt, band_icon = risk_band(last_rul)

            st.markdown(
                f"### Current Status (latest window): {s_icon} **{s_txt}** "
                f"(RUL ≈ {int(last_rul) if last_rul is not None else 'N/A'})"
            )
            st.markdown(f"**Operational Band:** {band_icon} **{band_txt}**")

            # Trend sanity note (non-fatal)
            if len(ys) >= 3:
                diffs = []
                for i in range(1, len(ys)):
                    if pd.isna(ys[i]) or pd.isna(ys[i - 1]):
                        continue
                    diffs.append(ys[i] - ys[i - 1])
                if diffs and max(diffs) > 15:
                    st.warning(
                        "RUL may fluctuate across windows due to sensor noise and model uncertainty. "
                        "Focus on the overall trend, not single-point jumps."
                    )

            # Endpoint distribution
            if endpoints_used:
                st.caption(
                    "Endpoints used (debug): "
                    + ", ".join([f"{e}:{endpoints_used.count(e)}" for e in sorted(set(endpoints_used))])
                )

        with st.expander("Show rolling inference details (debug)"):
            st.code(json.dumps(details, indent=2), language="json")


# ============================================================
# TAB 2 — Degradation curves
# ============================================================
with tab2:
    st.subheader("Degradation Curves (from uploaded CSV)")

    if cycle_col and cycle_col in df.columns:
        df_plot = safe_sort_by_cycle(df, cycle_col)
    else:
        df_plot = df.copy()
        df_plot["cycle"] = range(1, len(df_plot) + 1)
        cycle_col = "cycle"

    if unit_col and unit_col in df_plot.columns:
        units = sorted(pd.to_numeric(df_plot[unit_col], errors="coerce").dropna().unique().astype(int).tolist())
        if len(units) == 0:
            units = [1]
        unit_pick = st.selectbox("Select unit for degradation curves", units, index=0)
        df_plot = df_plot[df_plot[unit_col].astype(str) == str(unit_pick)].copy()
    else:
        unit_pick = 1

    exclude = [cycle_col]
    if unit_col and unit_col in df_plot.columns:
        exclude.append(unit_col)

    numeric_cols = pick_numeric_sensors(df_plot, exclude=exclude)
    preferred = [c for c in df_plot.columns if c.startswith("sensor") or c.startswith("setting")]
    preferred = [c for c in preferred if c in numeric_cols]
    rest = [c for c in numeric_cols if c not in preferred]
    options = preferred + rest

    if len(options) == 0:
        st.warning("No numeric sensor/setting columns found to plot.")
    else:
        selected_cols = st.multiselect(
            "Select sensors/settings to plot (2–6)",
            options=options,
            default=options[:3] if len(options) >= 3 else options,
        )

        normalize = st.checkbox("Normalize selected sensors (recommended)", value=True)

        if len(selected_cols) == 0:
            st.info("Select at least one sensor to plot.")
        else:
            fig = plt.figure()
            for c in selected_cols:
                y = pd.to_numeric(df_plot[c], errors="coerce")
                if normalize:
                    mu = y.mean()
                    sd = y.std()
                    if sd is None or sd == 0 or pd.isna(sd):
                        sd = 1.0
                    y = (y - mu) / sd
                plt.plot(df_plot[cycle_col], y, label=c)

            plt.xlabel("Cycle")
            plt.ylabel("Normalized value" if normalize else "Value")
            plt.title(f"{dataset} — Degradation Curves (Unit {unit_pick})" + (" [normalized]" if normalize else ""))
            plt.grid(True)
            plt.legend()
            st.pyplot(fig, clear_figure=True)

            st.caption(
                "These curves show how sensor readings drift across cycles. "
                "Normalization helps compare sensors with very different ranges on the same plot."
            )


# ============================================================
# TAB 3 — Multi-engine comparison + Fleet analytics
# ============================================================
with tab3:
    st.subheader("Multi-engine comparison (predict last-window RUL per unit)")
    st.caption("Now includes Fleet KPIs + Maintenance Queue + CSV export.")

    if not (unit_col and unit_col in df.columns):
        st.info("Your uploaded CSV has no `unit` column. Upload a multi-engine CSV to enable this section.")
        st.stop()

    if cycle_col and cycle_col in df.columns:
        df_sorted = safe_sort_by_cycle(df, cycle_col)
    else:
        df_sorted = df.copy()
        df_sorted["cycle"] = df_sorted.groupby(unit_col).cumcount() + 1
        cycle_col = "cycle"

    units = sorted(pd.to_numeric(df_sorted[unit_col], errors="coerce").dropna().unique().astype(int).tolist())
    if len(units) == 0:
        st.info("No valid units found.")
        st.stop()

    # ✅ FIX: slider cannot have min==max when only 1 unit
    n_units = len(units)
    if n_units <= 1:
        st.info("Only 1 unit detected in the uploaded CSV, so multi-unit evaluation is effectively single-unit.")
        max_units = 1
    else:
        max_units = st.slider(
            "Max units to evaluate (limit API calls)",
            min_value=1,
            max_value=min(30, n_units),
            value=min(10, n_units),
            step=1,
        )

    mode = st.radio(
        "Inference mode for multi-engine",
        ["Batch (recommended if each unit has ≥ seq_len)", "Single (fallback)"],
        index=0,
        horizontal=True,
    )

    run_multi = st.button("Run multi-engine comparison", type="primary")

    if run_multi:
        rows = []
        with st.spinner("Predicting last-window RUL for units..."):
            for u in units[:max_units]:
                df_u = df_sorted[df_sorted[unit_col].astype(str) == str(u)].copy()
                if len(df_u) < seq_len:
                    rows.append(
                        {
                            "unit": int(u),
                            "rul": None,
                            "status": "Too short",
                            "risk_band": "⚪ Unknown",
                            "cycles": None,
                            "endpoint": None,
                        }
                    )
                    continue

                win = df_u.tail(seq_len).copy()
                cmin = _safe_int(pd.to_numeric(win[cycle_col], errors="coerce").min(), 0)
                cmax = _safe_int(pd.to_numeric(win[cycle_col], errors="coerce").max(), 0)

                batch_payload = dataframe_to_batch_payload(win, unit_col=unit_col, cycle_col=cycle_col)
                single_payload = dataframe_to_single_payload(win, fd_name=dataset, unit_col=unit_col, cycle_col=cycle_col)

                try:
                    if mode.startswith("Batch"):
                        r = requests.post(batch_url, json=batch_payload, timeout=API_TIMEOUT)
                        used_ep = "batch"
                    else:
                        r = requests.post(single_url, json=single_payload, timeout=API_TIMEOUT)
                        used_ep = "single"

                    if r.status_code != 200:
                        rows.append(
                            {
                                "unit": int(u),
                                "rul": None,
                                "status": f"API {r.status_code}",
                                "risk_band": "⚪ Unknown",
                                "cycles": f"{cmin}-{cmax}",
                                "endpoint": used_ep,
                            }
                        )
                        continue

                    out = r.json()
                    _, _, display = extract_rul_any(out)

                    s_txt, s_icon = compute_status(display)
                    b_txt, b_icon = risk_band(display)

                    rows.append(
                        {
                            "unit": int(u),
                            "rul": display,
                            "status": f"{s_icon} {s_txt}",
                            "risk_band": f"{b_icon} {b_txt}",
                            "cycles": f"{cmin}-{cmax}",
                            "endpoint": used_ep,
                        }
                    )
                except Exception as e:
                    rows.append(
                        {
                            "unit": int(u),
                            "rul": None,
                            "status": f"Error: {e}",
                            "risk_band": "⚪ Unknown",
                            "cycles": f"{cmin}-{cmax}",
                            "endpoint": "unknown",
                        }
                    )

        res = pd.DataFrame(rows)
        st.dataframe(res, use_container_width=True)

        # ---------- Fleet KPIs + Maintenance Queue + Export ----------
        ok = res.dropna(subset=["rul"]).copy()
        if len(ok) == 0:
            st.warning("No valid RUL values to compute fleet analytics.")
            st.stop()

        ok["rul"] = ok["rul"].astype(float)

        # Fleet KPIs
        st.markdown("### Fleet KPIs")
        crit = int(ok["risk_band"].astype(str).str.contains("Critical").sum())
        warn = int(ok["risk_band"].astype(str).str.contains("Warning").sum())
        plan = int(ok["risk_band"].astype(str).str.contains("Plan").sum())
        healthy = int(ok["risk_band"].astype(str).str.contains("Healthy").sum())

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Units evaluated", len(ok))
        k2.metric("Critical", crit)
        k3.metric("Warning", warn)
        k4.metric("Plan", plan)
        k5.metric("Healthy", healthy)

        # Maintenance Queue
        st.markdown("### Maintenance Queue (lowest RUL first)")
        queue = ok.sort_values("rul", ascending=True)[["unit", "rul", "risk_band", "cycles", "endpoint"]].head(10)
        st.dataframe(queue, use_container_width=True)

        st.download_button(
            "⬇️ Download fleet snapshot CSV",
            data=ok.to_csv(index=False).encode("utf-8"),
            file_name=f"fleet_snapshot_{dataset}.csv",
            mime="text/csv",
        )

        # Plot (kept from your original)
        ok_sorted = ok.sort_values("rul", ascending=True)

        def _status_color(rval: float) -> str:
            # Keep original bar coloring logic (3-state) for continuity
            if rval <= STATUS_THRESHOLDS["critical"]:
                return "red"
            if rval <= STATUS_THRESHOLDS["warning"]:
                return "orange"
            return "green"

        colors = [_status_color(float(v)) for v in ok_sorted["rul"].tolist()]

        fig = plt.figure()
        bars = plt.bar(ok_sorted["unit"].astype(str), ok_sorted["rul"], color=colors)
        plt.xlabel("Unit")
        plt.ylabel("Predicted RUL (display used)")
        plt.title(f"{dataset} — Multi-engine comparison (last window per unit)")
        plt.grid(True, axis="y")

        # value labels (makes near-zero still visible)
        for b, v in zip(bars, ok_sorted["rul"].tolist()):
            plt.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{float(v):.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # zoom if everything is near-zero
        max_r = float(ok_sorted["rul"].max())
        if max_r <= 10:
            plt.ylim(0, max(1.0, max_r * 1.2))

        st.pyplot(fig, clear_figure=True)

        st.caption(
            "This compares multiple engines at the same moment (their latest window). "
            "Lower RUL indicates engines closer to failure → higher priority for maintenance. "
            "Fleet KPIs + queue help managers make quick decisions."
        )
