# ai_predictive_maintenance_copilot/streamlit_app/pages/2_Predict_RUL.py

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# ============================================================
# FD rules 
# ============================================================
FD_RULES = {
    "FD001": {"min_cycles_batch": 30},
    "FD002": {"min_cycles_batch": 100},
    "FD003": {"min_cycles_batch": 30},
    "FD004": {"min_cycles_batch": 100},
}

STATUS_THRESHOLDS = {"critical": 20, "warning": 60}


# ============================================================
# Helpers
# ============================================================



def get_api_timeout() -> int:
    t = st.session_state.get("API_TIMEOUT") or os.getenv("API_TIMEOUT", "30")
    try:
        return int(t)
    except Exception:
        return 30


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    return df2


def _infer_window_label_from_filename(filename: Optional[str]) -> str:
    """Label ONLY from filename hint; no guessing."""
    if not filename:
        return ""
    name = filename.lower()
    if "head" in name:
        return " (early life)"
    if "tail" in name or "last" in name:
        return " (latest window)"
    return ""


def detect_window_from_df(
    df: pd.DataFrame,
    unit_col: Optional[str],
    cycle_col: Optional[str],
) -> Tuple[Optional[int], Optional[int], str]:
    """Derive window ONLY from uploaded CSV."""
    df2 = _clean_columns(df)

    cycle_used = cycle_col if (cycle_col and cycle_col in df2.columns) else None
    if cycle_used is not None:
        s = pd.to_numeric(df2[cycle_used], errors="coerce")
        if s.notna().any():
            return int(s.min()), int(s.max()), "cycle column found"
        return None, None, "cycle column present but not numeric"

    n = len(df2)
    if n == 0:
        return None, None, "no rows"
    return 1, n, "cycle column missing; using row order (1..N)"


def dataframe_to_batch_payload(df: pd.DataFrame, unit_col: Optional[str], cycle_col: Optional[str]) -> Dict[str, Any]:
    """
    Batch endpoint expects: { "records": [ {unit, cycle, ...features...}, ... ] }
    """
    df2 = _clean_columns(df)

    # Unit normalize
    if unit_col and unit_col in df2.columns:
        if unit_col != "unit":
            df2 = df2.rename(columns={unit_col: "unit"})
    else:
        df2["unit"] = 1

    # Cycle normalize
    if cycle_col and cycle_col in df2.columns:
        if cycle_col != "cycle":
            df2 = df2.rename(columns={cycle_col: "cycle"})
    else:
        df2["cycle"] = df2.groupby("unit").cumcount() + 1

    # Drop targets
    drop_targets = [c for c in ["RUL", "rul", "target", "label"] if c in df2.columns]
    if drop_targets:
        df2 = df2.drop(columns=drop_targets)

    return {"records": df2.to_dict(orient="records")}


def dataframe_to_single_payload(df: pd.DataFrame, fd_name: str, unit_col: Optional[str], cycle_col: Optional[str]) -> Dict[str, Any]:
    """
    Single endpoint expects:
      { "fd_name": "FD004", "unit": 1, "rows": [ {...}, {...} ] }
    """
    df2 = _clean_columns(df)

    # Unit id (best effort)
    unit_id = 1
    if unit_col and unit_col in df2.columns:
        try:
            unit_id = int(pd.to_numeric(df2[unit_col], errors="coerce").dropna().iloc[0])
        except Exception:
            unit_id = 1

    # Cycle normalize
    if cycle_col and cycle_col in df2.columns:
        if cycle_col != "cycle":
            df2 = df2.rename(columns={cycle_col: "cycle"})
    else:
        df2["cycle"] = range(1, len(df2) + 1)

    # Remove unit column from rows payload (single endpoint does not need per-row unit)
    if unit_col and unit_col in df2.columns:
        df2 = df2.drop(columns=[unit_col], errors="ignore")
    df2 = df2.drop(columns=["unit"], errors="ignore")

    # Drop targets
    drop_targets = [c for c in ["RUL", "rul", "target", "label"] if c in df2.columns]
    if drop_targets:
        df2 = df2.drop(columns=drop_targets)

    return {"fd_name": fd_name.upper(), "unit": int(unit_id), "rows": df2.to_dict(orient="records")}


def get_batch_endpoint(fd_name: str) -> str:
    return f"/predict/{fd_name.strip().lower()}"


def get_single_endpoint() -> str:
    return "/single/predict"


def compute_status_from_rul(rul_value: float) -> Tuple[str, str]:
    if rul_value <= STATUS_THRESHOLDS["critical"]:
        return "Critical", "🔴"
    if rul_value <= STATUS_THRESHOLDS["warning"]:
        return "Warning", "🟠"
    return "Healthy", "🟢"


def _to_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def extract_display_rul(out: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (display_rul, calibrated, raw)

    ✅ Supports ALL schemas we've seen:
    - top-level: predicted_rul / rul
    - top-level: pred_rul_calibrated / pred_rul_raw   (FD004 single)
    - list: out["predictions"][0]["pred_rul_calibrated"] / ["pred_rul_raw"] (FD001 batch)
    - list: out["predictions"][0]["predicted_rul"] / ["rul"] (other possible)
    """
    calibrated = None
    raw = None
    display = None

    # --- Top-level common ---
    if "predicted_rul" in out:
        display = _to_float_or_none(out.get("predicted_rul"))
    if display is None and "rul" in out:
        display = _to_float_or_none(out.get("rul"))

    # --- Top-level calibrated/raw (single FD004) ---
    if "pred_rul_calibrated" in out:
        calibrated = _to_float_or_none(out.get("pred_rul_calibrated"))
    if "pred_rul_raw" in out:
        raw = _to_float_or_none(out.get("pred_rul_raw"))

    # --- Predictions list (batch FD001) ---
    preds = out.get("predictions")
    if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict):
        p0 = preds[0]
        if calibrated is None:
            calibrated = _to_float_or_none(p0.get("pred_rul_calibrated"))
        if raw is None:
            raw = _to_float_or_none(p0.get("pred_rul_raw"))

        # sometimes batch might have these names too
        if display is None:
            display = _to_float_or_none(p0.get("predicted_rul"))
        if display is None:
            display = _to_float_or_none(p0.get("rul"))

    # Choose display in safe order:
    # 1) predicted_rul/rul if present
    # 2) calibrated if present
    # 3) raw if present
    if display is None:
        display = calibrated if calibrated is not None else raw

    return display, calibrated, raw


def extract_model_name(out: Dict[str, Any]) -> str:
    for k in ["model_name", "model", "best_model", "best_model_name", "model_id"]:
        v = out.get(k)
        if v:
            return str(v)
    return "N/A"


def validate_min_cycles_per_unit(df: pd.DataFrame, unit_col: Optional[str], fd_name: str) -> Tuple[bool, str]:
    min_req = FD_RULES.get(fd_name.upper(), {}).get("min_cycles_batch", 30)

    if unit_col and unit_col in df.columns:
        counts = df.groupby(unit_col).size()
        bad = counts[counts < min_req]
        if len(bad) > 0:
            u = bad.index[0]
            c = int(bad.iloc[0])
            return False, f"{fd_name} batch requires ≥{min_req} cycles per unit. unit={u} has only {c}."
        return True, ""
    else:
        if len(df) < min_req:
            return False, f"{fd_name} batch requires ≥{min_req} cycles per unit. file has only {len(df)} rows."
        return True, ""


# ============================================================
# UI
# ============================================================
def get_api_base_url() -> str:
    url = st.session_state.get("API_BASE_URL") or os.getenv("API_BASE_URL", "").strip()
    if not url:
        st.error("API_BASE_URL is not set. Set it in Cloud Run env vars or in streamlit_app/.env.local for local.")
        st.stop()
    return str(url).strip().rstrip("/")


API_BASE_URL = get_api_base_url()
API_TIMEOUT = get_api_timeout()

dataset = st.session_state.get("selected_dataset", "FD001").upper()
batch_url = API_BASE_URL + get_batch_endpoint(dataset)
single_url = API_BASE_URL + get_single_endpoint()

st.title("🧪 Predict RUL")
st.caption(f"Current dataset: **{dataset}**")
st.caption(f"Batch endpoint: `{batch_url}`")
st.caption(f"Single endpoint (fallback): `{single_url}`")

tab_csv, tab_json = st.tabs(["📄 Upload CSV (recommended)", "🧾 Paste JSON (advanced)"])


# =========================
# CSV TAB
# =========================
with tab_csv:
    st.subheader("Upload CSV")
    st.write("Supports single-engine (no unit column) and multi-engine (unit column present).")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        df = _clean_columns(df)

        st.write("Preview:")
        st.dataframe(df.head(15), use_container_width=True)

        st.markdown("### Column mapping")
        st.caption("If your CSV uses different names, map them here (we will rename them internally).")

        col_options = ["(not present)"] + list(df.columns)
        default_unit = "unit" if "unit" in df.columns else "(not present)"
        default_cycle = "cycle" if "cycle" in df.columns else "(not present)"

        unit_choice = st.selectbox("Unit/Engine column (optional)", col_options, index=col_options.index(default_unit))
        cycle_choice = st.selectbox("Cycle/Time column (optional)", col_options, index=col_options.index(default_cycle))

        unit_col = None if unit_choice == "(not present)" else unit_choice
        cycle_col = None if cycle_choice == "(not present)" else cycle_choice

        st.write(f"Rows detected (records): **{len(df)}**")

        w_start, w_end, w_reason = detect_window_from_df(df, unit_col, cycle_col)
        label_suffix = _infer_window_label_from_filename(getattr(uploaded, "name", None))

        if w_start is not None and w_end is not None:
            st.caption(f"Window used: cycles **{w_start}–{w_end}**{label_suffix}")
        else:
            st.caption(f"Window used: (unable to detect) — {w_reason}")

        ok_batch, batch_msg = validate_min_cycles_per_unit(df, unit_col, dataset)
        min_req = FD_RULES.get(dataset, {}).get("min_cycles_batch", 30)

        if ok_batch:
            st.success(f"✅ Batch-ready: meets minimum cycles requirement (≥ {min_req} per unit).")
        else:
            st.warning(
                f"⚠️ Not batch-ready: {batch_msg}\n\n"
                f"👉 Fix options:\n"
                f"- Generate **head{min_req}/tail{min_req}** CSV for this FD, OR\n"
                f"- Use **Single Predict (/single/predict)** (recommended for demo on short windows)."
            )

        batch_payload = dataframe_to_batch_payload(df, unit_col=unit_col, cycle_col=cycle_col)
        single_payload = dataframe_to_single_payload(df, fd_name=dataset, unit_col=unit_col, cycle_col=cycle_col)

        with st.expander("Show generated batch request payload (from CSV)"):
            st.code(json.dumps(batch_payload, indent=2)[:25000], language="json")

        with st.expander("Show generated single request payload (fallback)"):
            st.code(json.dumps(single_payload, indent=2)[:25000], language="json")

        mode = st.radio(
            "Prediction mode",
            options=[
                "Auto (Batch if possible, else Single)",
                "Batch (/predict/{fd})",
                "Single (/single/predict) [recommended demo]",
            ],
            index=0,
        )

        if st.button("🤖 Predict RUL from CSV", type="primary"):
            try:
                if mode == "Batch (/predict/{fd})":
                    use_batch = True
                elif mode == "Single (/single/predict) [recommended demo]":
                    use_batch = False
                else:
                    use_batch = ok_batch  # Auto

                # --- Call API ---
                if use_batch:
                    r = requests.post(batch_url, json=batch_payload, timeout=API_TIMEOUT)
                    mode_used = "batch"
                else:
                    r = requests.post(single_url, json=single_payload, timeout=API_TIMEOUT)
                    mode_used = "single"

                if r.status_code != 200:
                    st.error(f"API error {r.status_code}: {r.text}")
                    st.stop()

                out = r.json()

                # --- Extract RUL robustly ---
                display_rul, cal_rul, raw_rul = extract_display_rul(out)
                model_name = extract_model_name(out)

                st.success("✅ Prediction received")

                # Unit show
                unit_show = 1
                if unit_col and unit_col in df.columns:
                    try:
                        unit_show = int(pd.to_numeric(df[unit_col], errors="coerce").dropna().iloc[0])
                    except Exception:
                        unit_show = 1

                # Cards
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown("**Dataset**")
                    st.markdown(f"# {dataset}")
                    st.caption(f"Mode used: **{mode_used}**")
                with c2:
                    st.markdown("**Model**")
                    st.markdown(f"# {model_name}")
                with c3:
                    st.markdown("**Unit**")
                    st.markdown(f"# {unit_show}")
                with c4:
                    st.markdown("**Predicted RUL**")
                    st.markdown(f"# {int(display_rul) if display_rul is not None else 'N/A'}")

                # Status
                if display_rul is None:
                    st.markdown("## Status: ⚪ **Unknown (no RUL)**")
                    st.info("Backend returned no usable RUL value. Check raw JSON and keys.")
                else:
                    status_text, status_icon = compute_status_from_rul(display_rul)
                    st.markdown(f"## Status: {status_icon} **{status_text}**")

                # Window line
                if w_start is not None and w_end is not None:
                    st.caption(f"Window used: cycles **{w_start}–{w_end}**{label_suffix}")
                else:
                    st.caption(f"Window used: (unable to detect) — {w_reason}")

                # Sequence length
                seq_len = out.get("seq_len") or out.get("sequence_length") or out.get("window_size")
                if seq_len is not None:
                    st.caption(f"Sequence length used by model: **{seq_len}**")

                with st.expander("Show extracted values (debug)"):
                    st.code(
                        json.dumps(
                            {
                                "display_rul_used": display_rul,
                                "pred_rul_calibrated": cal_rul,
                                "pred_rul_raw": raw_rul,
                            },
                            indent=2,
                        ),
                        language="json",
                    )

                with st.expander("Show raw response JSON"):
                    st.code(json.dumps(out, indent=2), language="json")

            except Exception as e:
                st.error(f"Request failed: {e}")


# =========================
# JSON TAB
# =========================
with tab_json:
    st.subheader("Paste JSON")

    st.caption(
        "Paste either:\n"
        "1) Batch schema: `{ \"records\": [...] }`  → calls `/predict/{fd}`\n"
        "2) Single schema: `{ \"fd_name\": \"FD001\", \"unit\": 1, \"rows\": [...] }` → calls `/single/predict`"
    )

    # Template only (so it’s not blank)
    default_payload = {"records": [{"unit": 1, "cycle": 1, "setting1": 0.0, "setting2": 0.0, "setting3": 100.0}]}
    raw = st.text_area("JSON payload", value=json.dumps(default_payload, indent=2), height=350)

    if st.button("🚀 Predict from JSON", type="primary"):
        try:
            payload = json.loads(raw)

            if isinstance(payload, dict) and "records" in payload:
                r = requests.post(batch_url, json=payload, timeout=API_TIMEOUT)
            elif isinstance(payload, dict) and {"fd_name", "unit", "rows"}.issubset(payload.keys()):
                r = requests.post(single_url, json=payload, timeout=API_TIMEOUT)
            else:
                st.error(
                    "Invalid schema. Paste either:\n"
                    "- `{ \"records\": [...] }` (batch)\n"
                    "- `{ \"fd_name\": \"FD001\", \"unit\": 1, \"rows\": [...] }` (single)"
                )
                st.stop()

            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {r.text}")
                st.stop()

            out = r.json()
            display_rul, cal_rul, raw_rul = extract_display_rul(out)
            model_name = extract_model_name(out)

            st.success("✅ Prediction received")

            if display_rul is not None:
                status_text, status_icon = compute_status_from_rul(display_rul)
            else:
                status_text, status_icon = "Unknown (no RUL)", "⚪"

            st.markdown(f"## Status: {status_icon} **{status_text}**")
            st.markdown(f"### Model: **{model_name}**")
            st.markdown(f"### Predicted RUL: **{int(display_rul) if display_rul is not None else 'N/A'}**")

            with st.expander("Show extracted values (debug)"):
                st.code(
                    json.dumps(
                        {
                            "display_rul_used": display_rul,
                            "pred_rul_calibrated": cal_rul,
                            "pred_rul_raw": raw_rul,
                        },
                        indent=2,
                    ),
                    language="json",
                )

            with st.expander("Show raw response JSON"):
                st.code(json.dumps(out, indent=2), language="json")

        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        except Exception as e:
            st.error(f"Request failed: {e}")
