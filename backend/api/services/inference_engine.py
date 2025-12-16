# backend/api/services/inference_engine.py
"""
Unified inference engine.

Responsibilities:
- End-to-end inference for FD001–FD004
- Apply feature engineering
- Build last-window sequences
- Run model prediction
- Apply NASA calibration and metrics

STRICT RULES:
- No FastAPI code here
- No disk writes here
- Deterministic behavior only
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

from backend.api.utils.config_reader import load_fd_config, load_unified_model_index
from backend.api.utils.logging_utils import get_logger
from backend.api.utils.nasa_metrics import (
    apply_nasa_calibration,
    nasa_asymmetric_score,
    rmse_mae,
)
from backend.api.services.model_loader import (
    load_model,
    load_feature_scaler,
    load_rul_scaler,
)
from backend.api.services.feature_engineering import apply_rolling_and_delta_features
from backend.api.services.sequence_builder import build_last_window_sequences

logger = get_logger("inference_engine")


def run_fd_inference(
    fd_name: str,
    df_input: pd.DataFrame,
    y_true: np.ndarray | None = None,
    allow_padding: bool = False,  # ✅ NEW
) -> Dict[str, Any]:
    """
    Runs inference for one FD dataset.

    Inputs:
    - fd_name: FD001 / FD002 / FD003 / FD004
    - df_input: dataframe containing unit, cycle, settings, sensors
    - y_true: optional true RUL (for metrics)
    - allow_padding: if True, pad short unit histories to seq_len (recommended only for single-engine demo/UI)

    Returns:
    - dict with predictions, metrics, metadata
    """

    fd_name = fd_name.upper().strip()
    logger.info(f"Starting inference for {fd_name}")

    # ---- Load configs (source of truth) ----
    cfg = load_fd_config(fd_name)

    # unified index is OPTIONAL metadata only
    unified_all = load_unified_model_index()
    unified_entry = unified_all.get(fd_name, {})

    # ---- Load artifacts ----
    model = load_model(fd_name)
    feature_scaler = load_feature_scaler(fd_name)
    rul_scaler = load_rul_scaler(fd_name)

    # ---- Required inference params from FD config ----
    final_feature_cols = cfg["final_feature_columns"]

    if "sequence_length" not in cfg:
        raise KeyError(f"{fd_name}: config missing required key 'sequence_length'")
    seq_len = int(cfg["sequence_length"])

    if "nasa_shift" not in cfg:
        raise KeyError(f"{fd_name}: config missing required key 'nasa_shift'")
    if "nasa_max_rul_cap" not in cfg:
        raise KeyError(f"{fd_name}: config missing required key 'nasa_max_rul_cap'")

    nasa_shift = float(cfg["nasa_shift"])
    nasa_max_rul = float(cfg["nasa_max_rul_cap"])

    model_name = cfg.get("best_model_name") or cfg.get("best_deep_model_name") or "UNKNOWN_MODEL"

    # ---- Identify base sensor columns (before roll/delta) ----
    base_sensor_cols = [
        c for c in final_feature_cols
        if not c.endswith(("_roll3_mean", "_roll3_std", "_roll5_mean", "_roll5_std", "_delta"))
        and c not in ("unit", "cycle")
    ]

    # ---- Feature engineering ----
    df_fe = apply_rolling_and_delta_features(
        df=df_input,
        sensor_cols=base_sensor_cols,
        unit_col="unit",
        cycle_col="cycle",
    )

    # ---- Validate engineered features ----
    missing = [c for c in final_feature_cols if c not in df_fe.columns]
    if missing:
        raise KeyError(f"{fd_name}: missing engineered features: {missing[:10]}")

    df_final = df_fe[["unit", "cycle"] + final_feature_cols].copy()

    # ---- Scale features ----
    X_scaled = feature_scaler.transform(df_final[final_feature_cols].values)
    df_scaled = df_final.copy()
    df_scaled[final_feature_cols] = X_scaled

    # ---- Build last-window sequences ----
    # ✅ IMPORTANT: allow_padding=True -> skip_short=False -> pad to seq_len
    X_seq, units = build_last_window_sequences(
        df=df_scaled,
        feature_cols=final_feature_cols,
        seq_len=seq_len,
        unit_col="unit",
        cycle_col="cycle",
        skip_short=(not allow_padding),
    )

    # ---- Predict ----
    y_pred_scaled = model.predict(X_seq, verbose=0).reshape(-1, 1)
    y_pred_raw = rul_scaler.inverse_transform(y_pred_scaled).ravel()
    y_pred_raw = np.clip(y_pred_raw, 0.0, nasa_max_rul)
    # ---- NASA calibration ----
    y_pred_cal = apply_nasa_calibration(
        y_pred=y_pred_raw,
        shift=nasa_shift,
        max_rul=nasa_max_rul,
    )

    result = {
        "fd_name": fd_name,
        "model_name": model_name,
        "units": units.tolist(),
        "pred_rul_raw": y_pred_raw.tolist(),
        "pred_rul_calibrated": y_pred_cal.tolist(),
        "sequence_length": seq_len,
        "nasa_shift": nasa_shift,
        "nasa_max_rul_cap": nasa_max_rul,
        "config_path": unified_entry.get("config_path", f"configs/{fd_name}_config.json"),
        "allow_padding": bool(allow_padding),  # ✅ optional debug metadata
    }

    # ---- Metrics (optional) ----
    if y_true is not None:
        rmse, mae = rmse_mae(y_true, y_pred_cal)
        nasa_score = nasa_asymmetric_score(y_true, y_pred_cal)
        result["metrics"] = {"rmse": rmse, "mae": mae, "nasa_score": nasa_score}

    logger.info(
        f"Inference complete for {fd_name} | engines={len(units)} | allow_padding={allow_padding}"
    )
    return result
