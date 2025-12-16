# backend/api/utils/nasa_metrics.py
"""
NASA scoring + calibration utilities.

We keep this separate so:
- Inference engine remains clean
- Score logic is testable
"""

from typing import Tuple
import numpy as np


def nasa_asymmetric_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard NASA CMAPSS asymmetric scoring.
    Penalizes late predictions (underestimation) more than early ones.

    score = sum( exp(-e/13)-1 if e<0 else exp(e/10)-1 )
    where e = y_pred - y_true
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    e = y_pred - y_true
    score = np.where(e < 0, np.exp(-e / 13.0) - 1.0, np.exp(e / 10.0) - 1.0)
    return float(np.sum(score))


def apply_nasa_calibration(y_pred: np.ndarray, shift: float, max_rul: float) -> np.ndarray:
    """
    Calibration used in your notebook:
    - y_cal = clip(y_pred - shift, 0, max_rul)
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_cal = y_pred - float(shift)
    y_cal = np.clip(y_cal, 0.0, float(max_rul))
    return y_cal


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return rmse, mae
