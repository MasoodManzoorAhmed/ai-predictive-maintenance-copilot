# backend/api/services/model_loader.py
"""
Model & scaler loader service.

Responsibilities:
- Load trained Keras models (.keras / .h5) for FD001–FD004
- Load feature scalers and RUL scalers
- Cache loaded artifacts to avoid reloading on every request

Design rules:
- No inference logic here
- No FastAPI imports here
- Pure loading + caching

SOURCE OF TRUTH:
- Paths come from the FD config JSON (configs/FD00X_config.json)
"""

from typing import Dict, Any
from pathlib import Path

import joblib
import tensorflow as tf

from backend.api.utils.config_reader import load_fd_config, project_root

# In-memory cache (process-level)
_MODEL_CACHE: Dict[str, Any] = {}
_SCALER_CACHE: Dict[str, Any] = {}


def _resolve_path(path_str: str) -> Path:
    """
    Resolve paths stored in configs.
    Supports both absolute and project-relative paths.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root() / p


def load_model(fd_name: str):
    """
    Load and cache Keras model for given FD.
    """
    fd_name = fd_name.upper()

    if fd_name in _MODEL_CACHE:
        return _MODEL_CACHE[fd_name]

    cfg = load_fd_config(fd_name)

    # Support both keys (FD004 previously used best_deep_model_path)
    model_path_str = cfg.get("model_path") or cfg.get("best_deep_model_path")
    if not model_path_str:
        raise KeyError(f"{fd_name} config missing 'model_path' (or 'best_deep_model_path').")

    model_path = _resolve_path(model_path_str)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # compile=False avoids custom loss/metric issues
    model = tf.keras.models.load_model(model_path, compile=False)

    _MODEL_CACHE[fd_name] = model
    return model


def load_feature_scaler(fd_name: str):
    """
    Load and cache feature scaler for given FD.
    """
    fd_name = fd_name.upper()
    key = f"{fd_name}_feature"

    if key in _SCALER_CACHE:
        return _SCALER_CACHE[key]

    cfg = load_fd_config(fd_name)

    scaler_path_str = cfg.get("feature_scaler_path")
    if not scaler_path_str:
        raise KeyError(f"{fd_name} config missing 'feature_scaler_path'.")

    scaler_path = _resolve_path(scaler_path_str)

    if not scaler_path.exists():
        raise FileNotFoundError(f"Feature scaler not found: {scaler_path}")

    scaler = joblib.load(scaler_path)
    _SCALER_CACHE[key] = scaler
    return scaler


def load_rul_scaler(fd_name: str):
    """
    Load and cache RUL scaler for given FD.
    """
    fd_name = fd_name.upper()
    key = f"{fd_name}_rul"

    if key in _SCALER_CACHE:
        return _SCALER_CACHE[key]

    cfg = load_fd_config(fd_name)

    scaler_path_str = cfg.get("rul_scaler_path")
    if not scaler_path_str:
        raise KeyError(f"{fd_name} config missing 'rul_scaler_path'.")

    scaler_path = _resolve_path(scaler_path_str)

    if not scaler_path.exists():
        raise FileNotFoundError(f"RUL scaler not found: {scaler_path}")

    scaler = joblib.load(scaler_path)
    _SCALER_CACHE[key] = scaler
    return scaler
