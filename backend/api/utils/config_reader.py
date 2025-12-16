# backend/api/utils/config_reader.py
"""
Config reader utilities.

Responsibilities:
- Load JSON configs from configs/ directory
- Validate required keys (light validation)
- Provide clean getters for FD configs + unified index

Design:
- Keep file paths relative to project root.
- Avoid hardcoding Google Drive paths.
"""

import json
from pathlib import Path
from typing import Dict, Any


def project_root() -> Path:
    """
    Returns repository root based on this file location.
    Assumes: backend/api/utils/config_reader.py
    Root is 4 levels up: utils -> api -> backend -> <root>
    """
    return Path(__file__).resolve().parents[3]


def configs_dir() -> Path:
    return project_root() / "configs"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_fd_config(fd_name: str) -> Dict[str, Any]:
    """
    Loads FD001_config.json, FD002_config.json, etc. from configs/.
    This FD config is the SINGLE SOURCE OF TRUTH for:
      - model_path
      - feature_scaler_path
      - rul_scaler_path
      - sequence_length
      - nasa params
      - feature columns
    """
    fd_name = fd_name.upper().strip()
    valid = {"FD001", "FD002", "FD003", "FD004"}
    if fd_name not in valid:
        raise ValueError(f"Invalid fd_name: {fd_name}. Must be one of {sorted(valid)}")

    path = configs_dir() / f"{fd_name}_config.json"
    cfg = load_json(path)

    # Light validation (strict validation happens in services)
    required = [
        "final_feature_columns",
        "sequence_length",
        "nasa_shift",
        "nasa_max_rul_cap",
        "model_path",
        "feature_scaler_path",
        "rul_scaler_path",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"{fd_name} config missing keys: {missing} in {path}")

    return cfg


def load_unified_model_index() -> Dict[str, Any]:
    """
    Loads unified_model_index.json from configs/.

    IMPORTANT:
    This file is OPTIONAL metadata ONLY.
    Example use: map FD -> config_path for UI/debugging.

    Do NOT depend on it for model_path / scaler paths / seq_len / nasa params.
    """
    path = configs_dir() / "unified_model_index.json"
    if not path.exists():
        # allow project to run even if you remove it
        return {}
    return load_json(path)
