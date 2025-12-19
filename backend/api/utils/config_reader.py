# backend/api/utils/config_reader.py
"""
Config reader utilities.

Responsibilities:
- Load JSON configs from configs/ directory
- Validate required keys (light validation)
- Provide clean getters for FD configs + unified index

Design:
- Prefer container-safe absolute paths when running in Docker/Cloud Run.
- Fall back to repo-relative paths for local development.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict


def project_root() -> Path:
    """
    Determine the repo root robustly.

    Cloud Run/Docker:
      - We copy configs to /app/configs
      - So /app is the effective "project root"

    Local dev:
      - Walk up from this file until we find a folder that contains "configs/"
    """
    # 1) Strong container default
    container_root = Path("/app")
    if (container_root / "configs").exists():
        return container_root

    # 2) Optional override (useful for tests or nonstandard layouts)
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if (p / "configs").exists():
            return p

    # 3) Local dev fallback: search upwards
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "configs").exists():
            return p

    # 4) Fail loudly with a helpful message
    raise FileNotFoundError(
        "Could not locate project root (configs/ not found). "
        "Expected /app/configs in container or a configs/ folder in repo. "
        "You can set PROJECT_ROOT env var if needed."
    )


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

    This file is OPTIONAL metadata ONLY.
    Example use: map FD -> config_path for UI/debugging.

    Do NOT depend on it for model_path / scaler paths / seq_len / nasa params.
    """
    path = configs_dir() / "unified_model_index.json"
    if not path.exists():
        return {}
    return load_json(path)
