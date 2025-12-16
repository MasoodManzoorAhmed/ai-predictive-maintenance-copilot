# backend/api/utils/logging_utils.py
"""
Logging utilities.

Why this exists:
- Central place to control logging format
- Easy to switch to cloud logging later (AWS/GCP)
- Avoid print() in production code
"""

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Returns a configured logger.
    """
    logger = logging.getLogger(name or "ai_predictive_maintenance")

    if logger.handlers:
        # Logger already configured
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger
