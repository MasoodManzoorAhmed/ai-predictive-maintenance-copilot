# backend/api/models/requests.py
"""
Pydantic request schemas (API input contracts).

Why this matters:
- Prevents invalid inputs reaching ML / RAG logic
- Generates clean Swagger docs
- Keeps API stable and professional

These schemas are SAFE for:
- Phase 6 (RUL inference)
- Phase 10 (RAG Copilot)
- Phase 11 (context-aware Copilot)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# ============================
# Shared Enums / Validators
# ============================

class FDName(str, Enum):
    FD001 = "FD001"
    FD002 = "FD002"
    FD003 = "FD003"
    FD004 = "FD004"


class CopilotStyle(str, Enum):
    """
    ONLY 3 styles supported across UI + backend.
    """
    CHECKLIST = "Checklist"
    CONCISE = "Concise"
    DETAILED = "Detailed"


class CopilotRole(str, Enum):
    """
    Optional persona for the answer.
    """
    MAINTENANCE_MANAGER = "Maintenance Manager"
    TECHNICIAN = "Technician"
    RELIABILITY_ENGINEER = "Reliability Engineer"


# ============================
# Batch Prediction Requests
# ============================

class FD001BatchRequest(BaseModel):
    """Batch prediction request for FD001."""
    records: List[Dict[str, Any]] = Field(
        ...,
        description="List of engine records (raw cycles or preprocessed rows).",
    )


class FD002BatchRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        description="List of engine records (raw cycles or preprocessed rows).",
    )


class FD003BatchRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        description="List of engine records (raw cycles or preprocessed rows).",
    )


class FD004BatchRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        description="List of engine records (raw cycles or preprocessed rows).",
    )


# ============================
# Single Engine Prediction
# ============================

class SingleEngineRequest(BaseModel):
    """
    Single-engine prediction request.

    Used when client already has cycle-level rows.
    """
    fd_name: FDName = Field(
        ..., description="One of: FD001, FD002, FD003, FD004"
    )
    unit: int = Field(
        ..., ge=1, description="Engine unit id (>=1)"
    )
    last_cycle: Optional[int] = Field(
        None, ge=1, description="Optional last observed cycle number."
    )
    rows: List[Dict[str, Any]] = Field(
        ...,
        description="List of cycle rows (dicts with sensor/setting values).",
    )


# ============================
# Copilot / RAG Request
# ============================

class CopilotQueryRequest(BaseModel):
    """
    Maintenance Copilot query request.

    - UI should send ONLY these 3 styles: Checklist / Concise / Detailed
    - Role is optional but recommended for consistent output
    """

    # Core question
    question: str = Field(
        ...,
        min_length=3,
        description="User question for the maintenance copilot.",
    )

    # Optional contextual conditioning (safe defaults)
    fd_name: Optional[FDName] = Field(
        None,
        description="Optional FD context (FD001–FD004).",
    )
    unit: Optional[int] = Field(
        None,
        ge=1,
        description="Optional engine unit context.",
    )

    # Optional: UI can set this based on predicted RUL band
    risk_band: Optional[str] = Field(
        None,
        description="Optional risk band (e.g., Critical/Warning/Watch/Healthy).",
    )

    # Output controls 
    style: CopilotStyle = Field(
        default=CopilotStyle.CHECKLIST,
        description="Answer style. Allowed: Checklist, Concise, Detailed.",
    )
    role: CopilotRole = Field(
        default=CopilotRole.MAINTENANCE_MANAGER,
        description="Answer role/persona (stable set).",
    )

    # Retrieval control
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of RAG chunks to retrieve.",
    )

    # Extra extensibility (future tools, metadata)
    extra: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional extra context payload (phase, flags, etc.).",
    )

    @validator("question")
    def _strip_question(cls, v: str) -> str:
        v2 = (v or "").strip()
        if len(v2) < 3:
            raise ValueError("question must be at least 3 characters")
        return v2
