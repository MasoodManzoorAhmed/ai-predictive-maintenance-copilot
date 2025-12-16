# backend/api/models/requests.py
"""
Pydantic request schemas (API input contracts).

Why this matters:
- Prevents garbage inputs reaching your ML models
- Generates clean Swagger docs
- Makes your API professional and testable

We keep these schemas stable.
Business logic goes in services/.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class FD001BatchRequest(BaseModel):
    """
    Minimal batch request for FD001.

    In Phase 6 we keep this flexible because you may send:
    - a list of per-engine records
    - or already-prepared sequences later

    The service layer will decide how to interpret `records`.
    """
    records: List[Dict[str, Any]] = Field(
        ...,
        description="List of engine records (raw cycles or preprocessed rows)."
    )


class FD002BatchRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        description="List of engine records (raw cycles or preprocessed rows)."
    )


class FD003BatchRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        description="List of engine records (raw cycles or preprocessed rows)."
    )


class FD004BatchRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        description="List of engine records (raw cycles or preprocessed rows)."
    )


class SingleEngineRequest(BaseModel):
    """
    Single-engine prediction request.

    You can call this with:
    - fd_name (FD001..FD004)
    - unit (engine id)
    - optional last_cycle
    - rows: list of cycle sensor rows (each row is a dict)
    """
    fd_name: str = Field(..., description="One of: FD001, FD002, FD003, FD004")
    unit: int = Field(..., ge=1, description="Engine unit id (>=1)")
    last_cycle: Optional[int] = Field(None, ge=1, description="Optional last cycle number.")
    rows: List[Dict[str, Any]] = Field(
        ...,
        description="List of cycle rows (dicts). Must contain required sensor/setting fields."
    )


class CopilotQueryRequest(BaseModel):
    """
    Copilot (Phase 10+) query request.
    Keep the schema now; implement logic later.
    """
    question: str = Field(..., min_length=3, description="User question for the maintenance copilot.")
    fd_name: Optional[str] = Field(None, description="Optional FD context (FD001..FD004).")
    unit: Optional[int] = Field(None, ge=1, description="Optional engine unit context.")
    extra: Optional[Dict[str, Any]] = Field(None, description="Optional extra context payload.")
