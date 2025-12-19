# backend/api/models/responses.py
"""
Pydantic response schemas (API output contracts).

Why this matters:
- Ensures consistent API responses
- Makes Swagger docs clean
- Protects frontend & clients from breaking changes

"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ============================
# Prediction responses
# ============================

class EnginePrediction(BaseModel):
    unit: int = Field(..., ge=1, description="Engine unit id (>=1)")
    pred_rul_raw: float = Field(..., description="Raw predicted RUL (post-safety clipping)")
    pred_rul_calibrated: float = Field(..., description="NASA calibrated RUL (shift + cap applied)")
    true_rul: Optional[float] = Field(None, description="True RUL (if available)")


class BaseFDResponse(BaseModel):
    # identity
    fd_name: str = Field(..., description="Dataset name: FD001/FD002/FD003/FD004")
    model_name: str = Field(..., description="Best model used for this dataset")

    # predictions
    predictions: List[EnginePrediction] = Field(
        default_factory=list,
        description="One prediction per unit (last-window)",
    )

    # debug metadata
    sequence_length: int = Field(..., ge=1, description="Sequence length used to build last-window sequences")
    nasa_shift: float = Field(..., description="NASA calibration shift applied to raw prediction")
    nasa_max_rul_cap: float = Field(..., description="NASA max RUL cap applied after calibration")
    allow_padding: bool = Field(False, description="If True, short sequences were padded (batch should be False)")
    units_count: int = Field(..., ge=0, description="Number of units predicted in this response")

    message: Optional[str] = Field(None, description="Optional status message")


class FD001PredictionResponse(BaseFDResponse):
    pass


class FD002PredictionResponse(BaseFDResponse):
    pass


class FD003PredictionResponse(BaseFDResponse):
    pass


class FD004PredictionResponse(BaseFDResponse):
    pass


class SingleEnginePredictionResponse(BaseModel):
    fd_name: str = Field(..., description="Dataset name: FD001/FD002/FD003/FD004")
    model_name: str = Field(..., description="Best model used for this dataset")

    unit: int = Field(..., ge=1, description="Engine unit id (>=1)")
    last_cycle: int = Field(..., ge=1, description="Last cycle index received for this unit (>=1)")

    pred_rul_raw: float = Field(..., description="Raw predicted RUL (post-safety clipping)")
    pred_rul_calibrated: float = Field(..., description="NASA calibrated RUL (shift + cap applied)")

    sequence_length: int = Field(..., ge=1, description="Sequence length used by the model")
    nasa_shift: float = Field(..., description="NASA calibration shift")
    nasa_max_rul_cap: float = Field(..., description="NASA max RUL cap")
    allow_padding: bool = Field(..., description="If True, input had < sequence_length cycles and was padded")


# ============================
# Copilot / RAG responses
# ============================

class CopilotSource(BaseModel):
    source: str = Field(..., description="Document name or identifier (e.g., PDF filename)")
    page: Optional[int] = Field(None, ge=1, description="Page number in the document (if available)")
    chunk_id: Optional[str] = Field(None, description="Chunk id used during indexing")
    score: Optional[float] = Field(None, description="Retriever similarity score (higher = more relevant)")
    text_preview: Optional[str] = Field(None, description="Short preview of the retrieved text")


class CopilotResponse(BaseModel):
    answer: str = Field(..., description="Final copilot answer")

    sources: List[CopilotSource] = Field(
        default_factory=list,
        description="Evidence chunks used (may be empty).",
    )

    tool_used: str = Field(
        default="rag",
        description="Which tool produced the answer (rag/no-llm/stub).",
    )

    # echo-back of controls 
    style: Optional[str] = Field(None, description="Answer style that was applied (Checklist/Concise/Detailed)")
    role: Optional[str] = Field(None, description="Role/persona used for the answer")
    top_k: Optional[int] = Field(None, description="Retriever top_k used")
