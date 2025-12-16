# backend/api/models/responses.py
"""
Pydantic response schemas (API output contracts).

Why this matters:
- Ensures consistent API responses
- Makes Swagger docs clean
- Protects frontend & clients from breaking changes

DO NOT put business logic here.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class EnginePrediction(BaseModel):
    unit: int = Field(..., description="Engine unit id")
    pred_rul_raw: float = Field(..., description="Raw predicted RUL (post-safety clipping)")
    pred_rul_calibrated: float = Field(..., description="NASA calibrated RUL (shift + cap applied)")
    true_rul: Optional[float] = Field(None, description="True RUL (if available)")


class BaseFDResponse(BaseModel):
    # identity
    fd_name: str = Field(..., description="Dataset name: FD001/FD002/FD003/FD004")
    model_name: str = Field(..., description="Best model used for this dataset")

    # predictions
    predictions: List[EnginePrediction] = Field(..., description="One prediction per unit (last-window)")

    # debug metadata (no business logic; just reporting)
    sequence_length: int = Field(..., description="Sequence length used to build last-window sequences")
    nasa_shift: float = Field(..., description="NASA calibration shift applied to raw prediction")
    nasa_max_rul_cap: float = Field(..., description="NASA max RUL cap applied after calibration")
    allow_padding: bool = Field(False, description="If True, short sequences were padded (batch should be False)")
    units_count: int = Field(..., description="Number of units predicted in this response")

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
    unit: int = Field(..., description="Engine unit id")
    last_cycle: int = Field(..., description="Last cycle index received for this unit")

    pred_rul_raw: float = Field(..., description="Raw predicted RUL (post-safety clipping)")
    pred_rul_calibrated: float = Field(..., description="NASA calibrated RUL (shift + cap applied)")

    sequence_length: int = Field(..., description="Sequence length used by the model")
    nasa_shift: float = Field(..., description="NASA calibration shift")
    nasa_max_rul_cap: float = Field(..., description="NASA max RUL cap")
    allow_padding: bool = Field(..., description="If True, input had < sequence_length cycles and was padded")


class CopilotResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    tool_used: Optional[str] = None
