# backend/api/routers/copilot.py
"""
Copilot router (Phase 10 will implement full RAG + tool routing).

For Phase 6:
- Provide stable endpoint contract
- Return a clear "not implemented" response
"""

from fastapi import APIRouter

from backend.api.models.requests import CopilotQueryRequest
from backend.api.models.responses import CopilotResponse

router = APIRouter()


@router.post("/query", response_model=CopilotResponse)
def copilot_query(request: CopilotQueryRequest):
    return CopilotResponse(
        answer="Copilot is not implemented yet. Phase 10 will add RAG + tool routing.",
        sources=[],
        tool_used="stub",
    )
