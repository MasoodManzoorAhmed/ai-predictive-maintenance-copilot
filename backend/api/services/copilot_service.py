# backend/api/services/copilot_service.py
"""
Copilot Service Layer.

Responsibilities:
- Act as a thin service wrapper over the RAG pipeline
- Normalize inputs before calling RAG
- Guarantee a stable response schema for API/UI layers
- Keep business logic out of routers

This file should NOT:
- Know about FastAPI request/response objects
- Contain UI logic
- Contain prompt construction logic
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.api.rag.rag_pipeline import answer_question


def run_copilot_query(
    question: str,
    fd_name: Optional[str] = None,
    unit: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a Copilot (RAG) query.

    Parameters
    ----------
    question : str
        User question for the Copilot.
    fd_name : str | None
        Dataset name (FD001–FD004). Optional.
    unit : int | None
        Engine/unit identifier. Optional.
    extra : dict | None
        Extra metadata (style, role, top_k, phase, etc.).

    Returns
    -------
    dict
        {
            "answer": str,
            "sources": list,
            "tool_used": "rag"
        }
    """

    # -------------------------
    # Normalize inputs
    # -------------------------
    question = (question or "").strip()
    if not question:
        return {
            "answer": "No question provided to Copilot.",
            "sources": [],
            "tool_used": "rag",
        }

    fd_name_norm = fd_name.strip().upper() if isinstance(fd_name, str) else None

    unit_norm: Optional[int]
    try:
        unit_norm = int(unit) if unit is not None else None
    except Exception:
        unit_norm = None

    extra_norm: Dict[str, Any] = dict(extra or {})

    # -------------------------
    # Call RAG pipeline
    # -------------------------
    try:
        result = answer_question(
            question=question,
            fd_name=fd_name_norm,
            unit=unit_norm,
            extra=extra_norm,
        ) or {}

        return {
            "answer": result.get("answer", "") or "",
            "sources": result.get("sources", []) or [],
            "tool_used": "rag",
        }

    except Exception as e:
        # IMPORTANT:
        # Router will return HTTP 200 with a safe error message.
        return {
            "answer": f"Copilot failed to generate a response: {str(e)}",
            "sources": [],
            "tool_used": "rag_error",
        }
