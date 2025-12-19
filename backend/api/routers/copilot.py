# backend/api/routers/copilot.py
"""
Copilot router (Phase 10+).

Implements:
- /copilot/query  (stable)
- /copilot/ask    (alias for legacy calls)

Calls the stable RAG entrypoint:
backend.api.rag.rag_pipeline.answer_question

Return schema matches CopilotResponse with structured sources.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request

from backend.api.models.requests import CopilotQueryRequest
from backend.api.models.responses import CopilotResponse, CopilotSource
from backend.api.rag.rag_config import RAGConfig
from backend.api.rag.rag_pipeline import answer_question

router = APIRouter()

log = logging.getLogger("copilot_router")

_ALLOWED_STYLES = {"Checklist", "Concise", "Detailed"}
_DEFAULT_STYLE = "Checklist"
_DEFAULT_ROLE = "Maintenance Manager"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _normalize_sources(raw_sources: Optional[List[Dict[str, Any]]]) -> List[CopilotSource]:
    """
    rag_pipeline returns dicts like:
      {source, page, chunk_id, score, text_preview}
    Convert them into CopilotSource objects.
    """
    out: List[CopilotSource] = []
    for s in raw_sources or []:
        out.append(
            CopilotSource(
                source=str(s.get("source", "unknown")),
                page=s.get("page"),
                chunk_id=s.get("chunk_id"),
                score=s.get("score"),
                text_preview=s.get("text_preview"),
            )
        )
    return out


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _pick_style(payload: CopilotQueryRequest) -> str:
    """
    Source of truth:
      1) payload.extra["style"]
      2) default
    """
    style = None
    if payload.extra and isinstance(payload.extra, dict):
        style = payload.extra.get("style")

    style = str(style).strip() if style is not None else ""
    if style not in _ALLOWED_STYLES:
        return _DEFAULT_STYLE
    return style


def _pick_role(payload: CopilotQueryRequest) -> str:
    role = None
    if payload.extra and isinstance(payload.extra, dict):
        role = payload.extra.get("role")

    role = str(role).strip() if role is not None else ""
    return role or _DEFAULT_ROLE


def _clean_error_message(e: Exception) -> str:
    """
    Keep error message useful but avoid leaking secrets.
    """
    msg = str(e) or e.__class__.__name__
    # very basic redaction patterns (safe, not perfect)
    for key in ["OPENROUTER_API_KEY", "API_KEY", "Authorization", "Bearer "]:
        if key in msg:
            msg = msg.replace(key, "[REDACTED]")
    return msg[:800]  # avoid massive stack text in HTTP response


# ------------------------------------------------------------
# Core handler (shared by /query and /ask)
# ------------------------------------------------------------
def _copilot_rag(payload: CopilotQueryRequest) -> CopilotResponse:
    cfg = RAGConfig.from_env()

    style = _pick_style(payload)
    role = _pick_role(payload)

    # basic validation (avoid confusing 500s)
    question = (payload.question or "").strip()
    if not question:
        raise ValueError("`question` is required and cannot be empty.")

    if not cfg.enabled:
        return CopilotResponse(
            answer="RAG is disabled. Set RAG_ENABLED=1 and redeploy.",
            sources=[],
            tool_used="rag_disabled",
            style=style,
            role=role,
            top_k=None,
        )

    # Merge extra safely
    extra: Dict[str, Any] = dict(payload.extra or {})
    extra["style"] = style
    extra["role"] = role

    if payload.fd_name:
        extra.setdefault("fd_name", payload.fd_name)
    if payload.unit is not None:
        extra.setdefault("unit", payload.unit)

    # top_k: request overrides config (bounded)
    req_top_k = getattr(payload, "top_k", None)
    top_k = _safe_int(req_top_k, default=_safe_int(getattr(cfg, "top_k", None), 5))
    top_k = max(1, min(top_k, 20))

    # Call stable entrypoint
    result = answer_question(
        question=question,
        fd_name=payload.fd_name,
        unit=payload.unit,
        extra=extra,
        top_k=top_k,
    ) or {}

    return CopilotResponse(
        answer=str(result.get("answer") or ""),
        sources=_normalize_sources(result.get("sources")),
        tool_used=str(result.get("tool_used") or "rag"),
        style=style,
        role=role,
        top_k=top_k,
    )


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@router.post("/query", response_model=CopilotResponse)
def copilot_query(payload: CopilotQueryRequest, request: Request):
    try:
        return _copilot_rag(payload)
    except Exception as e:
        # IMPORTANT: log full exception server-side (Cloud Run logs)
        log.exception("Copilot /query failed")
        raise HTTPException(status_code=500, detail=_clean_error_message(e))


@router.post("/ask", response_model=CopilotResponse)
def copilot_ask(payload: CopilotQueryRequest, request: Request):
    try:
        return _copilot_rag(payload)
    except Exception as e:
        log.exception("Copilot /ask failed")
        raise HTTPException(status_code=500, detail=_clean_error_message(e))
