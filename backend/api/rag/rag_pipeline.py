# backend/api/rag/rag_pipeline.py
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging

from .rag_config import RAGConfig
from .retriever import RAGRetriever
from .prompting import build_rag_prompt
from .llm_client import call_llm

log = logging.getLogger("rag_pipeline")


# -----------------------------
# Cache retriever by index path
# -----------------------------
@lru_cache(maxsize=4)
def _get_retriever_cached(index_dir: str) -> RAGRetriever:
    """
    Cache retriever by index_dir so Cloud Run doesn't reload FAISS every request.
    """
    return RAGRetriever.load(Path(index_dir))


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _bounded_top_k(raw: Any, default: int) -> int:
    top_k = _safe_int(raw, default)
    if top_k < 1:
        top_k = 1
    if top_k > 20:
        top_k = 20
    return top_k


def _check_index_files(index_dir: Path) -> Tuple[bool, str]:
    """
    Fail fast if index files are missing in Cloud Run image.
    """
    chunks = index_dir / "chunks.json"
    faiss = index_dir / "faiss.index"
    if not index_dir.exists():
        return False, f"RAG index_dir does not exist: {index_dir}"
    if not chunks.exists():
        return False, f"Missing RAG file: {chunks}"
    if not faiss.exists():
        return False, f"Missing RAG file: {faiss}"
    return True, "ok"


def rag_answer(
    question: str,
    fd_name: Optional[str],
    unit: Optional[int],
    extra: Dict[str, Any] | None,
) -> Dict[str, Any]:
    cfg = RAGConfig.from_env()

    # Minimal contract always
    if not cfg.enabled:
        return {
            "answer": "RAG is disabled (RAG_ENABLED=0).",
            "sources": [],
            "tool_used": "rag_disabled",
        }

    q = (question or "").strip()
    if not q:
        return {
            "answer": "Question is empty. Provide a non-empty question.",
            "sources": [],
            "tool_used": "bad_request",
        }

    # Ensure index exists in container
    index_dir = Path(cfg.index_dir)
    ok, msg = _check_index_files(index_dir)
    if not ok:
        # log full detail server-side
        log.error("RAG index not ready: %s", msg)
        return {
            "answer": f"RAG index not ready: {msg}",
            "sources": [],
            "tool_used": "rag_index_missing",
        }

    # Load retriever (cached by path)
    try:
        retriever = _get_retriever_cached(str(index_dir))
    except Exception as e:
        log.exception("Failed to load retriever from %s", index_dir)
        return {
            "answer": f"Failed to load RAG retriever: {e}",
            "sources": [],
            "tool_used": "retriever_load_failed",
        }

    # top_k: request overrides config (extra already merged by router)
    raw_top_k = (extra or {}).get("top_k", cfg.top_k)
    top_k = _bounded_top_k(raw_top_k, default=_safe_int(cfg.top_k, 5))

    # Retrieve
    try:
        retrieved = retriever.search(q, top_k=top_k)
    except Exception as e:
        log.exception("Retriever.search failed")
        return {
            "answer": f"Retriever failed: {e}",
            "sources": [],
            "tool_used": "retriever_search_failed",
        }

    # Build prompt
    try:
        prompt = build_rag_prompt(
            question=q,
            fd_name=fd_name,
            unit=unit,
            extra=extra,
            retrieved=retrieved,
        )
    except Exception as e:
        log.exception("build_rag_prompt failed")
        return {
            "answer": f"Prompt build failed: {e}",
            "sources": [],
            "tool_used": "prompt_build_failed",
        }

    # LLM call (common 500 cause in Cloud Run if OPENROUTER_API_KEY missing)
    try:
        answer = call_llm(prompt=prompt, provider=cfg.llm_provider, model=cfg.llm_model)
    except Exception as e:
        log.exception("call_llm failed (provider=%s model=%s)", cfg.llm_provider, cfg.llm_model)
        return {
            "answer": (
                "LLM call failed. Check LLM_PROVIDER / LLM_MODEL / OPENROUTER_API_KEY env vars. "
                f"Error: {e}"
            ),
            "sources": [],
            "tool_used": "llm_call_failed",
        }

    # Sources
    sources: List[Dict[str, Any]] = []
    for r in retrieved:
        c = r.chunk
        sources.append(
            {
                "source": getattr(c, "source", "unknown"),
                "page": getattr(c, "page", None),
                "chunk_id": getattr(c, "chunk_id", None),
                "score": getattr(r, "score", None),
                "text_preview": (getattr(c, "text", "") or "")[:280],
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "tool_used": "rag",
    }


def answer_question(
    question: str,
    fd_name: str | None = None,
    unit: int | None = None,
    extra: dict | None = None,
    top_k: int | None = None,
) -> Dict[str, Any]:
    """
    Stable public entry point for RAG.

    Backward compatible:
      - answer_question(question, top_k=5)
      - answer_question(question, fd_name="FD004", unit=1, extra={...})

    Rules:
      - If top_k is provided explicitly, it overrides extra["top_k"].
      - Always returns {"answer": str, "sources": list, "tool_used": str}.
    """
    extra_dict: Dict[str, Any] = dict(extra or {})

    if top_k is not None:
        extra_dict["top_k"] = top_k

    return rag_answer(
        question=question,
        fd_name=fd_name,
        unit=unit,
        extra=extra_dict,
    )
