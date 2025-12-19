# backend/api/rag/rag_config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _safe_int(value: Optional[str], default: int) -> int:
    try:
        if value is None:
            return default
        v = value.strip()
        if v == "":
            return default
        return int(v)
    except Exception:
        return default


def _in_container() -> bool:
    """
    Heuristics for Cloud Run / containers:
    - Cloud Run sets K_SERVICE
    - We also have WORKDIR=/app in Dockerfile
    """
    return bool(os.getenv("K_SERVICE")) or os.getenv("PWD", "") == "/app"


def _default_index_dir() -> Path:
    # In your Docker build you COPY rag_index -> /app/rag_index, so this is safest.
    return Path("/app/rag_index") if _in_container() else Path("rag_index")


@dataclass(frozen=True)
class RAGConfig:
    enabled: bool
    index_dir: Path
    top_k: int
    max_context_chars: int
    llm_provider: str  # "openai" | "openrouter" | "none"
    llm_model: str

    @staticmethod
    def from_env() -> "RAGConfig":
        enabled = (os.getenv("RAG_ENABLED", "1") or "1").strip() == "1"

        # -----------------------------
        # Index dir resolution
        # -----------------------------
        # If env var is set:
        #   - absolute: use as-is
        #   - relative: resolve from /app when in container, else from cwd
        raw_index_dir = (os.getenv("RAG_INDEX_DIR") or "").strip()

        if raw_index_dir:
            p = Path(raw_index_dir)
            if p.is_absolute():
                index_dir = p
            else:
                # In container, we want /app/<relative>
                base = Path("/app") if _in_container() else Path.cwd()
                index_dir = (base / p)
        else:
            index_dir = _default_index_dir()

        index_dir = index_dir.resolve()

        # -----------------------------
        # Retrieval controls
        # -----------------------------
        top_k = _safe_int(os.getenv("RAG_TOP_K"), default=4)
        if top_k < 1:
            top_k = 1
        if top_k > 20:
            top_k = 20

        max_context_chars = _safe_int(os.getenv("RAG_MAX_CONTEXT_CHARS"), default=8000)
        if max_context_chars < 500:
            max_context_chars = 500
        if max_context_chars > 50000:
            max_context_chars = 50000

        # -----------------------------
        # LLM controls
        # -----------------------------
        llm_provider = (os.getenv("LLM_PROVIDER", "none") or "none").strip().lower()

        # If using OpenRouter, the common pattern is "openai/<model>".
        # Default to a safe value per provider.
        if llm_provider == "openrouter":
            default_model = "openai/gpt-4o-mini"
        elif llm_provider == "openai":
            default_model = "gpt-4o-mini"
        else:
            default_model = "none"

        llm_model = (os.getenv("LLM_MODEL", default_model) or default_model).strip()

        return RAGConfig(
            enabled=enabled,
            index_dir=index_dir,
            top_k=top_k,
            max_context_chars=max_context_chars,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
