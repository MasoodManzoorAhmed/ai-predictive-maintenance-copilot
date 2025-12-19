# scripts/ask_rag.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# ============================================================
# Clean demo output: suppress TF/oneDNN logs
# ============================================================
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


# ============================================================
# Ensure project root is on sys.path so `import backend...` works
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from backend.api.rag.rag_config import RAGConfig  # noqa: E402
from backend.api.rag.rag_pipeline import answer_question  # noqa: E402


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def print_sources(sources: List[Dict[str, Any]]) -> None:
    """
    Pretty-print retrieved RAG sources.
    """
    if not sources:
        print("(no sources returned)")
        return

    for i, s in enumerate(sources, start=1):
        src = s.get("source", "unknown")
        page = s.get("page", "?")
        chunk_id = s.get("chunk_id", "?")
        score = s.get("score", 0.0)
        preview = (s.get("text_preview") or "").replace("\n", " ").strip()

        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0

        print(f"\n[{i}] source={src} | page={page} | chunk_id={chunk_id} | score={score_f:.4f}")
        if preview:
            print(f"    {preview}")


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask_rag.py "<question>" [top_k]')
        return 2

    question = sys.argv[1]
    top_k = _safe_int(sys.argv[2] if len(sys.argv) >= 3 else None, default=5)
    if top_k < 1:
        top_k = 1
    if top_k > 20:
        top_k = 20

    cfg = RAGConfig.from_env()

    # -------------------------
    # Safety & clarity checks
    # -------------------------
    if not cfg.enabled:
        print(" RAG is disabled.")
        print("   Set RAG_ENABLED=1 and try again.")
        return 1

    # Provider key checks 
    if cfg.llm_provider in ("openrouter", "openai"):
        if cfg.llm_provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
            print(" Missing OPENROUTER_API_KEY.")
            print("   Set it in your environment and retry.")
            return 1

    # -------------------------
    # Run RAG pipeline 
    # -------------------------
    result = answer_question(
        question=question,
        fd_name=None,
        unit=None,
        extra=None,
        top_k=top_k,
    ) or {}

    # -------------------------
    # Output
    # -------------------------
    print("\n=== QUESTION ===")
    print(question)

    print("\n=== ANSWER ===")
    print((result.get("answer") or "").strip())

    print("\n=== SOURCES USED ===")
    print_sources(result.get("sources") or [])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
