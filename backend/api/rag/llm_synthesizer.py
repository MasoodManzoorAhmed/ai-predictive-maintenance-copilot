# backend/api/rag/llm_synthesizer.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if (v is not None and v.strip() != "") else default


def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Build a grounded prompt:
    - Only use the provided context chunks.
    - If not enough info, say so and ask for more info.
    - Cite chunk sources (pdf + page + chunk_id).
    """
    context_blocks = []
    for i, c in enumerate(contexts, start=1):
        meta = c.get("meta", {})
        src = meta.get("source", "unknown_source")
        page = meta.get("page", "NA")
        chunk_id = meta.get("chunk_id", "NA")
        text = c.get("text", "")
        context_blocks.append(
            f"[{i}] SOURCE={src} PAGE={page} CHUNK_ID={chunk_id}\n{text}"
        )

    joined = "\n\n".join(context_blocks)

    return f"""You are a maintenance copilot. Answer the user's question using ONLY the provided context.

Rules:
- If the context does not contain the answer, say: "I don't have enough information in the provided documents."
- Do NOT invent procedures.
- Provide a short, practical answer.
- After the answer, include "Evidence:" with bullet citations like: (SOURCE, page, chunk_id).

Question:
{question}

Context:
{joined}
"""


def call_openrouter(prompt: str) -> str:
    """
    Calls OpenRouter Chat Completions API.
    Requires:
      - OPENROUTER_API_KEY
    Optional:
      - OPENROUTER_MODEL (default: openai/gpt-4o-mini)
    """
    api_key = _env("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY env var. Set it in PowerShell before running."
        )

    model = _env("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful, careful maintenance copilot."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    data = r.json()

    return data["choices"][0]["message"]["content"]
