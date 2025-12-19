# backend/api/rag/llm_client.py
from __future__ import annotations

import os
import re
import time
from typing import Dict, Optional, Tuple


# -----------------------------
# Env helpers
# -----------------------------
def _get_env_str(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _get_env_int(key: str, default: int) -> int:
    raw = _get_env_str(key, str(default))
    try:
        return int(raw)
    except Exception:
        return default


def _get_env_float(key: str, default: float) -> float:
    raw = _get_env_str(key, str(default))
    try:
        return float(raw)
    except Exception:
        return default


def _require_env(key: str, provider: str) -> str:
    val = _get_env_str(key)
    if not val:
        raise RuntimeError(f"Missing required environment variable `{key}` for LLM_PROVIDER={provider}")
    return val


# -----------------------------
# Style controls (3 options)
# -----------------------------
_STYLE_RE = re.compile(r"USER_PREFS:\s*role=.*?\s+style=([A-Za-z]+)", re.IGNORECASE)


def _infer_style_from_prompt(prompt: str) -> str:
    """
    I infer from  prompt header line:
      USER_PREFS: role=... style=Concise|Checklist|Detailed
    If missing, default to Checklist.
    """
    m = _STYLE_RE.search(prompt or "")
    if not m:
        return "Checklist"
    style = (m.group(1) or "").strip().lower()
    if style in ("concise", "checklist", "detailed"):
        return style.capitalize()
    return "Checklist"


def _max_tokens_for_style(style: str) -> int:
    """
    Conservative defaults.
      LLM_MAX_TOKENS_CONCISE, LLM_MAX_TOKENS_CHECKLIST, LLM_MAX_TOKENS_DETAILED
    """
    style = (style or "").strip().lower()
    if style == "concise":
        return _get_env_int("LLM_MAX_TOKENS_CONCISE", 280)
    if style == "detailed":
        return _get_env_int("LLM_MAX_TOKENS_DETAILED", 950)
    # Checklist (default)
    return _get_env_int("LLM_MAX_TOKENS_CHECKLIST", 520)


def _temperature() -> float:
    # Keeping low to reduce hallucination / variability
    return _get_env_float("LLM_TEMPERATURE", 0.2)


def _timeouts_and_retries() -> Tuple[float, int]:
    # OpenAI SDK uses httpx internally; these values are used for our retry loop + sensible wait.
    timeout_s = float(_get_env_int("LLM_TIMEOUT_SECONDS", 45))
    retries = _get_env_int("LLM_RETRIES", 2)  # total attempts = 1 + retries
    retries = max(0, min(5, retries))
    return timeout_s, retries


def _sleep_backoff(attempt: int) -> None:
    # attempt: 0,1,2... -> 0.8s, 1.6s, 3.2s ...
    time.sleep(min(6.0, 0.8 * (2**attempt)))


def _split_system_and_user(prompt: str) -> Tuple[str, str]:
    """
    prompting.py emits a "SYSTEM:" block at the top.
    I convert that into a real system message for better control.
    If missing, we fallback to putting everything as user content.
    """
    p = prompt or ""
    if "SYSTEM:" not in p:
        return "", p

    # Take everything after "SYSTEM:" until first blank line as system, rest as user.
    # If formatting changes, we still behave safely.
    
    parts = p.split("\n\n", 1)
    system_block = parts[0]
    user_block = parts[1] if len(parts) > 1 else ""

    system_text = system_block.replace("SYSTEM:", "", 1).strip()
    user_text = user_block.strip() if user_block else p
    return system_text, user_text


def call_llm(prompt: str, provider: str, model: str) -> str:
    """
    Unified LLM caller for RAG.

    Providers:
      - none       : deterministic fallback (no generation)
      - openai     : OpenAI official API
      - openrouter : OpenRouter (OpenAI-compatible)

    Style-aware length:
      - Infers style from prompt (USER_PREFS: ... style=Concise|Checklist|Detailed)
      - Sets max_tokens accordingly

    Always returns a string.
    Raises RuntimeError for configuration issues.
    """
    provider = (provider or "none").strip().lower()
    model = (model or "").strip()

    # --------------------------------------------------
    # 1) No-LLM fallback 
    # --------------------------------------------------
    if provider == "none":
        return (
            "RAG (no-LLM mode)\n"
            "-----------------\n"
            "Relevant maintenance documents were retrieved, but natural language "
            "generation is disabled.\n\n"
            "To enable generation:\n"
            "- Set LLM_PROVIDER=openrouter (recommended)\n"
            "- Set OPENROUTER_API_KEY\n\n"
            "Tip:\n"
            "- Ask a more specific question (component, symptom, fault, alarm, or cycle range)."
        )

    if not model:
        raise RuntimeError(f"LLM model is empty. Set LLM_MODEL for provider={provider}.")

    style = _infer_style_from_prompt(prompt)
    max_tokens = _max_tokens_for_style(style)
    temp = _temperature()
    timeout_s, retries = _timeouts_and_retries()

    system_text, user_text = _split_system_and_user(prompt)

    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})

    # --------------------------------------------------
    # OpenAI SDK import
    # --------------------------------------------------
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenAI-compatible SDK not installed. Install `openai>=1.0`.") from e

    # --------------------------------------------------
    # Provider client setup
    # --------------------------------------------------
    client_kwargs: Dict[str, object] = {}

    if provider == "openai":
        api_key = _require_env("OPENAI_API_KEY", provider)
        client_kwargs["api_key"] = api_key

    elif provider == "openrouter":
        api_key = _require_env("OPENROUTER_API_KEY", provider)
        base_url = _get_env_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        client_kwargs["api_key"] = api_key
        client_kwargs["base_url"] = base_url

        
        # If  SDK version doesn’t support default_headers, it will still work without them.
        app_name = _get_env_str("OPENROUTER_APP_NAME", "ai_predictive_maintenance_copilot")
        referrer = _get_env_str("OPENROUTER_REFERRER", "")
        headers: Dict[str, str] = {"X-Title": app_name}
        if referrer:
            headers["HTTP-Referer"] = referrer
        # Try attaching headers if supported
        try:
            client_kwargs["default_headers"] = headers  # type: ignore[assignment]
        except Exception:
            pass

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER='{provider}'")

    client = OpenAI(**client_kwargs)  # type: ignore[arg-type]

    # --------------------------------------------------
    # Request with small retry loop
    # --------------------------------------------------
    last_err: Optional[Exception] = None
    for attempt in range(0, 1 + retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
                # NOTE:  Intentionally kept it simple (no streaming) for server stability.
            )
            text = resp.choices[0].message.content or ""
            return text.strip()

        except Exception as e:
            last_err = e

            # Best-effort transient detection (429/5xx)
            # Different SDK versions expose status differently, so keep it defensive.
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            is_transient = False
            try:
                if status is not None and int(status) in (408, 409, 425, 429, 500, 502, 503, 504):
                    is_transient = True
            except Exception:
                is_transient = False

            if attempt < retries and is_transient:
                _sleep_backoff(attempt)
                continue

            # Non-transient or final attempt
            break

    # Final error message
    raise RuntimeError(
        f"{provider.upper()} request failed after {1 + retries} attempt(s). "
        f"style={style} max_tokens={max_tokens} timeout={timeout_s}s. "
        f"Error: {last_err}"
    ) from last_err
