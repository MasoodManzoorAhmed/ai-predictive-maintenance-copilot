# backend/api/rag/prompting.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .rag_config import RAGConfig
from .retriever import RetrievedChunk

_ALLOWED_STYLES = {"Checklist", "Concise", "Detailed"}
_DEFAULT_STYLE = "Checklist"
_DEFAULT_ROLE = "Maintenance Manager"


def _clip(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[TRUNCATED]"


def _norm(s: Any) -> str:
    return "" if s is None else str(s).strip()


def _sanitize_question(q: str) -> str:
    q = _norm(q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _normalize_style(style: Any) -> str:
    s = _norm(style)
    if not s:
        return _DEFAULT_STYLE

    # allow minor UI typos/casing
    s_low = s.lower()
    if "concise" in s_low:
        return "Concise"
    if "detail" in s_low:
        return "Detailed"
    if "check" in s_low:
        return "Checklist"

    # strict allow-list
    return s if s in _ALLOWED_STYLES else _DEFAULT_STYLE


def _normalize_role(role: Any) -> str:
    r = _norm(role)
    return r or _DEFAULT_ROLE


def _infer_component(question: str) -> str:
    """
    Lightweight component inference to tighten wording.
    DOES NOT add facts; it only narrows phrasing.
    """
    q = (question or "").lower()

    compressor_terms = [
        "compressor", "stall", "surge", "rotating stall", "surge line", "stall margin",
        "mass flow", "throttle", "pressure ratio", "bleed", "vsv", "igv", "variable stator",
    ]
    turbine_terms = [
        "turbine", "hot section", "blade creep", "thermal fatigue", "egt", "tgt", "hpt", "lpt",
    ]
    rotor_terms = [
        "vibration", "bearing", "imbalance", "misalignment", "shaft", "rub", "oil debris",
    ]
    fan_terms = ["fan", "inlet", "distortion", "fodd", "bird strike"]

    def any_term(terms: List[str]) -> bool:
        return any(t in q for t in terms)

    if any_term(compressor_terms):
        return "compressor"
    if any_term(turbine_terms):
        return "turbine"
    if any_term(rotor_terms):
        return "rotor/bearings"
    if any_term(fan_terms):
        return "fan/inlet"
    return "general"


def _role_rules(role: str) -> List[str]:
    r = (role or "").strip().lower()
    if "technician" in r:
        return [
            "- Write for a hands-on technician.",
            "- Prefer concrete inspection/measurement steps and safety-first actions.",
            "- Keep steps executable on the shop floor.",
        ]
    if "reliability" in r:
        return [
            "- Write for a reliability engineer.",
            "- Emphasize failure modes, leading indicators, monitoring, and controls (only if in evidence).",
            "- Separate supported conclusions vs assumptions.",
        ]
    # default: manager
    return [
        "- Write for a maintenance manager.",
        "- Emphasize prioritization, scheduling, risk level, and resource planning.",
        "- Keep actions decision-oriented and time-bounded where possible (only if supported).",
    ]


def _component_rules(component: str) -> List[str]:
    if component == "compressor":
        return [
            "- The question appears compressor-related (stall/surge/flow instability).",
            "- Prioritize compressor stability language ONLY if supported by evidence.",
            "- Prefer operational triggers like low mass flow / instability modes ONLY if present in snippets.",
        ]
    if component == "turbine":
        return [
            "- The question appears turbine/hot-section related.",
            "- Use thermal/efficiency/degradation language ONLY if supported by evidence.",
        ]
    if component == "rotor/bearings":
        return [
            "- The question appears rotor/bearing related.",
            "- Use vibration/tribology/imbalance language ONLY if supported by evidence.",
        ]
    if component == "fan/inlet":
        return [
            "- The question appears fan/inlet related.",
            "- Use inlet distortion / foreign object / flow issues ONLY if supported by evidence.",
        ]
    return ["- Keep the answer general and evidence-grounded."]


def _output_contract(style: str) -> List[str]:
    """
    Style-specific strict output formats.
    This is what makes the LLM actually respond differently.
    """
    if style == "Concise":
        return [
            "OUTPUT FORMAT (STRICT - CONCISE):",
            "A) Answer (max 3 sentences). Cite like [1], [2].",
            "B) Next actions (max 3 bullets, action-only). Cite where possible.",
            "C) Evidence limits (1–2 bullets): what is missing OR what cannot be concluded from snippets.",
        ]

    if style == "Detailed":
        return [
            "OUTPUT FORMAT (STRICT - DETAILED):",
            "1) Direct Answer (4–10 sentences). Cite like [1], [2].",
            "2) Why this is happening (2–6 bullets): explanation tied to evidence; no invented theory.",
            "3) Maintenance Plan (6–12 bullets): actions + checks + monitoring; cite where possible.",
            "4) Evidence & Limits:",
            "   - Supported by evidence (2–5 bullets with citations)",
            "   - Missing info/docs (1–6 bullets) if evidence is insufficient",
        ]

    # Default: Checklist
    return [
        "OUTPUT FORMAT (STRICT - CHECKLIST):",
        "1) Direct Answer (2–6 sentences). Cite like [1], [2].",
        "2) Maintenance Checklist (5–12 bullets). Each bullet MUST be an action. Cite where possible.",
        "3) Evidence & Limits:",
        "   - Supported evidence (1–3 bullets with citations)",
        "   - Missing info/docs (1–5 bullets) if evidence is insufficient",
    ]


def build_rag_prompt(
    question: str,
    fd_name: Optional[str],
    unit: Optional[int],
    extra: Dict[str, Any] | None,
    retrieved: List[RetrievedChunk],
) -> str:
    """
    Production RAG prompt with:
    - strong anti-hallucination rules
    - style + role conditioning (3 styles only)
    - component tightening
    - bounded evidence context
    """
    extra = extra or {}
    cfg = RAGConfig.from_env()

    q = _sanitize_question(question)
    dataset = _norm(fd_name).upper() if fd_name else "N/A"
    unit_s = str(unit) if unit is not None else "N/A"

    role = _normalize_role(extra.get("role"))
    style = _normalize_style(extra.get("style"))
    top_k = extra.get("top_k", None)

    component = _infer_component(q)

    # ---------------------------
    # SYSTEM / POLICY (STRICT)
    # ---------------------------
    header: List[str] = []
    header.append("SYSTEM:")
    header.append("You are an Industrial Predictive Maintenance Copilot for turbofan engine systems (NASA CMAPSS style).")
    header.append("You MUST follow these rules:")
    header.append("1) Use ONLY the EVIDENCE SNIPPETS provided. Treat them as the only source of truth.")
    header.append("2) Do NOT invent facts, numbers, procedures, page numbers, or citations.")
    header.append("3) If evidence is insufficient, explicitly say: 'Evidence insufficient' and list what is missing.")
    header.append("4) Every non-trivial technical claim MUST have at least one citation like [1], [2].")
    header.append("5) If a claim is not supported, mark it as: 'Not supported by evidence' (do not present as fact).")
    header.append("6) Keep the response maintenance-first and operational.")
    header.append("")
    header.append(f"CONTEXT: dataset={dataset} unit={unit_s}")
    header.append(f"USER_PREFS: role={role} style={style} top_k={top_k if top_k is not None else 'default'}")
    header.append(f"FOCUS_COMPONENT: {component}")
    header.append("")

    # ---------------------------
    # SCOPE RULES (component)
    # ---------------------------
    scope: List[str] = []
    scope.append("SCOPE RULES:")
    scope.extend(_component_rules(component))
    scope.append("")

    # ---------------------------
    # WRITING RULES (role)
    # ---------------------------
    writing: List[str] = []
    writing.append("WRITING RULES:")
    writing.extend(_role_rules(role))
    writing.append("- Avoid textbook filler. Tie explanations to actions/implications.")
    writing.append("")

    # ---------------------------
    # EVIDENCE BLOCK
    # ---------------------------
    evidence_lines: List[str] = ["EVIDENCE SNIPPETS:"]
    if not retrieved:
        evidence_lines.append("(none)")
    else:
        for j, r in enumerate(retrieved, start=1):
            c = r.chunk
            source = getattr(c, "source", "unknown")
            page = getattr(c, "page", None)
            text = (getattr(c, "text", "") or "").strip()

            evidence_lines.append(f"[{j}] source={source} page={page} score={r.score:.4f}")
            evidence_lines.append(text)
            evidence_lines.append("")

    evidence_text = _clip("\n".join(evidence_lines), cfg.max_context_chars)

    # ---------------------------
    # OUTPUT CONTRACT (style-specific)
    # ---------------------------
    contract: List[str] = []
    contract.extend(_output_contract(style))
    contract.append("")
    contract.append("CITATION RULES:")
    contract.append("- Only cite snippet indices that exist, e.g., [1], [2].")
    contract.append("- Do not cite document names or pages unless they appear in the snippet metadata above.")
    contract.append("")
    contract.append(f"QUESTION: {q}")

    return (
        "\n".join(header)
        + "\n"
        + "\n".join(scope)
        + "\n".join(writing)
        + "\n"
        + evidence_text
        + "\n"
        + "\n".join(contract)
    )
