"""
FallbackAgent: Handles unsupported requests, low confidence, no-match situations,
and recoverable errors. Asks ONE clarifying question or offers alternatives.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.schemas import FALLBACK_SYSTEM_PROMPT
from ..core.state import AgentState, Intent
from ..tools.df_tools import list_unique

logger = logging.getLogger(__name__)


def fallback_node(state: AgentState, llm: Any) -> dict:
    """
    Generate a helpful fallback response.
    - For unsupported intents: explain limitation + pivot to what's available.
    - For no-match: show top candidates.
    - For errors: explain and suggest.
    - Ask at most ONE clarifying question.
    """
    trace = list(state.get("node_trace", []))
    trace.append("fallback")

    intent = state.get("intent", "")
    user_query = state.get("user_query", "")

    # Debug: always log why fallback was triggered
    logger.warning(
        "FALLBACK triggered | intent=%s | confidence=%s | error=%s | no_match=%s | node_trace=%s",
        intent,
        state.get("confidence"),
        state.get("error"),
        state.get("no_match"),
        state.get("node_trace", []),
    )
    error = state.get("error")
    candidates = state.get("candidates", [])
    no_match = state.get("no_match", False)

    # Build context for the LLM
    context_parts = [f"User query: {user_query}"]

    if intent == Intent.UNSUPPORTED:
        context_parts.append("Reason: This type of question (valuations/prices/addresses) is not in scope.")

    if error:
        context_parts.append(f"Technical error: {error}")

    if no_match and candidates:
        cand_str = ", ".join(
            f"{c['value']} ({c['field']})" for c in candidates[:3]
        )
        context_parts.append(f"No exact match. Similar entries: {cand_str}")
    elif no_match:
        context_parts.append(
            "No data rows matched the filters. The dataset covers 2024-2025 (through Q1 2025)."
        )

    if state.get("missing_fields"):
        context_parts.append(
            f"Still need: {', '.join(state['missing_fields'])}"
        )

    context = "\n".join(context_parts)

    # Try LLM-generated response
    answer = _llm_fallback(context, llm)

    # Supplement with available data summary if helpful
    if no_match and not candidates:
        answer += _available_data_hint()

    return {
        "node_trace": trace,
        "final_answer": answer,
        "steps_performed": [
            "Request routed to FallbackAgent.",
            f"Reason: {_classify_reason(intent, error, no_match)}",
        ],
        "assumptions": [],
    }


def _llm_fallback(context: str, llm: Any) -> str:
    """Use LLM to produce a friendly fallback response."""
    try:
        response = llm.invoke([
            SystemMessage(content=FALLBACK_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ])
        return response.content.strip()
    except Exception as exc:
        logger.warning("FallbackAgent LLM failed: %s", exc)
        return _static_fallback()


def _classify_reason(intent: str, error: str | None, no_match: bool) -> str:
    if intent == Intent.UNSUPPORTED:
        return "Unsupported intent (valuations/prices/addresses not in dataset)."
    if no_match:
        return "No matching data rows after applying filters."
    if error:
        return f"Technical error: {error[:80]}"
    return "Low confidence or ambiguous request."


def _available_data_hint() -> str:
    """Append a hint about what data IS available."""
    try:
        properties = list_unique("property_name")
        tenants = list_unique("tenant_name")
        return (
            f"\n\n**Available data:** {len(properties)} properties "
            f"({', '.join(properties[:4])}{'...' if len(properties) > 4 else ''}), "
            f"{len(tenants)} tenants, years 2024–2025 (through Q1 2025)."
        )
    except Exception:
        return ""


def _static_fallback() -> str:
    """Static fallback when LLM is unavailable."""
    return (
        "I'm unable to answer that question with the available data. "
        "I can help with: total portfolio profit, profit by property or tenant, "
        "breakdowns by ledger type/group, comparisons between properties/tenants, "
        "and top/bottom rankings. Would you like to try one of these?"
    )
