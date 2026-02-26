"""
RouterAgent: Classify user intent and required fields.
Uses LLM with structured output; validates and recovers from malformed JSON.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.schemas import ROUTER_SYSTEM_PROMPT, RouterOutput
from ..core.state import AgentState, Intent

logger = logging.getLogger(__name__)


def router_node(state: AgentState, llm: Any) -> dict:
    """
    Classify the user's query into one of the supported intents.

    Returns partial state update with: intent, confidence, router_notes, node_trace.
    """
    trace = list(state.get("node_trace", []))
    trace.append("router")

    user_query = state.get("user_query", "")
    conv_ctx = state.get("conversation_context") or ""

    # Build the human message: include recent conversation so the router can
    # resolve pronouns ("he", "there", "that") before classifying intent.
    if conv_ctx:
        human_content = (
            f"Recent conversation:\n{conv_ctx}\n"
            f"---\n"
            f"New user query: {user_query}"
        )
    else:
        human_content = f"User query: {user_query}"

    try:
        structured_llm = llm.with_structured_output(RouterOutput)
        result: RouterOutput = structured_llm.invoke([
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ])
        intent = result.intent
        confidence = result.confidence
        notes = result.notes

        # Deterministic overrides (applied in priority order).
        if _is_ledger_reconciliation_query(user_query):
            intent = Intent.BREAKDOWN
            confidence = max(confidence, 0.95)
            notes = (notes or "") + " | deterministic override: ledger_reconciliation→breakdown"
        elif _is_tenant_listing_query(user_query):
            intent = Intent.COUNT
            confidence = max(confidence, 0.9)
            notes = (notes or "") + " | deterministic override: tenant_listing→count"
        elif _is_grouped_breakdown_query(user_query):
            intent = Intent.BREAKDOWN
            confidence = max(confidence, 0.85)
            notes = (notes or "") + " | deterministic override: grouped_metric→breakdown"
        elif _is_count_query(user_query):
            intent = Intent.COUNT
            confidence = max(confidence, 0.9)
            notes = (notes or "") + " | deterministic override: count"
        elif _is_margin_data_query(user_query):
            intent = Intent.MARGIN_ANALYSIS
            confidence = max(confidence, 0.8)
            notes = (notes or "") + " | deterministic override: margin_analysis"
        elif _is_time_bucket_extreme_query(user_query):
            intent = Intent.RANKING
            confidence = max(confidence, 0.8)
            notes = (notes or "") + " | deterministic override: time_bucket_ranking"

        return {
            "intent": intent,
            "confidence": confidence,
            "router_notes": notes,
            "node_trace": trace,
        }

    except Exception as exc:
        logger.warning("RouterAgent structured output failed: %s. Trying fallback.", exc)
        return _fallback_route(user_query, trace, str(exc))


def _fallback_route(query: str, trace: list[str], error_note: str) -> dict:
    """
    Heuristic fallback when LLM structured output fails.
    Looks for keywords to guess intent.
    """
    q = query.lower()

    intent = Intent.GENERAL_QUESTION
    confidence = 0.5

    # Only fire unsupported for explicit valuation/address keywords — NOT for
    # location pronouns like "where does he/she work/live/pay" which in context
    # of a tenant query means "which building".
    pronoun_location = any(w in q for w in ("where does", "where is", "where do", "where did"))
    has_unsupported_kw = any(w in q for w in ("price", "value", "worth", "appraisal", "valuation", "address"))

    if _is_ledger_reconciliation_query(query):
        intent = Intent.BREAKDOWN
        confidence = 0.95
    elif has_unsupported_kw and not pronoun_location:
        intent = Intent.UNSUPPORTED
        confidence = 0.9
    elif _is_tenant_listing_query(query):
        intent = Intent.COUNT
        confidence = 0.9
    elif _is_count_query(query):
        intent = Intent.COUNT
        confidence = 0.9
    elif _is_margin_data_query(query):
        intent = Intent.MARGIN_ANALYSIS
        confidence = 0.8
    elif _is_time_bucket_extreme_query(query):
        intent = Intent.RANKING
        confidence = 0.8
    elif pronoun_location:
        # "Where is/does X" in a real-estate P&L context → which property they're in
        intent = Intent.PNL_BY_PROPERTY
        confidence = 0.65
    elif any(w in q for w in ("top", "bottom", "best", "worst", "ranking", "rank")):
        intent = Intent.RANKING
        confidence = 0.7
    elif re.search(r"\bfrom\s+q[1-4]\b|\bfrom\s+20\d{2}\b|\bq[1-4]\s+to\s+q[1-4]\b|\b20\d{2}\s+to\s+20\d{2}\b", q):
        # "from Q1 to Q4", "from 2024 to 2025" → period-over-period comparison
        intent = Intent.PERIOD_COMPARISON
        confidence = 0.75
    elif any(w in q for w in ("increase", "decrease", "growth", "grew", "change", "trend")):
        # Likely period comparison if time context present
        if re.search(r"\bq[1-4]\b|\b20\d{2}\b", q):
            intent = Intent.PERIOD_COMPARISON
            confidence = 0.7
        else:
            intent = Intent.GENERAL_QUESTION
            confidence = 0.5
    elif any(w in q for w in ("what percentage", "what %", "what percent", "what share",
                                "what proportion", "what fraction",
                                "percentage of", "percent of", "share of", "proportion of",
                                "how much of", "what part of")):
        intent = Intent.SHARE_OF_TOTAL
        confidence = 0.8
    elif any(w in q for w in ("compare", "vs", "versus", "difference between")):
        intent = Intent.COMPARISON
        confidence = 0.7
    elif _is_grouped_breakdown_query(query):
        intent = Intent.BREAKDOWN
        confidence = 0.85
    elif any(w in q for w in ("breakdown", "break down", "by ledger", "by type", "by group")):
        intent = Intent.BREAKDOWN
        confidence = 0.7
    elif any(w in q for w in ("by property", "per property", "each building")):
        intent = Intent.PNL_BY_PROPERTY
        confidence = 0.7
    elif any(w in q for w in ("by tenant", "per tenant", "each tenant")):
        intent = Intent.PNL_BY_TENANT
        confidence = 0.7
    elif any(w in q for w in ("review", "full review", "overview", "tell me about",
                                "give me everything", "summarise", "summarize", "report on",
                                "all about", "what can you tell me about")):
        intent = Intent.FULL_REVIEW
        confidence = 0.7
    elif any(w in q for w in ("total", "overall", "profit", "pnl", "p&l", "portfolio")):
        intent = Intent.PNL_TOTAL
        confidence = 0.7

    return {
        "intent": intent,
        "confidence": confidence,
        "router_notes": f"Heuristic fallback (LLM error: {error_note[:80]})",
        "node_trace": trace,
        "error": f"Router LLM error; used heuristic: {error_note[:120]}",
    }


def _is_margin_data_query(query: str) -> bool:
    """
    Detect net/profit margin or expense ratio queries intended for portfolio data analysis.
    Avoid routing pure definition questions ("what is net profit margin?").
    """
    q = (query or "").lower().strip()
    has_ratio_keyword = bool(
        "margin" in q
        or re.search(r"\bexpense[_\s]?ratio\b|\bcost[_\s]?ratio\b", q)
    )
    if not has_ratio_keyword:
        return False

    # Definition-style questions should remain general questions unless data cues are present.
    is_definition = bool(re.match(r"^(what is|define|meaning of|explain)\b", q))
    has_data_cue = bool(
        re.search(r"\b(property|building|asset|tenant|portfolio|entity)\b", q)
        or re.search(r"\b20\d{2}\b|\bq[1-4]\b|\bthis year\b|\blast year\b", q)
        or re.search(r"\b(top|bottom|best|worst|highest|lowest|most|least|rank)\b", q)
        or re.search(r"\bby\s+(property|building|tenant|asset)\b", q)
        or re.search(r"\babove[- ]average\b|\bbelow[- ]average\b", q)
    )

    if is_definition and not has_data_cue:
        return False
    return has_data_cue


def _is_time_bucket_extreme_query(query: str) -> bool:
    """
    Detect questions like:
    - "Which month in 2024 had the lowest total net profit?"
    - "What quarter had the highest profit in 2025?"
    These should be ranking/group queries, not period_comparison.
    """
    q = (query or "").lower().strip()
    if not q:
        return False

    has_bucket = bool(re.search(r"\b(month|quarter|year)\b", q))
    has_superlative = bool(re.search(r"\b(lowest|highest|least|most|worst|best|top|bottom)\b", q))
    has_financial_metric = bool(re.search(
        r"\b(profit|net profit|pnl|p&l|expense|expenses|revenue|income|cost|costs)\b", q
    ))

    # True period-comparison cues ("from X to Y", "between A and B") should not be overridden.
    has_period_comparison_cue = bool(
        re.search(r"\bfrom\b.+\bto\b", q)
        or re.search(r"\bbetween\b.+\band\b", q)
        or re.search(r"\b(?:increase|decrease|change|growth|grew)\b", q)
    )

    return has_bucket and has_superlative and has_financial_metric and not has_period_comparison_cue


def _is_grouped_breakdown_query(query: str) -> bool:
    """
    Detect queries that name a specific ledger metric AND a grouping dimension.
    These should route to BREAKDOWN even if the user says 'review' or 'overview',
    because compute_full_review ignores both group_by and ledger_filters entirely.

    Examples that match:
      "review of total expenses for each quarter in 2024"
      "show me revenue per property"
      "overview of expenses by month in Q1"

    Examples that do NOT match (true full_review):
      "give me a full review of Building 120"
      "overview of PropCo"
    """
    q = (query or "").lower().strip()
    has_specific_metric = bool(re.search(
        r"\b(expenses?|costs?|revenue|income|rental\s+income|management\s+fees?)\b", q
    ))
    has_group_dim = bool(re.search(
        r"\b(by|per|each|for\s+each)\s+(quarter|month|year|property|building|tenant)\b"
        r"|\bfor\s+each\s+(quarter|month|year)\b"
        r"|\beach\s+(quarter|month|year)\b",
        q,
    ))
    return has_specific_metric and has_group_dim


def _is_tenant_listing_query(query: str) -> bool:
    """
    Detect queries asking which tenants/occupants are in a specific building.
    Examples: "Who lives in Building 120?", "Which tenants are in Building 140?",
              "Who are the tenants in Building 17?", "Who rents Building 120?"
    These should route to COUNT (listing tenants), not pnl_by_property.
    """
    q = (query or "").lower().strip()
    return bool(re.search(
        r"\b("
        r"who\s+(lives?|is|are|rents?|pays?\s+rent|occupies?)\s+in"
        r"|which\s+tenants?\s+(are\s+)?in"
        r"|what\s+tenants?\s+(are\s+)?in"
        r"|who\s+are\s+the\s+tenants?\s+(in|of|at)"
        r"|tenants?\s+in\s+(building|property)\b"
        r")\b",
        q,
    ))


def _is_ledger_reconciliation_query(query: str) -> bool:
    """
    Detect queries asking whether ledger_category totals reconcile with a ledger_group total.
    Example: "Does the sum of all ledger_category revenue totals equal the total from
              ledger_group = rental_income in 2024?"
    These require the full data pipeline (retrieve→compute) and must not be routed to
    general_question.
    """
    q = (query or "").lower()
    has_comparison = bool(
        re.search(r"\bdoes\b.{0,80}\bequal\b|\bdo\b.{0,40}\bmatch\b|\breconcil\w*\b", q)
    )
    has_category = bool(re.search(r"\bledger[_\s-]?categor(?:y|ies)\b", q))
    has_group = bool(re.search(r"ledger[_\s-]?group", q))
    return has_comparison and has_category and has_group


def _is_count_query(query: str) -> bool:
    """
    Detect "how many X" queries that ask for a count of distinct entities.
    Only fires when the subject is a countable portfolio entity, not a financial metric.
    """
    q = (query or "").lower().strip()
    if not re.search(r"\bhow\s+many\b", q):
        return False
    # Must ask about a countable entity type, not a financial figure
    has_entity = bool(re.search(
        r"\b(tenant|tenants|property|properties|building|buildings|asset|assets|entity|entities)\b", q
    ))
    return has_entity
