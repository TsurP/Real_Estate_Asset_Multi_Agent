"""
Shared state definition for the LangGraph multi-agent system.
Uses TypedDict with LangGraph-compatible annotations.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Intent constants (string enum to keep JSON-compatible)
# ---------------------------------------------------------------------------
class Intent:
    PNL_TOTAL = "pnl_total"
    PNL_BY_PROPERTY = "pnl_by_property"
    PNL_BY_TENANT = "pnl_by_tenant"
    BREAKDOWN = "breakdown"
    COMPARISON = "comparison"
    PERIOD_COMPARISON = "period_comparison"
    SHARE_OF_TOTAL = "share_of_total"
    RANKING = "ranking"
    MARGIN_ANALYSIS = "margin_analysis"
    FULL_REVIEW = "full_review"
    COUNT = "count"
    GENERAL_QUESTION = "general_question"
    UNSUPPORTED = "unsupported"

    ALL = [
        PNL_TOTAL, PNL_BY_PROPERTY, PNL_BY_TENANT, BREAKDOWN,
        COMPARISON, PERIOD_COMPARISON, SHARE_OF_TOTAL, RANKING, MARGIN_ANALYSIS,
        FULL_REVIEW, COUNT, GENERAL_QUESTION, UNSUPPORTED,
    ]

    # Intents that require data retrieval
    DATA_INTENTS = {
        PNL_TOTAL, PNL_BY_PROPERTY, PNL_BY_TENANT,
        BREAKDOWN, COMPARISON, PERIOD_COMPARISON, SHARE_OF_TOTAL, RANKING, MARGIN_ANALYSIS,
        FULL_REVIEW, COUNT,
    }


# ---------------------------------------------------------------------------
# Timeframe structure
# ---------------------------------------------------------------------------
class Timeframe(TypedDict, total=False):
    year: Optional[str]          # e.g. "2024"
    quarter: Optional[str]       # e.g. "2024-Q3"
    month: Optional[str]         # e.g. "2024-M06"
    ytd: Optional[bool]          # True if "YTD" / "this year"
    label: Optional[str]         # human-readable, e.g. "Q3 2024"


# ---------------------------------------------------------------------------
# Main agent state
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    # ---- Conversation ----
    messages: Annotated[list, add_messages]   # full message history
    user_query: str                            # current raw user text

    # ---- Router output ----
    intent: Optional[str]                      # one of Intent.*
    confidence: float                          # 0.0 – 1.0
    router_notes: Optional[str]

    # ---- Extraction output ----
    targets: list[str]            # resolved target names (1-2)
    target_fields: list[str]      # field keys: property_name / tenant_name / entity_name
    timeframe: Optional[dict]     # Timeframe dict (used for retrieval filter)
    timeframe_a: Optional[dict]   # Period A for period_comparison intent
    timeframe_b: Optional[dict]   # Period B for period_comparison intent
    group_by: Optional[str]       # e.g. "ledger_type", "property_name"
    query_condition: Optional[str]  # optional boolean/set condition (e.g., expenses_without_revenue)
    ledger_filters: dict          # {ledger_type, ledger_group, ledger_category, ledger_code, description_keyword}
    k: int                        # for ranking
    rank_direction: str           # "top" | "bottom"
    missing_fields: list[str]     # fields the user needs to supply
    ratio_type: Optional[str]     # "profit_margin" | "expense_ratio" (for margin_analysis)
    margin_pre_filter: Optional[str]  # "above_average_revenue" | "below_average_revenue"

    # ---- Retrieval output ----
    retrieved_rows: Optional[int]           # row count after filter
    candidates: list[dict]                  # [{field, value, score}] for disambiguation
    ambiguous: bool                         # True if multiple matches above threshold
    no_match: bool                          # True if nothing found

    # ---- Computation output ----
    computation_result: Optional[dict]      # structured result from compute agent

    # ---- Response ----
    final_answer: Optional[str]
    steps_performed: list[str]
    assumptions: list[str]

    # ---- Control / debug ----
    node_trace: list[str]
    error: Optional[str]
    clarification_question: Optional[str]

    # ---- Serialised dataframe (JSON records) ----
    # Stored as list[dict] to survive LangGraph serialisation
    retrieved_records: Optional[list[dict]]

    # ---- Multi-turn conversation context ----
    # Last 1-2 Q&A exchanges injected by the UI so pronouns ("he", "it", "that")
    # can be resolved by the extraction agent.
    conversation_context: Optional[str]


# ---------------------------------------------------------------------------
# Default factory for a clean initial state
# ---------------------------------------------------------------------------
def initial_state(user_query: str) -> AgentState:
    return AgentState(
        messages=[],
        user_query=user_query,
        intent=None,
        confidence=0.0,
        router_notes=None,
        targets=[],
        target_fields=[],
        timeframe=None,
        timeframe_a=None,
        timeframe_b=None,
        group_by=None,
        query_condition=None,
        ledger_filters={},
        k=5,
        rank_direction="top",
        missing_fields=[],
        ratio_type=None,
        margin_pre_filter=None,
        retrieved_rows=None,
        candidates=[],
        ambiguous=False,
        no_match=False,
        computation_result=None,
        final_answer=None,
        steps_performed=[],
        assumptions=[],
        node_trace=[],
        error=None,
        clarification_question=None,
        retrieved_records=None,
        conversation_context=None,
    )
