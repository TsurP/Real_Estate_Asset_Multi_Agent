"""
LangGraph multi-agent graph definition.

Graph flow:
    START
      └─► router_node
            ├─► respond_node     (general_question)
            ├─► fallback_node    (unsupported / low confidence)
            └─► extract_node
                  ├─► [interrupt] ClarificationInterrupt → resumed → extract_node continues
                  └─► retrieve_node
                        ├─► fallback_node  (no_match)
                        ├─► [interrupt] DisambiguationInterrupt → resumed → retrieve continues
                        └─► compute_node
                              ├─► fallback_node  (compute error)
                              └─► respond_node

Global error edge: any node exception → fallback_node
"""
from __future__ import annotations

import os
import functools
import logging
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .core.state import AgentState, Intent, initial_state
from .agents.router import router_node
from .agents.extract import extract_node
from .agents.retrieve import retrieve_node
from .agents.compute import compute_node
from .agents.respond import respond_node
from .agents.fallback import fallback_node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm() -> Any:
    """
    Create the LLM client based on environment variables.
    Priority: ANTHROPIC_API_KEY → OPENAI_API_KEY → raise.
    Model configurable via LLM_MODEL env var.
    """
    provider = os.getenv("LLM_PROVIDER", "").lower()

    # Auto-detect if provider not set
    if not provider:
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise EnvironmentError(
                "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
            )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
        return ChatAnthropic(model=model, temperature=0)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


# ---------------------------------------------------------------------------
# Node wrappers (bind LLM + handle global errors)
# ---------------------------------------------------------------------------

def _wrap_with_fallback(node_fn, llm_required: bool, llm: Any):
    """Wrap a node function; catch exceptions and route to error state."""
    @functools.wraps(node_fn)
    def wrapper(state: AgentState) -> dict:
        try:
            if llm_required:
                return node_fn(state, llm)
            return node_fn(state)
        except Exception as exc:
            # Don't swallow interrupts — re-raise them so LangGraph can handle them
            from langgraph.errors import GraphInterrupt
            if isinstance(exc, GraphInterrupt):
                raise
            logger.error("Node %s crashed: %s", node_fn.__name__, exc, exc_info=True)
            return {
                "error": f"{node_fn.__name__} failed: {exc}",
                "node_trace": list(state.get("node_trace", [])) + [f"{node_fn.__name__}:ERROR"],
            }
    return wrapper


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def _after_router(state: AgentState) -> str:
    """Route after RouterAgent based on intent and confidence."""
    intent = state.get("intent", Intent.GENERAL_QUESTION)
    confidence = state.get("confidence", 0.0)
    error = state.get("error")

    if error and intent not in Intent.DATA_INTENTS:
        logger.warning("_after_router → fallback (error + non-data-intent: %s)", error)
        return "fallback"

    if intent == Intent.UNSUPPORTED:
        logger.warning("_after_router → fallback (unsupported intent)")
        return "fallback"

    if intent == Intent.GENERAL_QUESTION:
        return "respond"

    if confidence < 0.4:
        logger.warning("_after_router → fallback (low confidence: %s, intent: %s)", confidence, intent)
        return "fallback"

    return "extract"


def _after_extract(state: AgentState) -> str:
    """Route after ExtractionAgent."""
    error = state.get("error")
    if error:
        return "fallback"
    # missing_fields will be empty if we passed through (interrupt already handled)
    return "retrieve"


def _after_retrieve(state: AgentState) -> str:
    """Route after RetrievalAgent."""
    error = state.get("error")
    no_match = state.get("no_match", False)

    if error or no_match:
        logger.warning("_after_retrieve → fallback (error=%s, no_match=%s)", error, no_match)
        return "fallback"

    return "compute"


def _after_compute(state: AgentState) -> str:
    """Route after ComputationAgent."""
    error = state.get("error")
    result = state.get("computation_result", {})

    if error or (isinstance(result, dict) and "error" in result):
        logger.warning("_after_compute → fallback (error=%s, result_error=%s)",
                       error, (result or {}).get("error"))
        return "fallback"

    return "respond"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(checkpointer: MemorySaver | None = None) -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Args:
        checkpointer: Optional MemorySaver for persistent conversation state
                      (required for interrupt/resume support).

    Returns:
        Compiled LangGraph application.
    """
    llm = _build_llm()

    # Create node functions with LLM bound
    _router = _wrap_with_fallback(router_node, llm_required=True, llm=llm)
    _extract = _wrap_with_fallback(extract_node, llm_required=True, llm=llm)
    _retrieve = _wrap_with_fallback(retrieve_node, llm_required=False, llm=None)
    _compute = _wrap_with_fallback(compute_node, llm_required=False, llm=None)
    _respond = _wrap_with_fallback(respond_node, llm_required=True, llm=llm)
    _fallback = _wrap_with_fallback(fallback_node, llm_required=True, llm=llm)

    # Build graph
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("router", _router)
    builder.add_node("extract", _extract)
    builder.add_node("retrieve", _retrieve)
    builder.add_node("compute", _compute)
    builder.add_node("respond", _respond)
    builder.add_node("fallback", _fallback)

    # Add edges
    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        _after_router,
        {
            "extract": "extract",
            "respond": "respond",
            "fallback": "fallback",
        },
    )

    builder.add_conditional_edges(
        "extract",
        _after_extract,
        {
            "retrieve": "retrieve",
            "fallback": "fallback",
        },
    )

    builder.add_conditional_edges(
        "retrieve",
        _after_retrieve,
        {
            "compute": "compute",
            "fallback": "fallback",
        },
    )

    builder.add_conditional_edges(
        "compute",
        _after_compute,
        {
            "respond": "respond",
            "fallback": "fallback",
        },
    )

    builder.add_edge("respond", END)
    builder.add_edge("fallback", END)

    # Compile with interrupt support
    interrupt_before: list[str] = []  # interrupts are inline via interrupt()
    compile_kwargs: dict = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    return builder.compile(**compile_kwargs)


# ---------------------------------------------------------------------------
# Convenience: run a single query (no streaming, no checkpointer)
# ---------------------------------------------------------------------------

def run_query(query: str, graph: Any | None = None) -> AgentState:
    """
    Run a single query through the graph synchronously.
    Returns final state. Suitable for testing.
    """
    if graph is None:
        graph = build_graph()

    state = initial_state(query)
    result = graph.invoke(state)
    return result
