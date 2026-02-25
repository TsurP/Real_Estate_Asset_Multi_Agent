"""
Streamlit chat UI for the Real Estate Portfolio Assistant.

Features:
- Multi-turn conversation with interrupt/resume support
- Debug panel showing intent, entities, graph trace, row count
- Sample prompt buttons (12 prompts)
- Persistent thread per session
"""
from __future__ import annotations

import os
import sys
import uuid
import warnings
from pathlib import Path
from typing import Optional

import streamlit as st

warnings.filterwarnings(
    "ignore",
    message=r"(?s)Pydantic serializer warnings:.*field_name='parsed'.*",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Path setup — allow running as `streamlit run src/app.py` from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Portfolio Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports after path setup
# ---------------------------------------------------------------------------
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.tools.df_tools import load_dataframe, list_unique
from src.graph import build_graph
from src.core.state import initial_state

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PARQUET_PATH = str(PROJECT_ROOT / "cortex (2) (1) (2) (1) (1) (3).parquet")

SAMPLE_PROMPTS = [
    "What is the total profit for 2024?",
    "Show me profit by property for Q1 2025",
    "Which tenant generated the most revenue in 2024?",
    "Compare Building 140 vs Building 160 for 2024",
    "Top 5 properties by profit in 2024",
    "Bottom 3 tenants by profit last year",
    "Break down expenses by ledger type for Q4 2024",
    "What was the rental income for Building 17 in 2024?",
    "Show profit breakdown by ledger group for 2024",
    "Compare revenue vs expenses for Building 180 in 2024",
    "What is YTD profit for 2025?",
    "What does 'ledger_group' mean in real-estate accounting?",
]


# ---------------------------------------------------------------------------
# Initialise session state
# ---------------------------------------------------------------------------

def _init_session():
    if "graph" not in st.session_state:
        with st.spinner("Loading data and building agent graph…"):
            try:
                load_dataframe(PARQUET_PATH)
                checkpointer = MemorySaver()
                st.session_state.graph = build_graph(checkpointer=checkpointer)
                st.session_state.checkpointer = checkpointer
            except Exception as e:
                st.error(f"Failed to initialise: {e}")
                st.stop()

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{role, content, debug}]

    if "awaiting_clarification" not in st.session_state:
        st.session_state.awaiting_clarification = False

    if "last_debug" not in st.session_state:
        st.session_state.last_debug = {}

    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False


# ---------------------------------------------------------------------------
# Graph interaction helpers
# ---------------------------------------------------------------------------

def _get_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _is_interrupted(thread_id: str) -> tuple[bool, Optional[str]]:
    """Check if the graph is paused at an interrupt. Returns (interrupted, question)."""
    try:
        config = _get_config(thread_id)
        state_snap = st.session_state.graph.get_state(config)
        if state_snap and state_snap.tasks:
            for task in state_snap.tasks:
                for intr in task.interrupts:
                    return True, str(intr.value)
    except Exception:
        pass
    return False, None


def _build_conversation_context() -> Optional[str]:
    """
    Build a short conversation-context string from the last 2 Q&A pairs so the
    extraction agent can resolve pronouns ("he", "it", "that building", etc.).
    """
    msgs = st.session_state.messages
    if not msgs:
        return None
    # Take the last 4 messages (2 exchanges) excluding the current user turn
    recent = msgs[-4:] if len(msgs) >= 4 else msgs[:]
    lines = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        # Truncate long assistant answers to keep context compact
        content = m["content"][:300] + "…" if len(m["content"]) > 300 else m["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else None


def _invoke_new_query(query: str, thread_id: str) -> dict:
    """Start a new query on a fresh thread, injecting recent conversation context."""
    config = _get_config(thread_id)
    state = initial_state(query)
    state["conversation_context"] = _build_conversation_context()
    try:
        final = st.session_state.graph.invoke(state, config)
        return final
    except Exception as exc:
        return {"final_answer": f"Error: {exc}", "node_trace": [], "error": str(exc)}


def _resume_with_answer(answer: str, thread_id: str) -> dict:
    """Resume an interrupted graph with the user's clarification."""
    config = _get_config(thread_id)
    try:
        final = st.session_state.graph.invoke(Command(resume=answer), config)
        return final
    except Exception as exc:
        return {"final_answer": f"Error during resume: {exc}", "node_trace": [], "error": str(exc)}


def _extract_debug(state: dict) -> dict:
    """Extract debug-relevant fields from final state."""
    return {
        "intent": state.get("intent", "—"),
        "confidence": f"{state.get('confidence', 0):.0%}",
        "targets": state.get("targets", []),
        "target_fields": state.get("target_fields", []),
        "timeframe": state.get("timeframe") or "—",
        "group_by": state.get("group_by") or "—",
        "ledger_filters": state.get("ledger_filters") or "—",
        "row_count": state.get("retrieved_rows", "—"),
        "node_trace": state.get("node_trace", []),
        "assumptions": state.get("assumptions", []),
        "computation_type": (state.get("computation_result") or {}).get("type", "—"),
    }


# ---------------------------------------------------------------------------
# UI: Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar():
    with st.sidebar:
        st.title("🏢 Portfolio Assistant")
        st.markdown("---")

        st.subheader("Sample Prompts")
        for prompt in SAMPLE_PROMPTS:
            if st.button(prompt, key=f"sp_{prompt[:20]}", use_container_width=True):
                st.session_state.pending_prompt = prompt

        st.markdown("---")
        st.checkbox("Show debug panel", key="show_debug")

        st.markdown("---")
        if st.button("🔄 New conversation", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.awaiting_clarification = False
            st.session_state.last_debug = {}
            st.rerun()

        st.markdown("---")
        st.caption(
            "Dataset: PropCo portfolio · 2024–2025 · "
            "5 properties · 18 tenants\n\n"
            "⚠ No valuations/prices in this dataset."
        )


# ---------------------------------------------------------------------------
# UI: Debug panel
# ---------------------------------------------------------------------------

def _render_debug(debug: dict):
    if not debug:
        return

    with st.expander("🔍 Debug Panel", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Intent & Confidence**")
            st.code(f"{debug.get('intent', '—')}  ({debug.get('confidence', '—')})")

            st.markdown("**Targets**")
            targets = debug.get("targets", [])
            fields = debug.get("target_fields", [])
            if targets:
                for t, f in zip(targets, fields):
                    st.code(f"{t} ({f})")
            else:
                st.caption("—")

            st.markdown("**Timeframe**")
            tf = debug.get("timeframe", "—")
            st.code(str(tf))

        with col2:
            st.markdown("**Graph Node Trace**")
            trace = debug.get("node_trace", [])
            if trace:
                st.code(" → ".join(trace))
            else:
                st.caption("—")

            st.markdown("**Rows used**")
            st.code(str(debug.get("row_count", "—")))

            st.markdown("**Ledger filters**")
            st.code(str(debug.get("ledger_filters", "—")))

        if debug.get("assumptions"):
            st.markdown("**Assumptions**")
            for a in debug["assumptions"]:
                st.caption(f"• {a}")


# ---------------------------------------------------------------------------
# UI: Chat area
# ---------------------------------------------------------------------------

def _render_chat():
    st.title("Real Estate Portfolio Assistant")
    st.caption(
        "Ask about portfolio P&L, property performance, tenant revenue, breakdowns, and comparisons. "
        "**Note:** Property valuations and market prices are not in this dataset."
    )

    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("steps"):
                with st.expander("How I computed this"):
                    for step in msg["steps"]:
                        st.markdown(f"• {step}")

    # Show debug panel for last response
    if st.session_state.show_debug and st.session_state.last_debug:
        _render_debug(st.session_state.last_debug)

    # Clarification hint
    if st.session_state.awaiting_clarification:
        st.info("Awaiting your clarification — please type your answer below.")

    # Handle pre-set prompt from sidebar button
    pending = st.session_state.pop("pending_prompt", None)

    user_input = st.chat_input("Ask about your portfolio…") or pending

    if user_input:
        _handle_user_input(user_input)


def _handle_user_input(user_input: str):
    """Process user input: new query or clarification answer."""
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            if st.session_state.awaiting_clarification:
                # Resume the interrupted graph
                final_state = _resume_with_answer(user_input, st.session_state.thread_id)
            else:
                # New query on current thread
                # For genuine new questions, start a fresh thread
                st.session_state.thread_id = str(uuid.uuid4())
                final_state = _invoke_new_query(user_input, st.session_state.thread_id)

        # Check if graph is STILL interrupted (e.g., disambiguation after clarification)
        interrupted, clarification_q = _is_interrupted(st.session_state.thread_id)

        if interrupted and clarification_q:
            st.session_state.awaiting_clarification = True
            answer_text = clarification_q
            steps = []
        else:
            st.session_state.awaiting_clarification = False
            answer_text = final_state.get("final_answer") or "I couldn't generate a response."
            steps = final_state.get("steps_performed", [])

        st.markdown(answer_text)
        if steps:
            with st.expander("How I computed this"):
                for step in steps:
                    st.markdown(f"• {step}")

    # Store in message history
    msg_record: dict = {"role": "assistant", "content": answer_text}
    if steps:
        msg_record["steps"] = steps
    st.session_state.messages.append(msg_record)

    # Update debug info
    if not interrupted:
        st.session_state.last_debug = _extract_debug(final_state)

    st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _init_session()
    _render_sidebar()
    _render_chat()


if __name__ == "__main__":
    main()
