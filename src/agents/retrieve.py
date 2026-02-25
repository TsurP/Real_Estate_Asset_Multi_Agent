"""
RetrievalAgent: Resolve names via fuzzy matching and fetch filtered DataFrame rows.
May interrupt for disambiguation when multiple close matches exist.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langgraph.types import interrupt

from ..core.state import AgentState, Intent
from ..tools.df_tools import (
    get_dataframe,
    list_unique,
    query_ledger,
    resolve_all_targets,
    resolve_name,
)

logger = logging.getLogger(__name__)

# Score above which we consider a match unambiguous
UNAMBIGUOUS_THRESHOLD = 90
# Score below which we consider it no match
NO_MATCH_THRESHOLD = 60
# Gap between top match and next to call it "unambiguous"
SCORE_GAP_THRESHOLD = 15


def retrieve_node(state: AgentState) -> dict:
    """
    1. Resolve target names via fuzzy matching.
    2. Handle ambiguous / no-match cases with interrupts.
    3. Query the filtered DataFrame.
    4. Store serialised records in state.
    """
    trace = list(state.get("node_trace", []))
    trace.append("retrieve")

    intent = state.get("intent", "")
    targets = state.get("targets", [])
    target_fields = state.get("target_fields", [])
    timeframe = state.get("timeframe")
    ledger_filters = state.get("ledger_filters", {})

    df = get_dataframe()

    # ---------------------------------------------------------------------------
    # Resolve target names (skip if no targets required)
    # ---------------------------------------------------------------------------
    resolved_targets: list[dict] = []
    if targets:
        resolved, unresolved = resolve_all_targets(targets, target_fields, df)

        # Handle unresolved targets
        if unresolved:
            for u in unresolved:
                field = u.get("field", "property_name")
                candidates = list_unique(field, df)[:8]
                q = interrupt(
                    f"I couldn't find '{u['target']}'. "
                    f"Did you mean one of: {', '.join(candidates)}? "
                    f"Please type the exact name."
                )
                # Re-resolve with user's clarification
                new_matches = resolve_name(q, field, df)
                if new_matches:
                    resolved.append({
                        "target": q,
                        "field": field,
                        "value": new_matches[0]["value"],
                        "score": new_matches[0]["score"],
                        "all_matches": new_matches,
                    })

        # Handle ambiguous matches (high score but multiple close candidates)
        for res in resolved:
            all_m = res.get("all_matches", [])
            if len(all_m) >= 2:
                top_score = all_m[0]["score"]
                second_score = all_m[1]["score"]
                is_exact = top_score == 100
                is_ambiguous = (
                    not is_exact
                    and top_score >= NO_MATCH_THRESHOLD
                    and (top_score - second_score) < SCORE_GAP_THRESHOLD
                )
                if is_ambiguous:
                    top3 = [m["value"] for m in all_m[:3]]
                    q = interrupt(
                        f"'{res['target']}' matches multiple entries: {', '.join(top3)}. "
                        f"Which one did you mean?"
                    )
                    # Use the disambiguation answer
                    new_matches = resolve_name(q, res["field"], df)
                    if new_matches:
                        res["value"] = new_matches[0]["value"]

        resolved_targets = resolved

    # ---------------------------------------------------------------------------
    # Query filtered dataframe
    # ---------------------------------------------------------------------------
    # For share_of_total: fetch the whole scope (no target filter) so compute can
    # derive both the target amount and the total. Targets are resolved above for
    # name validation but not applied as a row filter here.
    query_targets = [] if intent == Intent.SHARE_OF_TOTAL else resolved_targets

    try:
        filtered_df = query_ledger(
            timeframe=timeframe,
            ledger_filters=ledger_filters,
            resolved_targets=query_targets,
            df=df,
        )
    except Exception as exc:
        logger.error("query_ledger failed: %s", exc)
        return {
            "node_trace": trace,
            "error": f"Data retrieval error: {exc}",
            "no_match": True,
            "retrieved_rows": 0,
            "retrieved_records": [],
        }

    row_count = len(filtered_df)

    if row_count == 0:
        return {
            "node_trace": trace,
            "no_match": True,
            "retrieved_rows": 0,
            "retrieved_records": [],
            "candidates": _suggest_candidates(targets, target_fields, df),
        }

    # Serialise to list[dict] (safe across LangGraph state serialisation)
    records = filtered_df.to_dict(orient="records")
    # Convert any non-serialisable types
    records = _sanitize_records(records)

    # Store resolved target info back so ResponseAgent can reference it
    return {
        "node_trace": trace,
        "no_match": False,
        "ambiguous": False,
        "retrieved_rows": row_count,
        "retrieved_records": records,
        "targets": [r["value"] for r in resolved_targets] if resolved_targets else targets,
        "candidates": [],
        "error": None,
    }


def _sanitize_records(records: list[dict]) -> list[dict]:
    """Convert numpy / pandas types to plain Python for JSON serialisation."""
    import numpy as np
    clean = []
    for row in records:
        new_row = {}
        for k, v in row.items():
            if isinstance(v, float) and (v != v):  # NaN
                new_row[k] = None
            elif hasattr(v, "item"):  # numpy scalar
                new_row[k] = v.item()
            else:
                new_row[k] = v
        clean.append(new_row)
    return clean


def _suggest_candidates(
    targets: list[str],
    target_fields: list[str],
    df: Any,
) -> list[dict]:
    """Generate top-3 candidate suggestions for unmatched targets."""
    from rapidfuzz import fuzz, process as fuzz_process

    suggestions = []
    for name, field in zip(targets, target_fields):
        choices = list_unique(field, df)
        results = fuzz_process.extract(name, choices, scorer=fuzz.WRatio, limit=3)
        suggestions.extend([
            {"field": field, "value": r[0], "score": r[1]} for r in results
        ])
    return suggestions
