"""
ResponseAgent: Generate a concise, structured answer from computation results.
Uses deterministic formatting for data intents and LLM only for general questions.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from ..core.state import AgentState, Intent

logger = logging.getLogger(__name__)


def respond_node(state: AgentState, llm: Any) -> dict:
    """
    Produce the final answer for the user.
    For general_question intents: pure LLM answer (no computation).
    For data intents: format computation_result into readable prose.
    """
    trace = list(state.get("node_trace", []))
    trace.append("respond")

    intent = state.get("intent", "")
    user_query = state.get("user_query", "")
    computation_result = state.get("computation_result")
    timeframe = state.get("timeframe", {})
    targets = state.get("targets", [])
    target_fields = state.get("target_fields", [])
    row_count = state.get("retrieved_rows", 0)
    ledger_filters = state.get("ledger_filters", {})

    # ---- General question: no data needed ----
    if intent == Intent.GENERAL_QUESTION:
        answer = _llm_general(user_query, llm)
        return {
            "node_trace": trace,
            "final_answer": answer,
            "steps_performed": ["Answered from domain knowledge (no data query needed)."],
            "assumptions": [],
        }

    # ---- Data intent: format computation result ----
    if computation_result and "error" not in computation_result:
        answer = _format_data_answer(computation_result, timeframe, targets, target_fields, ledger_filters, row_count)
        steps = _derive_steps(intent, computation_result, timeframe, ledger_filters, row_count, targets)
        assumptions = _derive_assumptions(timeframe, targets)
    else:
        error_msg = (computation_result or {}).get("error", "Unknown error")
        answer = f"I was unable to compute the answer: {error_msg}"
        steps = ["Retrieval succeeded but computation failed."]
        assumptions = []

    return {
        "node_trace": trace,
        "final_answer": answer,
        "steps_performed": steps,
        "assumptions": assumptions,
    }


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

_PORTFOLIO_DATA_KEYWORDS = (
    # listing
    "which entities", "what entities", "list entities", "show entities",
    "which properties", "what properties", "list properties", "show properties",
    "which buildings", "what buildings", "list buildings", "show buildings",
    "which tenants", "what tenants", "list tenants", "show tenants",
    "in my database", "in the database", "in your database",
    "what's available", "what is available", "what data", "what do you have",
    "what can you tell me", "overview", "summary of",
    # counting
    "how many", "number of", "count of", "total number",
)


def _llm_general(query: str, llm: Any) -> str:
    """Answer a general real-estate / accounting question without data.
    Enriches with actual portfolio data when the query asks about portfolio contents.
    """
    q_lower = query.lower()

    # Enrich with actual data when user asks about what's in the database or counts
    data_context = ""
    if any(kw in q_lower for kw in _PORTFOLIO_DATA_KEYWORDS):
        try:
            from ..tools.df_tools import list_unique
            entities = list_unique("entity_name")
            properties = list_unique("property_name")
            tenants = list_unique("tenant_name")
            data_context = (
                "\n\nThis system's actual portfolio data contains:\n"
                f"- Entities ({len(entities)}): {', '.join(entities)}\n"
                f"- Properties ({len(properties)}): {', '.join(properties)}\n"
                f"- Tenants ({len(tenants)}): {', '.join(tenants)}\n"
                "- Data years: 2024 (full year) and 2025 (Q1 only)\n"
            )
        except Exception:
            pass

    try:
        prompt = (
            "You are a knowledgeable real-estate portfolio analyst. "
            "Answer the following question concisely (under 150 words). "
            "When actual portfolio data is provided below, use it to answer — "
            "do NOT give a generic real-estate textbook answer."
            f"{data_context}\n\n"
            f"Question: {query}"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as exc:
        logger.warning("LLM general question failed: %s", exc)
        return "I can answer general real-estate and accounting questions. Please rephrase your question."


def _format_data_answer(
    result: dict,
    timeframe: dict | None,
    targets: list[str],
    target_fields: list[str],
    ledger_filters: dict,
    row_count: int,
) -> str:
    """Render deterministic, non-LLM answers for data intents.
    Metadata (scope, targets, filters) belongs in 'How I computed this', not here.
    """
    rtype = result.get("type", "")
    lf = ledger_filters or {}
    lf_type = lf.get("ledger_type")
    lf_group = lf.get("ledger_group")
    tf_label = (timeframe or {}).get("label") or _fmt_timeframe(timeframe or {})

    if rtype == "pnl_total":
        target_phrase = f" for {', '.join(targets)}" if targets else ""
        time_phrase = f" in {tf_label}" if tf_label else ""

        if lf_type == "expenses":
            metric = lf_group.replace("_", " ").lower() if lf_group else "total expenses"
            amount = abs(result["total_expenses"])
            verb = "were" if "expenses" in metric else "was"
            return f"The {metric}{target_phrase}{time_phrase} {verb} €{amount:,.2f}."
        elif lf_type == "revenue":
            metric = lf_group.replace("_", " ").lower() if lf_group else "total revenue"
            amount = result["total_revenue"]
            return f"The {metric}{target_phrase}{time_phrase} was €{amount:,.2f}."
        else:
            # Full P&L — show profit prominently + revenue/expenses on second line
            profit = result["total_profit"]
            revenue = result["total_revenue"]
            expenses = abs(result["total_expenses"])
            intro = f"Total P&L{target_phrase}{time_phrase}:" if (target_phrase or time_phrase) else ""
            lines = ([intro] if intro else []) + [
                f"Profit: €{profit:,.2f}",
                f"Revenue: €{revenue:,.2f} | Expenses: €{expenses:,.2f}",
            ]
            return "\n".join(lines)

    if rtype.startswith("grouped_"):
        rows = result.get("rows", [])

        # Location query: grouped by property but filtered by a tenant target.
        # User wants to know WHICH buildings the tenant is in, not profit figures.
        is_location_query = (
            rtype == "grouped_property_name"
            and bool(targets)
            and "tenant_name" in (target_fields or [])
        )
        if is_location_query:
            buildings = [r["label"] for r in rows]
            tenant = targets[0]
            if not buildings:
                return f"{tenant} was not found in any building."
            if len(buildings) == 1:
                return f"{tenant} is in {buildings[0]}."
            return f"{tenant} appears in: {', '.join(buildings[:-1])} and {buildings[-1]}."

        # Single-value result — produce a natural sentence
        if len(rows) == 1:
            label = rows[0]["label"]
            profit = rows[0]["profit"]
            metric = (
                str(lf.get("ledger_category", "")).replace("_", " ").lower() if lf.get("ledger_category")
                else lf_group.replace("_", " ").lower() if lf_group
                else lf_type.lower() if lf_type
                else "profit"
            )
            target_phrase = f" for {label}" if label else ""
            if result.get("group_field") == "ledger_type":
                target_phrase = ""
            time_phrase = f" in {tf_label}" if tf_label else ""
            if lf_type == "expenses":
                verb = "were" if "expenses" in metric else "was"
                amount = abs(profit)
            else:
                verb = "was"
                amount = profit
            return f"The {metric}{target_phrase}{time_phrase} {verb} €{amount:,.2f}."

        # Multi-row table
        group_field = result.get("group_field", "group")
        result_k = result.get("k")
        direction = result.get("direction", "top")

        # For expenses: re-sort by absolute amount descending (most expensive first)
        # and show positive values. compute_grouped sorts by raw profit (least negative first).
        if lf_type == "expenses":
            rows = sorted(rows, key=lambda r: abs(r["profit"]), reverse=True)
            fmt_row = lambda p: abs(p)
            metric_label = "Expenses"
            total_display = abs(result.get("total_profit", 0))
        elif lf_type == "revenue":
            fmt_row = lambda p: p
            metric_label = "Revenue"
            total_display = result.get("total_profit", 0)
        else:
            fmt_row = lambda p: p
            metric_label = "Profit"
            total_display = result.get("total_profit", 0)

        # k=1: produce a single winner sentence instead of a table.
        # After sorting, rows[0] is always the "most/highest" end and rows[-1] is the
        # "least/lowest" end — pick based on direction.
        if result_k == 1 and rows:
            winner = rows[-1] if direction == "bottom" else rows[0]
            label = winner["label"].replace("_", " ")
            amount = fmt_row(winner["profit"])
            gf_human = group_field.replace("_name", "").replace("_", " ")
            if lf_type == "expenses":
                contrib = "least" if direction == "bottom" else "most"
                return (
                    f"The {gf_human} that contributed the {contrib} to total expenses "
                    f"was {label}, at €{amount:,.2f}."
                )
            elif lf_type == "revenue":
                adjective = "lowest" if direction == "bottom" else "highest"
                return (
                    f"The {gf_human} with the {adjective} total revenue "
                    f"was {label}, at €{amount:,.2f}."
                )
            else:
                adjective = "lowest" if direction == "bottom" else "highest"
                return (
                    f"The {gf_human} with the {adjective} total profit "
                    f"was {label}, at €{amount:,.2f}."
                )

        # Limit rows to k when set
        display_rows = rows[:result_k] if result_k and result_k < len(rows) else rows
        lines = [f"{metric_label} by {group_field}:"] + [
            f"- {r['label']}: €{fmt_row(r['profit']):,.2f}" for r in display_rows
        ]
        lines.append(f"Total: €{total_display:,.2f}")
        return "\n".join(lines)

    if rtype == "comparison":
        a = result["target_a"]
        b = result["target_b"]
        delta = result["delta"]
        pct = result["pct_delta"]
        pct_str = f"{pct:+.1f}%" if pct != float("inf") else "∞%"
        lines = [
            f"{a['target']}: €{a['profit']:,.2f}",
            f"{b['target']}: €{b['profit']:,.2f}",
            f"Delta (A-B): €{delta:+,.2f} ({pct_str})",
            f"Higher performer: {result['higher']}",
        ]
        return "\n".join(lines)

    if rtype == "ranking":
        # For expense rankings, profit is negative; display absolute values so amounts are positive.
        fmt_profit = (lambda p: abs(p)) if lf_type == "expenses" else (lambda p: p)
        rows = result.get("rows", [])
        direction = result.get("direction", "top")
        group_field = result.get("group_field", "group")

        if result.get("k") == 1 and rows:
            winner = rows[0]
            label_human = _humanize_rank_label(str(winner["label"]), group_field)
            adjective = "lowest" if direction == "bottom" else "highest"
            metric = "total net profit" if not lf_type else f"total {lf_type}"
            if group_field == "month":
                subject = f"The month with the {adjective} {metric}"
            elif group_field == "quarter":
                subject = f"The quarter with the {adjective} {metric}"
            elif group_field == "year":
                subject = f"The year with the {adjective} {metric}"
            else:
                subject = f"The {group_field.replace('_name', '').replace('_', ' ')} with the {adjective} {metric}"
            return f"{subject} was {label_human}, at €{fmt_profit(winner['profit']):,.2f}."

        lines = [f"{direction.capitalize()} {result['k']} by {group_field}:"] + [
            f"- #{r['rank']} {r['label']}: €{fmt_profit(r['profit']):,.2f}" for r in rows
        ]
        return "\n".join(lines)

    if rtype == "margin_analysis":
        rows = result.get("rows", [])
        if not rows:
            return "No margin values could be computed from the selected data."

        direction = result.get("direction", "top")
        group_field = result.get("group_field", "group")
        k = result.get("k", len(rows))

        if k == 1:
            winner = rows[0]
            descriptor = "Worst" if direction == "bottom" else "Best"
            return (
                f"{descriptor} net profit margin by {group_field}: "
                f"{winner['label']} at {winner['margin_pct']:.2f}% "
                f"(€{winner['total_profit']:,.2f} ÷ €{winner['total_revenue']:,.2f})."
            )

        lines = [f"{direction.capitalize()} {k} by net profit margin (profit ÷ revenue):"]
        for r in rows:
            lines.append(
                f"- #{r['rank']} {r['label']}: {r['margin_pct']:.2f}% "
                f"(€{r['total_profit']:,.2f} ÷ €{r['total_revenue']:,.2f})"
            )
        return "\n".join(lines)

    if rtype == "presence_condition":
        condition = result.get("condition", "")
        group_field = result.get("group_field", "group")
        rows = result.get("rows", [])
        tf_phrase = f" in {tf_label}" if tf_label else ""

        if group_field == "property_name":
            group_noun = "properties"
        elif group_field == "tenant_name":
            group_noun = "tenants"
        elif group_field == "entity_name":
            group_noun = "entities"
        else:
            group_noun = group_field

        if condition == "expenses_without_revenue":
            if not rows:
                return f"No {group_noun} had expenses but no revenue{tf_phrase}."
            labels = ", ".join(r["label"] for r in rows)
            return f"{group_noun.capitalize()} with expenses but no revenue{tf_phrase}: {labels}."

        if condition == "revenue_without_expenses":
            if not rows:
                return f"No {group_noun} had revenue but no expenses{tf_phrase}."
            labels = ", ".join(r["label"] for r in rows)
            return f"{group_noun.capitalize()} with revenue but no expenses{tf_phrase}: {labels}."

        return f"No results found for condition '{condition}'."

    if rtype == "stat_condition":
        condition = result.get("condition", "")
        group_field = result.get("group_field", "group")
        rows = result.get("rows", [])
        avg_profit = float(result.get("average_profit", 0.0))
        tf_phrase = f" in {tf_label}" if tf_label else ""

        if group_field == "month":
            singular, plural = "month", "months"
        elif group_field == "quarter":
            singular, plural = "quarter", "quarters"
        elif group_field == "year":
            singular, plural = "year", "years"
        elif group_field == "property_name":
            singular, plural = "property", "properties"
        elif group_field == "tenant_name":
            singular, plural = "tenant", "tenants"
        else:
            singular, plural = "group", "groups"

        if condition == "below_average_profit":
            if not rows:
                return f"No {plural} had below-average net profit{tf_phrase}."
            labels = ", ".join(_humanize_rank_label(r["label"], group_field) for r in rows)
            return (
                f"The average net profit per {singular}{tf_phrase} was €{avg_profit:,.2f}. "
                f"The {plural} below average were: {labels}."
            )
        if condition == "above_average_profit":
            if not rows:
                return f"No {plural} had above-average net profit{tf_phrase}."
            labels = ", ".join(_humanize_rank_label(r["label"], group_field) for r in rows)
            return (
                f"The average net profit per {singular}{tf_phrase} was €{avg_profit:,.2f}. "
                f"The {plural} above average were: {labels}."
            )

        return f"No results found for condition '{condition}'."

    if rtype == "count":
        field = result.get("field", "")
        count = result.get("count", 0)
        values = result.get("values", [])
        target_phrase = f" in {', '.join(targets)}" if targets else ""
        tf_phrase = f" in {tf_label}" if tf_label else ""
        entity = field.replace("_name", "").replace("_", " ")
        if count == 0:
            return f"No {entity}s found{target_phrase}{tf_phrase}."
        value_list = ", ".join(str(v) for v in values)
        plural = "" if count == 1 else "s"
        return (
            f"There {'was' if count == 1 else 'were'} {count} {entity}{plural}"
            f"{target_phrase}{tf_phrase}: {value_list}."
        )

    if rtype == "full_review":
        return _format_full_review(result)

    if rtype == "period_delta":
        return _format_period_delta(result)

    if rtype == "share_of_total":
        return _format_share_of_total(result, ledger_filters)

    return f"Result: {result}"


def _format_full_review(result: dict) -> str:
    """Format a comprehensive multi-section review report."""
    total = result.get("total", {})
    lines: list[str] = [
        "**Overall P&L**",
        f"Profit: €{total.get('total_profit', 0):,.2f}  |  "
        f"Revenue: €{total.get('total_revenue', 0):,.2f}  |  "
        f"Expenses: €{total.get('total_expenses', 0):,.2f}",
    ]

    by_property = result.get("by_property", {})
    if by_property.get("rows"):
        lines += ["", "**By Property**"]
        for r in by_property["rows"]:
            lines.append(f"- {r['label']}: €{r['profit']:,.2f}")

    by_tenant = result.get("by_tenant", {})
    if by_tenant.get("rows"):
        lines += ["", "**By Tenant**"]
        for r in by_tenant["rows"]:
            lines.append(f"- {r['label']}: €{r['profit']:,.2f}")

    by_year = result.get("by_year", {})
    if by_year.get("rows"):
        lines += ["", "**By Year**"]
        for r in by_year["rows"]:
            lines.append(f"- {r['label']}: €{r['profit']:,.2f}")

    return "\n".join(lines)


def _format_period_delta(result: dict) -> str:
    """Format a period-over-period profit change table."""
    period_a = result.get("period_a_label", "Period A")
    period_b = result.get("period_b_label", "Period B")
    rows = result.get("rows", [])
    group_field = result.get("group_field", "group")

    if not rows:
        return f"No data found for the comparison between {period_a} and {period_b}."

    winner = rows[0]
    sign = "+" if winner["delta"] >= 0 else ""
    summary = (
        f"Largest change from {period_a} to {period_b}: "
        f"{winner['label']} ({sign}€{winner['delta']:,.2f})\n"
    )

    lines = [summary, f"Full breakdown by {group_field}:"]
    for r in rows:
        s = "+" if r["delta"] >= 0 else ""
        pct = r.get("pct_delta", 0.0)
        pct_str = f"{pct:+.1f}%" if pct != float("inf") else "∞%"
        lines.append(
            f"- {r['label']}: {s}€{r['delta']:,.2f} ({pct_str})  "
            f"[{period_a}: €{r['period_a']:,.2f} → {period_b}: €{r['period_b']:,.2f}]"
        )
    return "\n".join(lines)


def _format_share_of_total(result: dict, ledger_filters: dict) -> str:
    """Format a share-of-total / percentage answer."""
    target = result.get("target", "")
    lf = ledger_filters or {}
    lf_type = lf.get("ledger_type")

    # Choose which metric to highlight based on ledger filter
    if lf_type == "revenue":
        pct = result["pct_revenue"]
        amount = result["target_revenue"]
        total = result["total_revenue"]
        metric = "revenue"
    elif lf_type == "expenses":
        pct = result["pct_expenses"]
        amount = result["target_expenses"]
        total = result["total_expenses"]
        metric = "expenses"
    else:
        pct = result["pct_profit"]
        amount = result["target_profit"]
        total = result["total_profit"]
        metric = "net profit"

    return (
        f"{target} accounted for {pct:.1f}% of total {metric}.\n"
        f"({target}: €{amount:,.2f} / Total: €{total:,.2f})"
    )


def _derive_steps(
    intent: str,
    result: dict,
    timeframe: dict | None,
    ledger_filters: dict,
    row_count: int,
    targets: list[str] | None = None,
) -> list[str]:
    """Generate bullet points describing what was computed (shown in 'How I computed this')."""
    steps = []

    tf_label = (timeframe or {}).get("label") or _fmt_timeframe(timeframe or {})
    if tf_label:
        steps.append(f"Filtered to timeframe: {tf_label} ({row_count:,} rows matched)")
    else:
        steps.append(f"No timeframe filter — used all available data ({row_count:,} rows)")

    if targets:
        steps.append(f"Filtered to: {', '.join(targets)}")

    if ledger_filters:
        lf_parts = [f"{k}={v}" for k, v in ledger_filters.items() if v is not None]
        if lf_parts:
            steps.append(f"Ledger filter: {', '.join(lf_parts)}")

    rtype = result.get("type", "")
    if rtype == "pnl_total":
        steps.append("Summed `profit` column across all matching rows.")
    elif rtype.startswith("grouped_"):
        field = result.get("group_field", "")
        steps.append(f"Grouped by `{field}` and summed `profit` per group.")
        steps.append("Sorted results by profit descending.")
    elif rtype == "comparison":
        steps.append("Computed profit sum for each target separately.")
        steps.append("Calculated absolute and percentage delta (A − B).")
    elif rtype == "ranking":
        steps.append(
            f"Grouped by `{result.get('group_field')}` and ranked by profit sum."
        )
        steps.append(f"Selected {result['direction']} {result['k']} entries (ties included).")
    elif rtype == "margin_analysis":
        steps.append(f"Grouped by `{result.get('group_field')}`.")
        steps.append("Computed net profit margin = total profit ÷ total revenue for each group.")
        steps.append(f"Ranked by margin and selected {result['direction']} {result['k']} entries (ties included).")
    elif rtype == "presence_condition":
        condition = result.get("condition", "")
        field = result.get("group_field", "")
        if condition == "expenses_without_revenue":
            steps.append(f"Grouped by `{field}` and checked ledger presence per group.")
            steps.append("Selected groups that have expenses rows and zero revenue rows.")
        elif condition == "revenue_without_expenses":
            steps.append(f"Grouped by `{field}` and checked ledger presence per group.")
            steps.append("Selected groups that have revenue rows and zero expenses rows.")
    elif rtype == "stat_condition":
        condition = result.get("condition", "")
        field = result.get("group_field", "")
        steps.append(f"Grouped by `{field}` and summed net profit per group.")
        steps.append("Computed the average net profit across all groups in scope.")
        if condition == "below_average_profit":
            steps.append("Selected groups with net profit strictly below the average.")
        elif condition == "above_average_profit":
            steps.append("Selected groups with net profit strictly above the average.")
    elif rtype == "count":
        field = result.get("field", "")
        steps.append(f"Counted distinct values of `{field}` in the filtered rows.")
    elif rtype == "full_review":
        steps.append("Computed total P&L (profit, revenue, expenses).")
        steps.append("Grouped by property_name and summed profit per property.")
        steps.append("Grouped by tenant_name and summed profit per tenant.")
        if result.get("by_year"):
            steps.append("Grouped by year and summed profit per year.")
    elif rtype == "period_delta":
        pa = result.get("period_a_label", "Period A")
        pb = result.get("period_b_label", "Period B")
        field = result.get("group_field", "")
        steps.append(f"Filtered retrieved data to {pa} and separately to {pb}.")
        steps.append(f"Grouped by `{field}` and summed profit for each period.")
        steps.append("Computed delta = Period B − Period A, sorted by delta descending.")
    elif rtype == "share_of_total":
        target = result.get("target", "")
        field = result.get("field", "")
        steps.append(f"Retrieved all data in scope (no {field} filter), then isolated {target}'s rows.")
        steps.append(f"Summed profit/revenue/expenses for {target} and for the full scope.")
        steps.append("Computed percentage = target amount / total amount × 100.")

    return steps


def _derive_assumptions(timeframe: dict | None, targets: list[str]) -> list[str]:
    """Note any implicit assumptions made."""
    assumptions = []
    if timeframe and timeframe.get("ytd"):
        year = timeframe.get("year", "")
        assumptions.append(
            f"Interpreted 'this year / YTD' as {year} up to the latest available data."
        )
    if not timeframe:
        assumptions.append("No time filter applied — all available data used.")
    return assumptions


def _fmt_timeframe(tf: dict) -> str:
    if tf.get("month"):
        return tf["month"]
    if tf.get("quarter"):
        return tf["quarter"]
    if tf.get("year"):
        return tf["year"]
    return ""


def _humanize_rank_label(label: str, group_field: str) -> str:
    """Convert coded time-bucket labels into natural language."""
    if group_field == "month":
        m = re.match(r"^(\d{4})-M(0[1-9]|1[0-2])$", label)
        if m:
            year = m.group(1)
            month = int(m.group(2))
            month_name = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December",
            ][month - 1]
            return f"{month_name} {year}"
    if group_field == "quarter":
        q = re.match(r"^(\d{4})-Q([1-4])$", label)
        if q:
            return f"Q{q.group(2)} {q.group(1)}"
    return label
