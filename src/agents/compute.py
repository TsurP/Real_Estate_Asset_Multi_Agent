"""
ComputationAgent: Deterministic aggregation/comparison/ranking using pandas only.
NO LLM calls — pure math.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd

from ..core.state import AgentState, Intent
from ..tools.df_tools import apply_timeframe_filter

logger = logging.getLogger(__name__)


def compute_node(state: AgentState) -> dict:
    """
    Execute deterministic computation based on intent and retrieved records.
    Returns structured computation_result dict.
    """
    trace = list(state.get("node_trace", []))
    trace.append("compute")

    intent = state.get("intent", "")
    records = state.get("retrieved_records", [])
    group_by = state.get("group_by")
    k = state.get("k", 5)
    rank_direction = state.get("rank_direction", "top")
    targets = state.get("targets", [])
    target_fields = state.get("target_fields", [])
    query_condition = state.get("query_condition")
    timeframe_a = state.get("timeframe_a")
    timeframe_b = state.get("timeframe_b")
    ledger_filters = state.get("ledger_filters", {})
    ratio_type = state.get("ratio_type") or "profit_margin"
    margin_pre_filter = state.get("margin_pre_filter")

    if not records:
        return {
            "node_trace": trace,
            "computation_result": {"error": "No data rows available for computation."},
            "error": "Empty dataset after filters.",
        }

    df = pd.DataFrame(records)

    try:
        result = _dispatch(
            intent,
            df,
            group_by,
            k,
            rank_direction,
            targets,
            target_fields,
            query_condition,
            timeframe_a,
            timeframe_b,
            ledger_filters,
            ratio_type,
            margin_pre_filter,
        )
        return {
            "node_trace": trace,
            "computation_result": result,
            "error": None,
        }
    except Exception as exc:
        logger.error("ComputationAgent error: %s", exc, exc_info=True)
        return {
            "node_trace": trace,
            "computation_result": {"error": str(exc)},
            "error": f"Computation failed: {exc}",
        }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _dispatch(
    intent: str,
    df: pd.DataFrame,
    group_by: Optional[str],
    k: int,
    rank_direction: str,
    targets: list[str],
    target_fields: list[str],
    query_condition: Optional[str] = None,
    timeframe_a: Optional[dict] = None,
    timeframe_b: Optional[dict] = None,
    ledger_filters: Optional[dict] = None,
    ratio_type: Optional[str] = None,
    margin_pre_filter: Optional[str] = None,
) -> dict:
    if query_condition:
        field = group_by or "property_name"
        return compute_presence_condition(df, field, query_condition)

    if intent == Intent.PNL_TOTAL:
        return compute_pnl_total(df)

    if intent == Intent.PNL_BY_PROPERTY:
        # If group_by is a non-default dimension (e.g. month), use it — the property
        # filter is already applied at retrieval; the user wants further breakdown.
        dim = group_by if group_by and group_by != "property_name" else "property_name"
        result = compute_grouped(df, dim)
        result["k"] = k
        result["direction"] = rank_direction
        return result

    if intent == Intent.PNL_BY_TENANT:
        # Same: respect group_by when it diverges from the default tenant dimension.
        dim = group_by if group_by and group_by != "tenant_name" else "tenant_name"
        result = compute_grouped(df, dim)
        result["k"] = k
        result["direction"] = rank_direction
        return result

    if intent == Intent.BREAKDOWN:
        dim = group_by or "ledger_type"  # default breakdown dimension
        result = compute_grouped(df, dim)
        # Pass k and direction so respond.py can render single-winner sentences
        # and limit/sort display rows correctly.
        result["k"] = k
        result["direction"] = rank_direction
        return result

    if intent == Intent.COMPARISON:
        return compute_comparison(df, targets, target_fields)

    if intent == Intent.RANKING:
        # group_by is the authoritative ranking dimension; target_fields[0] as fallback
        field = group_by or (target_fields[0] if target_fields else "property_name")
        # Expenses are stored as negative numbers; flip sort direction so "top expenses"
        # yields the most spent (most negative) entities first.
        lf = ledger_filters or {}
        effective_direction = rank_direction
        if lf.get("ledger_type") == "expenses":
            effective_direction = "bottom" if rank_direction == "top" else "top"
        result = compute_ranking(df, field, k, effective_direction)
        # Restore original user-facing direction label ("Top 3", not "Bottom 3")
        result["direction"] = rank_direction
        return result

    if intent == Intent.MARGIN_ANALYSIS:
        field = group_by or "property_name"
        return compute_margin_analysis(
            df, field, k, rank_direction,
            ratio_type=ratio_type or "profit_margin",
            pre_filter=margin_pre_filter,
        )

    if intent == Intent.PERIOD_COMPARISON:
        field = group_by or (target_fields[0] if target_fields else "property_name")
        return compute_period_delta(df, timeframe_a, timeframe_b, field, k=k, direction=rank_direction)

    if intent == Intent.SHARE_OF_TOTAL:
        return compute_share_of_total(df, targets, target_fields)

    if intent == Intent.FULL_REVIEW:
        return compute_full_review(df)

    if intent == Intent.COUNT:
        field = group_by or "tenant_name"
        return compute_count(df, field)

    # Fallback: just sum
    return compute_pnl_total(df)


# ---------------------------------------------------------------------------
# Individual compute functions
# ---------------------------------------------------------------------------

def compute_pnl_total(df: pd.DataFrame) -> dict:
    """Sum all profit — overall P&L total."""
    total = float(df["profit"].sum())
    revenue = float(df.loc[df["ledger_type"] == "revenue", "profit"].sum())
    expenses = float(df.loc[df["ledger_type"] == "expenses", "profit"].sum())

    return {
        "type": "pnl_total",
        "total_profit": total,
        "total_revenue": revenue,
        "total_expenses": expenses,
        "row_count": len(df),
        "timeframe_values": _timeframe_summary(df),
    }


def compute_grouped(df: pd.DataFrame, group_field: str) -> dict:
    """Profit grouped by a single field, sorted descending."""
    if group_field not in df.columns:
        raise ValueError(f"Column '{group_field}' not in dataset.")

    grp = (
        df.groupby(group_field, dropna=True)["profit"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    grp.columns = [group_field, "profit"]

    rows = []
    for _, r in grp.iterrows():
        key = r[group_field]
        rows.append({
            "label": str(key) if key is not None else "(no value)",
            "profit": float(r["profit"]),
        })

    return {
        "type": f"grouped_{group_field}",
        "group_field": group_field,
        "rows": rows,
        "total_profit": float(grp["profit"].sum()),
        "row_count": len(df),
    }


def compute_comparison(
    df: pd.DataFrame,
    targets: list[str],
    target_fields: list[str],
) -> dict:
    """Compare exactly two targets — both profit sums + delta + % delta."""
    if len(targets) < 2:
        raise ValueError("Comparison requires exactly 2 targets.")

    results = []
    for name, field in zip(targets, target_fields):
        mask = df[field] == name
        subset = df[mask]
        profit = float(subset["profit"].sum())
        results.append({"target": name, "field": field, "profit": profit, "rows": len(subset)})

    a, b = results[0], results[1]
    delta = a["profit"] - b["profit"]

    if b["profit"] != 0.0:
        pct_delta = (delta / abs(b["profit"])) * 100
    elif a["profit"] != 0.0:
        pct_delta = float("inf")
    else:
        pct_delta = 0.0

    winner = a["target"] if a["profit"] > b["profit"] else b["target"]

    return {
        "type": "comparison",
        "target_a": a,
        "target_b": b,
        "delta": delta,              # a - b
        "pct_delta": pct_delta,      # (a-b)/|b| * 100
        "higher": winner,
        "row_count": len(df),
    }


def compute_ranking(
    df: pd.DataFrame,
    group_field: str,
    k: int,
    direction: str = "top",
) -> dict:
    """Top/bottom K entities by profit. Handles ties by including all tied items."""
    if group_field not in df.columns:
        raise ValueError(f"Column '{group_field}' not in dataset.")

    grp = (
        df.groupby(group_field, dropna=True)["profit"]
        .sum()
        .reset_index()
    )
    grp.columns = [group_field, "profit"]
    grp = grp.sort_values("profit", ascending=(direction == "bottom"))

    # Include tied items at the k-th position
    if len(grp) >= k:
        cutoff = float(grp.iloc[k - 1]["profit"])
        if direction == "bottom":
            grp = grp[grp["profit"] <= cutoff]
        else:
            grp = grp[grp["profit"] >= cutoff]
    # Cap at reasonable max
    grp = grp.head(k + 5)

    rows = [
        {
            "rank": i + 1,
            "label": str(r[group_field]),
            "profit": float(r["profit"]),
        }
        for i, (_, r) in enumerate(grp.iterrows())
    ]

    return {
        "type": "ranking",
        "direction": direction,
        "k": k,
        "group_field": group_field,
        "rows": rows,
        "row_count": len(df),
    }


def compute_margin_analysis(
    df: pd.DataFrame,
    group_field: str,
    k: int,
    direction: str = "top",
    ratio_type: str = "profit_margin",
    pre_filter: Optional[str] = None,
) -> dict:
    """
    Rank entities by a ratio metric.

    ratio_type:
      - "profit_margin"  → net profit ÷ revenue  (default)
      - "expense_ratio"  → |expenses| ÷ revenue

    pre_filter:
      - "above_average_revenue"  → only include groups whose total revenue > mean revenue
      - "below_average_revenue"  → only include groups whose total revenue < mean revenue
    """
    if group_field not in df.columns:
        raise ValueError(f"Column '{group_field}' not in dataset.")
    if "ledger_type" not in df.columns:
        raise ValueError("Column 'ledger_type' not in dataset.")

    profit_by_group = df.groupby(group_field, dropna=True)["profit"].sum()
    revenue_by_group = (
        df.loc[df["ledger_type"] == "revenue"]
        .groupby(group_field, dropna=True)["profit"]
        .sum()
    )
    expenses_by_group = (
        df.loc[df["ledger_type"] == "expenses"]
        .groupby(group_field, dropna=True)["profit"]
        .sum()
    )

    all_labels = profit_by_group.index.union(revenue_by_group.index)

    # Optional pre-filter: keep only groups with above/below average revenue
    if pre_filter in ("above_average_revenue", "below_average_revenue"):
        rev_values = [float(revenue_by_group.get(lb, 0.0)) for lb in all_labels]
        avg_revenue = float(pd.Series(rev_values).mean()) if rev_values else 0.0
        if pre_filter == "above_average_revenue":
            all_labels = [lb for lb in all_labels if float(revenue_by_group.get(lb, 0.0)) > avg_revenue]
        else:
            all_labels = [lb for lb in all_labels if float(revenue_by_group.get(lb, 0.0)) < avg_revenue]
        pre_filter_avg_revenue: Optional[float] = avg_revenue
    else:
        pre_filter_avg_revenue = None

    rows = []
    for label in all_labels:
        total_profit = float(profit_by_group.get(label, 0.0))
        total_revenue = float(revenue_by_group.get(label, 0.0))
        total_expenses = float(expenses_by_group.get(label, 0.0))  # negative in dataset
        if total_revenue <= 0:
            ratio = None
            ratio_pct = None
        elif ratio_type == "expense_ratio":
            ratio = abs(total_expenses) / total_revenue
            ratio_pct = ratio * 100
        else:
            ratio = total_profit / total_revenue
            ratio_pct = ratio * 100
        rows.append({
            "label": str(label),
            "total_profit": total_profit,
            "total_revenue": total_revenue,
            "total_expenses": total_expenses,
            "margin": ratio,
            "margin_pct": ratio_pct,
        })

    rankable = [r for r in rows if r["margin"] is not None]
    if not rankable:
        raise ValueError("No groups have positive revenue, cannot compute ratio.")

    rankable = sorted(rankable, key=lambda r: r["margin"], reverse=(direction == "top"))

    # Include ties at K-th value
    if len(rankable) >= k:
        cutoff = rankable[k - 1]["margin"]
        if direction == "bottom":
            rankable = [r for r in rankable if r["margin"] <= cutoff]
        else:
            rankable = [r for r in rankable if r["margin"] >= cutoff]
    rankable = rankable[: k + 5]

    ranked_rows = [
        {
            "rank": i + 1,
            **r,
        }
        for i, r in enumerate(rankable)
    ]

    result = {
        "type": "margin_analysis",
        "ratio_type": ratio_type,
        "direction": direction,
        "k": k,
        "group_field": group_field,
        "rows": ranked_rows,
        "row_count": len(df),
    }
    if pre_filter:
        result["pre_filter"] = pre_filter
    if pre_filter_avg_revenue is not None:
        result["pre_filter_avg_revenue"] = pre_filter_avg_revenue
    return result


def compute_presence_condition(
    df: pd.DataFrame,
    group_field: str,
    condition: str,
) -> dict:
    """
    Compute groups satisfying query conditions like:
    - expenses_without_revenue
    - revenue_without_expenses
    - below_average_profit
    - above_average_profit
    """
    if group_field not in df.columns:
        raise ValueError(f"Column '{group_field}' not in dataset.")

    if condition.startswith("ledger_group_share:"):
        group_name = condition.split(":", 1)[1].strip()
        # numerator: rows belonging to the specific ledger_group
        numerator = float(df.loc[df["ledger_group"] == group_name, "profit"].sum())
        # denominator: all rows in scope (already filtered to ledger_type by retrieve)
        denominator = float(df["profit"].sum())
        # Signed ratio: when both are negative (expenses convention) the result is
        # already positive.  No abs() — hiding signs can mask accounting errors.
        pct = (numerator / denominator * 100) if denominator != 0.0 else 0.0
        return {
            "type": "ledger_group_share",
            "group_name": group_name,
            "group_total": numerator,
            "scope_total": denominator,
            "pct": pct,
            "row_count": len(df),
        }

    if condition.startswith("ledger_reconciliation:"):
        group_name = condition.split(":", 1)[1].strip()
        # category_total: sum of ALL revenue rows (across every ledger_category)
        category_total = float(df.loc[df["ledger_type"] == "revenue", "profit"].sum())
        # group_total: sum of all rows in the named ledger_group
        group_total = float(df.loc[df["ledger_group"] == group_name, "profit"].sum())
        diff = category_total - group_total
        matches = abs(diff) < 0.01
        return {
            "type": "reconciliation",
            "condition": condition,
            "group_name": group_name,
            "category_total": category_total,
            "group_total": group_total,
            "difference": diff,
            "matches": matches,
            "row_count": len(df),
        }

    if condition in {"below_average_profit", "above_average_profit"}:
        group_profit = df.groupby(group_field, dropna=True)["profit"].sum()
        avg_profit = float(group_profit.mean()) if len(group_profit) > 0 else 0.0
        if condition == "below_average_profit":
            filtered = group_profit[group_profit < avg_profit].sort_values(ascending=True)
        else:
            filtered = group_profit[group_profit > avg_profit].sort_values(ascending=False)
        rows = [{"label": str(label), "profit": float(value)} for label, value in filtered.items()]
        return {
            "type": "stat_condition",
            "condition": condition,
            "group_field": group_field,
            "average_profit": avg_profit,
            "rows": rows,
            "row_count": len(df),
        }

    if condition == "positive_revenue_negative_profit":
        # Groups where total revenue > 0 but total net profit (revenue + expenses) < 0.
        if "ledger_type" not in df.columns:
            raise ValueError("Column 'ledger_type' not in dataset.")
        revenue_by_group = (
            df.loc[df["ledger_type"] == "revenue"]
            .groupby(group_field, dropna=True)["profit"]
            .sum()
        )
        net_profit_by_group = df.groupby(group_field, dropna=True)["profit"].sum()
        all_groups = revenue_by_group.index.union(net_profit_by_group.index)
        rows = []
        for label in all_groups:
            rev = float(revenue_by_group.get(label, 0.0))
            profit = float(net_profit_by_group.get(label, 0.0))
            if rev > 0 and profit < 0:
                rows.append({"label": str(label), "total_revenue": rev, "total_profit": profit})
        rows = sorted(rows, key=lambda r: r["label"])
        return {
            "type": "presence_condition",
            "condition": condition,
            "group_field": group_field,
            "rows": rows,
            "row_count": len(df),
        }

    if condition == "mixed_sign_revenue":
        # Find groups that have at least one positive AND one negative revenue entry.
        if "ledger_type" not in df.columns:
            raise ValueError("Column 'ledger_type' not in dataset.")
        rev = df.loc[(df["ledger_type"] == "revenue") & df[group_field].notna()]
        if rev.empty:
            return {
                "type": "presence_condition",
                "condition": condition,
                "group_field": group_field,
                "rows": [],
                "row_count": len(df),
            }
        agg = rev.groupby(group_field)["profit"].agg(["min", "max"])
        mixed = agg[(agg["min"] < 0) & (agg["max"] > 0)]
        rows = sorted(
            [
                {
                    "label": str(label),
                    "min_revenue": float(row["min"]),
                    "max_revenue": float(row["max"]),
                }
                for label, row in mixed.iterrows()
            ],
            key=lambda r: r["label"],
        )
        return {
            "type": "presence_condition",
            "condition": condition,
            "group_field": group_field,
            "rows": rows,
            "row_count": len(df),
        }

    if "ledger_type" not in df.columns:
        raise ValueError("Column 'ledger_type' not in dataset.")

    sums = (
        df.groupby([group_field, "ledger_type"], dropna=True)["profit"]
        .sum()
        .unstack(fill_value=0.0)
    )
    counts = (
        df.groupby([group_field, "ledger_type"], dropna=True)
        .size()
        .unstack(fill_value=0)
    )

    rows = []
    for label in sums.index:
        revenue = float(sums.loc[label].get("revenue", 0.0))
        expenses = float(sums.loc[label].get("expenses", 0.0))
        has_revenue = int(counts.loc[label].get("revenue", 0)) > 0
        has_expenses = int(counts.loc[label].get("expenses", 0)) > 0

        match = False
        if condition == "expenses_without_revenue":
            match = has_expenses and not has_revenue
        elif condition == "revenue_without_expenses":
            match = has_revenue and not has_expenses
        else:
            raise ValueError(f"Unsupported query condition: {condition}")

        if match:
            rows.append({
                "label": str(label),
                "total_revenue": revenue,
                "total_expenses": expenses,
            })

    rows = sorted(rows, key=lambda r: r["label"])

    return {
        "type": "presence_condition",
        "condition": condition,
        "group_field": group_field,
        "rows": rows,
        "row_count": len(df),
    }


def compute_full_review(df: pd.DataFrame) -> dict:
    """
    Comprehensive multi-section report:
    - Overall P&L (total profit / revenue / expenses)
    - Profit by property
    - Profit by tenant
    - Profit by year (when multiple years present)
    """
    total = compute_pnl_total(df)
    by_property = compute_grouped(df, "property_name")
    by_tenant = compute_grouped(df, "tenant_name")

    result: dict = {
        "type": "full_review",
        "total": total,
        "by_property": by_property,
        "by_tenant": by_tenant,
        "row_count": len(df),
    }

    if "year" in df.columns and df["year"].nunique() > 1:
        result["by_year"] = compute_grouped(df, "year")

    return result


def compute_period_delta(
    df: pd.DataFrame,
    timeframe_a: Optional[dict],
    timeframe_b: Optional[dict],
    group_field: str,
    k: int = 999,
    direction: str = "top",
) -> dict:
    """
    Period-over-period comparison: profit grouped by group_field for period A vs period B.
    Returns rows sorted by delta descending (or ascending when direction='bottom'), limited to k.
    """
    if not timeframe_a or not timeframe_b:
        raise ValueError("period_comparison requires both timeframe_a and timeframe_b.")
    if group_field not in df.columns:
        raise ValueError(f"Column '{group_field}' not in dataset.")

    df_a = apply_timeframe_filter(df, timeframe_a)
    df_b = apply_timeframe_filter(df, timeframe_b)

    grp_a = df_a.groupby(group_field, dropna=True)["profit"].sum()
    grp_b = df_b.groupby(group_field, dropna=True)["profit"].sum()

    all_labels = grp_a.index.union(grp_b.index)
    rows = []
    for label in all_labels:
        a = float(grp_a.get(label, 0.0))
        b = float(grp_b.get(label, 0.0))
        delta = b - a  # positive = increased from period_a to period_b
        pct = (delta / abs(a) * 100) if a != 0.0 else (float("inf") if delta > 0 else 0.0)
        rows.append({
            "label": str(label),
            "period_a": a,
            "period_b": b,
            "delta": delta,
            "pct_delta": pct,
        })

    rows = sorted(rows, key=lambda r: r["delta"], reverse=(direction != "bottom"))
    if k < len(rows):
        rows = rows[:k]

    return {
        "type": "period_delta",
        "group_field": group_field,
        "period_a_label": timeframe_a.get("label", "Period A"),
        "period_b_label": timeframe_b.get("label", "Period B"),
        "rows": rows,
        "k": k,
        "direction": direction,
        "row_count": len(df),
    }


def compute_share_of_total(
    df: pd.DataFrame,
    targets: list[str],
    target_fields: list[str],
) -> dict:
    """
    Compute what fraction (%) a specific target contributes to the total.
    df must contain ALL rows in scope (no target filter applied at retrieval time).
    """
    if not targets or not target_fields:
        raise ValueError("share_of_total requires at least one target.")

    target = targets[0]
    field = target_fields[0]

    if field not in df.columns:
        raise ValueError(f"Column '{field}' not in dataset.")

    # Split: target rows vs. full scope
    mask = df[field] == target
    df_target = df[mask]
    if df_target.empty:
        raise ValueError(f"No data found for {target}.")

    total = float(df["profit"].sum())
    target_total = float(df_target["profit"].sum())

    # Revenue and expense subtotals
    total_rev = float(df.loc[df["ledger_type"] == "revenue", "profit"].sum())
    target_rev = float(df_target.loc[df_target["ledger_type"] == "revenue", "profit"].sum())
    total_exp = float(df.loc[df["ledger_type"] == "expenses", "profit"].sum())
    target_exp = float(df_target.loc[df_target["ledger_type"] == "expenses", "profit"].sum())

    def _pct(part: float, whole: float) -> float:
        return (part / whole * 100) if whole != 0.0 else 0.0

    return {
        "type": "share_of_total",
        "target": target,
        "field": field,
        "target_profit": target_total,
        "total_profit": total,
        "pct_profit": _pct(target_total, total),
        "target_revenue": target_rev,
        "total_revenue": total_rev,
        "pct_revenue": _pct(target_rev, total_rev),
        "target_expenses": abs(target_exp),
        "total_expenses": abs(total_exp),
        "pct_expenses": _pct(abs(target_exp), abs(total_exp)),
        "row_count": len(df),
    }


def compute_count(df: pd.DataFrame, field: str) -> dict:
    """Count distinct values of a field in the filtered dataset."""
    if field not in df.columns:
        raise ValueError(f"Column '{field}' not in dataset.")
    values = sorted(df[field].dropna().unique().tolist())
    return {
        "type": "count",
        "field": field,
        "count": len(values),
        "values": values,
        "row_count": len(df),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timeframe_summary(df: pd.DataFrame) -> dict:
    """Summarise the time coverage of the df."""
    result: dict = {}
    for col in ("year", "quarter", "month"):
        if col in df.columns:
            vals = sorted(df[col].dropna().unique().tolist())
            result[col] = vals
    return result
