"""
ExtractionAgent: Extract targets, timeframe, filters, and identify missing fields.
May call interrupt() when required fields are missing.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from ..core.schemas import EXTRACTION_SYSTEM_PROMPT, ExtractionOutput, LedgerFilters, TimeframeOutput
from ..core.state import AgentState, Intent
from ..tools.df_tools import list_unique

logger = logging.getLogger(__name__)

# Intents that REQUIRE at least one time dimension
# Ranking without a timeframe is valid — it means "across all available data"
TIMEFRAME_REQUIRED = {
    Intent.PNL_TOTAL, Intent.PNL_BY_PROPERTY, Intent.PNL_BY_TENANT,
    Intent.BREAKDOWN, Intent.COMPARISON,
}

# Intents that REQUIRE at least one target
TARGET_REQUIRED = {Intent.COMPARISON}

# Phrases the user may type to mean "no timeframe filter"
_ALL_TIME_PHRASES = ("all", "all time", "all times", "all data", "everything",
                     "any", "any time", "no filter", "no timeframe")


def extract_node(state: AgentState, llm: Any) -> dict:
    """
    Extract entities from user query, then check for missing required fields.
    If a required field is missing, interrupt the graph to ask ONE clarifying question.
    """
    trace = list(state.get("node_trace", []))
    trace.append("extract")

    user_query = state.get("user_query", "")
    intent = state.get("intent", Intent.GENERAL_QUESTION)
    clarification_answer = state.get("_clarification_answer")

    # Build context-enriched query for the LLM
    today = date.today()
    conv_ctx = state.get("conversation_context") or ""
    context = (
        f"Intent: {intent}\n"
        f"Current date: {today.isoformat()} "
        f"(current year={today.year}, current quarter=Q{(today.month-1)//3+1}, "
        f"current month=M{today.month:02d})\n"
    )
    if conv_ctx:
        context += f"Recent conversation (use to resolve pronouns like 'he/she/it/they/that'):\n{conv_ctx}\n"
    context += f"User query: {user_query}"
    if clarification_answer:
        context += f"\nUser clarification: {clarification_answer}"

    extracted = _run_llm_extraction(context, llm)

    if extracted is None:
        extracted = _heuristic_extraction(user_query, intent, today, conv_ctx)

    # Build partial state update from extraction
    update: dict = {
        "node_trace": trace,
        "targets": extracted.targets,
        "target_fields": extracted.target_fields,
        "group_by": extracted.group_by,
        "query_condition": None,
        "ledger_filters": _ledger_filters_to_dict(extracted.ledger_filters),
        "k": extracted.k,
        "rank_direction": extracted.rank_direction,
        "missing_fields": [],
    }

    # Preserve explicit named entities/properties/tenants from user text even if LLM omits them.
    explicit_targets, explicit_target_fields = _extract_explicit_targets_from_query(user_query)
    if explicit_targets:
        merged_pairs = list(zip(update.get("targets", []), update.get("target_fields", [])))
        for t, f in zip(explicit_targets, explicit_target_fields):
            if (t, f) not in merged_pairs:
                merged_pairs.append((t, f))
        update["targets"] = [t for t, _ in merged_pairs]
        update["target_fields"] = [f for _, f in merged_pairs]

    # Detect boolean/statistical conditions (e.g., "expenses but no revenue", "below-average net profit").
    condition = _detect_query_condition(user_query)
    if condition:
        update["query_condition"] = condition
        # Condition queries need full rows in timeframe scope for compute checks.
        update["ledger_filters"] = {}
        q = user_query.lower()
        if re.search(r"\bmonth\b", q):
            update["group_by"] = "month"
        elif re.search(r"\bquarter\b|\bq[1-4]\b", q):
            update["group_by"] = "quarter"
        elif re.search(r"\byear\b", q):
            update["group_by"] = "year"
        elif re.search(r"\btenant\b", q):
            update["group_by"] = "tenant_name"
        elif re.search(r"\b(property|properties|building|buildings|asset|assets)\b", q):
            update["group_by"] = "property_name"
        elif re.search(r"\b(entity|entities|propco)\b", q):
            update["group_by"] = "entity_name"
        elif not update.get("group_by"):
            update["group_by"] = "property_name"

    # Explicit ledger grouping phrases in the user query should override noisy LLM output.
    explicit_ledger_group_by = _infer_explicit_ledger_group_by(user_query)
    if explicit_ledger_group_by:
        update["group_by"] = explicit_ledger_group_by

    # Ranking "winner" questions should return one result unless user explicitly asks for top/bottom N.
    explicit_k = _extract_explicit_k(user_query)
    if explicit_k is not None:
        update["k"] = explicit_k
    elif intent == Intent.RANKING and _is_single_winner_ranking_query(user_query):
        update["k"] = 1
    if intent == Intent.RANKING:
        inferred_dir = _infer_rank_direction_from_query(user_query)
        if inferred_dir:
            update["rank_direction"] = inferred_dir
        inferred_time_gb = _infer_time_group_by(user_query)
        if inferred_time_gb:
            update["group_by"] = inferred_time_gb

    # Single-winner BREAKDOWN: "which category/type contributed the most/least"
    # Mirrors the ranking single-winner detection but for breakdown intent.
    if intent == Intent.BREAKDOWN:
        q_lower_bd = user_query.lower()
        has_which = bool(re.search(r"\b(which|what)\b", q_lower_bd))
        has_superlative = bool(re.search(
            r"\b(most|highest|largest|biggest|best|lowest|least|smallest|worst)\b", q_lower_bd
        ))
        if has_which and has_superlative and explicit_k is None:
            update["k"] = 1
            inferred_dir = _infer_rank_direction_from_query(user_query)
            if inferred_dir:
                update["rank_direction"] = inferred_dir

    # COUNT intent: determine which entity type to count (overrides LLM group_by noise).
    if intent == Intent.COUNT:
        q_c = user_query.lower()
        if re.search(r"\b(tenant|tenants)\b", q_c):
            update["group_by"] = "tenant_name"
        elif re.search(r"\b(property|properties|building|buildings|asset|assets)\b", q_c):
            update["group_by"] = "property_name"
        elif re.search(r"\b(entity|entities)\b", q_c):
            update["group_by"] = "entity_name"
        else:
            update["group_by"] = "tenant_name"  # safe default

    # Margin analysis needs deterministic normalisation (LLM often returns breakdown/group_by month).
    if intent == Intent.MARGIN_ANALYSIS:
        _normalise_margin_analysis_extraction(update, user_query)

    # Resolve timeframe
    tf_dict = _timeframe_model_to_dict(extracted.timeframe)
    if not tf_dict:
        # LLMs can miss explicit dates for ranking/comparison prompts; apply deterministic parse.
        tf_dict = _parse_timeframe_from_text(user_query, today)
    if not tf_dict and clarification_answer:
        tf_dict = _parse_timeframe_from_text(clarification_answer, today)
    update["timeframe"] = tf_dict

    # --- Period comparison: detect "from X to Y" pattern and extract two timeframes ---
    # For period_comparison intent, timeframe_a and timeframe_b drive compute; timeframe
    # (used by retrieve) is set to the containing year so all needed rows are fetched.
    update["timeframe_a"] = None
    update["timeframe_b"] = None
    if intent == Intent.PERIOD_COMPARISON:
        period_a, period_b = _parse_period_pair(user_query, today)
        if period_a and period_b:
            update["timeframe_a"] = period_a
            update["timeframe_b"] = period_b
            # Retrieve the containing year (drop quarter/month so we get both periods)
            year_a = period_a.get("year") or (period_a.get("quarter") or "")[:4] or (period_a.get("month") or "")[:4]
            update["timeframe"] = {"year": year_a, "label": year_a} if year_a else None
        else:
            logger.warning("EXTRACT | period_comparison but couldn't parse two periods from %r", user_query)
            # Self-heal: "Which month/quarter/year had the lowest/highest profit in 2024?"
            # is a ranking over time buckets, not a period-to-period delta query.
            if _is_time_bucket_extreme_query_local(user_query):
                update["intent"] = Intent.RANKING
                update["group_by"] = _infer_time_group_by(user_query) or update.get("group_by") or "month"
                inferred_dir = _infer_rank_direction_from_query(user_query)
                if inferred_dir:
                    update["rank_direction"] = inferred_dir
                if explicit_k is None:
                    update["k"] = 1
                logger.warning(
                    "EXTRACT | converted period_comparison -> ranking for temporal-superlative query %r",
                    user_query,
                )

    effective_intent = update.get("intent", intent)

    # Defensive: if group_by is a time dimension AND the timeframe is at the SAME granularity,
    # clear the timeframe — it would be redundant (e.g. filtering to year=2024 while grouping
    # by year only shows 2024, pointless). But keep it when the timeframe is a CONTAINING
    # scope: "monthly breakdown in 2024" → group_by=month, timeframe=year → KEEP so retrieve
    # filters to 2024 rows before compute groups them by month.
    if update.get("group_by") in ("year", "quarter", "month") and update.get("timeframe"):
        gb = update["group_by"]
        tf = update["timeframe"]
        same_granularity = (
            (gb == "year" and tf.get("year") and not tf.get("quarter") and not tf.get("month"))
            or (gb == "quarter" and bool(tf.get("quarter")))
            or (gb == "month" and bool(tf.get("month")))
        )
        if same_granularity:
            logger.warning(
                "EXTRACT | group_by=%s matches timeframe granularity %s — clearing timeframe (redundant)",
                gb, tf,
            )
            update["timeframe"] = None

    # Defensive: for ranking intent, targets should be empty — we rank ALL entities.
    # The LLM may hallucinate a target from the dataset context ("Tenant 1" from
    # "Tenants: Tenant 1–18"), but ranking means comparing everyone, not filtering to one.
    if effective_intent == Intent.RANKING and update.get("targets"):
        pairs = list(zip(update.get("targets", []), update.get("target_fields", [])))
        explicit_pairs = set(zip(explicit_targets, explicit_target_fields))
        kept = [(t, f) for (t, f) in pairs if (t, f) in explicit_pairs]
        if kept:
            if len(kept) != len(pairs):
                logger.warning(
                    "EXTRACT | ranking intent: cleared non-explicit targets; kept %s",
                    kept,
                )
            update["targets"] = [t for t, _ in kept]
            update["target_fields"] = [f for _, f in kept]
        else:
            logger.warning(
                "EXTRACT | ranking intent: clearing targets %s (should rank all entities)",
                update["targets"],
            )
            update["targets"] = []
            update["target_fields"] = []

    # Defensive: for ranking/pnl_total/pnl_by_* without explicit ledger keywords,
    # clear ledger_filters that were inferred from vague words like "paid"/"earned".
    # Only keep filters if the user's query actually contains ledger terminology.
    _LEDGER_KEYWORDS = (
        "ledger", "ledger_type", "ledger_group", "ledger_category",
        "rental income", "rent", "management fee", "bank charge",
        "expense", "revenue", "income", "fee", "charge",
    )
    if update.get("ledger_filters") and not any(kw in user_query.lower() for kw in _LEDGER_KEYWORDS):
        logger.warning(
            "EXTRACT | clearing ledger_filters %s — no explicit ledger keywords in query %r",
            update["ledger_filters"], user_query,
        )
        update["ledger_filters"] = {}

    # Defensive: if ledger_group is already set, strip ledger_category and ledger_code.
    # "rental income" → ledger_group=rental_income is the right granularity; the LLM
    # should not infer sub-category codes (e.g. revenue_rent_taxed) that the user never mentioned.
    lf = update.get("ledger_filters") or {}
    if lf.get("ledger_group") and (lf.get("ledger_category") or lf.get("ledger_code")):
        logger.warning(
            "EXTRACT | clearing ledger_category/code — ledger_group already set, "
            "no explicit category terms in query %r",
            user_query,
        )
        lf.pop("ledger_category", None)
        lf.pop("ledger_code", None)
        update["ledger_filters"] = lf

    # Defensive: clear ledger_group unless the query contains a group-specific term.
    # Generic type words ("revenue", "expenses") justify ledger_type only, NOT ledger_group.
    # e.g. "total revenue from Building 120" → ledger_type=revenue only, NOT rental_income.
    _LEDGER_GROUP_KEYWORDS = (
        "rental income", "rent income", "rent revenue",
        "management fee", "management fees",
        "bank charge", "bank charges", "bank fee",
        "service charge", "service charges",
        "maintenance", "insurance",
        "utilities", "interest income",
        "legal fee", "accounting fee",
    )
    lf = update.get("ledger_filters") or {}
    if lf.get("ledger_group"):
        lg_value = lf["ledger_group"]
        # Accept the group if: (a) a keyword from the whitelist appears in the query, OR
        # (b) the actual ledger_group value itself (underscores → spaces) appears in the query.
        # This lets "general expenses" match ledger_group=general_expenses without needing
        # to enumerate every possible group name in the whitelist.
        lg_humanized = lg_value.replace("_", " ").lower()
        kw_match = any(kw in user_query.lower() for kw in _LEDGER_GROUP_KEYWORDS)
        value_match = lg_humanized in user_query.lower()
        if not kw_match and not value_match:
            logger.warning(
                "EXTRACT | clearing ledger_group=%s — no group-specific keywords in query %r",
                lg_value, user_query,
            )
            lf.pop("ledger_group", None)
            update["ledger_filters"] = lf

    # Defensive: "revenues minus/less expenses" or explicit "net profit" in the query means the
    # user is defining full P&L, not filtering to revenue only. Clear a mistakenly set
    # ledger_type=revenue so the ranking/computation uses both sides of the P&L.
    lf = update.get("ledger_filters") or {}
    if lf.get("ledger_type") == "revenue":
        q_lc = user_query.lower()
        defines_net_profit = bool(
            re.search(r"\bnet\s+profit\b", q_lc)
            or re.search(r"\brevenu\w*\s+(minus|less|subtract\w*)\s+expens", q_lc)
            or re.search(r"\bexpens\w*\s+(subtracted|deducted)\s+from\s+revenu", q_lc)
        )
        if defines_net_profit:
            logger.warning(
                "EXTRACT | clearing ledger_type=revenue — query defines net profit "
                "(revenue-expenses), not a revenue-only filter: %r",
                user_query,
            )
            lf.pop("ledger_type", None)
            update["ledger_filters"] = lf

    # Canonicalize common natural-language ledger terms to actual dataset enums.
    _canonicalize_ledger_filters(update, user_query)

    # Defensive: for share_of_total intent, group_by is not used by computation — clear it.
    if effective_intent == Intent.SHARE_OF_TOTAL and update.get("group_by"):
        logger.warning(
            "EXTRACT | share_of_total: clearing group_by=%s (not applicable for share computation)",
            update["group_by"],
        )
        update["group_by"] = None

    # Supplementary pronoun resolution: if targets are still empty after LLM extraction
    # and the query contains a pronoun, scan conversation context for entity mentions.
    # This runs even when LLM extraction succeeds but fails to resolve pronouns
    # (e.g. "Where does he live?" after "Tenant 7 paid the most").
    # NOTE: Use word-boundary regex to avoid false matches like "he" inside "the".
    # NOTE: Skip for ranking intent — ranking always operates on ALL entities.
    _PRONOUN_RE = re.compile(r"\b(he|she|they|his|her|their|him|it)\b")
    q_lower = user_query.lower()
    if (intent != Intent.RANKING
            and not update.get("targets") and conv_ctx
            and _PRONOUN_RE.search(q_lower)):
        tenants = list_unique("tenant_name")
        properties = list_unique("property_name")
        ctx_lower = conv_ctx.lower()
        for t in tenants:
            if t.lower() in ctx_lower:
                update["targets"] = [t]
                update["target_fields"] = ["tenant_name"]
                logger.warning("EXTRACT | pronoun resolved → %s (tenant) from conv_ctx", t)
                break
        if not update.get("targets"):
            for p in properties:
                if p.lower() in ctx_lower:
                    update["targets"] = [p]
                    update["target_fields"] = ["property_name"]
                    logger.warning("EXTRACT | pronoun resolved → %s (property) from conv_ctx", p)
                    break
        if not update.get("targets"):
            entities = list_unique("entity_name")
            for e in entities:
                if e.lower() in ctx_lower:
                    update["targets"] = [e]
                    update["target_fields"] = ["entity_name"]
                    logger.warning("EXTRACT | pronoun resolved → %s (entity) from conv_ctx", e)
                    break

    # Debug: log what was extracted
    logger.warning(
        "EXTRACT | intent=%s | targets=%s | group_by=%s | condition=%s | timeframe=%s | ledger_filters=%s",
        effective_intent,
        list(zip(update.get("targets", []), update.get("target_fields", []))),
        update.get("group_by"),
        update.get("query_condition"),
        update.get("timeframe"),
        update.get("ledger_filters"),
    )

    # Defensive: validate timeframe has actual data — prevents 0-row retrieval
    # when LLM defaults to current year (e.g. 2026) which has no data in the dataset.
    if update.get("timeframe"):
        from ..tools.df_tools import query_ledger, get_dataframe
        try:
            test_rows = len(query_ledger(timeframe=update["timeframe"], df=get_dataframe()))
            if test_rows == 0:
                logger.warning(
                    "EXTRACT | timeframe %s has 0 rows in dataset — clearing it",
                    update["timeframe"],
                )
                update["timeframe"] = None
        except Exception:
            pass

    # ---------------------------------------------------------------------------
    # Check for missing required fields → interrupt if needed
    # ---------------------------------------------------------------------------
    missing = _find_missing_fields(effective_intent, update)

    if missing:
        question = _build_clarification_question(missing[0], effective_intent)
        update["missing_fields"] = missing
        update["clarification_question"] = question

        # Interrupt: pause graph, surface question to user, resume with answer
        answer = interrupt(question)

        # When resumed, merge the user's answer into state and re-extract
        update["_clarification_answer"] = answer
        update["missing_fields"] = []

        # Handle "all times" style answers without needing LLM
        if missing[0] == "timeframe" and any(ph in answer.lower() for ph in _ALL_TIME_PHRASES):
            update["timeframe"] = {"label": "all time"}
        else:
            # Re-invoke extraction with the answer to fill missing field
            context_with_answer = context + f"\nUser clarification: {answer}"
            re_extracted = _run_llm_extraction(context_with_answer, llm)
            if re_extracted is None:
                # Heuristic fallback for re-extraction (LLM unavailable)
                re_extracted = _heuristic_extraction(answer, intent, date.today())
            if re_extracted:
                if re_extracted.timeframe and "timeframe" in missing[0]:
                    tf = re_extracted.timeframe
                    if tf.year or tf.quarter or tf.month:
                        tf_dict = {k: v for k, v in {
                            "year": tf.year, "quarter": tf.quarter,
                            "month": tf.month, "ytd": tf.ytd, "label": tf.label,
                        }.items() if v is not None and v != ""}
                        update["timeframe"] = tf_dict
                if re_extracted.targets and "target" in missing[0]:
                    update["targets"] = re_extracted.targets
                    update["target_fields"] = re_extracted.target_fields
                # Always sync group_by from re-extraction if it provides one
                if re_extracted.group_by:
                    update["group_by"] = re_extracted.group_by
            if "timeframe" in missing[0] and not update.get("timeframe"):
                parsed = _parse_timeframe_from_text(answer, date.today())
                if parsed:
                    update["timeframe"] = parsed

    else:
        update["missing_fields"] = []
        update["clarification_question"] = None

    return update


def _ledger_filters_to_dict(lf: Any) -> dict:
    """Serialise LedgerFilters model (or plain dict) to a plain dict for state storage."""
    if hasattr(lf, "model_dump"):
        return lf.model_dump(exclude_none=True)
    return lf or {}


def _timeframe_model_to_dict(tf: TimeframeOutput | None) -> dict | None:
    """Convert TimeframeOutput to compact dict; return None if empty."""
    if not tf:
        return None
    if not (tf.year or tf.quarter or tf.month):
        return None
    tf_dict = {
        "year": tf.year,
        "quarter": tf.quarter,
        "month": tf.month,
        "ytd": tf.ytd,
        "label": tf.label,
    }
    return {k: v for k, v in tf_dict.items() if v is not None and v != ""}


def _dict_to_timeframe_model(tf: dict | None) -> TimeframeOutput | None:
    """Convert timeframe dict to TimeframeOutput model for heuristic extraction."""
    if not tf:
        return None
    return TimeframeOutput(
        year=tf.get("year"),
        quarter=tf.get("quarter"),
        month=tf.get("month"),
        ytd=bool(tf.get("ytd", False)),
        label=tf.get("label", ""),
    )


def _parse_timeframe_from_text(text: str, today: date) -> dict | None:
    """Deterministically parse explicit timeframe phrases from user text."""
    q = (text or "").strip()
    if not q:
        return None
    ql = q.lower()

    # Explicit month: 2024-M06 / 2024 M06
    m_match = re.search(r"\b(20\d{2})\s*[- ]?\s*m(0[1-9]|1[0-2])\b", ql, re.IGNORECASE)
    if m_match:
        year, month = m_match.group(1), m_match.group(2)
        month_key = f"{year}-M{month}"
        return {"month": month_key, "label": month_key}

    # Explicit quarter: Q3 2024 / 2024 Q3 / 2024-Q3
    q_match = re.search(r"\bq([1-4])\s*(20\d{2})\b|\b(20\d{2})\s*[- ]?\s*q([1-4])\b", ql, re.IGNORECASE)
    if q_match:
        if q_match.group(1):
            quarter, year = q_match.group(1), q_match.group(2)
        else:
            year, quarter = q_match.group(3), q_match.group(4)
        quarter_key = f"{year}-Q{quarter}"
        return {"quarter": quarter_key, "label": f"Q{quarter} {year}"}

    # Relative phrases
    if "last year" in ql:
        year = str(today.year - 1)
        return {"year": year, "label": year}
    if "this year" in ql or "ytd" in ql or "year to date" in ql:
        year = str(today.year)
        return {"year": year, "ytd": True, "label": f"{year} YTD"}
    if "this quarter" in ql:
        quarter_num = (today.month - 1) // 3 + 1
        quarter_key = f"{today.year}-Q{quarter_num}"
        return {"quarter": quarter_key, "label": quarter_key}
    if "last quarter" in ql:
        quarter_num = (today.month - 1) // 3 + 1
        if quarter_num == 1:
            year = today.year - 1
            quarter_num = 4
        else:
            year = today.year
            quarter_num -= 1
        quarter_key = f"{year}-Q{quarter_num}"
        return {"quarter": quarter_key, "label": quarter_key}

    # Explicit year (least specific)
    y_match = re.search(r"\b(20\d{2})\b", q)
    if y_match:
        year = y_match.group(1)
        return {"year": year, "label": year}

    return None


def _parse_period_pair(text: str, today: date) -> tuple[dict | None, dict | None]:
    """
    Detect period-over-period patterns and return (period_a, period_b).
    Handles:
      "from Q1 2024 to Q4 2024"
      "from 2024 to 2025"
      "Q1 2024 to Q4 2024"
      "between Q1 and Q4 2024"
    Returns (None, None) if not a two-period query.
    """
    ql = (text or "").lower()

    # "from Q1 YYYY to Q4 YYYY" or "Q1 YYYY to Q4 YYYY"
    q_to_q = re.search(
        r"\bq([1-4])\s*(20\d{2})\s+(?:to|and|through)\s+q([1-4])\s*(20\d{2})\b",
        ql,
    )
    if q_to_q:
        qa_num, ya = q_to_q.group(1), q_to_q.group(2)
        qb_num, yb = q_to_q.group(3), q_to_q.group(4)
        pa = {"quarter": f"{ya}-Q{qa_num}", "year": ya, "label": f"Q{qa_num} {ya}"}
        pb = {"quarter": f"{yb}-Q{qb_num}", "year": yb, "label": f"Q{qb_num} {yb}"}
        return pa, pb

    # "from Q1 to Q4 2024" (single year, two quarters)
    q_to_q_sy = re.search(
        r"\bq([1-4])\s+(?:to|and|through)\s+q([1-4])\s+(20\d{2})\b",
        ql,
    )
    if not q_to_q_sy:
        # Also "from Q1 to Q4 of 2024" or "between Q1 and Q4 2024"
        q_to_q_sy = re.search(
            r"\b(from\s+)?q([1-4])\s+(?:to|and|through)\s+q([1-4])(?:\s+(?:of\s+)?(20\d{2}))?\b",
            ql,
        )
        if q_to_q_sy and q_to_q_sy.group(4):
            qa_num = q_to_q_sy.group(2)
            qb_num = q_to_q_sy.group(3)
            year = q_to_q_sy.group(4)
            pa = {"quarter": f"{year}-Q{qa_num}", "year": year, "label": f"Q{qa_num} {year}"}
            pb = {"quarter": f"{year}-Q{qb_num}", "year": year, "label": f"Q{qb_num} {year}"}
            return pa, pb
        # fallback: find ANY year nearby
        y_near = re.search(r"\b(20\d{2})\b", ql)
        if q_to_q_sy and y_near:
            qa_num = q_to_q_sy.group(2)
            qb_num = q_to_q_sy.group(3)
            year = y_near.group(1)
            pa = {"quarter": f"{year}-Q{qa_num}", "year": year, "label": f"Q{qa_num} {year}"}
            pb = {"quarter": f"{year}-Q{qb_num}", "year": year, "label": f"Q{qb_num} {year}"}
            return pa, pb
    elif q_to_q_sy:
        qa_num, qb_num, year = q_to_q_sy.group(1), q_to_q_sy.group(2), q_to_q_sy.group(3)
        pa = {"quarter": f"{year}-Q{qa_num}", "year": year, "label": f"Q{qa_num} {year}"}
        pb = {"quarter": f"{year}-Q{qb_num}", "year": year, "label": f"Q{qb_num} {year}"}
        return pa, pb

    # "from YYYY to YYYY" (year-over-year)
    y_to_y = re.search(r"\b(from\s+)?(20\d{2})\s+(?:to|and|through|vs\.?|versus)\s+(20\d{2})\b", ql)
    if y_to_y:
        ya, yb = y_to_y.group(2), y_to_y.group(3)
        pa = {"year": ya, "label": ya}
        pb = {"year": yb, "label": yb}
        return pa, pb

    return None, None


def _extract_explicit_k(text: str) -> int | None:
    """Extract explicit K from phrases like 'top 5', 'bottom 3', '5 best'."""
    q = (text or "").lower()
    match = re.search(r"\btop\s*(\d+)\b|\bbottom\s*(\d+)\b|\b(\d+)\s*(?:best|worst|top|bottom)\b", q)
    if not match:
        return None
    return int(match.group(1) or match.group(2) or match.group(3))


def _is_single_winner_ranking_query(text: str) -> bool:
    """True for singular winner questions like 'Which tenant paid the most?'."""
    q = (text or "").lower().strip()
    if not q:
        return False
    if _extract_explicit_k(q) is not None:
        return False

    singular_subject = bool(
        re.search(r"\b(which|what)\s+(tenant|property|building|asset|month|quarter|year)\b", q)
        or re.search(r"\bwho\b", q)
    )
    superlative = bool(
        re.search(r"\b(most|highest|best|lowest|least|worst)\b", q)
        or re.search(r"\bnumber\s+one\b", q)
        or re.search(r"#1\b", q)
        or re.search(r"\b1st\b", q)
    )

    return singular_subject and superlative


def _normalise_margin_analysis_extraction(update: dict, user_query: str) -> None:
    """Normalise extraction output for margin-analysis questions."""
    q = (user_query or "").lower()

    # Margin should use full P&L scope; do not keep revenue/expense-only filters.
    update["ledger_filters"] = {}

    # Determine grouping dimension for margin ranking.
    if re.search(r"\btenant\b", q):
        update["group_by"] = "tenant_name"
    elif re.search(r"\b(property|building|asset)\b", q):
        update["group_by"] = "property_name"
    elif update.get("group_by") not in ("property_name", "tenant_name", "entity_name"):
        update["group_by"] = "property_name"

    # Direction from superlative language.
    if re.search(r"\b(worst|lowest|least)\b", q):
        update["rank_direction"] = "bottom"
    elif re.search(r"\b(best|highest|most)\b", q):
        update["rank_direction"] = "top"

    explicit_k = _extract_explicit_k(user_query)
    if explicit_k is not None:
        update["k"] = explicit_k
    elif _is_single_winner_ranking_query(user_query):
        update["k"] = 1

    # If no real named target appears in the query, clear hallucinated targets for ranking-style margin queries.
    if _is_single_winner_ranking_query(user_query) and not _query_mentions_known_target(user_query):
        update["targets"] = []
        update["target_fields"] = []


def _canonicalize_ledger_filters(update: dict, user_query: str) -> None:
    """
    Map natural language expense phrases to dataset-specific ledger values.
    """
    lf = dict(update.get("ledger_filters") or {})
    q = (user_query or "").lower()

    # "insurance expenses" should target the concrete category value in dataset.
    if re.search(r"\binsurance\b", q):
        if not lf.get("ledger_type"):
            lf["ledger_type"] = "expenses"
        lf["ledger_category"] = "insurance_in_general"
        # Avoid broader group filter (taxes+insurances) when user explicitly asked insurance.
        if lf.get("ledger_group") and "insurance" in str(lf["ledger_group"]).lower():
            lf.pop("ledger_group", None)

    update["ledger_filters"] = lf


def _query_mentions_known_target(query: str) -> bool:
    """Check if query explicitly contains any known property/tenant/entity name."""
    q = (query or "").lower()
    for field in ("property_name", "tenant_name", "entity_name"):
        for value in list_unique(field):
            if str(value).lower() in q:
                return True
    return False


def _detect_query_condition(query: str) -> str | None:
    """
    Detect query conditions that require deterministic post-aggregation filtering.
    """
    q = (query or "").lower()
    if not q:
        return None

    has_expense = bool(re.search(r"\b(expense|expenses|cost|costs)\b", q))
    has_revenue = bool(re.search(r"\b(revenue|income)\b", q))
    has_profit = bool(re.search(r"\b(net profit|profit|pnl|p&l)\b", q))
    no_revenue = bool(re.search(r"\b(no|without)\s+revenue\b|\bbut\s+no\s+revenue\b|\band\s+no\s+revenue\b", q))
    no_expenses = bool(re.search(r"\b(no|without)\s+expenses?\b|\bbut\s+no\s+expenses?\b|\band\s+no\s+expenses?\b", q))
    below_avg = bool(re.search(r"\bbelow[- ]average\b|\bunder[- ]average\b|\bless than average\b", q))
    above_avg = bool(re.search(r"\babove[- ]average\b|\bover[- ]average\b|\bmore than average\b|\bhigher than average\b", q))

    if has_expense and no_revenue:
        return "expenses_without_revenue"
    if has_revenue and no_expenses:
        return "revenue_without_expenses"
    if has_profit and below_avg:
        return "below_average_profit"
    if has_profit and above_avg:
        return "above_average_profit"
    return None


def _infer_rank_direction_from_query(query: str) -> str | None:
    """Infer ranking direction from superlative language."""
    q = (query or "").lower()
    if re.search(r"\b(lowest|least|worst|bottom)\b", q):
        return "bottom"
    if re.search(r"\b(highest|most|best|top)\b", q):
        return "top"
    return None


def _infer_time_group_by(query: str) -> str | None:
    """Infer time bucket group_by from query text."""
    q = (query or "").lower()
    if re.search(r"\bmonth\b", q):
        return "month"
    if re.search(r"\bquarter\b|\bq[1-4]\b", q):
        return "quarter"
    if re.search(r"\byear\b", q):
        return "year"
    return None


def _infer_explicit_ledger_group_by(query: str) -> str | None:
    """Infer explicit ledger grouping dimension from user wording."""
    q = (query or "").lower()
    if not q:
        return None

    if re.search(r"\bledger[\s_-]*categor(?:y|ies)\b", q):
        return "ledger_category"
    # "expense category", "cost category", "by category", "per category" → ledger_category
    if re.search(r"\b(expense|cost|revenue|income)[\s_-]*categor(?:y|ies)\b", q):
        return "ledger_category"
    if re.search(r"\b(by|per|each)\s+categor(?:y|ies)\b", q):
        return "ledger_category"
    if re.search(r"\bledger[\s_-]*group(?:s)?\b", q):
        return "ledger_group"
    if re.search(r"\bledger[\s_-]*type(?:s)?\b", q):
        return "ledger_type"
    return None


def _is_time_bucket_extreme_query_local(query: str) -> bool:
    """Local detection for month/quarter/year extreme net-profit queries."""
    q = (query or "").lower().strip()
    if not q:
        return False
    has_bucket = bool(re.search(r"\b(month|quarter|year)\b", q))
    has_superlative = bool(re.search(r"\b(lowest|highest|least|most|worst|best|top|bottom)\b", q))
    has_financial_metric = bool(re.search(
        r"\b(profit|net profit|pnl|p&l|expense|expenses|revenue|income|cost|costs)\b", q
    ))
    has_period_compare = bool(
        re.search(r"\bfrom\b.+\bto\b", q)
        or re.search(r"\bbetween\b.+\band\b", q)
        or re.search(r"\b(?:increase|decrease|change|growth|grew)\b", q)
    )
    return has_bucket and has_superlative and has_financial_metric and not has_period_compare


def _extract_explicit_targets_from_query(query: str) -> tuple[list[str], list[str]]:
    """Extract explicit known target names mentioned verbatim in the query."""
    q = (query or "").lower()
    if not q:
        return [], []

    matches: list[tuple[int, str, str]] = []
    for field in ("entity_name", "property_name", "tenant_name"):
        for value in list_unique(field):
            v = str(value)
            idx = q.find(v.lower())
            if idx >= 0:
                matches.append((idx, v, field))

    if not matches:
        return [], []

    matches.sort(key=lambda x: (x[0], -len(x[1])))
    seen: set[tuple[str, str]] = set()
    targets: list[str] = []
    fields: list[str] = []
    for _, v, f in matches:
        key = (v, f)
        if key in seen:
            continue
        seen.add(key)
        targets.append(v)
        fields.append(f)
    return targets, fields


def _run_llm_extraction(context: str, llm: Any) -> ExtractionOutput | None:
    """Run LLM structured extraction, return None on failure."""
    try:
        structured_llm = llm.with_structured_output(ExtractionOutput)
        result: ExtractionOutput = structured_llm.invoke([
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ])
        return result
    except Exception as exc:
        logger.warning("ExtractionAgent LLM failed: %s", exc)
        return None


def _heuristic_extraction(
    query: str, intent: str, today: date, conv_ctx: str = ""
) -> ExtractionOutput:
    """
    Keyword-based fallback extraction when LLM fails.
    When the query contains pronouns and no explicit entity is found, scan
    conversation context for the most recently mentioned entity.
    """
    q = query.lower()

    targets: list[str] = []
    target_fields: list[str] = []
    timeframe = None
    group_by = None
    rank_direction = "top"
    k = 5

    # Detect property references in query
    properties = list_unique("property_name")
    for p in properties:
        if p.lower() in q:
            targets.append(p)
            target_fields.append("property_name")

    # Detect tenant references in query
    tenants = list_unique("tenant_name")
    for t in tenants:
        if t.lower() in q:
            targets.append(t)
            target_fields.append("tenant_name")

    # If no targets found and query contains pronouns, scan conversation context
    _pronouns = ("he ", "she ", "they ", "his ", "her ", "their ", "him ", "it ",
                  "he'", "she'", "they'")
    if not targets and conv_ctx and any(p in q for p in _pronouns):
        ctx_lower = conv_ctx.lower()
        # Prefer tenant match (more specific) over property match
        for t in tenants:
            if t.lower() in ctx_lower:
                targets.append(t)
                target_fields.append("tenant_name")
                break
        if not targets:
            for p in properties:
                if p.lower() in ctx_lower:
                    targets.append(p)
                    target_fields.append("property_name")
                    break

    timeframe = _dict_to_timeframe_model(_parse_timeframe_from_text(query, today))

    # Detect ranking direction
    if any(w in q for w in ("bottom", "worst", "lowest")):
        rank_direction = "bottom"

    # Detect K
    explicit_k = _extract_explicit_k(q)
    if explicit_k is not None:
        k = explicit_k
    elif intent == Intent.RANKING and _is_single_winner_ranking_query(q):
        k = 1

    # Infer group_by from intent + keywords
    if intent in (Intent.RANKING, Intent.BREAKDOWN):
        if "tenant" in q:
            group_by = "tenant_name"
        elif "property" in q or "building" in q or "asset" in q:
            group_by = "property_name"
        elif re.search(r"\bledger[\s_-]*categor(?:y|ies)\b", q):
            group_by = "ledger_category"
        elif re.search(r"\b(expense|cost|revenue|income)[\s_-]*categor(?:y|ies)\b", q):
            group_by = "ledger_category"
        elif re.search(r"\b(by|per|each)\s+categor(?:y|ies)\b", q):
            group_by = "ledger_category"
        elif re.search(r"\bledger[\s_-]*group(?:s)?\b", q):
            group_by = "ledger_group"
        elif re.search(r"\bledger[\s_-]*type(?:s)?\b", q):
            group_by = "ledger_type"
        elif intent == Intent.RANKING:
            group_by = "property_name"  # safe default for ranking
    elif intent == Intent.PNL_BY_PROPERTY:
        group_by = "property_name"
    elif intent == Intent.PNL_BY_TENANT:
        group_by = "tenant_name"

    return ExtractionOutput(
        targets=targets,
        target_fields=target_fields,
        timeframe=timeframe,
        group_by=group_by,
        ledger_filters=LedgerFilters(),
        k=k,
        rank_direction=rank_direction,
        missing_fields=[],
    )


def _find_missing_fields(intent: str, update: dict) -> list[str]:
    """Return list of missing required fields for the given intent."""
    missing = []

    # Timeframe required for most data intents
    if intent in TIMEFRAME_REQUIRED and not update.get("timeframe"):
        # Exception: pnl_by_property with a tenant target is a "which building is X in?"
        # query — timeframe is optional because we're asking about location, not profit magnitude.
        if intent == Intent.PNL_BY_PROPERTY and any(
            f == "tenant_name" for f in update.get("target_fields", [])
        ):
            pass  # no timeframe required for tenant-location queries
        else:
            missing.append("timeframe")

    # Comparison requires exactly 2 targets
    if intent == Intent.COMPARISON:
        targets = update.get("targets", [])
        if len(targets) == 0:
            missing.append("target")
        elif len(targets) == 1:
            missing.append("comparison_target")

    return missing


def _build_clarification_question(missing_field: str, intent: str) -> str:
    """Build a single focused clarification question."""
    if missing_field == "timeframe":
        return (
            "Which time period are you interested in? "
            "For example: '2024', 'Q3 2024', '2025 Q1', 'last year', or a specific month like '2024-M06'."
        )
    if missing_field == "target":
        if intent == Intent.COMPARISON:
            properties = list_unique("property_name")
            tenants = list_unique("tenant_name")
            examples = (properties[:3] if properties else []) + (tenants[:2] if tenants else [])
            return (
                f"Which two targets would you like to compare? "
                f"Please name them (properties or tenants). "
                f"Available examples: {', '.join(examples)}."
            )
        return (
            "Which property, tenant, or entity are you asking about? "
            f"Properties: {', '.join(list_unique('property_name')[:5])}."
        )
    if missing_field == "comparison_target":
        return (
            "You mentioned one target. What would you like to compare it against? "
            f"Available properties: {', '.join(list_unique('property_name'))}."
        )
    return f"Could you provide more information about {missing_field}?"
