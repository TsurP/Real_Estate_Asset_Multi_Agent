"""
Deterministic data-access tools. These are the ONLY way agents touch the DataFrame.
No LLM calls here — pure pandas + fuzzy matching.
"""
from __future__ import annotations

import math
import re
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# Module-level singleton DF (loaded once at startup)
# ---------------------------------------------------------------------------
_DF: Optional[pd.DataFrame] = None
_PARQUET_PATH: Optional[str] = None

VALID_RESOLVE_FIELDS = {"property_name", "tenant_name", "entity_name"}

# Fuzzy-match threshold (0-100); below this we report no match
FUZZY_THRESHOLD = 60


# ---------------------------------------------------------------------------
# Load / access
# ---------------------------------------------------------------------------

def load_dataframe(path: str) -> pd.DataFrame:
    """Load parquet file and cache as module singleton."""
    global _DF, _PARQUET_PATH
    _PARQUET_PATH = path
    _DF = pd.read_parquet(path)
    # Normalise string columns to str (handles ArrowStringArray etc.)
    str_cols = [
        "entity_name", "property_name", "tenant_name", "ledger_type",
        "ledger_group", "ledger_category", "ledger_description",
        "month", "quarter", "year",
    ]
    for col in str_cols:
        if col in _DF.columns:
            _DF[col] = _DF[col].astype("object")  # convert to plain Python str
    _DF["profit"] = _DF["profit"].astype(float)
    return _DF


def get_dataframe() -> pd.DataFrame:
    if _DF is None:
        raise RuntimeError("DataFrame not loaded. Call load_dataframe() first.")
    return _DF


# ---------------------------------------------------------------------------
# Name resolution (fuzzy)
# ---------------------------------------------------------------------------

def resolve_name(
    name: str,
    field: str,
    df: Optional[pd.DataFrame] = None,
    threshold: int = FUZZY_THRESHOLD,
    limit: int = 5,
) -> list[dict]:
    """
    Fuzzy-match *name* against unique values in *field*.

    Returns a list of dicts sorted by score descending:
        [{"value": "Building 140", "score": 95}, ...]
    Empty list if nothing meets threshold.
    """
    if field not in VALID_RESOLVE_FIELDS:
        raise ValueError(f"Invalid field '{field}'. Must be one of {VALID_RESOLVE_FIELDS}")

    df = df if df is not None else get_dataframe()

    candidates = df[field].dropna().unique().tolist()
    if not candidates:
        return []

    # Exact match first
    name_lower = name.strip().lower()
    for c in candidates:
        if c.strip().lower() == name_lower:
            return [{"value": c, "score": 100}]

    # Fuzzy match
    results = process.extract(
        name,
        candidates,
        scorer=fuzz.WRatio,
        limit=limit,
    )
    return [
        {"value": match, "score": score}
        for match, score, _ in results
        if score >= threshold
    ]


def resolve_all_targets(
    targets: list[str],
    target_fields: list[str],
    df: Optional[pd.DataFrame] = None,
) -> tuple[list[dict], list[dict]]:
    """
    Resolve multiple targets.

    Returns:
        resolved: [{"target": name, "field": field, "value": resolved, "score": s}, ...]
        unresolved: targets that had no match above threshold
    """
    df = df if df is not None else get_dataframe()
    resolved = []
    unresolved = []

    for name, field in zip(targets, target_fields):
        matches = resolve_name(name, field, df)
        if matches:
            best = matches[0]
            resolved.append({
                "target": name,
                "field": field,
                "value": best["value"],
                "score": best["score"],
                "all_matches": matches,
            })
        else:
            unresolved.append({"target": name, "field": field})

    return resolved, unresolved


# ---------------------------------------------------------------------------
# Timeframe filtering helpers
# ---------------------------------------------------------------------------

def apply_timeframe_filter(df: pd.DataFrame, timeframe: Optional[dict]) -> pd.DataFrame:
    """Filter df by timeframe dict. Returns unfiltered df if no timeframe."""
    if not timeframe:
        return df

    # Most specific first: month > quarter > year
    if timeframe.get("month"):
        return df[df["month"] == timeframe["month"]]
    if timeframe.get("quarter"):
        return df[df["quarter"] == timeframe["quarter"]]
    if timeframe.get("year"):
        year = timeframe["year"]
        if timeframe.get("ytd"):
            # YTD: all rows up to current month within that year
            current_month = date.today().strftime("%Y-M%m")
            mask = (df["year"] == year) & (df["month"] <= current_month)
            return df[mask]
        return df[df["year"] == year]
    return df


def _apply_ledger_filters(df: pd.DataFrame, ledger_filters: dict) -> pd.DataFrame:
    """Apply ledger dimension filters."""
    if not ledger_filters:
        return df

    ledger_filters = _canonicalize_ledger_filters(ledger_filters)

    if lt := ledger_filters.get("ledger_type"):
        df = df[df["ledger_type"].str.lower() == lt.lower()]
    if lg := ledger_filters.get("ledger_group"):
        df = df[df["ledger_group"].str.lower() == lg.lower()]
    if lc := ledger_filters.get("ledger_category"):
        df = df[df["ledger_category"].str.lower() == lc.lower()]
    if code := ledger_filters.get("ledger_code"):
        df = df[df["ledger_code"] == int(code)]
    if kw := ledger_filters.get("description_keyword"):
        df = df[df["ledger_description"].str.contains(kw, case=False, na=False)]

    return df


def _canonicalize_ledger_filters(ledger_filters: dict) -> dict:
    """
    Canonicalize natural-language filter values to known dataset enums.
    """
    lf = dict(ledger_filters or {})

    lt = str(lf.get("ledger_type", "")).lower().strip()
    if lt in {"expense", "expenses", "cost", "costs"}:
        lf["ledger_type"] = "expenses"
    elif lt in {"revenue", "revenues", "income"}:
        lf["ledger_type"] = "revenue"

    # Direct natural-language alias for insurance.
    lg = str(lf.get("ledger_group", "")).lower().strip()
    if lg in {"insurance", "insurances", "insurance_expense", "insurance_expenses"}:
        lf.pop("ledger_group", None)
        lf["ledger_category"] = "insurance_in_general"
        lf.setdefault("ledger_type", "expenses")

    # Normalize ledger_group/category spellings (spaces/hyphens/plurals) to exact enums.
    if lg := lf.get("ledger_group"):
        resolved_group = _resolve_ledger_enum(str(lg), list_unique("ledger_group"))
        if resolved_group:
            lf["ledger_group"] = resolved_group
    if lc := lf.get("ledger_category"):
        resolved_category = _resolve_ledger_enum(str(lc), list_unique("ledger_category"))
        if resolved_category:
            lf["ledger_category"] = resolved_category

    return lf


def _norm_enum_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return re.sub(r"_+", "_", key)


def _resolve_ledger_enum(raw_value: str, choices: list[str]) -> str | None:
    """Resolve raw value to closest known enum value by normalized key match."""
    if not raw_value or not choices:
        return None

    raw_lower = raw_value.lower().strip()
    for choice in choices:
        if choice.lower() == raw_lower:
            return choice

    norm_raw = _norm_enum_key(raw_value)
    norm_map = {_norm_enum_key(c): c for c in choices}
    if norm_raw in norm_map:
        return norm_map[norm_raw]

    # Small singularization fallback
    if norm_raw.endswith("s") and norm_raw[:-1] in norm_map:
        return norm_map[norm_raw[:-1]]
    if f"{norm_raw}s" in norm_map:
        return norm_map[f"{norm_raw}s"]

    # Containment fallback (e.g., "insurance" -> "insurance_in_general")
    candidates = [c for c in choices if norm_raw in _norm_enum_key(c)]
    if candidates:
        return sorted(candidates, key=len)[0]

    return None


def _apply_target_filters(
    df: pd.DataFrame,
    resolved_targets: list[dict],
) -> pd.DataFrame:
    """Filter df to rows matching any of the resolved targets."""
    if not resolved_targets:
        return df

    masks = []
    for rt in resolved_targets:
        field = rt["field"]
        value = rt["value"]
        masks.append(df[field] == value)

    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m
    return df[combined]


# ---------------------------------------------------------------------------
# Main query function
# ---------------------------------------------------------------------------

def query_ledger(
    timeframe: Optional[dict] = None,
    ledger_filters: Optional[dict] = None,
    resolved_targets: Optional[list[dict]] = None,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Apply timeframe, ledger, and target filters to the portfolio DataFrame.

    Returns a filtered DataFrame slice (no aggregation — that's ComputationAgent's job).
    """
    df = df if df is not None else get_dataframe()

    df = apply_timeframe_filter(df, timeframe)
    df = _apply_ledger_filters(df, ledger_filters or {})
    df = _apply_target_filters(df, resolved_targets or [])

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Utility: list unique values
# ---------------------------------------------------------------------------

def list_unique(field: str, df: Optional[pd.DataFrame] = None) -> list:
    """Return sorted list of unique non-null values for *field*."""
    df = df if df is not None else get_dataframe()
    if field not in df.columns:
        return []
    return sorted(df[field].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# Timeframe helpers for relative expressions
# ---------------------------------------------------------------------------

def resolve_relative_timeframe(label: str) -> dict:
    """
    Convert relative label to a concrete timeframe dict.
    Used to supplement LLM extraction for common expressions.
    """
    today = date.today()
    year = today.year
    month = today.month
    quarter = (month - 1) // 3 + 1

    label_lower = label.lower().strip()

    if label_lower in ("this year", "ytd", "year to date"):
        return {
            "year": str(year),
            "ytd": True,
            "label": f"{year} YTD",
        }
    if label_lower == "last year":
        return {
            "year": str(year - 1),
            "ytd": False,
            "label": str(year - 1),
        }
    if label_lower == "this quarter":
        q_str = f"{year}-Q{quarter}"
        return {"quarter": q_str, "label": q_str}
    if label_lower == "last quarter":
        if quarter == 1:
            prev_q = f"{year - 1}-Q4"
        else:
            prev_q = f"{year}-Q{quarter - 1}"
        return {"quarter": prev_q, "label": prev_q}
    if label_lower == "this month":
        m_str = f"{year}-M{month:02d}"
        return {"month": m_str, "label": m_str}

    return {}
