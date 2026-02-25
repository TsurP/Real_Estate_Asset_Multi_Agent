"""
Tests for the data tools layer (df_tools.py).
Tests run against the real parquet dataset.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.df_tools import (
    list_unique,
    load_dataframe,
    query_ledger,
    resolve_name,
    resolve_relative_timeframe,
)

PARQUET = Path(__file__).parent.parent / "cortex (2) (1) (2) (1) (1) (3).parquet"


@pytest.fixture(scope="module")
def df():
    return load_dataframe(str(PARQUET))


# ---------------------------------------------------------------------------
# load_dataframe
# ---------------------------------------------------------------------------

class TestLoadDataframe:
    def test_loads_successfully(self, df):
        assert df is not None
        assert len(df) > 0

    def test_has_required_columns(self, df):
        required = [
            "entity_name", "property_name", "tenant_name", "ledger_type",
            "ledger_group", "ledger_category", "ledger_code", "ledger_description",
            "month", "quarter", "year", "profit",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_profit_is_float(self, df):
        assert df["profit"].dtype == float

    def test_year_values(self, df):
        years = set(df["year"].dropna().unique())
        assert "2024" in years
        assert "2025" in years

    def test_expected_row_count(self, df):
        # Dataset should have ~3924 rows
        assert len(df) > 3000


# ---------------------------------------------------------------------------
# resolve_name
# ---------------------------------------------------------------------------

class TestResolveName:
    def test_exact_match_property(self, df):
        results = resolve_name("Building 140", "property_name", df)
        assert results, "Should find exact match"
        assert results[0]["value"] == "Building 140"
        assert results[0]["score"] == 100

    def test_fuzzy_match_property(self, df):
        results = resolve_name("building 140", "property_name", df)
        assert results, "Lowercase should still match"
        assert results[0]["value"] == "Building 140"

    def test_partial_name_property(self, df):
        results = resolve_name("Bldg 140", "property_name", df)
        assert results, "Abbreviated name should match"
        assert results[0]["value"] == "Building 140"

    def test_exact_match_tenant(self, df):
        results = resolve_name("Tenant 5", "tenant_name", df)
        assert results
        assert results[0]["value"] == "Tenant 5"
        assert results[0]["score"] == 100

    def test_no_match_below_threshold(self, df):
        results = resolve_name("XYZ999NOMATCH", "property_name", df)
        assert results == [] or results[0]["score"] < 60

    def test_invalid_field_raises(self, df):
        with pytest.raises(ValueError, match="Invalid field"):
            resolve_name("foo", "invalid_field", df)

    def test_entity_resolution(self, df):
        results = resolve_name("PropCo", "entity_name", df)
        assert results
        assert results[0]["value"] == "PropCo"


# ---------------------------------------------------------------------------
# query_ledger
# ---------------------------------------------------------------------------

class TestQueryLedger:
    def test_year_filter(self, df):
        result = query_ledger(timeframe={"year": "2024"}, df=df)
        assert len(result) > 0
        assert all(result["year"] == "2024")

    def test_quarter_filter(self, df):
        result = query_ledger(timeframe={"quarter": "2024-Q1"}, df=df)
        assert len(result) > 0
        assert all(result["quarter"] == "2024-Q1")

    def test_month_filter(self, df):
        result = query_ledger(timeframe={"month": "2024-M06"}, df=df)
        assert len(result) > 0
        assert all(result["month"] == "2024-M06")

    def test_ledger_type_filter(self, df):
        result = query_ledger(ledger_filters={"ledger_type": "revenue"}, df=df)
        assert len(result) > 0
        assert all(result["ledger_type"] == "revenue")

    def test_ledger_group_filter(self, df):
        result = query_ledger(ledger_filters={"ledger_group": "rental_income"}, df=df)
        assert len(result) > 0
        assert all(result["ledger_group"] == "rental_income")

    def test_insurance_alias_maps_to_insurance_category(self, df):
        result = query_ledger(
            timeframe={"year": "2024"},
            ledger_filters={"ledger_type": "expenses", "ledger_group": "insurance"},
            df=df,
        )
        assert len(result) > 0
        assert all(result["ledger_category"] == "insurance_in_general")

    def test_target_filter_property(self, df):
        resolved = [{"field": "property_name", "value": "Building 140"}]
        result = query_ledger(resolved_targets=resolved, df=df)
        assert len(result) > 0
        assert all(result["property_name"] == "Building 140")

    def test_combined_filters(self, df):
        resolved = [{"field": "property_name", "value": "Building 140"}]
        result = query_ledger(
            timeframe={"year": "2024"},
            ledger_filters={"ledger_type": "revenue"},
            resolved_targets=resolved,
            df=df,
        )
        assert len(result) > 0
        assert all(result["year"] == "2024")
        assert all(result["ledger_type"] == "revenue")
        assert all(result["property_name"] == "Building 140")

    def test_no_filter_returns_all(self, df):
        result = query_ledger(df=df)
        assert len(result) == len(df)

    def test_nonexistent_target_returns_empty(self, df):
        resolved = [{"field": "property_name", "value": "Building 999"}]
        result = query_ledger(resolved_targets=resolved, df=df)
        assert len(result) == 0

    def test_description_keyword_filter(self, df):
        result = query_ledger(
            ledger_filters={"description_keyword": "rent"},
            df=df,
        )
        assert len(result) > 0
        assert all(result["ledger_description"].str.lower().str.contains("rent"))


# ---------------------------------------------------------------------------
# list_unique
# ---------------------------------------------------------------------------

class TestListUnique:
    def test_properties(self, df):
        props = list_unique("property_name", df)
        assert len(props) == 5
        assert "Building 140" in props

    def test_tenants(self, df):
        tenants = list_unique("tenant_name", df)
        assert len(tenants) == 18

    def test_years(self, df):
        years = list_unique("year", df)
        assert "2024" in years
        assert "2025" in years

    def test_invalid_field_returns_empty(self, df):
        result = list_unique("nonexistent_column", df)
        assert result == []


# ---------------------------------------------------------------------------
# resolve_relative_timeframe
# ---------------------------------------------------------------------------

class TestResolveRelativeTimeframe:
    def test_this_year(self):
        from datetime import date
        result = resolve_relative_timeframe("this year")
        assert result.get("year") == str(date.today().year)
        assert result.get("ytd") is True

    def test_last_year(self):
        from datetime import date
        result = resolve_relative_timeframe("last year")
        assert result.get("year") == str(date.today().year - 1)
        assert not result.get("ytd")

    def test_this_quarter(self):
        from datetime import date
        result = resolve_relative_timeframe("this quarter")
        assert "quarter" in result
        today = date.today()
        expected_q = (today.month - 1) // 3 + 1
        assert f"Q{expected_q}" in result["quarter"]

    def test_unknown_label(self):
        result = resolve_relative_timeframe("some random phrase")
        assert result == {}
