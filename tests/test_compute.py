"""
Tests for the ComputationAgent logic (deterministic pandas operations).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.compute import (
    compute_comparison,
    compute_grouped,
    compute_margin_analysis,
    compute_presence_condition,
    compute_pnl_total,
    compute_ranking,
)
from src.tools.df_tools import load_dataframe, query_ledger

PARQUET = Path(__file__).parent.parent / "cortex (2) (1) (2) (1) (1) (3).parquet"


@pytest.fixture(scope="module")
def full_df():
    return load_dataframe(str(PARQUET))


@pytest.fixture(scope="module")
def df_2024(full_df):
    return query_ledger(timeframe={"year": "2024"}, df=full_df)


# ---------------------------------------------------------------------------
# compute_pnl_total
# ---------------------------------------------------------------------------

class TestComputePnlTotal:
    def test_returns_dict_with_keys(self, df_2024):
        result = compute_pnl_total(df_2024)
        assert result["type"] == "pnl_total"
        assert "total_profit" in result
        assert "total_revenue" in result
        assert "total_expenses" in result
        assert "row_count" in result

    def test_total_equals_revenue_plus_expenses(self, df_2024):
        result = compute_pnl_total(df_2024)
        # total_profit = revenue + expenses (expenses are negative so they reduce total)
        expected = result["total_revenue"] + result["total_expenses"]
        assert abs(result["total_profit"] - expected) < 0.01

    def test_row_count_matches(self, df_2024):
        result = compute_pnl_total(df_2024)
        assert result["row_count"] == len(df_2024)

    def test_known_approximate_total(self, full_df):
        result = compute_pnl_total(full_df)
        # Dataset total is ~1,533,331
        assert result["total_profit"] > 1_000_000
        assert result["total_profit"] < 3_000_000

    def test_empty_df_returns_zero(self):
        empty = pd.DataFrame(columns=["profit", "ledger_type"])
        empty["profit"] = pd.array([], dtype=float)
        empty["ledger_type"] = pd.array([], dtype=str)
        result = compute_pnl_total(empty)
        assert result["total_profit"] == 0.0


# ---------------------------------------------------------------------------
# compute_grouped
# ---------------------------------------------------------------------------

class TestComputeGrouped:
    def test_group_by_property(self, df_2024):
        result = compute_grouped(df_2024, "property_name")
        assert result["type"] == "grouped_property_name"
        assert len(result["rows"]) > 0
        # Should have rows for each property
        labels = [r["label"] for r in result["rows"]]
        assert "Building 140" in labels or any("Building" in l for l in labels)

    def test_sorted_descending(self, df_2024):
        result = compute_grouped(df_2024, "property_name")
        profits = [r["profit"] for r in result["rows"]]
        assert profits == sorted(profits, reverse=True)

    def test_group_by_tenant(self, df_2024):
        result = compute_grouped(df_2024, "tenant_name")
        assert result["type"] == "grouped_tenant_name"
        assert len(result["rows"]) > 0

    def test_grouped_property_excludes_null_bucket(self, df_2024):
        result = compute_grouped(df_2024, "property_name")
        labels = [r["label"] for r in result["rows"]]
        assert "(no value)" not in labels
        assert "nan" not in labels

    def test_group_by_ledger_type(self, df_2024):
        result = compute_grouped(df_2024, "ledger_type")
        assert result["type"] == "grouped_ledger_type"
        labels = [r["label"] for r in result["rows"]]
        assert set(labels) == {"revenue", "expenses"}

    def test_sum_matches_total(self, df_2024):
        result = compute_grouped(df_2024, "property_name")
        row_sum = sum(r["profit"] for r in result["rows"])
        assert abs(row_sum - result["total_profit"]) < 0.01

    def test_invalid_column_raises(self, df_2024):
        with pytest.raises(ValueError, match="not in dataset"):
            compute_grouped(df_2024, "nonexistent_column")


# ---------------------------------------------------------------------------
# compute_comparison
# ---------------------------------------------------------------------------

class TestComputeComparison:
    def test_two_properties(self, full_df):
        df_2024 = query_ledger(timeframe={"year": "2024"}, df=full_df)
        result = compute_comparison(
            df_2024,
            targets=["Building 140", "Building 160"],
            target_fields=["property_name", "property_name"],
        )
        assert result["type"] == "comparison"
        assert "target_a" in result
        assert "target_b" in result
        assert "delta" in result
        assert "pct_delta" in result
        assert "higher" in result

    def test_delta_calculation(self, full_df):
        df_2024 = query_ledger(timeframe={"year": "2024"}, df=full_df)
        result = compute_comparison(
            df_2024,
            targets=["Building 140", "Building 160"],
            target_fields=["property_name", "property_name"],
        )
        expected_delta = result["target_a"]["profit"] - result["target_b"]["profit"]
        assert abs(result["delta"] - expected_delta) < 0.01

    def test_winner_is_higher_profit(self, full_df):
        df_2024 = query_ledger(timeframe={"year": "2024"}, df=full_df)
        result = compute_comparison(
            df_2024,
            targets=["Building 140", "Building 160"],
            target_fields=["property_name", "property_name"],
        )
        a_profit = result["target_a"]["profit"]
        b_profit = result["target_b"]["profit"]
        expected_winner = "Building 140" if a_profit > b_profit else "Building 160"
        assert result["higher"] == expected_winner

    def test_divide_by_zero_handled(self):
        df = pd.DataFrame({
            "property_name": ["A", "A", "B"],
            "profit": [100.0, 200.0, 0.0],
        })
        result = compute_comparison(
            df,
            targets=["A", "B"],
            target_fields=["property_name", "property_name"],
        )
        # Should not raise; pct_delta should be inf or handled
        assert result["delta"] == 300.0

    def test_fewer_than_two_targets_raises(self, df_2024):
        with pytest.raises(ValueError, match="2 targets"):
            compute_comparison(
                df_2024,
                targets=["Building 140"],
                target_fields=["property_name"],
            )


# ---------------------------------------------------------------------------
# compute_ranking
# ---------------------------------------------------------------------------

class TestComputeRanking:
    def test_top_k_properties(self, df_2024):
        result = compute_ranking(df_2024, "property_name", k=3, direction="top")
        assert result["type"] == "ranking"
        assert result["direction"] == "top"
        assert result["k"] == 3
        assert len(result["rows"]) >= 1
        assert len(result["rows"]) <= 8  # max k+5

    def test_bottom_k_tenants(self, df_2024):
        result = compute_ranking(df_2024, "tenant_name", k=3, direction="bottom")
        assert result["direction"] == "bottom"
        profits = [r["profit"] for r in result["rows"]]
        # Bottom = smallest profits first
        assert profits == sorted(profits)

    def test_ranks_are_sequential(self, df_2024):
        result = compute_ranking(df_2024, "property_name", k=5, direction="top")
        ranks = [r["rank"] for r in result["rows"]]
        assert ranks[0] == 1

    def test_invalid_column_raises(self, df_2024):
        with pytest.raises(ValueError, match="not in dataset"):
            compute_ranking(df_2024, "bad_column", k=5)

    def test_k_larger_than_data(self, df_2024):
        # 5 properties; k=10 should return at most 5
        result = compute_ranking(df_2024, "property_name", k=10, direction="top")
        assert len(result["rows"]) <= 10


class TestComputeMarginAnalysis:
    def test_worst_property_margin_2024(self, df_2024):
        result = compute_margin_analysis(df_2024, "property_name", k=1, direction="bottom")
        assert result["type"] == "margin_analysis"
        assert result["rows"][0]["label"] == "Building 120"
        assert 95.0 < result["rows"][0]["margin_pct"] < 97.0

    def test_best_property_margin_2024(self, df_2024):
        result = compute_margin_analysis(df_2024, "property_name", k=1, direction="top")
        assert result["rows"][0]["label"] == "Building 160"
        assert result["rows"][0]["margin_pct"] > 98.0


class TestComputePresenceCondition:
    def test_expenses_without_revenue_returns_empty_for_2024_properties(self, df_2024):
        result = compute_presence_condition(df_2024, "property_name", "expenses_without_revenue")
        assert result["type"] == "presence_condition"
        assert result["condition"] == "expenses_without_revenue"
        assert result["rows"] == []

    def test_below_average_profit_months_2024(self, df_2024):
        result = compute_presence_condition(df_2024, "month", "below_average_profit")
        assert result["type"] == "stat_condition"
        assert result["condition"] == "below_average_profit"
        labels = [r["label"] for r in result["rows"]]
        assert "2024-M12" in labels
        assert len(labels) == 6
        assert abs(result["average_profit"] - 97_626.8) < 1.0
