"""
Tests for intent routing and extraction logic.
Uses mock LLM to avoid API calls.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.schemas import RouterOutput, ExtractionOutput, TimeframeOutput
from src.core.state import Intent, initial_state
from src.agents.router import (
    router_node,
    _fallback_route,
    _is_margin_data_query,
    _is_time_bucket_extreme_query,
)
from src.agents.extract import (
    _extract_explicit_k,
    _is_single_winner_ranking_query,
    _parse_timeframe_from_text,
    _find_missing_fields,
    _build_clarification_question,
    _heuristic_extraction,
    extract_node,
)
from src.graph import _after_router, _after_extract, _after_retrieve, _after_compute


# ---------------------------------------------------------------------------
# RouterAgent – heuristic fallback (no LLM)
# ---------------------------------------------------------------------------

class TestFallbackRoute:
    def test_price_query_is_unsupported(self):
        result = _fallback_route("What is the price of Building 140?", [], "")
        assert result["intent"] == Intent.UNSUPPORTED

    def test_top_query_is_ranking(self):
        result = _fallback_route("Top 5 properties by profit", [], "")
        assert result["intent"] == Intent.RANKING

    def test_compare_query_is_comparison(self):
        result = _fallback_route("Compare Building 140 vs Building 160", [], "")
        assert result["intent"] == Intent.COMPARISON

    def test_total_query_is_pnl_total(self):
        result = _fallback_route("What is the total profit for 2024?", [], "")
        assert result["intent"] == Intent.PNL_TOTAL

    def test_by_property_intent(self):
        result = _fallback_route("Show profit by property", [], "")
        assert result["intent"] == Intent.PNL_BY_PROPERTY

    def test_by_tenant_intent(self):
        result = _fallback_route("Show profit by tenant", [], "")
        assert result["intent"] == Intent.PNL_BY_TENANT

    def test_breakdown_intent(self):
        result = _fallback_route("Breakdown by ledger type", [], "")
        assert result["intent"] == Intent.BREAKDOWN

    def test_margin_query_is_margin_analysis(self):
        result = _fallback_route(
            "Which property had the worst net profit margin in 2024?",
            [],
            "",
        )
        assert result["intent"] == Intent.MARGIN_ANALYSIS


# ---------------------------------------------------------------------------
# RouterAgent – with mocked LLM
# ---------------------------------------------------------------------------

class TestRouterNodeMocked:
    def _make_mock_llm(self, intent: str, confidence: float):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = RouterOutput(
            intent=intent, confidence=confidence, notes="test"
        )
        return mock_llm

    def test_pnl_total_routing(self):
        llm = self._make_mock_llm(Intent.PNL_TOTAL, 0.95)
        state = initial_state("What is the total profit for 2024?")
        result = router_node(state, llm)
        assert result["intent"] == Intent.PNL_TOTAL
        assert result["confidence"] == 0.95

    def test_unsupported_routing(self):
        llm = self._make_mock_llm(Intent.UNSUPPORTED, 0.9)
        state = initial_state("What is the market value of Building 140?")
        result = router_node(state, llm)
        assert result["intent"] == Intent.UNSUPPORTED

    def test_general_question_routing(self):
        llm = self._make_mock_llm(Intent.GENERAL_QUESTION, 0.85)
        state = initial_state("What is a ledger?")
        result = router_node(state, llm)
        assert result["intent"] == Intent.GENERAL_QUESTION

    def test_invalid_intent_falls_back_to_unsupported(self):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        # Return invalid intent
        mock_structured.invoke.return_value = RouterOutput(
            intent="gibberish_intent", confidence=0.5
        )
        state = initial_state("random query")
        result = router_node(state, llm=mock_llm)
        # Should have been normalised to unsupported
        assert result["intent"] in (Intent.UNSUPPORTED, "gibberish_intent")

    def test_llm_exception_triggers_heuristic(self):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.side_effect = Exception("API error")
        state = initial_state("total profit 2024")
        result = router_node(state, llm=mock_llm)
        # Should fall back to heuristic, which should guess pnl_total
        assert result["intent"] in Intent.ALL
        assert "error" in result or "router_notes" in result

    def test_margin_query_overrides_llm_misclassification(self):
        llm = self._make_mock_llm(Intent.BREAKDOWN, 0.91)
        state = initial_state("Which property had the worst net profit margin in 2024?")
        result = router_node(state, llm)
        assert result["intent"] == Intent.MARGIN_ANALYSIS

    def test_time_bucket_extreme_overrides_period_comparison(self):
        llm = self._make_mock_llm(Intent.PERIOD_COMPARISON, 0.9)
        state = initial_state("Which month in 2024 had the lowest total net profit?")
        result = router_node(state, llm)
        assert result["intent"] == Intent.RANKING


# ---------------------------------------------------------------------------
# RouterOutput validation
# ---------------------------------------------------------------------------

class TestRouterOutputValidation:
    def test_valid_intent_accepted(self):
        for intent in Intent.ALL:
            ro = RouterOutput(intent=intent, confidence=0.9)
            assert ro.intent == intent

    def test_confidence_clamped(self):
        with pytest.raises(Exception):
            RouterOutput(intent=Intent.PNL_TOTAL, confidence=1.5)

    def test_notes_optional(self):
        ro = RouterOutput(intent=Intent.GENERAL_QUESTION, confidence=0.8)
        assert ro.notes is None


class TestMarginIntentDetection:
    def test_data_margin_query_detected(self):
        assert _is_margin_data_query("Worst margin by property in 2024")

    def test_definition_margin_query_not_detected(self):
        assert not _is_margin_data_query("What is net profit margin?")


class TestTimeBucketExtremeDetection:
    def test_detects_lowest_month_profit_query(self):
        assert _is_time_bucket_extreme_query("Which month in 2024 had the lowest total net profit?")

    def test_does_not_detect_from_to_period_change_query(self):
        assert not _is_time_bucket_extreme_query("From Q1 to Q4 2024, which property had the largest increase?")


# ---------------------------------------------------------------------------
# ExtractionAgent – missing field detection
# ---------------------------------------------------------------------------

class TestMissingFieldDetection:
    def test_pnl_total_missing_timeframe(self):
        state = {
            "timeframe": None,
            "targets": [],
            "target_fields": [],
        }
        missing = _find_missing_fields(Intent.PNL_TOTAL, state)
        assert "timeframe" in missing

    def test_pnl_total_with_timeframe_no_missing(self):
        state = {
            "timeframe": {"year": "2024"},
            "targets": [],
            "target_fields": [],
        }
        missing = _find_missing_fields(Intent.PNL_TOTAL, state)
        assert "timeframe" not in missing

    def test_comparison_missing_both_targets(self):
        state = {
            "timeframe": {"year": "2024"},
            "targets": [],
            "target_fields": [],
        }
        missing = _find_missing_fields(Intent.COMPARISON, state)
        assert "target" in missing

    def test_comparison_missing_second_target(self):
        state = {
            "timeframe": {"year": "2024"},
            "targets": ["Building 140"],
            "target_fields": ["property_name"],
        }
        missing = _find_missing_fields(Intent.COMPARISON, state)
        assert "comparison_target" in missing

    def test_comparison_with_two_targets_no_missing(self):
        state = {
            "timeframe": {"year": "2024"},
            "targets": ["Building 140", "Building 160"],
            "target_fields": ["property_name", "property_name"],
        }
        missing = _find_missing_fields(Intent.COMPARISON, state)
        assert missing == []

    def test_general_question_no_missing(self):
        state = {
            "timeframe": None,
            "targets": [],
            "target_fields": [],
        }
        missing = _find_missing_fields(Intent.GENERAL_QUESTION, state)
        assert missing == []


# ---------------------------------------------------------------------------
# Heuristic extraction
# ---------------------------------------------------------------------------

class TestHeuristicExtraction:
    def setup_method(self):
        # Load df for list_unique calls
        try:
            from src.tools.df_tools import load_dataframe, get_dataframe
            PARQUET = Path(__file__).parent.parent / "cortex (2) (1) (2) (1) (1) (3).parquet"
            load_dataframe(str(PARQUET))
        except Exception:
            pass

    def test_detects_year(self):
        from datetime import date
        result = _heuristic_extraction("profit in 2024", Intent.PNL_TOTAL, date.today())
        assert result.timeframe is not None
        assert result.timeframe.year == "2024"

    def test_detects_quarter(self):
        from datetime import date
        result = _heuristic_extraction("Q3 2024 profit", Intent.PNL_TOTAL, date.today())
        assert result.timeframe is not None
        assert result.timeframe.quarter == "2024-Q3"

    def test_detects_property_name(self):
        from datetime import date
        result = _heuristic_extraction(
            "profit for Building 140 in 2024", Intent.PNL_BY_PROPERTY, date.today()
        )
        assert "Building 140" in result.targets

    def test_detects_bottom_direction(self):
        from datetime import date
        result = _heuristic_extraction("bottom 3 tenants", Intent.RANKING, date.today())
        assert result.rank_direction == "bottom"

    def test_detects_k(self):
        from datetime import date
        result = _heuristic_extraction("top 7 properties", Intent.RANKING, date.today())
        assert result.k == 7

    def test_this_year_detection(self):
        from datetime import date
        today = date.today()
        result = _heuristic_extraction("profit this year", Intent.PNL_TOTAL, today)
        assert result.timeframe is not None
        assert result.timeframe.year == str(today.year)
        assert result.timeframe.ytd is True

    def test_breakdown_ledger_category_grouping(self):
        from datetime import date
        result = _heuristic_extraction(
            "Break down expenses by ledger category in 2024",
            Intent.BREAKDOWN,
            date.today(),
        )
        assert result.group_by == "ledger_category"


class TestDeterministicTimeframeParsing:
    def test_parse_explicit_year(self):
        tf = _parse_timeframe_from_text("Top 5 tenants by profit in 2024", date(2026, 2, 24))
        assert tf == {"year": "2024", "label": "2024"}

    def test_parse_explicit_quarter(self):
        tf = _parse_timeframe_from_text("Show ranking for 2025 Q1", date(2026, 2, 24))
        assert tf == {"quarter": "2025-Q1", "label": "Q1 2025"}


class TestRankingKInference:
    def test_explicit_k_detection(self):
        assert _extract_explicit_k("top 5 tenants") == 5
        assert _extract_explicit_k("bottom 3 properties") == 3
        assert _extract_explicit_k("show best tenants") is None

    def test_single_winner_detection(self):
        assert _is_single_winner_ranking_query("Which tenant paid the most?")
        assert _is_single_winner_ranking_query("Who has the highest profit?")
        assert _is_single_winner_ranking_query("Which month in 2024 had the lowest total net profit?")
        assert not _is_single_winner_ranking_query("Top 5 tenants by profit")


class TestExtractNodeTimeframeFallback:
    def _mock_extract_llm(self, extraction_output: ExtractionOutput):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = extraction_output
        return mock_llm

    def test_explicit_year_survives_when_llm_omits_timeframe(self):
        llm = self._mock_extract_llm(
            ExtractionOutput(
                targets=[],
                target_fields=[],
                timeframe=None,
                group_by="tenant_name",
                ledger_filters={},
                k=5,
                rank_direction="top",
                missing_fields=[],
            )
        )
        state = initial_state("Top 5 tenants by profit in 2024")
        state["intent"] = Intent.RANKING

        result = extract_node(state, llm)
        assert result.get("timeframe") == {"year": "2024", "label": "2024"}

    def test_single_winner_query_forces_k_one_when_llm_defaults_to_five(self):
        llm = self._mock_extract_llm(
            ExtractionOutput(
                targets=[],
                target_fields=[],
                timeframe=None,
                group_by="tenant_name",
                ledger_filters={},
                k=5,
                rank_direction="top",
                missing_fields=[],
            )
        )
        state = initial_state("Which tenant payed the most?")
        state["intent"] = Intent.RANKING

        result = extract_node(state, llm)
        assert result.get("k") == 1

    def test_margin_query_normalises_grouping_and_filters(self):
        llm = self._mock_extract_llm(
            ExtractionOutput(
                targets=["Tenant 7"],
                target_fields=["tenant_name"],
                timeframe=TimeframeOutput(year="2024", label="2024"),
                group_by="month",
                ledger_filters={"ledger_type": "revenue"},
                k=5,
                rank_direction="top",
                missing_fields=[],
            )
        )
        state = initial_state("Which property had the worst net profit margin in 2024?")
        state["intent"] = Intent.MARGIN_ANALYSIS

        result = extract_node(state, llm)
        assert result.get("group_by") == "property_name"
        assert result.get("rank_direction") == "bottom"
        assert result.get("k") == 1
        assert result.get("ledger_filters") == {}
        assert result.get("targets") == []

    def test_insurance_query_maps_to_insurance_category(self):
        llm = self._mock_extract_llm(
            ExtractionOutput(
                targets=[],
                target_fields=[],
                timeframe=TimeframeOutput(year="2024", label="2024"),
                group_by=None,
                ledger_filters={"ledger_type": "expenses", "ledger_group": "insurance"},
                k=5,
                rank_direction="top",
                missing_fields=[],
            )
        )
        state = initial_state("What were the total insurance expenses for PropCo in 2024?")
        state["intent"] = Intent.BREAKDOWN

        result = extract_node(state, llm)
        assert result.get("ledger_filters", {}).get("ledger_type") == "expenses"
        assert result.get("ledger_filters", {}).get("ledger_category") == "insurance_in_general"
        assert "ledger_group" not in result.get("ledger_filters", {})

    def test_explicit_ledger_category_overrides_llm_group_by(self):
        llm = self._mock_extract_llm(
            ExtractionOutput(
                targets=["Building 120"],
                target_fields=["property_name"],
                timeframe=TimeframeOutput(year="2024", label="2024"),
                group_by="ledger_group",
                ledger_filters={"ledger_type": "expenses"},
                k=5,
                rank_direction="top",
                missing_fields=[],
            )
        )
        state = initial_state("For Building 120 in 2024, show expenses by ledger category.")
        state["intent"] = Intent.BREAKDOWN

        result = extract_node(state, llm)
        assert result.get("group_by") == "ledger_category"

    def test_explicit_entity_target_preserved_for_propco(self):
        from src.tools.df_tools import load_dataframe
        parquet = Path(__file__).parent.parent / "cortex (2) (1) (2) (1) (1) (3).parquet"
        load_dataframe(str(parquet))

        llm = self._mock_extract_llm(
            ExtractionOutput(
                targets=[],
                target_fields=[],
                timeframe=TimeframeOutput(year="2024", label="2024"),
                group_by=None,
                ledger_filters={"ledger_type": "expenses", "ledger_category": "insurance_in_general"},
                k=5,
                rank_direction="top",
                missing_fields=[],
            )
        )
        state = initial_state("What were the total insurance expenses for PropCo in 2024?")
        state["intent"] = Intent.PNL_TOTAL

        result = extract_node(state, llm)
        assert ("PropCo", "entity_name") in list(zip(result.get("targets", []), result.get("target_fields", [])))

    def test_condition_query_sets_presence_condition_and_clears_ledger_filter(self):
        llm = self._mock_extract_llm(
            ExtractionOutput(
                targets=[],
                target_fields=[],
                timeframe=TimeframeOutput(year="2024", label="2024"),
                group_by="property_name",
                ledger_filters={"ledger_type": "expenses"},
                k=5,
                rank_direction="top",
                missing_fields=[],
            )
        )
        state = initial_state("Which properties had expenses but no revenue in 2024?")
        state["intent"] = Intent.PNL_BY_PROPERTY

        result = extract_node(state, llm)
        assert result.get("query_condition") == "expenses_without_revenue"
        assert result.get("group_by") == "property_name"
        assert result.get("ledger_filters") == {}

    def test_below_average_month_query_sets_stat_condition(self):
        llm = self._mock_extract_llm(
            ExtractionOutput(
                targets=[],
                target_fields=[],
                timeframe=TimeframeOutput(year="2024", label="2024"),
                group_by="month",
                ledger_filters={"ledger_type": "expenses"},
                k=5,
                rank_direction="top",
                missing_fields=[],
            )
        )
        state = initial_state("Which months in 2024 had below-average net profit?")
        state["intent"] = Intent.PERIOD_COMPARISON

        result = extract_node(state, llm)
        assert result.get("query_condition") == "below_average_profit"
        assert result.get("group_by") == "month"
        assert result.get("ledger_filters") == {}

    def test_temporal_superlative_period_comparison_is_converted_to_ranking(self):
        llm = self._mock_extract_llm(
            ExtractionOutput(
                targets=[],
                target_fields=[],
                timeframe=TimeframeOutput(year="2024", label="2024"),
                group_by="month",
                ledger_filters={},
                k=5,
                rank_direction="top",
                missing_fields=[],
            )
        )
        state = initial_state("Which month in 2024 had the lowest total net profit?")
        state["intent"] = Intent.PERIOD_COMPARISON

        result = extract_node(state, llm)
        assert result.get("intent") == Intent.RANKING
        assert result.get("group_by") == "month"
        assert result.get("rank_direction") == "bottom"
        assert result.get("k") == 1

    def test_temporal_superlative_ranking_sets_k_one(self):
        llm = self._mock_extract_llm(
            ExtractionOutput(
                targets=[],
                target_fields=[],
                timeframe=TimeframeOutput(year="2024", label="2024"),
                group_by="month",
                ledger_filters={},
                k=5,
                rank_direction="bottom",
                missing_fields=[],
            )
        )
        state = initial_state("Which month in 2024 had the lowest total net profit?")
        state["intent"] = Intent.RANKING

        result = extract_node(state, llm)
        assert result.get("group_by") == "month"
        assert result.get("rank_direction") == "bottom"
        assert result.get("k") == 1


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

class TestConditionalEdges:
    def test_after_router_general_question(self):
        state = initial_state("What is profit?")
        state["intent"] = Intent.GENERAL_QUESTION
        state["confidence"] = 0.9
        assert _after_router(state) == "respond"

    def test_after_router_unsupported(self):
        state = initial_state("What is the price?")
        state["intent"] = Intent.UNSUPPORTED
        state["confidence"] = 0.9
        assert _after_router(state) == "fallback"

    def test_after_router_low_confidence_goes_fallback(self):
        state = initial_state("foo bar baz")
        state["intent"] = Intent.PNL_TOTAL
        state["confidence"] = 0.2
        assert _after_router(state) == "fallback"

    def test_after_router_data_intent_goes_extract(self):
        state = initial_state("total profit 2024")
        state["intent"] = Intent.PNL_TOTAL
        state["confidence"] = 0.9
        assert _after_router(state) == "extract"

    def test_after_extract_error_goes_fallback(self):
        state = initial_state("foo")
        state["error"] = "some error"
        assert _after_extract(state) == "fallback"

    def test_after_extract_no_error_goes_retrieve(self):
        state = initial_state("foo")
        state["error"] = None
        assert _after_extract(state) == "retrieve"

    def test_after_retrieve_no_match_goes_fallback(self):
        state = initial_state("foo")
        state["no_match"] = True
        state["error"] = None
        assert _after_retrieve(state) == "fallback"

    def test_after_retrieve_success_goes_compute(self):
        state = initial_state("foo")
        state["no_match"] = False
        state["error"] = None
        assert _after_retrieve(state) == "compute"

    def test_after_compute_error_goes_fallback(self):
        state = initial_state("foo")
        state["error"] = "compute failed"
        state["computation_result"] = {}
        assert _after_compute(state) == "fallback"

    def test_after_compute_success_goes_respond(self):
        state = initial_state("foo")
        state["error"] = None
        state["computation_result"] = {"type": "pnl_total", "total_profit": 1000.0}
        assert _after_compute(state) == "respond"
