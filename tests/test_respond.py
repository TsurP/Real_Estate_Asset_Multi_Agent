"""
Tests for ResponseAgent formatting behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.respond import respond_node
from src.core.state import Intent, initial_state


class TestRespondNode:
    def test_data_intent_is_deterministic_and_skips_llm(self):
        llm = MagicMock()
        llm.invoke.side_effect = AssertionError("LLM should not be called for data intents")

        state = initial_state("What is total profit for 2024?")
        state["intent"] = Intent.PNL_TOTAL
        state["timeframe"] = {"year": "2024", "label": "2024"}
        state["retrieved_rows"] = 10
        state["computation_result"] = {
            "type": "pnl_total",
            "total_profit": 1234.5,
            "total_revenue": 2000.0,
            "total_expenses": -765.5,
        }

        result = respond_node(state, llm)
        assert "€1,234.50" in result["final_answer"]
        assert "€2,000.00" in result["final_answer"]
        assert "€765.50" in result["final_answer"]  # expenses shown as positive amount

    def test_general_question_still_uses_llm(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="NOI is net operating income.")

        state = initial_state("What is NOI?")
        state["intent"] = Intent.GENERAL_QUESTION

        result = respond_node(state, llm)
        assert result["final_answer"] == "NOI is net operating income."
        assert llm.invoke.call_count == 1

    def test_margin_analysis_single_winner_format(self):
        llm = MagicMock()
        llm.invoke.side_effect = AssertionError("LLM should not be called for data intents")

        state = initial_state("Which property had the worst net profit margin in 2024?")
        state["intent"] = Intent.MARGIN_ANALYSIS
        state["timeframe"] = {"year": "2024", "label": "2024"}
        state["retrieved_rows"] = 3181
        state["computation_result"] = {
            "type": "margin_analysis",
            "direction": "bottom",
            "k": 1,
            "group_field": "property_name",
            "rows": [
                {
                    "rank": 1,
                    "label": "Building 120",
                    "total_profit": 675640.08,
                    "total_revenue": 703009.03,
                    "margin": 0.961069,
                    "margin_pct": 96.11,
                }
            ],
        }

        result = respond_node(state, llm)
        assert "Worst net profit margin" in result["final_answer"]
        assert "Building 120" in result["final_answer"]
        assert "96.11%" in result["final_answer"]

    def test_presence_condition_no_rows_format(self):
        llm = MagicMock()
        llm.invoke.side_effect = AssertionError("LLM should not be called for data intents")

        state = initial_state("Which properties had expenses but no revenue in 2024?")
        state["intent"] = Intent.PNL_BY_PROPERTY
        state["timeframe"] = {"year": "2024", "label": "2024"}
        state["retrieved_rows"] = 3181
        state["computation_result"] = {
            "type": "presence_condition",
            "condition": "expenses_without_revenue",
            "group_field": "property_name",
            "rows": [],
        }

        result = respond_node(state, llm)
        assert result["final_answer"] == "No properties had expenses but no revenue in 2024."

    def test_ranking_k1_month_is_natural_language(self):
        llm = MagicMock()
        llm.invoke.side_effect = AssertionError("LLM should not be called for data intents")

        state = initial_state("Which month in 2024 had the lowest total net profit?")
        state["intent"] = Intent.RANKING
        state["timeframe"] = {"year": "2024", "label": "2024"}
        state["retrieved_rows"] = 3181
        state["computation_result"] = {
            "type": "ranking",
            "direction": "bottom",
            "k": 1,
            "group_field": "month",
            "rows": [
                {"rank": 1, "label": "2024-M12", "profit": 77849.67},
            ],
        }

        result = respond_node(state, llm)
        assert "December 2024" in result["final_answer"]
        assert "lowest total net profit" in result["final_answer"]
        assert "€77,849.67" in result["final_answer"]

    def test_stat_condition_below_average_months_format(self):
        llm = MagicMock()
        llm.invoke.side_effect = AssertionError("LLM should not be called for data intents")

        state = initial_state("Which months in 2024 had below-average net profit?")
        state["intent"] = Intent.PERIOD_COMPARISON
        state["timeframe"] = {"year": "2024", "label": "2024"}
        state["retrieved_rows"] = 3181
        state["computation_result"] = {
            "type": "stat_condition",
            "condition": "below_average_profit",
            "group_field": "month",
            "average_profit": 97626.8,
            "rows": [
                {"label": "2024-M12", "profit": 77849.67},
                {"label": "2024-M02", "profit": 83045.11},
            ],
        }

        result = respond_node(state, llm)
        assert "average net profit per month in 2024 was €97,626.80" in result["final_answer"]
        assert "December 2024" in result["final_answer"]
        assert "February 2024" in result["final_answer"]
