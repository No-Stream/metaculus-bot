"""TOOLS_USED=true/false marker emission in _create_unified_explanation.

When ``PROBABILISTIC_TOOLS_ENABLED`` is set, comments gain a
``<!-- TOOLS_USED=true -->`` marker alongside the existing STACKED marker.
Off by default → ``<!-- TOOLS_USED=false -->``.

Fold logic sits in ``TemplateForecaster._create_unified_explanation`` in
``main.py``, gated on the aggregation strategy the same way the STACKED
marker is.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from forecasting_tools import BinaryQuestion, ForecastBot, GeneralLlm, MultipleChoiceQuestion, NumericQuestion

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.comment_markers import TOOLS_USED_MARKER_FALSE, TOOLS_USED_MARKER_TRUE
from metaculus_bot.tool_runner import FEATURE_FLAG_ENV, TYPES_ENV


def _make_bot(strategy: AggregationStrategy) -> TemplateForecaster:
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    llms: dict = {
        "forecasters": [test_llm, test_llm],
        "stacker": test_llm,
        "analyzer": test_llm,
        "default": test_llm,
        "parser": test_llm,
        "researcher": test_llm,
        "summarizer": test_llm,
    }
    return TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        aggregation_strategy=strategy,
        llms=llms,  # type: ignore[arg-type]
        is_benchmarking=True,
    )


def _make_q(spec=BinaryQuestion) -> MagicMock:
    q = MagicMock(spec=spec)
    q.id_of_question = 99
    q.question_text = "Will X happen?"
    q.page_url = "https://metaculus.com/q/99/"
    q.background_info = ""
    q.resolution_criteria = ""
    q.fine_print = ""
    q.open_time = datetime.now() - timedelta(days=30)
    q.scheduled_resolution_time = datetime.now() + timedelta(days=365)
    return q


_BASE_EXPLANATION = "# SUMMARY\n\nBase body."


class TestToolsUsedMarker:
    """Verifies the TOOLS_USED marker emission logic against Phase 1's
    _stacker_outcome tri-state (``primary`` | ``fallback_llm`` |
    ``fallback_median`` | ``skipped``) rather than the retired
    _question_was_stacked boolean."""

    def test_flag_on_emits_true_marker_with_stacking(self, monkeypatch):
        monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_q()
        bot._stacker_outcome[q.id_of_question] = "primary"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert TOOLS_USED_MARKER_TRUE in out
        assert TOOLS_USED_MARKER_FALSE not in out

    def test_flag_off_emits_false_marker_with_stacking(self, monkeypatch):
        monkeypatch.delenv(FEATURE_FLAG_ENV, raising=False)
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_q()
        bot._stacker_outcome[q.id_of_question] = "primary"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert TOOLS_USED_MARKER_FALSE in out
        assert TOOLS_USED_MARKER_TRUE not in out

    def test_flag_on_conditional_stacking_emits_true(self, monkeypatch):
        monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        q = _make_q()
        # fallback_median: stacker LLMs both failed, MEDIAN used. Was previously
        # labelled _question_was_stacked=False by the boolean marker.
        bot._stacker_outcome[q.id_of_question] = "fallback_median"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert TOOLS_USED_MARKER_TRUE in out

    def test_non_stacking_strategy_no_marker(self, monkeypatch):
        # Like STACKED, TOOLS_USED only makes sense for stacking-family
        # strategies (the stacker is the only consumer of the aggregated tool
        # output). For MEAN/MEDIAN we preserve the existing behavior of
        # emitting no marker at all — keeps residual-analysis parsing
        # unambiguous.
        monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
        bot = _make_bot(AggregationStrategy.MEAN)
        q = _make_q()

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert TOOLS_USED_MARKER_TRUE not in out
        assert TOOLS_USED_MARKER_FALSE not in out


class TestToolsUsedMarkerPerTypeDispatch:
    """F21 regression: marker reflects actual per-type dispatch via
    PROBABILISTIC_TOOLS_TYPES, not just the global flag.

    Previously the marker was set by ``env_flag_enabled(FEATURE_FLAG_ENV)``
    alone, so a numeric question with TYPES="binary,multiple_choice" would
    emit TOOLS_USED=true even though tool_runner._feature_enabled("numeric")
    returned False and no tool ran. That broke residual-analysis bucketing.
    """

    def test_numeric_q_with_binary_only_types_emits_false(self, monkeypatch):
        monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
        monkeypatch.setenv(TYPES_ENV, "binary,multiple_choice")
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_q(spec=NumericQuestion)
        bot._stacker_outcome[q.id_of_question] = "primary"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert TOOLS_USED_MARKER_FALSE in out
        assert TOOLS_USED_MARKER_TRUE not in out

    def test_binary_q_with_binary_only_types_emits_true(self, monkeypatch):
        monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
        monkeypatch.setenv(TYPES_ENV, "binary")
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_q(spec=BinaryQuestion)
        bot._stacker_outcome[q.id_of_question] = "primary"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert TOOLS_USED_MARKER_TRUE in out
        assert TOOLS_USED_MARKER_FALSE not in out

    def test_global_flag_off_emits_false_regardless_of_type(self, monkeypatch):
        monkeypatch.delenv(FEATURE_FLAG_ENV, raising=False)
        # Even with the per-type allow-list explicitly including binary, a
        # disabled global flag must short-circuit to FALSE.
        monkeypatch.setenv(TYPES_ENV, "binary,numeric,multiple_choice")
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_q(spec=BinaryQuestion)
        bot._stacker_outcome[q.id_of_question] = "primary"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert TOOLS_USED_MARKER_FALSE in out
        assert TOOLS_USED_MARKER_TRUE not in out

    def test_mc_q_with_numeric_only_types_emits_false(self, monkeypatch):
        # Belt-and-suspenders for the third question type: MC question with
        # numeric-only allow-list still emits FALSE.
        monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
        monkeypatch.setenv(TYPES_ENV, "numeric")
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_q(spec=MultipleChoiceQuestion)
        bot._stacker_outcome[q.id_of_question] = "primary"

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert TOOLS_USED_MARKER_FALSE in out
        assert TOOLS_USED_MARKER_TRUE not in out


_ = pytest
