"""Tests for producer-side comment construction in main.py and the
performance-analysis collector.

Covers three just-added paths that were previously untested:

1. ``TemplateForecaster._create_unified_explanation`` appends a
   ``<!-- STACKED=true/false -->`` marker only when the aggregation strategy
   is STACKING or CONDITIONAL_STACKING, and the value always comes from
   ``self._question_was_stacked.pop(qid, False)``.

2. ``TemplateForecaster._format_and_expand_research_summary`` annotates
   ``*Forecaster N*`` bullets with the model name pulled from each forecast's
   ``Model: ...`` reasoning prefix. This has to survive comment trimming so
   downstream parsers can recover per-model attribution from the summary
   alone.

3. ``metaculus_bot.performance_analysis.collector._process_single_question``
   wires the new post-data/was-stacked/per-model-numeric-percentiles/
   score-data fields onto each record.

The end-to-end test at the bottom round-trips the producer (main.py) and
consumer (performance_analysis.parsing) together — the critical confidence
check that the two sides actually work in concert.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    ReasonedPrediction,
)
from forecasting_tools.data_models.forecast_report import ResearchWithPredictions

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.comment_markers import STACKED_MARKER_FALSE, STACKED_MARKER_TRUE
from metaculus_bot.performance_analysis.collector import _process_single_question
from metaculus_bot.performance_analysis.parsing import (
    parse_per_model_forecasts,
    parse_per_model_numeric_percentiles,
    parse_per_model_reasoning_text,
    parse_stacked_marker,
)
from metaculus_bot.stacking import combine_stacker_and_base_reasoning

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_bot(strategy: AggregationStrategy) -> TemplateForecaster:
    """Create a TemplateForecaster with the minimal LLM config for the strategy.

    STACKING and CONDITIONAL_STACKING require a stacker LLM; CONDITIONAL_STACKING
    additionally requires an analyzer. MEAN/MEDIAN just need forecasters and the
    default helpers. We always pass an analyzer to keep the helper uniform — the
    extra key is harmless for non-conditional strategies.
    """
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    llms: dict[str, str | GeneralLlm | list[GeneralLlm]] = {
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


def _make_binary_question(qid: int = 12345) -> MagicMock:
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = qid
    q.question_text = "Will it happen?"
    q.page_url = f"https://metaculus.com/questions/{qid}/"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    return q


_BASE_EXPLANATION = "# SUMMARY\n\nBase explanation body."


# ---------------------------------------------------------------------------
# _create_unified_explanation marker injection
# ---------------------------------------------------------------------------


class TestStackedMarkerInjection:
    """Exercises TemplateForecaster._create_unified_explanation.

    The parent ForecastBot._create_unified_explanation is patched to return a
    fixed base string so the test doesn't need a real report/prediction stack.
    """

    @pytest.mark.parametrize(
        "strategy",
        [AggregationStrategy.MEAN, AggregationStrategy.MEDIAN],
    )
    def test_no_marker_for_non_stacking_strategies(self, strategy: AggregationStrategy):
        bot = _make_bot(strategy)
        q = _make_binary_question()

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert "STACKED=" not in out
        assert STACKED_MARKER_TRUE not in out
        assert STACKED_MARKER_FALSE not in out

    def test_non_stacking_cleans_stale_state_defensively(self):
        # If the dict accidentally has an entry for this question, the pop
        # must still clear it regardless of strategy, to avoid leaking state
        # across questions within a run.
        bot = _make_bot(AggregationStrategy.MEAN)
        q = _make_binary_question()
        bot._question_was_stacked[q.id_of_question] = True

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert q.id_of_question not in bot._question_was_stacked

    def test_stacking_true_emits_true_marker(self):
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question()
        bot._question_was_stacked[q.id_of_question] = True

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert STACKED_MARKER_TRUE in out
        assert STACKED_MARKER_FALSE not in out
        assert parse_stacked_marker(out) is True
        # State cleaned
        assert q.id_of_question not in bot._question_was_stacked

    def test_stacking_false_emits_false_marker(self):
        # Invariant under test: the marker value comes from the dict. In
        # production STACKING always sets True, but the plumbing must honor
        # whatever the dict says — otherwise CONDITIONAL_STACKING's skip-path
        # value would silently get overridden.
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question()
        bot._question_was_stacked[q.id_of_question] = False

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert STACKED_MARKER_FALSE in out
        assert STACKED_MARKER_TRUE not in out
        assert parse_stacked_marker(out) is False
        assert q.id_of_question not in bot._question_was_stacked

    def test_conditional_stacking_true_emits_true_marker(self):
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        q = _make_binary_question()
        bot._question_was_stacked[q.id_of_question] = True

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert STACKED_MARKER_TRUE in out
        assert parse_stacked_marker(out) is True

    def test_conditional_stacking_false_emits_false_marker(self):
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        q = _make_binary_question()
        bot._question_was_stacked[q.id_of_question] = False

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert STACKED_MARKER_FALSE in out
        assert parse_stacked_marker(out) is False

    def test_conditional_stacking_missing_dict_entry_defaults_to_false(self):
        # Production code path ALWAYS populates the dict before reaching
        # _create_unified_explanation. But the ``pop(..., False)`` default
        # is a defensive fallback — if some future change drops a branch,
        # we should emit STACKED=false rather than crash or emit nothing.
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        q = _make_binary_question()
        assert q.id_of_question not in bot._question_was_stacked

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=_BASE_EXPLANATION):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert STACKED_MARKER_FALSE in out
        assert parse_stacked_marker(out) is False

    def test_marker_survives_trim_comment(self):
        # trim_comment preserves the tail when truncating; the marker is
        # appended at the very end, so it must survive even when the base
        # text is pushed over the comment char limit.
        from metaculus_bot.constants import COMMENT_CHAR_LIMIT

        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question()
        bot._question_was_stacked[q.id_of_question] = True
        huge_base = "# SUMMARY\n" + ("X" * (COMMENT_CHAR_LIMIT + 1000))

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=huge_base):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        assert len(out) <= COMMENT_CHAR_LIMIT
        assert STACKED_MARKER_TRUE in out
        assert parse_stacked_marker(out) is True

    def test_annotated_bullets_survive_trim(self):
        # Load-bearing invariant: when a comment overflows COMMENT_CHAR_LIMIT,
        # the structured summary-and-tail preserving trim keeps the Forecasts
        # summary bullets intact (so downstream parse_per_model_forecasts still
        # gets per-model attribution) AND the tail STACKED marker.
        #
        # Build a full fake base comment whose rationale body is huge enough to
        # force trim_comment to fire. The summary has three annotated bullets
        # and the "### Research Summary" marker that _trim_preserving_summary_and_tail
        # anchors on. If annotation wiring ran before trim (as in production),
        # the annotated bullets are already present in the base_text; the trim
        # just needs to preserve them.
        from metaculus_bot.constants import COMMENT_CHAR_LIMIT

        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question()
        bot._question_was_stacked[q.id_of_question] = True

        summary_head = (
            "# SUMMARY\n"
            "*Question*: Will X?\n\n"
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (gpt-5.5)*: 72.0%\n"
            "*Forecaster 2 (claude-opus-4.7)*: 68.0%\n"
            "*Forecaster 3 (gemini-3.1-pro-preview)*: 80.0%\n\n"
            "### Research Summary\nshort research.\n\n"
            "================================================================================\n"
            "FORECAST SECTION:\n\n"
            "## R1: Forecaster 1 Reasoning\nModel: openrouter/openai/gpt-5.5\n\n"
        )
        huge_rationale = "X" * (COMMENT_CHAR_LIMIT + 50_000)
        base_text = summary_head + huge_rationale

        with patch.object(ForecastBot, "_create_unified_explanation", return_value=base_text):
            out = bot._create_unified_explanation(q, [], 0.5, 0.01, 1.0)

        # Trim fired (output shorter than input).
        assert len(out) < len(base_text)
        assert len(out) <= COMMENT_CHAR_LIMIT
        # Summary head preserved through the trim.
        assert "*Forecaster 1 (gpt-5.5)*: 72.0%" in out
        assert "*Forecaster 2 (claude-opus-4.7)*: 68.0%" in out
        assert "*Forecaster 3 (gemini-3.1-pro-preview)*: 80.0%" in out
        # Marker appended at the end survives.
        assert out.rstrip().endswith(STACKED_MARKER_TRUE)
        assert parse_stacked_marker(out) is True
        # Parser recovers per-model attribution keyed by model name, not "Forecaster N".
        per_model = parse_per_model_forecasts(out)
        assert per_model == {
            "gpt-5.5": "72.0%",
            "claude-opus-4.7": "68.0%",
            "gemini-3.1-pro-preview": "80.0%",
        }


# ---------------------------------------------------------------------------
# _format_and_expand_research_summary model annotation
# ---------------------------------------------------------------------------


_PARENT_SUMMARY = (
    "## Report 1 Summary\n"
    "### Forecasts\n"
    "*Forecaster 1*: 72.0%\n"
    "*Forecaster 2*: 68.0%\n"
    "*Forecaster 3*: 80.0%\n\n"
    "### Research Summary\n"
    "Some research.\n"
)


def _make_prediction_with_model(
    prob: float,
    model_tag: str | None,
    body: str = "analysis...",
) -> ReasonedPrediction:
    if model_tag is None:
        reasoning = body
    else:
        reasoning = f"Model: {model_tag}\n\n{body}"
    return ReasonedPrediction(prediction_value=prob, reasoning=reasoning)


class TestFormatAndExpandResearchSummaryAnnotation:
    """Exercises TemplateForecaster._format_and_expand_research_summary.

    The parent ForecastBot._format_and_expand_research_summary is patched to
    return a canned summary so we can focus on the annotation wiring.
    """

    def _call(self, predictions: list[ReasonedPrediction], parent_return: str = _PARENT_SUMMARY) -> str:
        research = ResearchWithPredictions(
            research_report="raw",
            summary_report="summary",
            errors=[],
            predictions=predictions,
        )
        from forecasting_tools.data_models.binary_report import BinaryReport

        with patch.object(
            ForecastBot,
            "_format_and_expand_research_summary",
            return_value=parent_return,
        ):
            return TemplateForecaster._format_and_expand_research_summary(
                report_number=1,
                report_type=BinaryReport,
                predicted_research=research,
            )

    def test_all_three_forecasts_get_annotated(self):
        predictions = [
            _make_prediction_with_model(0.72, "openrouter/openai/gpt-5.5"),
            _make_prediction_with_model(0.68, "openrouter/anthropic/claude-opus-4.7"),
            _make_prediction_with_model(0.80, "openrouter/google/gemini-3.1-pro-preview"),
        ]
        out = self._call(predictions)
        assert "*Forecaster 1 (gpt-5.5)*: 72.0%" in out
        assert "*Forecaster 2 (claude-opus-4.7)*: 68.0%" in out
        assert "*Forecaster 3 (gemini-3.1-pro-preview)*: 80.0%" in out

    def test_one_forecast_missing_model_prefix_leaves_that_bullet_unannotated(self):
        predictions = [
            _make_prediction_with_model(0.72, "openrouter/openai/gpt-5.5"),
            _make_prediction_with_model(0.68, None),
            _make_prediction_with_model(0.80, "openrouter/google/gemini-3.1-pro-preview"),
        ]
        out = self._call(predictions)
        assert "*Forecaster 1 (gpt-5.5)*: 72.0%" in out
        assert "*Forecaster 2*: 68.0%" in out
        assert "*Forecaster 2 (" not in out
        assert "*Forecaster 3 (gemini-3.1-pro-preview)*: 80.0%" in out

    def test_all_forecasts_missing_model_prefix_leaves_all_bullets_unannotated(self):
        predictions = [
            _make_prediction_with_model(0.72, None),
            _make_prediction_with_model(0.68, None),
            _make_prediction_with_model(0.80, None),
        ]
        out = self._call(predictions)
        assert "*Forecaster 1*: 72.0%" in out
        assert "*Forecaster 2*: 68.0%" in out
        assert "*Forecaster 3*: 80.0%" in out
        assert "(" not in out.split("### Forecasts")[1].split("### Research Summary")[0]

    def test_bullet_count_less_than_forecast_count_known_indices_still_annotated(self):
        # Edge case: stacking collapses predictions down to a single bullet
        # even though multiple base models fed in. Indices beyond what the
        # parent returned simply have no bullet to annotate — no crash.
        parent_return = "## Report 1 Summary\n### Forecasts\n*Forecaster 1*: 72.0%\n\n### Research Summary\nstuff\n"
        predictions = [
            _make_prediction_with_model(0.72, "openrouter/openai/gpt-5.5"),
            _make_prediction_with_model(0.68, "openrouter/anthropic/claude-opus-4.7"),
            _make_prediction_with_model(0.80, "openrouter/google/gemini-3.1-pro-preview"),
        ]
        out = self._call(predictions, parent_return=parent_return)
        assert "*Forecaster 1 (gpt-5.5)*: 72.0%" in out
        assert "Forecaster 2" not in out
        assert "Forecaster 3" not in out

    def test_bullet_count_more_than_forecast_count_unknown_indices_left_alone(self):
        # Converse edge case: parent returned more bullets than we have
        # predictions for. Known indices (1, 2) annotated, unknown (3) left alone.
        parent_return = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1*: 72.0%\n"
            "*Forecaster 2*: 68.0%\n"
            "*Forecaster 3*: 80.0%\n"
            "*Forecaster 4*: 65.0%\n\n"
            "### Research Summary\nstuff\n"
        )
        predictions = [
            _make_prediction_with_model(0.72, "openrouter/openai/gpt-5.5"),
            _make_prediction_with_model(0.68, "openrouter/anthropic/claude-opus-4.7"),
        ]
        out = self._call(predictions, parent_return=parent_return)
        assert "*Forecaster 1 (gpt-5.5)*: 72.0%" in out
        assert "*Forecaster 2 (claude-opus-4.7)*: 68.0%" in out
        assert "*Forecaster 3*: 80.0%" in out
        assert "*Forecaster 4*: 65.0%" in out


# ---------------------------------------------------------------------------
# Collector _process_single_question
# ---------------------------------------------------------------------------


def _make_post_data(category_name: str = "Politics") -> dict:
    return {
        "id": 999,
        "title": "Some question",
        "projects": {"category": [{"name": category_name}]},
    }


def _make_binary_q_dict(
    qid: int = 101,
    resolution: str | None = "yes",
    forecast_values: list[float] | None = None,
    score_data: dict | None = None,
) -> dict:
    if forecast_values is None:
        forecast_values = [0.3, 0.7]
    my_forecasts = {
        "latest": {"forecast_values": forecast_values},
    }
    if score_data is not None:
        my_forecasts["score_data"] = score_data
    return {
        "id": qid,
        "type": "binary",
        "resolution": resolution,
        "my_forecasts": my_forecasts,
        "scaling": {},
        "open_lower_bound": False,
        "open_upper_bound": False,
        "options": None,
        "title": "Will X?",
        "nr_forecasters": 42,
        "open_time": "2024-01-01T00:00:00Z",
        "actual_resolve_time": "2024-06-01T00:00:00Z",
        "scheduled_resolve_time": "2024-06-01T00:00:00Z",
    }


def _make_numeric_q_dict(
    qid: int = 202,
    resolution: str = "42.5",
    cdf: list[float] | None = None,
) -> dict:
    # Build a plausible 201-point CDF roughly centered at 50 with range [0, 100]
    if cdf is None:
        cdf = [i / 200.0 for i in range(201)]  # uniform, 0.0 .. 1.0
    return {
        "id": qid,
        "type": "numeric",
        "resolution": resolution,
        "my_forecasts": {"latest": {"forecast_values": cdf}},
        "scaling": {"range_min": 0.0, "range_max": 100.0, "zero_point": None},
        "open_lower_bound": False,
        "open_upper_bound": False,
        "options": None,
        "title": "What will X be?",
        "nr_forecasters": 10,
        "open_time": "2024-01-01T00:00:00Z",
        "actual_resolve_time": "2024-06-01T00:00:00Z",
        "scheduled_resolve_time": "2024-06-01T00:00:00Z",
    }


def _make_mc_q_dict(
    qid: int = 303,
    resolution: str = "Option A",
    options: list[str] | None = None,
    forecast_values: list[float] | None = None,
) -> dict:
    if options is None:
        options = ["Option A", "Option B"]
    if forecast_values is None:
        forecast_values = [0.8, 0.2]
    return {
        "id": qid,
        "type": "multiple_choice",
        "resolution": resolution,
        "my_forecasts": {"latest": {"forecast_values": forecast_values}},
        "scaling": {},
        "open_lower_bound": False,
        "open_upper_bound": False,
        "options": options,
        "title": "Which option?",
        "nr_forecasters": 10,
        "open_time": "2024-01-01T00:00:00Z",
        "actual_resolve_time": "2024-06-01T00:00:00Z",
        "scheduled_resolve_time": "2024-06-01T00:00:00Z",
    }


class TestCollectorProcessSingleQuestion:
    """Exercises metaculus_bot.performance_analysis.collector._process_single_question.

    The contract under test:
    - Binary/numeric/MC scoring is populated when resolution + forecast exist.
    - was_stacked, per_model_forecasts, per_model_numeric_percentiles, and
      metaculus_scores are carried onto the record verbatim.
    - category comes from post_data.projects.category[0].name.
    - None is returned when resolution_raw is missing or parse_resolution
      flags the question for skipping.
    """

    def test_binary_question_populates_scores_and_prob_yes(self):
        post = _make_post_data()
        q = _make_binary_q_dict(forecast_values=[0.3, 0.7])
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="# SUMMARY\nfoo\n",
            comment_id=5,
            per_model={"gpt-5.5": "70.0%"},
            per_model_numeric_percentiles={},
            was_stacked=True,
            post_data=post,
        )
        assert rec is not None
        assert rec["type"] == "binary"
        assert rec["our_prob_yes"] == 0.7
        assert rec["brier_score"] is not None
        assert rec["log_score"] is not None
        assert rec["was_stacked"] is True
        assert rec["per_model_forecasts"] == {"gpt-5.5": "70.0%"}
        assert rec["per_model_numeric_percentiles"] == {}

    def test_numeric_question_populates_numeric_log_score_and_percentiles(self):
        post = _make_post_data()
        q = _make_numeric_q_dict(resolution="42.5")
        per_model_percentiles = {"gpt-5.5": [(2.5, 10.0), (50.0, 42.0), (97.5, 85.0)]}
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="# SUMMARY\n",
            comment_id=7,
            per_model={},
            per_model_numeric_percentiles=per_model_percentiles,
            was_stacked=False,
            post_data=post,
        )
        assert rec is not None
        assert rec["type"] == "numeric"
        assert rec["numeric_log_score"] is not None
        assert rec["our_prob_yes"] is None
        assert rec["per_model_numeric_percentiles"] == per_model_percentiles
        assert rec["was_stacked"] is False

    def test_multiple_choice_question_populates_mc_log_score(self):
        post = _make_post_data()
        q = _make_mc_q_dict(
            resolution="Option A",
            options=["Option A", "Option B"],
            forecast_values=[0.8, 0.2],
        )
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="# SUMMARY\n",
            comment_id=9,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=None,
            post_data=post,
        )
        assert rec is not None
        assert rec["type"] == "multiple_choice"
        assert rec["mc_log_score"] is not None
        assert rec["was_stacked"] is None

    @pytest.mark.parametrize("flag", [True, False, None])
    def test_was_stacked_carried_verbatim(self, flag: bool | None):
        post = _make_post_data()
        q = _make_binary_q_dict()
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="",
            comment_id=1,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=flag,
            post_data=post,
        )
        assert rec is not None
        assert rec["was_stacked"] is flag

    def test_metaculus_scores_populated_from_my_forecasts_score_data(self):
        post = _make_post_data()
        score_data = {
            "peer_score": 5.2,
            "spot_peer_score": 3.1,
            "baseline_score": 12.0,
            "spot_baseline_score": 10.0,
            "coverage": 0.95,
            "weighted_coverage": 0.88,
            "relative_legacy_score": 0.05,
        }
        q = _make_binary_q_dict(score_data=score_data)
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="",
            comment_id=1,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=True,
            post_data=post,
        )
        assert rec is not None
        assert rec["metaculus_scores"] == score_data

    def test_metaculus_scores_none_when_score_data_missing(self):
        post = _make_post_data()
        q = _make_binary_q_dict()
        assert "score_data" not in q["my_forecasts"]
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="",
            comment_id=1,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=True,
            post_data=post,
        )
        assert rec is not None
        assert rec["metaculus_scores"] is None

    def test_category_pulled_from_projects_first_entry(self):
        post = _make_post_data(category_name="Sports")
        q = _make_binary_q_dict()
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="",
            comment_id=1,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=True,
            post_data=post,
        )
        assert rec is not None
        assert rec["metadata"]["category"] == "Sports"

    def test_category_none_when_projects_empty(self):
        post = {"id": 1, "title": "t", "projects": {"category": []}}
        q = _make_binary_q_dict()
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="",
            comment_id=1,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=True,
            post_data=post,
        )
        assert rec is not None
        assert rec["metadata"]["category"] is None

    def test_returns_none_when_resolution_missing(self):
        post = _make_post_data()
        q = _make_binary_q_dict(resolution=None)
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="",
            comment_id=1,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=True,
            post_data=post,
        )
        assert rec is None

    def test_returns_none_when_parse_resolution_flags_skip(self):
        # parse_resolution marks "annulled" and "ambiguous" with should_skip=True.
        post = _make_post_data()
        q = _make_binary_q_dict(resolution="annulled")
        rec = _process_single_question(
            post_id=post["id"],
            title=post["title"],
            q=q,
            comment_text="",
            comment_id=1,
            per_model={},
            per_model_numeric_percentiles={},
            was_stacked=True,
            post_data=post,
        )
        assert rec is None


# ---------------------------------------------------------------------------
# End-to-end integration: producer (main.py) + consumer (parsing.py)
# ---------------------------------------------------------------------------


class TestProducerConsumerRoundTrip:
    """The critical integration check: run a fake stacked pipeline through
    main.py's comment construction, then parse the output back with the
    performance_analysis parser and confirm per-model attributions survive."""

    def test_stacked_binary_roundtrip_recovers_models_and_marker(self):
        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question(qid=555)
        bot._question_was_stacked[q.id_of_question] = True

        predictions = [
            _make_prediction_with_model(0.72, "openrouter/openai/gpt-5.5", body="gpt analysis"),
            _make_prediction_with_model(0.68, "openrouter/anthropic/claude-opus-4.7", body="claude analysis"),
            _make_prediction_with_model(0.80, "openrouter/google/gemini-3.1-pro-preview", body="gemini analysis"),
        ]
        research = ResearchWithPredictions(
            research_report="raw research",
            summary_report="summary",
            errors=[],
            predictions=predictions,
        )

        # Produce the annotated summary via the annotation wiring.
        from forecasting_tools.data_models.binary_report import BinaryReport

        parent_summary = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1*: 72.0%\n"
            "*Forecaster 2*: 68.0%\n"
            "*Forecaster 3*: 80.0%\n\n"
            "### Research Summary\nr\n"
        )
        with patch.object(
            ForecastBot,
            "_format_and_expand_research_summary",
            return_value=parent_summary,
        ):
            annotated_summary = TemplateForecaster._format_and_expand_research_summary(
                report_number=1,
                report_type=BinaryReport,
                predicted_research=research,
            )

        # Wrap the annotated summary in a full base explanation so
        # _create_unified_explanation treats it like a real comment.
        base = f"# SUMMARY\n*Question*: ?\n\n{annotated_summary}\n# RESEARCH\nbody\n"
        with patch.object(ForecastBot, "_create_unified_explanation", return_value=base):
            full_comment = bot._create_unified_explanation(q, [research], 0.5, 0.01, 1.0)

        # Producer: marker present, summary annotated.
        assert STACKED_MARKER_TRUE in full_comment
        # Consumer: parse it back.
        assert parse_stacked_marker(full_comment) is True
        per_model = parse_per_model_forecasts(full_comment)
        assert per_model == {
            "gpt-5.5": "72.0%",
            "claude-opus-4.7": "68.0%",
            "gemini-3.1-pro-preview": "80.0%",
        }

    def test_roundtrip_with_stacker_combined_reasoning(self):
        # Real stacking shape: the framework collapses base predictions into ONE
        # aggregated prediction whose reasoning is produced by
        # combine_stacker_and_base_reasoning(). The per-base-model percentiles
        # are only recoverable from the combined reasoning body (the summary
        # bullet shows only the stacker's aggregate).
        from forecasting_tools.data_models.binary_report import BinaryReport

        bot = _make_bot(AggregationStrategy.STACKING)
        q = _make_binary_question(qid=777)
        bot._question_was_stacked[q.id_of_question] = True

        # Base predictions, one per ensemble member, each tagged with Model: by
        # _make_prediction in production (here we build the tag manually).
        base_predictions = [
            ReasonedPrediction(
                prediction_value=0.40,
                reasoning=(
                    "Model: openrouter/openai/gpt-5.5\n\n"
                    "gpt body.\n\n"
                    "Percentile 2.5: 10.0\n"
                    "Percentile 50: 40.0\n"
                    "Percentile 97.5: 80.0\n"
                ),
            ),
            ReasonedPrediction(
                prediction_value=0.50,
                reasoning=(
                    "Model: openrouter/anthropic/claude-opus-4.7\n\n"
                    "claude body.\n\n"
                    "Percentile 2.5: 15.0\n"
                    "Percentile 50: 50.0\n"
                    "Percentile 97.5: 90.0\n"
                ),
            ),
            ReasonedPrediction(
                prediction_value=0.60,
                reasoning=(
                    "Model: openrouter/google/gemini-3.1-pro-preview\n\n"
                    "gemini body.\n\n"
                    "Percentile 2.5: 20.0\n"
                    "Percentile 50: 60.0\n"
                    "Percentile 97.5: 100.0\n"
                ),
            ),
        ]
        meta_text = (
            "Stacker consolidated the three base forecasts.\n\n"
            "Percentile 2.5: 15.0\n"
            "Percentile 50: 50.0\n"
            "Percentile 97.5: 90.0\n"
        )
        combined = combine_stacker_and_base_reasoning(meta_text, base_predictions)
        aggregated = ReasonedPrediction(prediction_value=0.50, reasoning=combined)

        # Production: ResearchWithPredictions has length 1 for stacked questions.
        research = ResearchWithPredictions(
            research_report="raw research",
            summary_report="summary",
            errors=[],
            predictions=[aggregated],
        )
        assert len(research.predictions) == 1

        # Summary bullet shows only the stacker's aggregate (not per-base-model).
        # _format_and_expand_research_summary wraps a parent-returned summary
        # and tries to annotate based on per-prediction Model: prefixes; the
        # aggregated prediction starts with "## Stacker Meta-Analysis", no
        # Model: prefix, so Forecaster 1 stays unannotated.
        parent_summary = "## Report 1 Summary\n### Forecasts\n*Forecaster 1*: 50.0%\n\n### Research Summary\nr\n"
        with patch.object(
            ForecastBot,
            "_format_and_expand_research_summary",
            return_value=parent_summary,
        ):
            annotated_summary = TemplateForecaster._format_and_expand_research_summary(
                report_number=1,
                report_type=BinaryReport,
                predicted_research=research,
            )
        # No Model: prefix on stacker meta → no annotation; bullet left alone.
        assert "*Forecaster 1*: 50.0%" in annotated_summary

        # Build the base unified comment as the framework would: summary +
        # rationales section with a single R1 block wrapping the combined body.
        base_unified = (
            "# SUMMARY\n*Question*: ?\n\n"
            f"{annotated_summary}\n"
            "================================================================================\n"
            "FORECAST SECTION:\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            f"{combined}\n"
        )
        with patch.object(ForecastBot, "_create_unified_explanation", return_value=base_unified):
            # Prevent accidental use of aggregated_prediction_value by passing a
            # placeholder; the bot's method just concatenates the marker.
            full_comment = bot._create_unified_explanation(
                q,
                [research],
                0.5,
                0.01,
                1.0,
            )

        # Producer: stacked marker present.
        assert STACKED_MARKER_TRUE in full_comment
        assert parse_stacked_marker(full_comment) is True

        # Consumer 1: per_model_forecasts reflects ONLY the stacker's bullet
        # (no per-base attribution in the summary).
        forecasts = parse_per_model_forecasts(full_comment)
        assert forecasts == {"Forecaster 1": "50.0%"}

        # Consumer 2: percentiles recover the 3 base models via the
        # stacker-combined-body handler that splits on the delimiter.
        percentiles = parse_per_model_numeric_percentiles(full_comment)
        # Expect at minimum the 3 base models keyed by display name.
        assert "gpt-5.5" in percentiles
        assert "claude-opus-4.7" in percentiles
        assert "gemini-3.1-pro-preview" in percentiles
        assert percentiles["gpt-5.5"] == [(2.5, 10.0), (50.0, 40.0), (97.5, 80.0)]
        assert percentiles["claude-opus-4.7"] == [(2.5, 15.0), (50.0, 50.0), (97.5, 90.0)]
        assert percentiles["gemini-3.1-pro-preview"] == [
            (2.5, 20.0),
            (50.0, 60.0),
            (97.5, 100.0),
        ]

        # Consumer 3: reasoning text recovers 4 entries — stacker + 3 base.
        # The stacker entry is keyed anonymously as "Forecaster 1" since the
        # combined body begins with "## Stacker Meta-Analysis" (no Model: line
        # at the top of the R1 section for the stacker).
        reasoning = parse_per_model_reasoning_text(full_comment)
        assert len(reasoning) == 4
        assert "gpt-5.5" in reasoning
        assert "claude-opus-4.7" in reasoning
        assert "gemini-3.1-pro-preview" in reasoning
        # Stacker key — falls back since stacker's meta has no Model: prefix.
        assert "Forecaster 1" in reasoning
        assert "Stacker consolidated" in reasoning["Forecaster 1"]
        assert "gpt body." in reasoning["gpt-5.5"]
        assert "claude body." in reasoning["claude-opus-4.7"]
        assert "gemini body." in reasoning["gemini-3.1-pro-preview"]
