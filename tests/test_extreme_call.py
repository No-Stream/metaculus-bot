"""Tests for the per-member EXTREME_CALL telemetry (metaculus_bot/extreme_call.py).

Two layers, deliberately:

* the pure formatter (`format_extreme_call_markers`) — band edges, the lone/accompanied
  split, the survivor count, and the unknown-model sentinel;
* the real `_research_and_make_predictions` path — that the lines land next to
  FORECASTERS_SURVIVED on a binary question and are absent on a multiple-choice one.

The second layer is what pins the SCOPE of the marker. The formatter can be correct
while the wiring emits on every question type, and the finding it feeds (gemini's
lone-extreme rate, `scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md` §2)
is binary-only, so an MC line would silently pollute the numerator.
"""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from forecasting_tools import (
    MultipleChoiceQuestion,
    NumericDistribution,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
)

from metaculus_bot.constants import EXTREME_CALL_HIGH, EXTREME_CALL_LOW
from metaculus_bot.extreme_call import format_extreme_call_markers
from metaculus_bot.forecaster import TemplateForecaster
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from tests.conftest import make_mock_general_llm, make_mock_numeric_question

QID = 44874  # the question the finding is built on: a lone 0.03 on a YES resolution


def _fields(line: str) -> dict[str, str]:
    """Split one marker line's ``key=value`` tail into a dict (prefix dropped)."""
    head, _, tail = line.partition(": ")
    assert head == "EXTREME_CALL", f"unexpected marker head: {head!r}"
    return dict(token.split("=", 1) for token in tail.split())


class TestExtremeBandEdges:
    def test_low_and_high_calls_each_emit_one_line_with_their_side(self):
        lines = format_extreme_call_markers(QID, [("gemini-3.1-pro-preview", 0.03), ("gpt-5.6-sol", 0.97)])

        assert len(lines) == 2
        low, high = (_fields(line) for line in lines)
        assert low["model"] == "gemini-3.1-pro-preview"
        assert low["p"] == "0.0300"
        assert low["side"] == "low"
        assert high["model"] == "gpt-5.6-sol"
        assert high["p"] == "0.9700"
        assert high["side"] == "high"
        assert [low["question"], high["question"]] == [str(QID), str(QID)]

    def test_confident_but_unextreme_members_emit_nothing(self):
        # The denominator for a rate claim is FORECASTERS_SURVIVED + question type, so
        # a non-extreme member deliberately leaves no line at all.
        assert format_extreme_call_markers(QID, [("m1", 0.10), ("m2", 0.90), ("m3", 0.5)]) == []

    def test_band_membership_is_inclusive_at_both_edges(self):
        # The band is `p <= EXTREME_CALL_LOW or p >= EXTREME_CALL_HIGH`, so a call
        # sitting exactly on an edge IS extreme. Pinned because the neighbouring
        # value one tick inside must stay silent, and an off-by-one here would move
        # every future rate read.
        on_edges = format_extreme_call_markers(QID, [("m1", EXTREME_CALL_LOW), ("m2", EXTREME_CALL_HIGH)])
        assert [_fields(line)["side"] for line in on_edges] == ["low", "high"]
        assert format_extreme_call_markers(QID, [("m1", 0.0501), ("m2", 0.9499)]) == []

    def test_no_members_yields_no_lines(self):
        assert format_extreme_call_markers(QID, []) == []


class TestLoneVersusAccompanied:
    def test_two_extremes_on_the_same_side_are_both_accompanied(self):
        lines = format_extreme_call_markers(QID, [("m1", 0.03), ("m2", 0.04), ("m3", 0.40)])

        assert len(lines) == 2
        assert all(_fields(line)["lone"] == "false" for line in lines)

    def test_an_extreme_on_the_opposite_side_does_not_accompany(self):
        # The actual finding is same-SIDE agreement: two members straddling the middle
        # at 0.03 and 0.97 disagree maximally, so each is lone. Reading "another member
        # is extreme" as accompaniment would erase the split entirely.
        lines = format_extreme_call_markers(QID, [("m1", 0.03), ("m2", 0.97)])

        assert len(lines) == 2
        assert all(_fields(line)["lone"] == "true" for line in lines)
        assert [_fields(line)["side"] for line in lines] == ["low", "high"]

    def test_lone_extreme_among_moderate_members(self):
        lines = format_extreme_call_markers(QID, [("gemini-3.1-pro-preview", 0.03), ("m2", 0.35), ("m3", 0.30)])

        assert len(lines) == 1
        assert _fields(lines[0])["lone"] == "true"
        assert _fields(lines[0])["survivors"] == "3"


class TestSurvivorCount:
    def test_survivors_reports_every_member_not_just_the_extreme_ones(self):
        lines = format_extreme_call_markers(QID, [("m1", 0.02), ("m2", 0.5), ("m3", 0.5)])

        assert _fields(lines[0])["survivors"] == "3"

    def test_at_one_survivor_lone_is_vacuous_and_survivors_says_so(self):
        # A 1-of-3 publish makes "nobody else agreed" unavoidable rather than
        # informative, which is why the RECOMMENDATION excludes the degraded-window
        # records from the lone test. survivors=1 is the field that lets an analysis
        # condition on it instead of re-deriving it from another marker.
        lines = format_extreme_call_markers(QID, [("gemini-3.1-pro-preview", 0.03)])

        assert len(lines) == 1
        fields = _fields(lines[0])
        assert fields["lone"] == "true"
        assert fields["survivors"] == "1"


class TestUnknownModelName:
    def test_a_missing_model_prefix_renders_the_unknown_sentinel(self):
        # extract_model_display_name_from_reasoning returns None when a forecaster's
        # reasoning carries no "Model:" prefix. FORECASTERS_SURVIVED spells that
        # "unknown"; the same sentinel here keeps the two lines joinable, and it is
        # not a marker-parser None sentinel, so it survives harvesting as a string.
        lines = format_extreme_call_markers(QID, [(None, 0.03)])

        assert _fields(lines[0])["model"] == "unknown"

    def test_an_empty_model_name_is_also_unknown(self):
        assert _fields(format_extreme_call_markers(QID, [("", 0.97)])[0])["model"] == "unknown"


def _bot_with_forecasters(n: int) -> TemplateForecaster:
    """A minimal bot whose research and fan-out the caller stubs out.

    Mirrors the construction in tests/test_template_forecaster.py: default MEAN
    aggregation (no stacker, so no provider call), min_forecasters_to_publish=1 so a
    single survivor still publishes.
    """
    llm = make_mock_general_llm()
    llms_config: dict[str, Any] = {
        "forecasters": [llm] * n,
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=1)
    bot._get_notepad = AsyncMock(
        return_value=MagicMock(total_research_reports_attempted=0, total_predictions_attempted=0)
    )
    bot.run_research = AsyncMock(return_value="mock research")
    return bot


def _serve_predictions(bot: TemplateForecaster, predictions: list[ReasonedPrediction]) -> None:
    """Hand the fan-out a fixed prediction list, one per configured forecaster."""
    served = iter(predictions)
    bot._forecaster_with_soft_deadline = cast(Any, AsyncMock(side_effect=lambda *_args, **_kwargs: next(served)))


def _mc_question(question_id: int = 45244) -> MagicMock:
    question = MagicMock(spec=MultipleChoiceQuestion)
    question.id_of_question = question_id
    question.page_url = f"https://example.com/{question_id}"
    question.question_text = "Which option occurs?"
    question.background_info = "MC background"
    question.resolution_criteria = "Resolves to the realised option"
    question.fine_print = ""
    question.options = ["A", "B"]
    return question


def _mc_prediction(model: str, probability_a: float) -> ReasonedPrediction:
    return ReasonedPrediction(
        prediction_value=PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=probability_a),
                PredictedOption(option_name="B", probability=1.0 - probability_a),
            ]
        ),
        reasoning=f"Model: openrouter/vendor/{model}\n\nreasoning body",
    )


def _numeric_prediction(model: str) -> ReasonedPrediction:
    """A well-formed declared-percentile distribution on the shared [0, 100] mock question."""
    offsets = [-24, -20, -17, -14, -10, -4, 0, 4, 10, 14, 17, 20, 24]
    return ReasonedPrediction(
        prediction_value=NumericDistribution(
            declared_percentiles=[
                Percentile(value=50.0 + offset, percentile=pct)
                for offset, pct in zip(offsets, STANDARD_PERCENTILES, strict=True)
            ],
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
        ),
        reasoning=f"Model: openrouter/vendor/{model}\n\nreasoning body",
    )


class TestEmissionThroughTheRealForecastPath:
    @pytest.mark.asyncio
    async def test_binary_lone_extreme_is_logged_after_the_survivor_count(
        self, mock_binary_question: MagicMock, caplog: pytest.LogCaptureFixture
    ):
        bot = _bot_with_forecasters(3)
        _serve_predictions(
            bot,
            [
                ReasonedPrediction(
                    prediction_value=0.03, reasoning="Model: openrouter/google/gemini-3.1-pro-preview\n\nlow"
                ),
                ReasonedPrediction(prediction_value=0.30, reasoning="Model: openrouter/openai/gpt-5.6-sol\n\nmid"),
                ReasonedPrediction(
                    prediction_value=0.35, reasoning="Model: openrouter/anthropic/claude-opus-4.8\n\nmid"
                ),
            ],
        )

        with caplog.at_level("INFO"):
            result = await bot._research_and_make_predictions(mock_binary_question)

        assert len(result.predictions) == 3
        messages = [record.getMessage() for record in caplog.records]
        extreme = [m for m in messages if m.startswith("EXTREME_CALL:")]
        assert len(extreme) == 1, f"expected exactly one extreme member logged; got {extreme}"
        fields = _fields(extreme[0])
        assert fields == {
            "question": str(mock_binary_question.id_of_question),
            "model": "gemini-3.1-pro-preview",
            "p": "0.0300",
            "side": "low",
            "lone": "true",
            "survivors": "3",
        }
        # Ordered after FORECASTERS_SURVIVED, so one grep of the log reads the survivor
        # count and then the extreme calls it contained.
        assert messages.index(extreme[0]) > next(
            i for i, m in enumerate(messages) if m.startswith("FORECASTERS_SURVIVED:")
        )

    @pytest.mark.asyncio
    async def test_binary_accompanied_extremes_log_one_line_each(
        self, mock_binary_question: MagicMock, caplog: pytest.LogCaptureFixture
    ):
        bot = _bot_with_forecasters(2)
        _serve_predictions(
            bot,
            [
                ReasonedPrediction(prediction_value=0.96, reasoning="Model: openrouter/openai/gpt-5.6-sol\n\nhigh"),
                ReasonedPrediction(
                    prediction_value=0.97, reasoning="Model: openrouter/google/gemini-3.1-pro-preview\n\nhigh"
                ),
            ],
        )

        with caplog.at_level("INFO"):
            await bot._research_and_make_predictions(mock_binary_question)

        extreme = [_fields(m) for m in (r.getMessage() for r in caplog.records) if m.startswith("EXTREME_CALL:")]
        assert [(f["model"], f["side"], f["lone"]) for f in extreme] == [
            ("gpt-5.6-sol", "high", "false"),
            ("gemini-3.1-pro-preview", "high", "false"),
        ]

    @pytest.mark.asyncio
    async def test_multiple_choice_question_emits_no_extreme_call_line(self, caplog: pytest.LogCaptureFixture):
        # A dominant MC option is not an extreme binary call, and MC concentration is a
        # separate cut the operator did not include. An option probability of 0.97 must
        # therefore leave no EXTREME_CALL line — the marker is binary-scoped.
        bot = _bot_with_forecasters(2)
        _serve_predictions(bot, [_mc_prediction("gemini-3.1-pro-preview", 0.97), _mc_prediction("gpt-5.6-sol", 0.96)])

        with caplog.at_level("INFO"):
            await bot._research_and_make_predictions(_mc_question())

        messages = [record.getMessage() for record in caplog.records]
        assert any(m.startswith("FORECASTERS_SURVIVED:") for m in messages), (
            "the MC path did not reach the emission site"
        )
        assert not [m for m in messages if m.startswith("EXTREME_CALL:")]

    @pytest.mark.asyncio
    async def test_numeric_question_emits_no_extreme_call_line(self, caplog: pytest.LogCaptureFixture):
        # Same scope guard on the other non-binary type: a numeric member carries a
        # distribution rather than a probability, so there is nothing for the band to
        # read and the marker must stay silent.
        bot = _bot_with_forecasters(2)
        _serve_predictions(bot, [_numeric_prediction("gemini-3.1-pro-preview"), _numeric_prediction("gpt-5.6-sol")])

        with caplog.at_level("INFO"):
            await bot._research_and_make_predictions(make_mock_numeric_question(with_open_resolve_times=True))

        messages = [record.getMessage() for record in caplog.records]
        assert any(m.startswith("FORECASTERS_SURVIVED:") for m in messages), (
            "the numeric path did not reach the emission site"
        )
        assert not [m for m in messages if m.startswith("EXTREME_CALL:")]
