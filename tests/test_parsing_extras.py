from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import BinaryQuestion, GeneralLlm, MultipleChoiceQuestion, PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile

from main import TemplateForecaster
from metaculus_bot.exceptions import ValueExtractionError
from metaculus_bot.numeric.discrete_snap import OutcomeTypeResult
from metaculus_bot.value_extraction import ExtractionOutcome, McForecast


def _stub_open_time() -> datetime:
    return datetime.now() - timedelta(days=30)


def _stub_resolve_time() -> datetime:
    return datetime.now() + timedelta(days=365)


@pytest.mark.asyncio
async def test_binary_parsing_clamps_extremes():
    bot = TemplateForecaster(
        llms={
            "default": "mock",
            "parser": "mock",
            "researcher": "mock",
            "summarizer": "mock",
        }
    )

    # Minimal binary question
    q = MagicMock(spec=BinaryQuestion)
    q.page_url = "http://example.com"
    q.question_text = "?"
    q.background_info = ""
    q.resolution_criteria = ""
    q.fine_print = ""
    q.id_of_question = 1
    q.open_time = _stub_open_time()
    q.scheduled_resolution_time = _stub_resolve_time()

    llm = MagicMock(spec=GeneralLlm)
    llm.model = "parser-test"
    llm.invoke = AsyncMock(return_value="reasoning")

    def _outcome(v: float) -> ExtractionOutcome[float]:
        return ExtractionOutcome(value=v, rung="block", block_present=True)

    # 0.0 gets clamped to BINARY_PROB_MIN (0.02 since 2026-05-12; Atlas-inspired).
    # See scratch_docs_and_planning/atlas_inspired_improvements.md Workstream B.
    with patch(
        "metaculus_bot.forecaster_runners.extract_binary",
        new=AsyncMock(return_value=_outcome(0.0)),
    ):
        res = await bot._run_forecast_on_binary(q, "", llm)
        assert res.prediction_value == 0.02

    # 1.0 gets clamped to BINARY_PROB_MAX (0.98).
    with patch(
        "metaculus_bot.forecaster_runners.extract_binary",
        new=AsyncMock(return_value=_outcome(1.0)),
    ):
        res = await bot._run_forecast_on_binary(q, "", llm)
        assert res.prediction_value == 0.98


@pytest.mark.asyncio
async def test_numeric_parsing_raises_on_wrong_count():
    bot = TemplateForecaster(
        llms={
            "default": "mock",
            "parser": "mock",
            "researcher": "mock",
            "summarizer": "mock",
        }
    )

    q = SimpleNamespace(
        question_text="num?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        unit_of_measure=None,
        open_upper_bound=False,
        open_lower_bound=False,
        lower_bound=0,
        upper_bound=100,
        page_url="http://ex/q",
        zero_point=None,
        id_of_question=2,
        cdf_size=201,
        open_time=_stub_open_time(),
        scheduled_resolution_time=_stub_resolve_time(),
    )

    # Only 5 percentiles returned -> ladder validation drops the wrong count and
    # surfaces the failure as ValueExtractionError (the new failure-type contract
    # after the extraction-ladder refactor; the old parser ValidationError path
    # is gone).
    bad = [Percentile(value=v, percentile=p) for v, p in zip([1, 2, 3, 4, 5], [0.1, 0.2, 0.4, 0.6, 0.8], strict=False)]

    llm = SimpleNamespace(model="dummy")
    llm.invoke = AsyncMock(return_value="rationale")

    # Two independent module bindings are patched here: the C3 outcome_type read
    # goes through forecaster_runners.parse_structured, while the ladder's
    # rung-3 salvage calls value_extraction.parse_structured. The salvage rung
    # returns the wrong-count list (block+repair rungs fail — "rationale" has no
    # fenced block), _validate_numeric rejects the non-13 set, and rung 4 raises
    # ValueExtractionError.
    with (
        patch(
            "metaculus_bot.forecaster_runners.parse_structured",
            return_value=OutcomeTypeResult(is_discrete_integer=False),
        ),
        patch(
            "metaculus_bot.value_extraction.parse_structured",
            new=AsyncMock(return_value=bad),
        ),
        pytest.raises(ValueExtractionError),
    ):
        await bot._run_forecast_on_numeric(q, "", llm)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_parser_llm_used_for_structured_output():
    bot = TemplateForecaster(
        llms={
            "default": "mock",
            "parser": "mock",
            "researcher": "mock",
            "summarizer": "mock",
        }
    )

    sentinel_parser_model = object()
    original_get_llm = bot.get_llm
    bot.get_llm = MagicMock(
        side_effect=lambda purpose, *_: sentinel_parser_model if purpose == "parser" else original_get_llm(purpose)
    )  # type: ignore[method-assign]

    captured = {}

    percentiles_return = [
        Percentile(value=v, percentile=p)
        for v, p in zip(
            [0.25, 0.5, 1, 2, 4, 6, 7, 8, 10, 12, 13, 14, 15],
            [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99],
            strict=False,
        )
    ]

    async def _fake_extract_binary(text, parser_llm, **kwargs):
        # extract_binary signature: (text, parser_llm, *, prompt_notes, question_id, model_name)
        captured["model"] = parser_llm
        return ExtractionOutcome(value=0.5, rung="block", block_present=True)

    async def _fake_extract_mc(text, options, parser_llm, **kwargs):
        # extract_mc signature: (text, options, parser_llm, *, ...)
        captured["model"] = parser_llm

        pol = PredictedOptionList(
            predicted_options=[PredictedOption(option_name=name, probability=1.0 / len(options)) for name in options]
        )
        return ExtractionOutcome(
            value=McForecast(pol, [o.probability for o in pol.predicted_options]), rung="block", block_present=True
        )

    async def _fake_extract_numeric(text, parser_llm, **kwargs):
        # extract_numeric signature: (text, parser_llm, *, ...)
        captured["model"] = parser_llm
        return ExtractionOutcome(value=percentiles_return, rung="block", block_present=True)

    # Minimal binary question
    bq = MagicMock(spec=BinaryQuestion)
    bq.page_url = "url"
    bq.question_text = "?"
    bq.background_info = ""
    bq.resolution_criteria = ""
    bq.fine_print = ""
    bq.id_of_question = 10
    bq.open_time = _stub_open_time()
    bq.scheduled_resolution_time = _stub_resolve_time()
    llm = MagicMock(spec=GeneralLlm)
    llm.model = "m"
    llm.invoke = AsyncMock(return_value="r")

    with patch("metaculus_bot.forecaster_runners.extract_binary", _fake_extract_binary):
        await bot._run_forecast_on_binary(bq, "", llm)
        assert captured["model"] is sentinel_parser_model

    # Minimal multiple-choice question
    mcq = MagicMock(spec=MultipleChoiceQuestion)
    mcq.page_url = "url"
    mcq.question_text = "?"
    mcq.options = ["A", "B"]
    mcq.background_info = ""
    mcq.resolution_criteria = ""
    mcq.fine_print = ""
    mcq.id_of_question = 20
    mcq.open_time = _stub_open_time()
    mcq.scheduled_resolution_time = _stub_resolve_time()
    captured.clear()
    with patch("metaculus_bot.forecaster_runners.extract_mc", _fake_extract_mc):
        await bot._run_forecast_on_multiple_choice(mcq, "", llm)
        assert captured["model"] is sentinel_parser_model

    # Minimal numeric question
    nq = SimpleNamespace(
        question_text="num?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        unit_of_measure=None,
        open_upper_bound=False,
        open_lower_bound=False,
        lower_bound=0,
        upper_bound=100,
        page_url="url",
        zero_point=None,
        id_of_question=11,
        cdf_size=201,
        open_time=_stub_open_time(),
        scheduled_resolution_time=_stub_resolve_time(),
    )
    captured.clear()
    # C3 outcome_type path still calls parse_structured for OutcomeTypeResult when the
    # rationale has no fenced block — stub it to skip that leg.
    with (
        patch(
            "metaculus_bot.forecaster_runners.parse_structured",
            return_value=OutcomeTypeResult(is_discrete_integer=False),
        ),
        patch("metaculus_bot.forecaster_runners.extract_numeric", _fake_extract_numeric),
    ):
        await bot._run_forecast_on_numeric(nq, "", llm)  # type: ignore[arg-type]
        assert captured["model"] is sentinel_parser_model


@pytest.mark.asyncio
async def test_mc_additional_instructions_include_options():
    bot = TemplateForecaster(
        llms={
            "default": "mock",
            "parser": "mock",
            "researcher": "mock",
            "summarizer": "mock",
        }
    )
    q = MagicMock(spec=MultipleChoiceQuestion)
    q.page_url = "url"
    q.question_text = "who?"
    q.options = ["Alpha", "Beta"]
    q.background_info = ""
    q.resolution_criteria = ""
    q.fine_print = ""
    q.id_of_question = 21
    q.open_time = _stub_open_time()
    q.scheduled_resolution_time = _stub_resolve_time()

    llm = MagicMock(spec=GeneralLlm)
    llm.model = "m"
    llm.invoke = AsyncMock(return_value="r")

    seen = {}

    async def _fake_extract_mc(*args, **kwargs):
        seen["prompt_notes"] = kwargs.get("prompt_notes", "")
        pol = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Alpha", probability=0.5),
                PredictedOption(option_name="Beta", probability=0.5),
            ]
        )
        return ExtractionOutcome(
            value=McForecast(pol, [o.probability for o in pol.predicted_options]), rung="block", block_present=True
        )

    with patch("metaculus_bot.forecaster_runners.extract_mc", _fake_extract_mc):
        await bot._run_forecast_on_multiple_choice(q, "", llm)

    ai = seen["prompt_notes"] or ""
    assert "Alpha" in ai
    assert "Beta" in ai
