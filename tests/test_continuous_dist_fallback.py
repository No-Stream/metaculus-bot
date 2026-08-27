from datetime import datetime, timedelta
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools.data_models.numeric_report import Percentile as FTPercentile

from main import TemplateForecaster
from metaculus_bot.exceptions import ValueExtractionError
from metaculus_bot.numeric.discrete_snap import OutcomeTypeResult
from metaculus_bot.value_extraction import ExtractionOutcome


def _stub_open_time() -> datetime:
    return datetime.now() - timedelta(days=30)


def _stub_resolve_time() -> datetime:
    return datetime.now() + timedelta(days=365)


class DummyLLM:  # minimal async LLM for tests
    def __init__(self, reasoning: str):
        self._reasoning = reasoning
        self.model = "dummy-test-model"

    async def invoke(self, prompt: str):
        return self._reasoning


# Lightweight dummy object with the attrs _run_forecast_on_numeric needs.
def make_dummy_numeric_question():
    return SimpleNamespace(
        question_text="dummy numeric",
        background_info="",
        resolution_criteria="",
        fine_print="",
        unit_of_measure=None,
        open_upper_bound=True,
        open_lower_bound=True,
        lower_bound=0,
        upper_bound=9999,
        page_url="https://example.com/q",
        zero_point=0,
        id_of_question=123,  # Added for testing purposes
        cdf_size=201,
        open_time=_stub_open_time(),
        scheduled_resolution_time=_stub_resolve_time(),
    )


@pytest.fixture
def dummy_forecaster():
    # Provide the bare minimum llm config so TemplateForecaster initialises.
    dummy_llm = MagicMock()
    dummy_llm.model = "dummy"
    return TemplateForecaster(
        llms={
            "default": dummy_llm,
            "parser": "mock",
            "researcher": "mock",
            "summarizer": "mock",
        },
        publish_reports_to_metaculus=False,
    )


@pytest.mark.asyncio
async def test_numeric_parsing_success_without_fallback(dummy_forecaster):
    # We expect to use structured-output only; provide a valid structured parse.
    rationale = "irrelevant reasoning; parser output is mocked"
    q = make_dummy_numeric_question()
    llm = DummyLLM(rationale)

    fake_percentiles = [
        FTPercentile(value=v, percentile=p)
        for v, p in zip(
            [90, 95, 100, 110, 120, 130, 135, 140, 150, 160, 170, 175, 180],
            [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99],
            strict=True,
        )
    ]

    with (
        patch(
            "metaculus_bot.forecaster_runners.parse_structured",
            return_value=OutcomeTypeResult(is_discrete_integer=False),
        ),
        patch(
            "metaculus_bot.forecaster_runners.extract_numeric",
            new=AsyncMock(return_value=ExtractionOutcome(value=fake_percentiles, rung="block", block_present=True)),
        ),
    ):
        result = await dummy_forecaster._run_forecast_on_numeric(q, "", llm)  # type: ignore[arg-type]

    values = [p.value for p in result.prediction_value.declared_percentiles]
    # Basic sanity with tail widening enabled: monotone, median unchanged, tails not narrower
    assert len(values) == 13
    assert all(b > a for a, b in pairwise(values)), values
    assert values[6] == pytest.approx(135.0)
    assert values[0] <= 95.0
    assert values[-1] >= 175.0


@pytest.mark.asyncio
async def test_fallback_reraises_when_insufficient_numbers(dummy_forecaster):
    # Rationale has neither a fenced JSON block nor rescuable braces, so the
    # ladder's block+repair rungs fail. The salvage rung's parser
    # (value_extraction.parse_structured — an independent binding from the
    # C3 outcome_type read in forecaster_runners) returns an insufficient
    # 2-percentile list; _validate_numeric rejects the non-13 set and the
    # ladder's terminal rung raises its typed ValueExtractionError. The old
    # text-extraction "no declared_percentiles available" fallback is gone
    # with the F5 router.
    rationale = "Percentile 10: 5\nPercentile 20: 6\n"
    insufficient = [
        FTPercentile(value=5.0, percentile=0.1),
        FTPercentile(value=6.0, percentile=0.2),
    ]

    q = make_dummy_numeric_question()
    llm = DummyLLM(rationale)
    with (
        patch(
            "metaculus_bot.forecaster_runners.parse_structured",
            return_value=OutcomeTypeResult(is_discrete_integer=False),
        ),
        patch(
            "metaculus_bot.value_extraction.parse_structured",
            new=AsyncMock(return_value=insufficient),
        ),
        pytest.raises(ValueExtractionError),
    ):
        await dummy_forecaster._run_forecast_on_numeric(q, "", llm)  # type: ignore[arg-type]
