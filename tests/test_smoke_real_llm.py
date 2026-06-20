"""Real-LLM smoke tests using a free OpenRouter model.

Exercises the actual inference path end-to-end with no mocks.
Every test requires OPENROUTER_API_KEY and is gated behind @pytest.mark.live
so it never runs in CI or during `make test` (addopts excludes `live` by default).

Opt-in: pytest -m live
"""

import asyncio
import os
import re
from datetime import datetime, timedelta
from typing import cast

import pytest
from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericQuestion,
    structure_output,
)
from forecasting_tools.data_models.binary_report import BinaryPrediction

from metaculus_bot.prompts import binary_prompt, multiple_choice_prompt, numeric_prompt

FREE_MODEL = "openrouter/google/gemma-4-31b-it:free"
SKIP_REASON = "OPENROUTER_API_KEY not set"

skip_no_key = pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason=SKIP_REASON)

_NOW = datetime.now()
_OPEN = _NOW - timedelta(days=30)
_RESOLVE = _NOW + timedelta(days=180)


def _make_binary_question() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will the global average temperature in 2026 exceed the 2025 average?",
        id_of_post=99999,
        id_of_question=99999,
        background_info="Climate scientists track annual global mean surface temperature anomalies.",
        resolution_criteria="Resolves YES if the NASA GISS annual mean for 2026 exceeds 2025.",
        fine_print="Uses the January release of the prior year's annual mean.",
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


def _make_numeric_question() -> NumericQuestion:
    return NumericQuestion(
        question_text="What will the US unemployment rate be in December 2026?",
        id_of_post=99998,
        id_of_question=99998,
        background_info="The BLS publishes monthly unemployment figures.",
        resolution_criteria="Resolves to the seasonally-adjusted U-3 rate published by BLS.",
        fine_print="",
        unit_of_measure="percent",
        lower_bound=0.0,
        upper_bound=20.0,
        open_lower_bound=False,
        open_upper_bound=True,
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


def _make_mc_question() -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Which party will win the next UK general election?",
        id_of_post=99997,
        id_of_question=99997,
        options=["Labour", "Conservative", "Liberal Democrats", "Other"],
        background_info="UK general elections use first-past-the-post constituency voting.",
        resolution_criteria="Resolves to the party winning the most seats.",
        fine_print="",
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


def _make_llm() -> GeneralLlm:
    return GeneralLlm(model=FREE_MODEL, temperature=0.0, timeout=120)


@pytest.mark.live
@skip_no_key
async def test_free_model_binary_forecast():
    llm = _make_llm()
    question = _make_binary_question()
    prompt = binary_prompt(question, research="No relevant research found.")

    response = await llm.invoke(prompt)

    assert len(response) > 100, f"Response too short ({len(response)} chars)"
    prob_pattern = re.compile(r"(\d{1,3})%|0\.\d+|1\.0")
    assert prob_pattern.search(response), "No probability-like value found in response"


@pytest.mark.live
@skip_no_key
async def test_free_model_numeric_percentiles():
    llm = _make_llm()
    question = _make_numeric_question()
    prompt = numeric_prompt(
        question,
        research="No relevant research found.",
        lower_bound_message="The lower bound is 0.0% (hard floor).",
        upper_bound_message="The upper bound is 20.0% (open — values above are possible).",
    )

    response = await llm.invoke(prompt)

    percentile_pattern = re.compile(r"Percentile\s+[\d.]+:\s*([\d.]+)", re.IGNORECASE)
    matches = percentile_pattern.findall(response)
    assert len(matches) >= 5, f"Expected >=5 percentile lines, found {len(matches)}"

    values = [float(m) for m in matches]
    assert all(isinstance(v, float) for v in values)
    inversions = sum(1 for i in range(len(values) - 1) if values[i] > values[i + 1])
    assert inversions <= 2, f"Too many inversions ({inversions}) — values not roughly monotonic"


@pytest.mark.live
@skip_no_key
async def test_free_model_mc_forecast():
    llm = _make_llm()
    question = _make_mc_question()
    prompt = multiple_choice_prompt(question, research="No relevant research found.")

    response = await llm.invoke(prompt)

    options_mentioned = sum(1 for opt in question.options if opt.lower() in response.lower())
    assert options_mentioned >= 2, f"Only {options_mentioned} options mentioned in response"

    pct_pattern = re.compile(r"\d{1,3}%")
    assert pct_pattern.search(response), "No percentage-like number found"


@pytest.mark.live
@skip_no_key
async def test_free_model_structured_output_parseable():
    llm = _make_llm()
    question = _make_binary_question()
    prompt = binary_prompt(question, research="No relevant research found.")

    raw_response = await llm.invoke(prompt)

    parser_llm = _make_llm()
    prediction = await structure_output(
        text_to_structure=raw_response,
        output_type=BinaryPrediction,
        model=parser_llm,
    )

    assert isinstance(prediction, BinaryPrediction)
    assert 0.0 < prediction.prediction_in_decimal < 1.0


@pytest.mark.live
@skip_no_key
async def test_full_pipeline_single_question_free_model():
    from forecasting_tools.data_models.forecast_report import ForecastReport

    from main import TemplateForecaster
    from metaculus_bot.aggregation_strategies import AggregationStrategy

    llm = _make_llm()

    async def fake_research(question):
        await asyncio.sleep(0)
        return "No relevant research found. The question is speculative."

    # The "forecasters" slot accepts a list[GeneralLlm] at runtime (consumed by
    # llm_setup), which the parent ForecastBot's dict[str, str | GeneralLlm]
    # signature can't express; cast to satisfy the type checker.
    llms = cast(
        "dict[str, str | GeneralLlm]",
        {
            "forecasters": [llm],
            "default": llm,
            "parser": llm,
            "researcher": llm,
            "summarizer": llm,
        },
    )
    forecaster = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,
        aggregation_strategy=AggregationStrategy.MEAN,
        research_provider=fake_research,
        llms=llms,
        min_forecasters_to_publish=1,
    )

    question = _make_binary_question()
    reports = await forecaster.forecast_questions([question])

    assert len(reports) == 1
    report = reports[0]
    assert isinstance(report, ForecastReport)
    assert isinstance(report.prediction, float)
    assert 0.0 < report.prediction < 1.0
    assert report.explanation and len(report.explanation) > 0
