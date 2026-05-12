"""Runtime-activation tests: _make_prediction calls run_tools_for_forecaster.

These tests exercise the wiring step in ``main.py:_make_prediction``: once
the feature flag is ON, a rationale that carries a valid structured JSON
block should come back with a ``## Computed quantities`` section appended.

We mock ``forecasting-tools`` framework pieces (no real LLM calls) — the
point is to assert the *wiring*, not the math (the math is already covered
by ``tests/test_tool_runner.py``).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from forecasting_tools import BinaryPrediction, BinaryQuestion, GeneralLlm, ReasonedPrediction

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.tool_runner import FEATURE_FLAG_ENV


def _binary_payload() -> dict:
    return {
        "question_type": "binary",
        "prior": {"prob": 0.2, "source": "history"},
        "base_rate": {"k": 3, "n": 12, "ref_class": "years matching precondition"},
        "hazard": {
            "rate_per_unit": 0.25,
            "unit": "year",
            "window_duration_units": 1.0,
            "elapsed_fraction": 0.3,
            "remaining_fraction": 0.7,
        },
        "evidence": [{"summary": "policy shift", "direction": "up", "strength": "moderate"}],
        "scenarios": [],
        "posterior_prob": 0.35,
    }


def _rationale_with_json(payload: dict) -> str:
    return (
        "PHASE 1: OUTSIDE VIEW.\n\n"
        "Historical base rate 3 of 12.\n\n"
        f"```json\n{json.dumps(payload)}\n```\n\n"
        "Probability: 35%"
    )


def _make_bot() -> TemplateForecaster:
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    llms: dict = {
        "forecasters": [test_llm],
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
        aggregation_strategy=AggregationStrategy.MEAN,
        llms=llms,  # type: ignore[arg-type]
        is_benchmarking=True,
    )


def _make_binary_q() -> BinaryQuestion:
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = 12345
    q.id_of_post = 12345
    q.question_text = "Will X happen?"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.page_url = "https://example.com/q/12345"
    q.open_time = datetime.now() - timedelta(days=30)
    q.scheduled_resolution_time = datetime.now() + timedelta(days=365)
    return q


class _FakeNotepad:
    """Minimal notepad stub — real _get_notepad touches the API."""

    def __init__(self) -> None:
        self.total_predictions_attempted = 0
        self.total_research_reports_attempted = 0


async def _fake_get_notepad(_self, _q):
    await asyncio.sleep(0)
    return _FakeNotepad()


def _run_make_prediction(bot: TemplateForecaster, q, llm, rationale: str) -> ReasonedPrediction:
    """Drive _make_prediction end-to-end with the forecaster + notepad stubbed.

    Patching _get_notepad + _run_forecast_on_binary together because:
    - _get_notepad hits the framework's notepad cache / upstream API;
    - _run_forecast_on_binary is the real LLM call we want to short-circuit.
    """

    async def fake_forecast(_q, _r, _llm):
        await asyncio.sleep(0)
        return ReasonedPrediction(prediction_value=0.35, reasoning=rationale)

    with (
        patch.object(TemplateForecaster, "_run_forecast_on_binary", side_effect=fake_forecast),
        patch.object(TemplateForecaster, "_get_notepad", _fake_get_notepad),
    ):
        return asyncio.run(bot._make_prediction(q, "research", llm))


class TestMakePredictionAppendsComputedQuantities:
    def test_flag_on_appends_computed_quantities_section(self, monkeypatch):
        monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
        bot = _make_bot()
        q = _make_binary_q()
        rationale = _rationale_with_json(_binary_payload())

        llm = GeneralLlm(model="gpt-5-test", temperature=0.0)
        out: ReasonedPrediction = _run_make_prediction(bot, q, llm, rationale)

        assert "## Computed quantities" in out.reasoning
        # A binary block with prior + base_rate + hazard should light up
        # at least the Beta-binomial line.
        assert "Beta-binomial" in out.reasoning

    def test_flag_off_no_computed_quantities_section(self, monkeypatch):
        monkeypatch.delenv(FEATURE_FLAG_ENV, raising=False)
        bot = _make_bot()
        q = _make_binary_q()
        rationale = _rationale_with_json(_binary_payload())

        llm = GeneralLlm(model="gpt-5-test", temperature=0.0)
        out: ReasonedPrediction = _run_make_prediction(bot, q, llm, rationale)

        assert "## Computed quantities" not in out.reasoning
        assert "Beta-binomial" not in out.reasoning

    def test_flag_on_no_json_block_no_section(self, monkeypatch):
        # Flag on but rationale has no JSON block → no section appended.
        monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
        bot = _make_bot()
        q = _make_binary_q()
        rationale = "Just prose.\n\nProbability: 30%"

        llm = GeneralLlm(model="gpt-5-test", temperature=0.0)
        out: ReasonedPrediction = _run_make_prediction(bot, q, llm, rationale)

        assert "## Computed quantities" not in out.reasoning

    def test_model_tag_still_prepended(self, monkeypatch):
        # Regression guard: computed quantities should NOT displace the
        # "Model: ..." header that downstream attribution parsers rely on.
        monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
        bot = _make_bot()
        q = _make_binary_q()
        rationale = _rationale_with_json(_binary_payload())

        llm = GeneralLlm(model="gpt-5-test", temperature=0.0)
        out: ReasonedPrediction = _run_make_prediction(bot, q, llm, rationale)

        assert out.reasoning.startswith("Model: gpt-5-test")


# Silence "imported but unused" for pytest / BinaryPrediction — BinaryPrediction
# is used implicitly to guide type-checkers aware of forecasting_tools.
_ = BinaryPrediction
_ = pytest
