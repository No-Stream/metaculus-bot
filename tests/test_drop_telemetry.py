"""Per-model forecaster-drop attribution (systematic-failure observability).

Two halves, deliberately tested at different levels:

- The three DROP SITES (soft deadline, a raised exception in the gather done-loop,
  wall-clock cancellation) run through the real bot, because what's under test is
  whether each site knows the model slug, the question id, and a determinable cause
  at the moment it drops a forecaster.
- EMISSION is a pure function of the recorded drops, so those tests call
  ``emit_drop_telemetry`` directly. One end-to-end test covers the delegation and
  the end-of-run call site so the pure tests can't pass while the wiring rots.
"""

import asyncio
import json
import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from forecasting_tools import GeneralLlm, ReasonedPrediction

from main import TemplateForecaster
from metaculus_bot.drop_telemetry import (
    DROP_CAUSE_ERROR_OTHER,
    DROP_CAUSE_TIMEOUT_SOFT_DEADLINE,
    DROP_CAUSE_TIMEOUT_WALL_CLOCK,
    DROP_CAUSE_ZERO_OUTPUT,
    ForecasterDrop,
    emit_drop_telemetry,
)
from metaculus_bot.exceptions import ValueExtractionError

# The emitting logger. Scoped rather than root because caplog spans every logger
# that propagates there, and these tests assert on OUR code staying quiet.
_LOGGER = "metaculus_bot.drop_telemetry"


def _distinct_forecaster_llms(models: list[str]) -> list[MagicMock]:
    """Build mock GeneralLlm forecasters with DISTINCT model slugs so drop
    attribution can be asserted per model."""
    out: list[MagicMock] = []
    for slug in models:
        llm = MagicMock(spec=GeneralLlm)
        llm.model = slug
        llm.invoke = AsyncMock(return_value="mock reasoning")
        out.append(llm)
    return out


def _bot_with_distinct_forecasters(models: list[str], **kwargs: Any) -> TemplateForecaster:
    llms_config = {
        "forecasters": _distinct_forecaster_llms(models),
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config, **kwargs)
    bot._get_notepad = AsyncMock(
        return_value=MagicMock(total_research_reports_attempted=0, total_predictions_attempted=0)
    )
    bot.run_research = AsyncMock(return_value="mock research")
    return bot


class TestDropSiteAttribution:
    """Each site that drops a forecaster must name the model, the question, and why."""

    @pytest.mark.asyncio
    async def test_soft_deadline_records_attribution(self, mock_binary_question, monkeypatch: pytest.MonkeyPatch):
        """Soft-deadline site: cause=timeout_soft_deadline — and the scalar still bumps."""
        bot = _bot_with_distinct_forecasters(["prov/model-slow"], min_forecasters_to_publish=1)
        monkeypatch.setattr("metaculus_bot.forecaster.FORECASTER_SOFT_DEADLINE", 0.05)

        async def slow_make_prediction(question, research, llm, chart_b64=None):
            await asyncio.sleep(5)
            return ReasonedPrediction(prediction_value=0.5, reasoning="never")

        bot._make_prediction = AsyncMock(side_effect=slow_make_prediction)

        with pytest.raises(asyncio.TimeoutError):
            await bot._forecaster_with_soft_deadline(mock_binary_question, "research", bot._forecaster_llms[0], qid=999)

        assert bot._forecasters_dropped_count == 1
        assert bot._forecaster_drops == [
            ForecasterDrop(model="prov/model-slow", qid=999, cause=DROP_CAUSE_TIMEOUT_SOFT_DEADLINE)
        ]

    @pytest.mark.parametrize(
        ("exc", "expected_cause"),
        [
            (ValueExtractionError("all extraction rungs failed"), "parse_extraction"),
            (RuntimeError("LLM answer is an empty string. The model was prov/model-b"), "zero_output"),
            (ValueError("some unexpected boom"), "error_other"),
        ],
    )
    @pytest.mark.asyncio
    async def test_raised_exception_records_cause(self, exc: BaseException, expected_cause: str, mock_binary_question):
        """Gather done-loop site: the raised-exception drop is attributed to the right
        model and classified by inspecting the already-caught exception — zero_output
        reuses llm_retry's own classifier so telemetry agrees with retry."""
        bot = _bot_with_distinct_forecasters(["prov/model-a", "prov/model-b"], min_forecasters_to_publish=1)

        async def per_model(question, research, llm, qid, chart_b64=None):
            if llm.model == "prov/model-a":
                return ReasonedPrediction(prediction_value=0.5, reasoning="ok")
            raise exc

        bot._forecaster_with_soft_deadline = cast(Any, per_model)

        result = await bot._research_and_make_predictions(mock_binary_question)

        assert len(result.predictions) == 1  # survivor still publishes
        assert bot._forecasters_dropped_count == 1
        assert bot._forecaster_drops == [
            ForecasterDrop(model="prov/model-b", qid=mock_binary_question.id_of_question, cause=expected_cause)
        ]

    @pytest.mark.asyncio
    async def test_wall_clock_records_attribution(self, mock_binary_question, monkeypatch: pytest.MonkeyPatch):
        """Wall-clock abort: a cancelled-at-deadline forecaster is recorded with its
        model slug and cause=timeout_wall_clock (distinct from soft-deadline)."""
        monkeypatch.setattr("metaculus_bot.forecaster.PER_QUESTION_WALL_CLOCK_DEADLINE", 0.2)
        monkeypatch.setattr("metaculus_bot.stacking_route.WALL_CLOCK_STACKING_MIN_BUDGET", 0.0)
        bot = _bot_with_distinct_forecasters(["prov/model-fast", "prov/model-slow"], min_forecasters_to_publish=1)

        async def mixed(question, research, llm, qid, chart_b64=None):
            if llm.model == "prov/model-fast":
                return ReasonedPrediction(prediction_value=0.5, reasoning="ok")
            await asyncio.sleep(10)
            return ReasonedPrediction(prediction_value=0.5, reasoning="never")

        bot._forecaster_with_soft_deadline = cast(Any, mixed)

        await bot._research_and_make_predictions(mock_binary_question)

        assert bot._forecaster_drops == [
            ForecasterDrop(
                model="prov/model-slow",
                qid=mock_binary_question.id_of_question,
                cause=DROP_CAUSE_TIMEOUT_WALL_CLOCK,
            )
        ]
        # Continuity: the scalar equals the attributed-drops length.
        assert bot._forecasters_dropped_count == len(bot._forecaster_drops) == 1


class TestDropTelemetryEmission:
    """``systematic`` keys on DISTINCT questions per model, not on raw drop count."""

    def test_marker_and_systematic_warning(self, caplog):
        """A single model dropping across >=2 DISTINCT questions is systematic: it
        surfaces in the marker's systematic= field AND fires a WARNING. The
        FORECASTER_DROPS marker carries a JSON model->cause->count detail blob
        answerable in one grep."""
        drops = [
            ForecasterDrop("prov/model-a", 111, DROP_CAUSE_ZERO_OUTPUT),
            ForecasterDrop("prov/model-a", 222, DROP_CAUSE_ZERO_OUTPUT),
        ]

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            emit_drop_telemetry(drops)

        marker = next(line for line in caplog.messages if line.startswith("FORECASTER_DROPS:"))
        assert "total=2" in marker
        assert "systematic=prov/model-a" in marker
        detail = json.loads(marker.split("detail=", 1)[1])
        assert detail == {"prov/model-a": {"zero_output": 2}}
        assert any(
            rec.levelno == logging.WARNING and rec.message.startswith("SYSTEMATIC_FORECASTER_FAILURE:")
            for rec in caplog.records
        )

    def test_scattered_is_not_systematic(self, caplog):
        """Several models each dropping ONCE is provider-wide scatter, not a single
        model going bad: no systematic model, no WARNING."""
        drops = [
            ForecasterDrop("prov/model-a", 111, DROP_CAUSE_ZERO_OUTPUT),
            ForecasterDrop("prov/model-b", 222, DROP_CAUSE_TIMEOUT_SOFT_DEADLINE),
            ForecasterDrop("prov/model-c", 333, DROP_CAUSE_ERROR_OTHER),
        ]

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            emit_drop_telemetry(drops)

        marker = next(line for line in caplog.messages if line.startswith("FORECASTER_DROPS:"))
        assert "total=3" in marker
        assert "systematic=none" in marker
        assert not any(rec.message.startswith("SYSTEMATIC_FORECASTER_FAILURE:") for rec in caplog.records)

    def test_same_model_one_question_not_systematic(self, caplog):
        """One model dropped multiple times on the SAME question (e.g. a wall-clock
        abort of several members) is not systematic — systematic keys on DISTINCT
        questions, not raw drop count."""
        drops = [
            ForecasterDrop("prov/model-a", 111, DROP_CAUSE_TIMEOUT_WALL_CLOCK),
            ForecasterDrop("prov/model-a", 111, DROP_CAUSE_TIMEOUT_WALL_CLOCK),
        ]

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            emit_drop_telemetry(drops)

        marker = next(line for line in caplog.messages if line.startswith("FORECASTER_DROPS:"))
        assert "systematic=none" in marker
        assert not any(rec.message.startswith("SYSTEMATIC_FORECASTER_FAILURE:") for rec in caplog.records)

    def test_clean_run_emits_zero_marker_no_warning(self, caplog):
        """A clean run (zero drops) emits the marker at total=0 for archive presence,
        but no per-model summary and no spurious WARNING."""
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            emit_drop_telemetry([])

        marker = next(line for line in caplog.messages if line.startswith("FORECASTER_DROPS:"))
        assert "total=0" in marker
        assert "systematic=none" in marker
        assert not any(line.startswith("Forecaster drops by model:") for line in caplog.messages)
        # Scoped to our own loggers: caplog.records spans every logger that propagates
        # to root, so a third-party WARNING (forecasting-tools' missing-METACULUS_TOKEN
        # notice on CI) would otherwise fail an assertion that is about OUR code
        # staying quiet.
        our_warnings = [
            r for r in caplog.records if r.levelno >= logging.WARNING and r.name.startswith("metaculus_bot")
        ]
        assert not our_warnings, [r.getMessage() for r in our_warnings]


@pytest.mark.asyncio
async def test_end_of_run_emits_recorded_drops(caplog):
    """The delegation + call site, so the pure emission tests above can't pass while
    the wiring rots: ``forecast_questions`` must drain the bot's recorded drops
    through the module at end of run."""
    bot = _bot_with_distinct_forecasters(["prov/model-a"], min_forecasters_to_publish=1)
    bot._record_forecaster_drop(model="prov/model-a", qid=111, cause=DROP_CAUSE_ZERO_OUTPUT)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        await bot.forecast_questions([])

    marker = next(line for line in caplog.messages if line.startswith("FORECASTER_DROPS:"))
    assert "total=1" in marker
    assert json.loads(marker.split("detail=", 1)[1]) == {"prov/model-a": {"zero_output": 1}}
