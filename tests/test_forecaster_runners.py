"""Tests for metaculus_bot.forecaster_runners — extracted per-type forecast functions.

Exercises the three public functions (run_binary_forecast, run_mc_forecast,
run_numeric_forecast) to verify they produce the same results as the original
TemplateForecaster methods they replaced.

The runners now delegate value extraction to the deterministic-first
``value_extraction`` ladder (block → repair → llm). Tests here mostly patch the
top-level ladder entrypoints (``extract_binary``, ``extract_mc``,
``extract_numeric``) so we exercise the caller's post-processing without
re-testing ladder internals. A handful of integration-style tests feed a
rationale with a real fenced JSON block so the ladder runs end-to-end and we
can assert an ``EXTRACTION_RUNG`` telemetry line was emitted.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import litellm.exceptions as litellm_exc
import pytest
from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.ai_models.ai_utils.openai_utils import VisionMessageData
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile
from pydantic import ValidationError

from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN
from metaculus_bot.exceptions import UnitMismatchError, ValueExtractionError
from metaculus_bot.forecaster_runners import run_binary_forecast, run_mc_forecast, run_numeric_forecast
from metaculus_bot.llm_retry import TRANSIENT_RETRY_MAX_ELAPSED_S
from metaculus_bot.numeric.discrete_snap import OutcomeTypeResult
from metaculus_bot.value_extraction import ExtractionOutcome


@pytest.fixture
def parser_llm():
    return GeneralLlm(model="test-parser")


@pytest.fixture
def forecaster_llm():
    return GeneralLlm(model="test-forecaster")


@pytest.fixture
def binary_question():
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = 1001
    q.page_url = "https://metaculus.com/questions/1001"
    q.question_text = "Will X happen?"
    return q


@pytest.fixture
def mc_question():
    q = MagicMock(spec=MultipleChoiceQuestion)
    q.id_of_question = 2001
    q.page_url = "https://metaculus.com/questions/2001"
    q.question_text = "Which outcome?"
    q.options = ["Option A", "Option B", "Option C"]
    return q


@pytest.fixture
def numeric_question():
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = 3001
    q.page_url = "https://metaculus.com/questions/3001"
    q.lower_bound = 0
    q.upper_bound = 1000
    q.open_lower_bound = False
    q.open_upper_bound = True
    q.unit_of_measure = "widgets"
    return q


def _binary_outcome(value: float) -> ExtractionOutcome[float]:
    return ExtractionOutcome(value=value, rung="block", block_present=True)


def _mc_outcome(pol: PredictedOptionList) -> ExtractionOutcome[PredictedOptionList]:
    return ExtractionOutcome(value=pol, rung="block", block_present=True)


def _make_option_list(options: list[tuple[str, float]]) -> PredictedOptionList:
    return PredictedOptionList(predicted_options=[PredictedOption(option_name=n, probability=p) for n, p in options])


_STANDARD_PERCENTILES: list[Percentile] = [
    Percentile(percentile=p / 100, value=v)
    for p, v in zip(
        [2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5],
        [50, 100, 150, 200, 350, 450, 550, 700, 800, 900, 950],
        strict=True,
    )
]


class TestRunBinaryForecast:
    @pytest.mark.asyncio
    async def test_returns_reasoned_prediction_with_clamped_value(
        self, binary_question, forecaster_llm, parser_llm
    ) -> None:
        """Binary forecast clamps to [BINARY_PROB_MIN, BINARY_PROB_MAX] and returns ReasonedPrediction."""
        reasoning_text = "Analysis: likely yes.\n\nProbability: 75%"

        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
            patch(
                "metaculus_bot.forecaster_runners.extract_binary",
                new=AsyncMock(return_value=_binary_outcome(0.75)),
            ),
        ):
            result = await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert isinstance(result, ReasonedPrediction)
        assert result.prediction_value == 0.75
        assert result.reasoning == reasoning_text

    @pytest.mark.asyncio
    async def test_clamps_below_minimum(self, binary_question, forecaster_llm, parser_llm) -> None:
        """Values below BINARY_PROB_MIN are clamped up."""
        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="Very unlikely")),
            patch(
                "metaculus_bot.forecaster_runners.extract_binary",
                new=AsyncMock(return_value=_binary_outcome(0.001)),
            ),
        ):
            result = await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert result.prediction_value == BINARY_PROB_MIN

    @pytest.mark.asyncio
    async def test_clamps_above_maximum(self, binary_question, forecaster_llm, parser_llm) -> None:
        """Values above BINARY_PROB_MAX are clamped down."""
        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="Nearly certain")),
            patch(
                "metaculus_bot.forecaster_runners.extract_binary",
                new=AsyncMock(return_value=_binary_outcome(0.999)),
            ),
        ):
            result = await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert result.prediction_value == BINARY_PROB_MAX

    @pytest.mark.asyncio
    async def test_extraction_rung_logged_via_real_ladder(
        self, binary_question, forecaster_llm, parser_llm, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Integration-style: no ladder patch; a rationale with a valid binary block
        goes through ``extract_binary`` end-to-end, clamps to BINARY_PROB_MAX, and
        emits an ``EXTRACTION_RUNG`` INFO line on the ``metaculus_bot.value_extraction``
        logger.
        """
        caplog.set_level(logging.INFO, logger="metaculus_bot.value_extraction")
        reasoning_text = 'Nearly certain.\n\n```json\n{"question_type": "binary", "posterior_prob": 0.99}\n```'

        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
        ):
            result = await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert result.prediction_value == BINARY_PROB_MAX  # 0.99 clamped down
        rung_msgs = [r.getMessage() for r in caplog.records if "EXTRACTION_RUNG" in r.getMessage()]
        assert len(rung_msgs) == 1
        msg = rung_msgs[0]
        assert "question=1001" in msg
        assert "model=test-forecaster" in msg
        assert "qtype=binary" in msg
        assert "rung=block" in msg
        assert "block_present=True" in msg


class TestWindowDeclaredTelemetry:
    """``WINDOW_DECLARED`` reports the pricing window a member SAYS it used beside the
    window it actually had. The optional ``remaining_window_days`` block field is the
    source; a member that omits it leaves no line, so the archive is a declaration
    census rather than a coverage one. On qid 43837 six members priced the full question
    window when 16 days of it had already elapsed event-free, which this line makes a
    query instead of a re-read of every rationale.

    The block is parsed for telemetry only — nothing downstream reads the field — so
    these tests run the real ``parse_structured_block`` and assert on the log.
    """

    _FUTURE_CLOSE = datetime(2026, 12, 31, tzinfo=UTC)

    @staticmethod
    def _window_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [r.getMessage() for r in caplog.records if "WINDOW_DECLARED" in r.getMessage()]

    @staticmethod
    def _binary_rationale(*, declared: int | None) -> str:
        window = "" if declared is None else f', "remaining_window_days": {declared}'
        return f'Analysis.\n\n```json\n{{"question_type": "binary", "posterior_prob": 0.4{window}}}\n```'

    @pytest.mark.asyncio
    async def test_declared_window_is_logged_against_the_real_one(
        self, binary_question, forecaster_llm, parser_llm, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster_runners")
        binary_question.close_time = datetime.now(UTC) + timedelta(days=10)

        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=self._binary_rationale(declared=90))),
        ):
            await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        (msg,) = self._window_lines(caplog)
        assert "question=1001" in msg
        assert "model=test-forecaster" in msg
        assert "declared_days=90" in msg
        # The real remaining window, to one decimal — the ratio against declared_days is
        # the cut (90 declared against 10 real is the full-window pricing shape). Rounding
        # holds to one decimal until the test itself takes over an hour.
        assert "actual_days=10.0" in msg

    @pytest.mark.asyncio
    async def test_no_line_when_the_forecaster_declares_nothing(
        self, binary_question, forecaster_llm, parser_llm, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The field is optional, so silence must stay silence: an absent record means
        "did not declare", never "priced the window correctly"."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster_runners")
        binary_question.close_time = self._FUTURE_CLOSE

        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=self._binary_rationale(declared=None))),
        ):
            await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert self._window_lines(caplog) == []

    @pytest.mark.asyncio
    async def test_no_line_when_the_rationale_has_no_block_at_all(
        self, binary_question, forecaster_llm, parser_llm, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster_runners")
        binary_question.close_time = self._FUTURE_CLOSE

        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="Analysis.\n\nProbability: 40%")),
            patch(
                "metaculus_bot.forecaster_runners.extract_binary",
                new=AsyncMock(return_value=_binary_outcome(0.40)),
            ),
        ):
            await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert self._window_lines(caplog) == []

    @pytest.mark.asyncio
    async def test_missing_close_time_logs_the_none_sentinel(
        self, binary_question, forecaster_llm, parser_llm, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Backtests and ablations forecast resolved questions and can carry no close
        time. ``n/a`` is what the marker parser coerces to None, so a ratio cut skips the
        record instead of reading a measured zero."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster_runners")
        binary_question.close_time = None

        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=self._binary_rationale(declared=30))),
        ):
            await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        (msg,) = self._window_lines(caplog)
        assert "declared_days=30" in msg
        assert "actual_days=n/a" in msg

    @pytest.mark.asyncio
    async def test_mc_runner_logs_its_own_declaration(
        self, mc_question, forecaster_llm, parser_llm, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster_runners")
        mc_question.close_time = datetime.now(UTC) + timedelta(days=5)
        ballot = '{"Option A": 0.5, "Option B": 0.3, "Option C": 0.2}'
        rationale = (
            "Analysis.\n\n```json\n"
            f'{{"question_type": "multiple_choice", "option_probs": {ballot}, "remaining_window_days": 60}}\n```'
        )

        with (
            patch("metaculus_bot.forecaster_runners.multiple_choice_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=rationale)),
        ):
            await run_mc_forecast(mc_question, "research", forecaster_llm, parser_llm)

        (msg,) = self._window_lines(caplog)
        assert "question=2001" in msg
        assert "declared_days=60" in msg

    @pytest.mark.asyncio
    async def test_line_survives_an_extraction_failure(
        self, binary_question, forecaster_llm, parser_llm, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The declaration is logged before extraction, so it still reports what the
        model wrote on a question whose forecaster is ultimately dropped."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.forecaster_runners")
        binary_question.close_time = self._FUTURE_CLOSE
        boom = ValueExtractionError("no usable value")

        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=self._binary_rationale(declared=15))),
            patch("metaculus_bot.forecaster_runners.extract_binary", new=AsyncMock(side_effect=boom)),
            pytest.raises(ValueExtractionError),
        ):
            await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        (msg,) = self._window_lines(caplog)
        assert "declared_days=15" in msg


class TestForecasterBroadRetry:
    """The forecaster invoke is wrapped in the broad, 30s-gated retry (Round-2).

    Forecaster GeneralLlm instances are set allowed_tries=1 in llm_configs.py, so
    this wrapper is their SOLE retry layer: it recovers a fast blip but obeys the
    universal "no retry after 30s" deadline-safety rule that forecasting-tools'
    un-gated tenacity could not.
    """

    @pytest.mark.asyncio
    async def test_fast_blip_on_forecaster_invoke_retries_then_succeeds(
        self, binary_question, forecaster_llm, parser_llm
    ) -> None:
        """A fast litellm.Timeout on forecaster_llm.invoke is retried; the next call wins."""
        invoke = AsyncMock(
            side_effect=[
                litellm_exc.Timeout("blip", model="m", llm_provider="openrouter"),
                "Analysis: likely yes.\n\nProbability: 70%",
            ]
        )
        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch("metaculus_bot.llm_retry.asyncio.sleep", new=AsyncMock()),
            patch.object(forecaster_llm, "invoke", new=invoke),
            patch(
                "metaculus_bot.forecaster_runners.extract_binary",
                new=AsyncMock(return_value=_binary_outcome(0.70)),
            ),
        ):
            result = await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert result.prediction_value == 0.70
        assert invoke.await_count == 2

    @pytest.mark.asyncio
    async def test_slow_forecaster_failure_not_retried(self, binary_question, forecaster_llm, parser_llm) -> None:
        """A forecaster invoke that fails past the 30s gate is NOT retried — it propagates."""
        invoke = AsyncMock(side_effect=litellm_exc.Timeout("stall", model="m", llm_provider="openrouter"))
        clock = iter([0.0] + [TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0] * 20)

        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch("metaculus_bot.llm_retry.time.monotonic", lambda: next(clock)),
            patch.object(forecaster_llm, "invoke", new=invoke),
            pytest.raises(litellm_exc.Timeout),
        ):
            await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert invoke.await_count == 1


def _last_invoke_arg(invoke: AsyncMock) -> object:
    """Return the single positional arg of the most recent await, narrowing the
    Optional ``await_args`` for the type checker."""
    await_args = invoke.await_args
    assert await_args is not None, "invoke was never awaited"
    (arg,) = await_args.args
    return arg


class TestForecasterChartVision:
    """The optional TS-anchor chart image (chart_b64) turns the forecaster invoke
    input into a ``VisionMessageData``; absent it stays a bare prompt string. All
    three runners share the ``_forecaster_input`` helper, so binary covers the wiring
    and one numeric case confirms it threads through the numeric path too.
    """

    @pytest.mark.asyncio
    async def test_no_chart_invokes_with_bare_prompt(self, binary_question, forecaster_llm, parser_llm) -> None:
        invoke = AsyncMock(return_value="Analysis.\n\nProbability: 40%")
        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="PROMPT"),
            patch.object(forecaster_llm, "invoke", new=invoke),
            patch(
                "metaculus_bot.forecaster_runners.extract_binary",
                new=AsyncMock(return_value=_binary_outcome(0.40)),
            ),
        ):
            await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert _last_invoke_arg(invoke) == "PROMPT"  # plain string, not VisionMessageData

    @pytest.mark.asyncio
    async def test_chart_wraps_invoke_in_vision_message(self, binary_question, forecaster_llm, parser_llm) -> None:
        invoke = AsyncMock(return_value="Analysis.\n\nProbability: 40%")
        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="PROMPT"),
            patch.object(forecaster_llm, "invoke", new=invoke),
            patch(
                "metaculus_bot.forecaster_runners.extract_binary",
                new=AsyncMock(return_value=_binary_outcome(0.40)),
            ),
        ):
            await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm, chart_b64="ZmFrZQ==")

        called_arg = _last_invoke_arg(invoke)
        assert isinstance(called_arg, VisionMessageData)
        assert called_arg.prompt == "PROMPT"
        assert called_arg.b64_image == "ZmFrZQ=="
        assert called_arg.image_resolution == "low"

    @pytest.mark.asyncio
    async def test_numeric_runner_wraps_chart(self, numeric_question, forecaster_llm, parser_llm) -> None:
        invoke = AsyncMock(return_value="reasoning")
        mock_parse_structured = AsyncMock(return_value=OutcomeTypeResult(is_discrete_integer=False))
        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="PROMPT"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper", "lower")),
            patch.object(forecaster_llm, "invoke", new=invoke),
            patch("metaculus_bot.forecaster_runners.parse_structured", new=mock_parse_structured),
            patch(
                "metaculus_bot.forecaster_runners.extract_numeric",
                new=AsyncMock(
                    return_value=ExtractionOutcome(value=_STANDARD_PERCENTILES, rung="block", block_present=True)
                ),
            ),
            patch("metaculus_bot.forecaster_runners.sanitize_percentiles", return_value=(_STANDARD_PERCENTILES, None)),
            patch(
                "metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=MagicMock()
            ) as mock_build,
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(False, "")),
            patch("metaculus_bot.forecaster_runners.log_final_prediction"),
            patch("metaculus_bot.forecaster_runners.log_open_bound_piling_diagnostics"),
        ):
            await run_numeric_forecast(numeric_question, "research", forecaster_llm, parser_llm, chart_b64="ZmFrZQ==")

        called_arg = _last_invoke_arg(invoke)
        assert isinstance(called_arg, VisionMessageData)
        assert called_arg.b64_image == "ZmFrZQ=="
        # A lost model_name kwarg attributes CDF_MAXSTEP_CLIP to model=unknown silently.
        assert mock_build.call_args.kwargs["model_name"] == forecaster_llm.model


class TestRunMcForecast:
    @pytest.mark.asyncio
    async def test_returns_reasoned_prediction_with_option_list(self, mc_question, forecaster_llm, parser_llm) -> None:
        """MC forecast returns a ReasonedPrediction with PredictedOptionList."""
        reasoning_text = "Option A most likely."
        option_list = _make_option_list([("Option A", 0.6), ("Option B", 0.3), ("Option C", 0.1)])

        with (
            patch("metaculus_bot.forecaster_runners.multiple_choice_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
            patch(
                "metaculus_bot.forecaster_runners.extract_mc",
                new=AsyncMock(return_value=_mc_outcome(option_list)),
            ),
            patch("metaculus_bot.forecaster_runners.clamp_and_renormalize_mc", return_value=option_list),
        ):
            result = await run_mc_forecast(mc_question, "research", forecaster_llm, parser_llm)

        assert isinstance(result, ReasonedPrediction)
        assert result.prediction_value == option_list
        assert result.reasoning == reasoning_text

    @pytest.mark.asyncio
    async def test_real_ladder_maps_block_options_onto_question(self, mc_question, forecaster_llm, parser_llm) -> None:
        """Integration-style: rationale with a valid MC block goes through the real
        ``extract_mc`` ladder (no patch) and the returned option list maps onto
        ``question.options`` in canonical order.
        """
        reasoning_text = (
            "Option A is most likely.\n\n"
            "```json\n"
            '{"question_type": "multiple_choice", "option_probs": '
            '{"Option A": 0.6, "Option B": 0.3, "Option C": 0.1}}\n'
            "```"
        )

        with (
            patch("metaculus_bot.forecaster_runners.multiple_choice_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
        ):
            result = await run_mc_forecast(mc_question, "research", forecaster_llm, parser_llm)

        assert isinstance(result, ReasonedPrediction)
        pol = result.prediction_value
        assert isinstance(pol, PredictedOptionList)
        names = [o.option_name for o in pol.predicted_options]
        assert set(names) == {"Option A", "Option B", "Option C"}
        total = sum(o.probability for o in pol.predicted_options)
        assert abs(total - 1.0) < 1e-6


class TestRunNumericForecast:
    @pytest.mark.asyncio
    async def test_percentile_branch_returns_prediction_and_discrete_vote(
        self, numeric_question, forecaster_llm, parser_llm
    ) -> None:
        """Numeric forecast returns (prediction, discrete_vote) tuple via the percentile branch."""
        reasoning_text = "OUTCOME_TYPE: DISCRETE\n\nPercentile 2.5: 50"
        mock_prediction = MagicMock(spec=NumericDistribution)

        mock_parse_structured = AsyncMock(return_value=OutcomeTypeResult(is_discrete_integer=True))

        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="prompt"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper msg", "lower msg")),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
            patch("metaculus_bot.forecaster_runners.parse_structured", new=mock_parse_structured),
            patch(
                "metaculus_bot.forecaster_runners.extract_numeric",
                new=AsyncMock(
                    return_value=ExtractionOutcome(value=_STANDARD_PERCENTILES, rung="block", block_present=True)
                ),
            ),
            patch(
                "metaculus_bot.forecaster_runners.sanitize_percentiles",
                return_value=(_STANDARD_PERCENTILES, None),
            ),
            patch(
                "metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=mock_prediction
            ) as mock_build,
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(False, "")),
            patch("metaculus_bot.forecaster_runners.log_final_prediction"),
        ):
            prediction, discrete_vote = await run_numeric_forecast(
                numeric_question, "research", forecaster_llm, parser_llm
            )

        assert prediction.prediction_value == mock_prediction
        assert discrete_vote is True
        # A lost model_name kwarg attributes CDF_MAXSTEP_CLIP to model=unknown silently.
        assert mock_build.call_args.kwargs["model_name"] == forecaster_llm.model

    @pytest.mark.asyncio
    async def test_sanitize_percentiles_receives_the_forecaster_model_name(
        self, numeric_question, forecaster_llm, parser_llm
    ) -> None:
        """The NUMERIC_DEGENERATE_DECLARATION marker attributes a collapsed declaration
        to the forecaster that made it, and that attribution exists only because this
        caller passes ``model_name`` — a ``model=unknown`` in the archive means a caller
        forgot to (see the ``sanitize_percentiles`` docstring)."""
        mock_parse_structured = AsyncMock(return_value=OutcomeTypeResult(is_discrete_integer=False))

        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="prompt"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper msg", "lower msg")),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="reasoning")),
            patch("metaculus_bot.forecaster_runners.parse_structured", new=mock_parse_structured),
            patch(
                "metaculus_bot.forecaster_runners.extract_numeric",
                new=AsyncMock(
                    return_value=ExtractionOutcome(value=_STANDARD_PERCENTILES, rung="block", block_present=True)
                ),
            ),
            patch(
                "metaculus_bot.forecaster_runners.sanitize_percentiles",
                return_value=(_STANDARD_PERCENTILES, None),
            ) as mock_sanitize,
            patch("metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=MagicMock()),
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(False, "")),
            patch("metaculus_bot.forecaster_runners.log_final_prediction"),
        ):
            await run_numeric_forecast(numeric_question, "research", forecaster_llm, parser_llm)

        assert mock_sanitize.call_args.kwargs["model_name"] == forecaster_llm.model

    @pytest.mark.asyncio
    async def test_unit_mismatch_raises(self, numeric_question, forecaster_llm, parser_llm) -> None:
        """When detect_unit_mismatch returns True, raises UnitMismatchError."""
        mock_parse_structured = AsyncMock(return_value=OutcomeTypeResult(is_discrete_integer=False))

        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="prompt"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper msg", "lower msg")),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="reasoning")),
            patch("metaculus_bot.forecaster_runners.parse_structured", new=mock_parse_structured),
            patch(
                "metaculus_bot.forecaster_runners.extract_numeric",
                new=AsyncMock(
                    return_value=ExtractionOutcome(value=_STANDARD_PERCENTILES, rung="block", block_present=True)
                ),
            ),
            patch(
                "metaculus_bot.forecaster_runners.sanitize_percentiles",
                return_value=(_STANDARD_PERCENTILES, None),
            ),
            patch("metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=MagicMock()),
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(True, "off by 1000x")),
            pytest.raises(UnitMismatchError, match="off by 1000x"),
        ):
            await run_numeric_forecast(numeric_question, "research", forecaster_llm, parser_llm)

    @pytest.mark.asyncio
    async def test_c3_block_read_skips_parser_call_for_outcome_type(
        self, numeric_question, forecaster_llm, parser_llm
    ) -> None:
        """F7: when the rationale's structured JSON block declares outcome_type,
        discrete_vote is read from the block and parse_structured is NOT called for
        OutcomeTypeResult. Percentiles come from the ladder (patched); parse_structured
        is patched to a sentinel that fails the test if invoked.
        """
        reasoning_text = (
            "Some rationale text.\n"
            "```json\n"
            '{"question_type": "numeric", "outcome_type": "discrete_integer",'
            ' "declared_percentiles": {"0.1": 10.0, "0.5": 50.0, "0.9": 90.0}}\n'
            "```\n"
        )

        mock_parse_structured = AsyncMock(
            side_effect=AssertionError("parse_structured should not be called for outcome_type when block declares it")
        )

        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="prompt"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper msg", "lower msg")),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
            patch("metaculus_bot.forecaster_runners.parse_structured", new=mock_parse_structured),
            patch(
                "metaculus_bot.forecaster_runners.extract_numeric",
                new=AsyncMock(
                    return_value=ExtractionOutcome(value=_STANDARD_PERCENTILES, rung="block", block_present=True)
                ),
            ),
            patch(
                "metaculus_bot.forecaster_runners.sanitize_percentiles",
                return_value=(_STANDARD_PERCENTILES, None),
            ),
            patch("metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=MagicMock()),
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(False, "")),
            patch("metaculus_bot.forecaster_runners.log_final_prediction"),
        ):
            _, discrete_vote = await run_numeric_forecast(numeric_question, "research", forecaster_llm, parser_llm)

        assert discrete_vote is True
        assert mock_parse_structured.await_count == 0  # confirms parse_structured was never called

    @pytest.mark.asyncio
    async def test_discrete_vote_none_when_parse_fails(self, numeric_question, forecaster_llm, parser_llm) -> None:
        """When OUTCOME_TYPE parsing fails, discrete_vote is None."""
        mock_parse_structured = AsyncMock(side_effect=ValidationError.from_exception_data(title="test", line_errors=[]))

        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="prompt"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper msg", "lower msg")),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="reasoning")),
            patch("metaculus_bot.forecaster_runners.parse_structured", new=mock_parse_structured),
            patch(
                "metaculus_bot.forecaster_runners.extract_numeric",
                new=AsyncMock(
                    return_value=ExtractionOutcome(value=_STANDARD_PERCENTILES, rung="block", block_present=True)
                ),
            ),
            patch(
                "metaculus_bot.forecaster_runners.sanitize_percentiles",
                return_value=(_STANDARD_PERCENTILES, None),
            ),
            patch("metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=MagicMock()),
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(False, "")),
            patch("metaculus_bot.forecaster_runners.log_final_prediction"),
        ):
            _, discrete_vote = await run_numeric_forecast(numeric_question, "research", forecaster_llm, parser_llm)

        assert discrete_vote is None
