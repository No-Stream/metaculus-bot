"""
Additional tests to cover PCHIP fallback/smoothing paths and input validation.
Concise scenarios to increase confidence in complex numeric forecast flow.
"""

from datetime import datetime, timedelta
from itertools import pairwise
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from forecasting_tools import GeneralLlm
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion
from pydantic import ValidationError

from metaculus_bot.numeric.discrete_snap import OutcomeTypeResult
from metaculus_bot.numeric.pipeline import _apply_jitter_and_clamp as apply_jitter_and_clamp
from metaculus_bot.value_extraction import ExtractionOutcome


def _numeric_extract_mock(plist):
    return AsyncMock(return_value=ExtractionOutcome(value=plist, rung="block", block_present=True))


def _as_numeric_question(q: SimpleNamespace) -> NumericQuestion:
    return cast(NumericQuestion, q)


def _as_general_llm(llm: object) -> GeneralLlm:
    return cast(GeneralLlm, llm)


def _stub_open_time() -> datetime:
    return datetime.now() - timedelta(days=30)


def _stub_resolve_time() -> datetime:
    return datetime.now() + timedelta(days=365)


def _make_forecaster():
    from forecasting_tools import GeneralLlm

    from main import TemplateForecaster

    mock_llms: dict[str, str | GeneralLlm] = {
        "default": MagicMock(),
        "parser": MagicMock(),
        "researcher": MagicMock(),
        "summarizer": MagicMock(),
    }
    return TemplateForecaster(llms=mock_llms, publish_reports_to_metaculus=False)


def _make_question(**overrides):
    opts = {
        "open_upper_bound": False,
        "open_lower_bound": False,
        "upper_bound": 100.0,
        "lower_bound": 0.0,
        "zero_point": None,
        "id_of_question": 4242,
        "question_text": "num?",
        "background_info": "",
        "resolution_criteria": "",
        "fine_print": "",
        "unit_of_measure": "units",
        "page_url": "https://example/q/4242",
        "cdf_size": 201,
        "open_time": _stub_open_time(),
        "scheduled_resolution_time": _stub_resolve_time(),
    }
    opts.update(overrides)
    return SimpleNamespace(**opts)


class DummyLLM:
    def __init__(self, reasoning: str = "r"):
        self._reasoning = reasoning
        self.model = "dummy"

    async def invoke(self, prompt: str):
        return self._reasoning


@pytest.mark.asyncio
@patch("metaculus_bot.numeric.pchip_cdf.generate_pchip_cdf", side_effect=RuntimeError("boom"))
@patch("metaculus_bot.numeric.pchip_cdf.percentiles_to_pchip_format", return_value={})
async def test_pchip_fallback_success(mock_format, mock_generate, caplog):
    f = _make_forecaster()
    q = _make_question()

    # Valid 13-percentile set
    plist = [
        Percentile(percentile=p, value=v)
        for p, v in zip(
            [0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.975, 0.99],
            [1, 2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5, 99],
            strict=False,
        )
    ]

    with (
        patch(
            "metaculus_bot.forecaster_runners.parse_structured",
            return_value=OutcomeTypeResult(is_discrete_integer=False),
        ),
        patch("metaculus_bot.forecaster_runners.extract_numeric", new=_numeric_extract_mock(plist)),
    ):
        caplog.clear()
        caplog.set_level("WARNING")
        result = await f._run_forecast_on_numeric(_as_numeric_question(q), "", _as_general_llm(DummyLLM()))

    # Fallback warning emitted
    assert any("PCHIP CDF construction failed" in rec.message for rec in caplog.records)

    # Fallback NumericDistribution returns a cdf that is monotone
    c = result.prediction_value.cdf  # type: ignore[attr-defined]
    probs = [p.percentile for p in c]
    assert all(a <= b for a, b in pairwise(probs))


@pytest.mark.asyncio
@patch("metaculus_bot.numeric.pchip_cdf.generate_pchip_cdf", side_effect=RuntimeError("boom"))
@patch("metaculus_bot.numeric.pchip_cdf.percentiles_to_pchip_format", return_value={})
async def test_pchip_fallback_failure_diagnostics(mock_format, mock_generate, caplog):
    f = _make_forecaster()
    q = _make_question()

    plist = [
        Percentile(percentile=p, value=v)
        for p, v in zip(
            [0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.975, 0.99],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            strict=False,
        )
    ]

    # Force fallback CDF construction to raise via a fake class. ft 0.2.92 builds
    # the CDF in get_cdf() (the fallback path and diagnostics call it directly);
    # cdf is only a deprecated alias delegating to it. Mirror that surface so the
    # spacing AssertionError surfaces where the production code actually reads it.
    class FakeND:
        def __init__(self, *args, **kwargs):
            self.declared_percentiles = plist

        def get_cdf(self):
            raise AssertionError("Percentiles at indices are too close")

        @property
        def cdf(self):
            return self.get_cdf()

    with (
        patch(
            "metaculus_bot.forecaster_runners.parse_structured",
            return_value=OutcomeTypeResult(is_discrete_integer=False),
        ),
        patch("metaculus_bot.forecaster_runners.extract_numeric", new=_numeric_extract_mock(plist)),
        patch("metaculus_bot.numeric.pchip_processing.NumericDistribution", FakeND),
    ):
        caplog.clear()
        caplog.set_level("ERROR")
        with pytest.raises(AssertionError):
            await f._run_forecast_on_numeric(_as_numeric_question(q), "", _as_general_llm(DummyLLM()))

    # Rich diagnostics logged
    msgs = [r.message for r in caplog.records]
    assert any("Numeric CDF spacing assertion" in m for m in msgs)
    assert any("Bounds=" in m and "Declared percentiles" in m for m in msgs)


@pytest.mark.asyncio
@patch("metaculus_bot.numeric.pchip_cdf.percentiles_to_pchip_format", return_value={})
async def test_smoothing_respects_open_bounds(mock_format, caplog):
    f = _make_forecaster()
    # Open bounds question
    q = _make_question(open_upper_bound=True, open_lower_bound=True)

    # Tiny deltas to trigger smoothing
    base = np.linspace(0.0, 1.0, 201)
    base[50:55] = base[50] + np.linspace(0, 1e-8, 5)

    with patch(
        "metaculus_bot.numeric.pchip_cdf.generate_pchip_cdf",
        return_value=(base.tolist(), False),
    ):
        plist = [
            Percentile(percentile=p, value=v)
            for p, v in zip(
                [0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.975, 0.99],
                [0, 2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5, 100],
                strict=False,
            )
        ]
        with (
            patch(
                "metaculus_bot.forecaster_runners.parse_structured",
                return_value=OutcomeTypeResult(is_discrete_integer=False),
            ),
            patch("metaculus_bot.forecaster_runners.extract_numeric", new=_numeric_extract_mock(plist)),
        ):
            caplog.clear()
            caplog.set_level("WARNING")
            result = await f._run_forecast_on_numeric(_as_numeric_question(q), "", _as_general_llm(DummyLLM()))

    # Smoothing log
    assert any("CDF ramp smoothing" in rec.message for rec in caplog.records)

    # Endpoints pinned for open bounds
    c = result.prediction_value.cdf  # type: ignore[attr-defined]
    probs = [p.percentile for p in c]
    assert probs[0] >= 0.001
    assert probs[-1] <= 0.999


@pytest.mark.asyncio
async def test_numeric_percentile_set_validation():
    f = _make_forecaster()
    q = _make_question()

    # 13 items but wrong set (0.03 instead of 0.025): filtering to the standard
    # set drops the non-standard 0.03, so the count falls to 12 and validation
    # fails on the count check — same "wrong label" intent as the old 11-set.
    bad = [
        Percentile(percentile=p, value=v)
        for p, v in zip(
            [0.01, 0.03, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.975, 0.99],
            [1, 3, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5, 99],
            strict=False,
        )
    ]

    # The bad set must arrive through the EXTRACTION LADDER, and parse_structured must
    # return an OutcomeTypeResult. This test previously patched parse_structured with the
    # percentile list, but at that point in run_numeric_forecast that call serves the C3
    # outcome_type read, so production died on `outcome_result.is_discrete_integer` with an
    # AttributeError before validation ran — which the bare `raises(Exception)` absorbed,
    # leaving the test green while asserting nothing about percentile validation. Assert the
    # specific ValidationError, not Exception, so an unrelated crash can't satisfy it again.
    with (
        patch(
            "metaculus_bot.forecaster_runners.parse_structured",
            return_value=OutcomeTypeResult(is_discrete_integer=False),
        ),
        patch(
            "metaculus_bot.forecaster_runners.extract_numeric",
            new=AsyncMock(return_value=ExtractionOutcome(value=bad, rung="block", block_present=True)),
        ),
        pytest.raises(ValidationError),
    ):
        await f._run_forecast_on_numeric(_as_numeric_question(q), "", _as_general_llm(DummyLLM()))


@pytest.mark.asyncio
@patch("metaculus_bot.numeric.pipeline.generate_pchip_cdf")
@patch("metaculus_bot.numeric.pipeline.percentiles_to_pchip_format", return_value={})
async def test_discrete_zero_point_override(mock_format, mock_generate):
    f = _make_forecaster()
    # Discrete (non-201) and zero_point provided → should pass zero_point=None into pchip.
    # Patch pipeline's OWN bindings: the discrete question builds directly on its coarse
    # grid via build_numeric_distribution, which resolves both names at module top here.
    q = _make_question(cdf_size=101, zero_point=0.0)

    # Return a valid coarse-grid CDF to avoid fallback.
    mock_generate.return_value = (np.linspace(0.0, 1.0, 101).tolist(), False)

    plist = [
        Percentile(percentile=p, value=v)
        for p, v in zip(
            [0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.975, 0.99],
            [1, 2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5, 99],
            strict=False,
        )
    ]

    with (
        patch(
            "metaculus_bot.forecaster_runners.parse_structured",
            return_value=OutcomeTypeResult(is_discrete_integer=False),
        ),
        patch("metaculus_bot.forecaster_runners.extract_numeric", new=_numeric_extract_mock(plist)),
    ):
        await f._run_forecast_on_numeric(_as_numeric_question(q), "", _as_general_llm(DummyLLM()))

    # Capture the call arguments to ensure zero_point=None was used
    _args, kwargs = mock_generate.call_args
    assert kwargs.get("zero_point", "sentinel") is None


def test_lower_bound_adjacent_cluster(caplog):
    _make_forecaster()
    # Closed lower bound; cluster near lower
    q = _make_question(
        open_upper_bound=False,
        open_lower_bound=False,
        lower_bound=0.0,
        upper_bound=100.0,
    )
    raw = [
        Percentile(percentile=0.01, value=0.0),
        Percentile(percentile=0.025, value=0.0),
        Percentile(percentile=0.05, value=0.0),
        Percentile(percentile=0.10, value=0.0),
        Percentile(percentile=0.20, value=0.1),
        Percentile(percentile=0.40, value=5.0),
        Percentile(percentile=0.50, value=10.0),
        Percentile(percentile=0.60, value=15.0),
        Percentile(percentile=0.80, value=20.0),
        Percentile(percentile=0.90, value=30.0),
        Percentile(percentile=0.95, value=40.0),
        Percentile(percentile=0.975, value=50.0),
        Percentile(percentile=0.99, value=60.0),
    ]

    caplog.clear()
    caplog.set_level("WARNING")
    adjusted = apply_jitter_and_clamp(raw, _as_numeric_question(q))

    vals = [p.value for p in adjusted]
    assert all(q.lower_bound <= v <= q.upper_bound for v in vals)
    assert all(b > a for a, b in pairwise(vals)), vals
    msgs = [rec.message for rec in caplog.records]
    assert any("Cluster spread applied" in m for m in msgs)
    assert any("Corrected numeric distribution" in m for m in msgs)


@pytest.mark.asyncio
async def test_binary_parse_additional_instructions_capture():
    from forecasting_tools import BinaryQuestion, GeneralLlm

    from main import TemplateForecaster

    bot = TemplateForecaster(llms={"default": "m", "parser": "p", "researcher": "r", "summarizer": "s"})
    q = MagicMock(spec=BinaryQuestion)
    q.page_url = "http://ex"
    q.question_text = "?"
    q.background_info = ""
    q.resolution_criteria = ""
    q.fine_print = ""
    q.id_of_question = 7
    q.open_time = _stub_open_time()
    q.scheduled_resolution_time = _stub_resolve_time()

    # Dummy forecaster LLM
    llm = MagicMock(spec=GeneralLlm)
    llm.model = "m"
    llm.invoke = AsyncMock(return_value="rationale")

    seen = {}

    async def _fake_extract_binary(*args, **kwargs):
        seen["prompt_notes"] = kwargs.get("prompt_notes", "")
        return ExtractionOutcome(value=0.5, rung="block", block_present=True)

    with patch("metaculus_bot.forecaster_runners.extract_binary", _fake_extract_binary):
        await bot._run_forecast_on_binary(q, "", llm)

    ai = seen.get("prompt_notes", "")
    assert "decimal in [0,1]" in ai
    assert "NN%" in ai
    assert "NN/100" in ai
