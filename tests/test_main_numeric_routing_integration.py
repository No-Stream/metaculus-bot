"""Integration test for Workstream E: numeric routing in main._run_forecast_on_numeric.

Mocks the LLM + the parser LLM (structure_output) to feed three rationales
end-to-end through ``TemplateForecaster._run_forecast_on_numeric``:

1. Percentiles-only -> percentile pipeline (sanitize + build_numeric_distribution).

Must produce a 201-point CDF. (The ``numeric_format=...`` log line was
removed 2026-07-08 with the mixture-branch collapse — the router is
percentile-only now.)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from forecasting_tools import GeneralLlm, NumericQuestion
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.numeric.discrete_snap import OutcomeTypeResult
from tests.conftest import make_mock_numeric_question


def _make_bot() -> TemplateForecaster:
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    llms: dict[str, str | GeneralLlm | list[GeneralLlm]] = {
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
        publish_reports_to_metaculus=False,
        aggregation_strategy=AggregationStrategy.MEAN,
        llms=llms,  # type: ignore[arg-type]
        is_benchmarking=True,
    )


def _make_numeric_question() -> NumericQuestion:
    return make_mock_numeric_question(
        id_of_question=9999,
        question_text="Numeric Q",
        background_info="",
        resolution_criteria="",
        with_open_resolve_times=True,
    )


# Thirteen canonical percentiles a real forecaster would emit.
_DECLARED_PERCENTILES: list[tuple[float, float]] = [
    (0.01, 2.0),
    (0.025, 5.0),
    (0.05, 10.0),
    (0.10, 15.0),
    (0.20, 25.0),
    (0.40, 35.0),
    (0.50, 45.0),
    (0.60, 55.0),
    (0.80, 70.0),
    (0.90, 80.0),
    (0.95, 88.0),
    (0.975, 95.0),
    (0.99, 98.0),
]


def _percentile_lines() -> str:
    return "\n".join(
        f"Percentile {int(p * 100) if p * 100 == int(p * 100) else p * 100}: {v}" for p, v in _DECLARED_PERCENTILES
    )


def _percentile_objs() -> list[Percentile]:
    return [Percentile(percentile=p, value=v) for p, v in _DECLARED_PERCENTILES]


def _percentiles_only_rationale() -> str:
    payload = {
        "question_type": "numeric",
        "declared_percentiles": {str(p): v for p, v in _DECLARED_PERCENTILES},
    }
    return f"Analysis.\n\nOUTCOME_TYPE: CONTINUOUS\n\n```json\n{json.dumps(payload)}\n```\n\n{_percentile_lines()}"


def _structure_output_mock(percentile_objs_for_parser: list[Percentile] | None) -> Any:
    """Build an async mock that responds to both structure_output call sites.

    First call expects OutcomeTypeResult; second call expects list[Percentile].
    If percentile_objs_for_parser is None, the second call raises ValueError
    to simulate a case where the parser finds no percentile lines.
    """

    call_count = {"n": 0}

    async def _impl(_reasoning: str, _return_type: Any, *_args: Any, **_kwargs: Any) -> Any:
        # Async checkpoint so flake8-async ASYNC910 is satisfied; the parser
        # LLM in production is async, so this matches the real signature.
        await asyncio.sleep(0)
        call_count["n"] += 1
        if call_count["n"] == 1:
            return OutcomeTypeResult(is_discrete_integer=False)
        if percentile_objs_for_parser is None:
            raise ValueError("no percentiles found")
        return percentile_objs_for_parser

    return _impl


@pytest.mark.asyncio
async def test_percentiles_only_path_logs_format_and_returns_distribution(caplog: pytest.LogCaptureFixture) -> None:
    bot = _make_bot()
    question = _make_numeric_question()
    rationale = _percentiles_only_rationale()

    with patch("metaculus_bot.forecaster_runners.parse_structured", new=_structure_output_mock(_percentile_objs())):
        with patch.object(GeneralLlm, "invoke", new=AsyncMock(return_value=rationale)):
            with caplog.at_level(logging.INFO, logger="metaculus_bot.forecaster_runners"):
                pred = await bot._run_forecast_on_numeric(
                    question, research="r", llm_to_use=GeneralLlm(model="test-model")
                )

    assert isinstance(pred.prediction_value, NumericDistribution)
    cdf = pred.prediction_value.cdf
    assert len(cdf) == 201


@pytest.mark.asyncio
async def test_percentile_path_default_k_tail_is_identity_no_op() -> None:
    """Gap 4: with the new default k_tail=1.0, the percentile pipeline produces
    a NumericDistribution whose declared percentile values match the input
    declarations (modulo deterministic sanitize jitter / clamp). The previous
    k_tail=1.25 default actively widened the tails — this regression catches
    silent reactivation of the old default."""
    bot = _make_bot()
    question = _make_numeric_question()
    rationale = _percentiles_only_rationale()

    with patch("metaculus_bot.forecaster_runners.parse_structured", new=_structure_output_mock(_percentile_objs())):
        with patch.object(GeneralLlm, "invoke", new=AsyncMock(return_value=rationale)):
            pred = await bot._run_forecast_on_numeric(question, research="r", llm_to_use=GeneralLlm(model="test-model"))

    declared = pred.prediction_value.declared_percentiles
    declared_by_pct: dict[float, float] = {round(float(p.percentile), 4): float(p.value) for p in declared}

    # Outer-tail anchors are the ones that would move under k_tail>1; they must
    # match the input declarations to within float precision (jitter is only
    # applied to duplicate values, which our test inputs don't have).
    for input_pct, input_val in _DECLARED_PERCENTILES:
        observed = declared_by_pct.get(round(input_pct, 4))
        assert observed is not None, f"missing declared pct {input_pct}"
        assert abs(observed - input_val) < 1e-6, (
            f"k_tail=1.0 default failed: pct {input_pct} declared {input_val}, got {observed}"
        )


@pytest.mark.asyncio
async def test_percentile_path_with_old_k_tail_visibly_widens_tails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gap 4: monkeypatch the active k_tail to the old 1.25 default and verify
    the tails *do* move. Together with the test above, this proves the new
    k_tail=1.0 default is a true no-op (not silently broken)."""
    monkeypatch.setattr("metaculus_bot.numeric.pipeline.TAIL_WIDEN_K_TAIL", 1.25)

    bot = _make_bot()
    question = _make_numeric_question()
    rationale = _percentiles_only_rationale()

    with patch("metaculus_bot.forecaster_runners.parse_structured", new=_structure_output_mock(_percentile_objs())):
        with patch.object(GeneralLlm, "invoke", new=AsyncMock(return_value=rationale)):
            pred = await bot._run_forecast_on_numeric(question, research="r", llm_to_use=GeneralLlm(model="test-model"))

    declared = pred.prediction_value.declared_percentiles
    declared_by_pct: dict[float, float] = {round(float(p.percentile), 4): float(p.value) for p in declared}

    # k_tail=1.25 widens distance-from-median outside the central 60%. The
    # 2.5% / 97.5% anchors must move at least a touch (>0.1 unit on the [0, 100]
    # range — comfortably above any sanitize-pass jitter).
    p025 = declared_by_pct[0.025]
    p975 = declared_by_pct[0.975]
    assert abs(p025 - 5.0) > 0.1, f"k_tail=1.25 should widen p025 but got {p025} vs declared 5.0"
    assert abs(p975 - 95.0) > 0.1, f"k_tail=1.25 should widen p975 but got {p975} vs declared 95.0"
