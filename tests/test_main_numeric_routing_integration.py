"""Integration test for Workstream E: numeric routing in main._run_forecast_on_numeric.

Mocks the LLM + the parser LLM (structure_output) to feed three rationales
end-to-end through ``TemplateForecaster._run_forecast_on_numeric``:

1. Percentiles-only -> percentile pipeline (sanitize + build_numeric_distribution).
2. Mixture-only -> mixture path (percentiles_to_metaculus_cdf_via_mixture).
3. Both -> mixture wins; warning logged.

Each path must produce a 201-point CDF and emit a ``numeric_format=...``
INFO log line for residual analysis.
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


# Eleven canonical percentiles a real forecaster would emit.
_DECLARED_PERCENTILES: list[tuple[float, float]] = [
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
        "distribution_family_hint": "normal",
    }
    return f"Analysis.\n\nOUTCOME_TYPE: CONTINUOUS\n\n```json\n{json.dumps(payload)}\n```\n\n{_percentile_lines()}"


def _mixture_only_rationale() -> str:
    payload = {
        "question_type": "numeric",
        "declared_percentiles": {str(p): v for p, v in _DECLARED_PERCENTILES},
        "mixture_components": [
            {"weight": 0.25, "mean": 20.0, "sd": 7.0},
            {"weight": 0.50, "mean": 50.0, "sd": 10.0},
            {"weight": 0.25, "mean": 80.0, "sd": 6.0},
        ],
    }
    # No trailing percentile lines — pure mixture path.
    return f"Analysis.\n\nOUTCOME_TYPE: CONTINUOUS\n\n```json\n{json.dumps(payload)}\n```\n"


def _both_rationale() -> str:
    payload = {
        "question_type": "numeric",
        "declared_percentiles": {str(p): v for p, v in _DECLARED_PERCENTILES},
        "mixture_components": [
            {"weight": 0.5, "mean": 30.0, "sd": 5.0},
            {"weight": 0.5, "mean": 70.0, "sd": 5.0},
        ],
    }
    return f"Analysis.\n\nOUTCOME_TYPE: CONTINUOUS\n\n```json\n{json.dumps(payload)}\n```\n\n{_percentile_lines()}"


def _structure_output_mock(percentile_objs_for_parser: list[Percentile] | None) -> Any:
    """Build an async mock that responds to both structure_output call sites.

    First call expects OutcomeTypeResult; second call expects list[Percentile].
    If percentile_objs_for_parser is None, the second call raises ValueError
    to simulate a mixture-only rationale where the percentile parser finds
    nothing.
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

    with patch("metaculus_bot.forecaster_runners.structure_output", new=_structure_output_mock(_percentile_objs())):
        with patch.object(GeneralLlm, "invoke", new=AsyncMock(return_value=rationale)):
            with caplog.at_level(logging.INFO, logger="metaculus_bot.forecaster_runners"):
                pred = await bot._run_forecast_on_numeric(
                    question, research="r", llm_to_use=GeneralLlm(model="test-model")
                )

    assert isinstance(pred.prediction_value, NumericDistribution)
    cdf = pred.prediction_value.cdf
    assert len(cdf) == 201
    assert any("numeric_format=percentiles" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_mixture_only_path_logs_format_and_returns_distribution(caplog: pytest.LogCaptureFixture) -> None:
    bot = _make_bot()
    question = _make_numeric_question()
    rationale = _mixture_only_rationale()

    with patch("metaculus_bot.forecaster_runners.structure_output", new=_structure_output_mock(None)):
        with patch.object(GeneralLlm, "invoke", new=AsyncMock(return_value=rationale)):
            with caplog.at_level(logging.INFO, logger="metaculus_bot.forecaster_runners"):
                pred = await bot._run_forecast_on_numeric(
                    question, research="r", llm_to_use=GeneralLlm(model="test-model")
                )

    assert isinstance(pred.prediction_value, NumericDistribution)
    cdf = pred.prediction_value.cdf
    assert len(cdf) == 201
    # Pure mixture rationale + percentile-parser failure => "mixture".
    assert any(
        "numeric_format=mixture" in rec.message or "numeric_format=both" in rec.message for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_both_path_uses_mixture_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    bot = _make_bot()
    question = _make_numeric_question()
    rationale = _both_rationale()

    with patch("metaculus_bot.forecaster_runners.structure_output", new=_structure_output_mock(_percentile_objs())):
        with patch.object(GeneralLlm, "invoke", new=AsyncMock(return_value=rationale)):
            with caplog.at_level(logging.WARNING, logger="metaculus_bot.numeric_format_router"):
                pred = await bot._run_forecast_on_numeric(
                    question, research="r", llm_to_use=GeneralLlm(model="test-model")
                )

    assert isinstance(pred.prediction_value, NumericDistribution)
    cdf = pred.prediction_value.cdf
    assert len(cdf) == 201
    assert any("both" in rec.message.lower() for rec in caplog.records)


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

    with patch("metaculus_bot.forecaster_runners.structure_output", new=_structure_output_mock(_percentile_objs())):
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

    with patch("metaculus_bot.forecaster_runners.structure_output", new=_structure_output_mock(_percentile_objs())):
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


@pytest.mark.asyncio
async def test_mixture_path_produces_different_cdf_than_percentile_path() -> None:
    """Sanity: mixture branch is not silently falling back to percentiles.

    The mixture is bimodal at 20/80; its CDF at value=50 should differ
    meaningfully from a near-normal percentile-derived CDF.
    """
    bot_a = _make_bot()
    bot_b = _make_bot()
    question = _make_numeric_question()

    pct_rationale = _percentiles_only_rationale()
    mix_rationale = _mixture_only_rationale()

    with patch("metaculus_bot.forecaster_runners.structure_output", new=_structure_output_mock(_percentile_objs())):
        with patch.object(GeneralLlm, "invoke", new=AsyncMock(return_value=pct_rationale)):
            pred_pct = await bot_a._run_forecast_on_numeric(
                question, research="r", llm_to_use=GeneralLlm(model="test-model")
            )

    with patch("metaculus_bot.forecaster_runners.structure_output", new=_structure_output_mock(None)):
        with patch.object(GeneralLlm, "invoke", new=AsyncMock(return_value=mix_rationale)):
            pred_mix = await bot_b._run_forecast_on_numeric(
                question, research="r", llm_to_use=GeneralLlm(model="test-model")
            )

    cdf_pct = pred_pct.prediction_value.cdf
    cdf_mix = pred_mix.prediction_value.cdf

    midpoint_idx = 100  # value=50 on the [0, 100] grid with 201 points
    pct_at_mid = cdf_pct[midpoint_idx].percentile
    mix_at_mid = cdf_mix[midpoint_idx].percentile

    assert abs(pct_at_mid - mix_at_mid) > 0.001, (
        f"mixture CDF at midpoint ({mix_at_mid}) is too close to percentile CDF "
        f"({pct_at_mid}) — mixture branch may be a silent no-op"
    )
