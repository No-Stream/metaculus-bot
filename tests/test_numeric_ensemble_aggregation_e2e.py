"""End-to-end ensemble aggregation tests covering the mixture × percentile mix.

Gap 3: when MULTIPLE forecasters produce CDFs from a mix of mixture-derived
and percentile-derived branches, ``combine_numeric_predictions`` must
aggregate them into a single Metaculus-compliant 201-point CDF without
loss of monotonicity, min-step compliance, or correct endpoint pinning.

The tests build three forecasters' rationales, push each through
``_run_forecast_on_numeric``, then aggregate. We choose three rationales
because that is the smallest count that exercises non-trivial mean/median
divergence (n=2 produces identical mean/median).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from forecasting_tools import GeneralLlm, NumericQuestion
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy, combine_numeric_predictions
from metaculus_bot.numeric.discrete_snap import OutcomeTypeResult
from tests.conftest import make_mock_numeric_question

# ---------------------------------------------------------------------------
# Fixtures: a question + helpers to build rationales for each branch
# ---------------------------------------------------------------------------


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
        id_of_question=4242,
        question_text="Numeric Q for ensemble",
        background_info="",
        resolution_criteria="",
        with_open_resolve_times=True,
    )


_DECLARED_PERCENTILES_A: list[tuple[float, float]] = [
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

_DECLARED_PERCENTILES_B: list[tuple[float, float]] = [
    (0.025, 8.0),
    (0.05, 14.0),
    (0.10, 19.0),
    (0.20, 27.0),
    (0.40, 38.0),
    (0.50, 50.0),
    (0.60, 60.0),
    (0.80, 75.0),
    (0.90, 83.0),
    (0.95, 90.0),
    (0.975, 96.0),
]

_DECLARED_PERCENTILES_C: list[tuple[float, float]] = [
    (0.025, 3.0),
    (0.05, 7.0),
    (0.10, 12.0),
    (0.20, 22.0),
    (0.40, 32.0),
    (0.50, 40.0),
    (0.60, 50.0),
    (0.80, 65.0),
    (0.90, 75.0),
    (0.95, 85.0),
    (0.975, 92.0),
]


def _percentile_lines(decl: list[tuple[float, float]]) -> str:
    return "\n".join(f"Percentile {int(p * 100) if p * 100 == int(p * 100) else p * 100}: {v}" for p, v in decl)


def _percentile_objs(decl: list[tuple[float, float]]) -> list[Percentile]:
    return [Percentile(percentile=p, value=v) for p, v in decl]


def _percentiles_only_rationale(decl: list[tuple[float, float]]) -> str:
    payload = {
        "question_type": "numeric",
        "declared_percentiles": {str(p): v for p, v in decl},
        "distribution_family_hint": "normal",
    }
    return f"Analysis.\n\nOUTCOME_TYPE: CONTINUOUS\n\n```json\n{json.dumps(payload)}\n```\n\n{_percentile_lines(decl)}"


def _mixture_only_rationale(decl: list[tuple[float, float]], components: list[dict[str, float]]) -> str:
    payload = {
        "question_type": "numeric",
        "declared_percentiles": {str(p): v for p, v in decl},
        "mixture_components": components,
    }
    # No trailing percentile lines — pure mixture branch.
    return f"Analysis.\n\nOUTCOME_TYPE: CONTINUOUS\n\n```json\n{json.dumps(payload)}\n```\n"


def _structure_output_mock(percentile_objs_for_parser: list[Percentile] | None) -> Any:
    """Mocks the two structure_output calls in _run_forecast_on_numeric: first
    OutcomeTypeResult, then list[Percentile]. Pass None for percentile_objs to
    simulate a mixture-only rationale where the parser finds no lines."""

    call_count = {"n": 0}

    async def _impl(_reasoning: str, _return_type: Any, *_args: Any, **_kwargs: Any) -> Any:
        await asyncio.sleep(0)
        call_count["n"] += 1
        if call_count["n"] == 1:
            return OutcomeTypeResult(is_discrete_integer=False)
        if percentile_objs_for_parser is None:
            raise ValueError("no percentiles found")
        return percentile_objs_for_parser

    return _impl


async def _drive_forecast(
    bot: TemplateForecaster,
    question: NumericQuestion,
    rationale: str,
    parser_percentiles: list[Percentile] | None,
) -> NumericDistribution:
    """Run a single forecaster end-to-end through _run_forecast_on_numeric."""
    with patch("metaculus_bot.forecaster_runners.structure_output", new=_structure_output_mock(parser_percentiles)):
        with patch.object(GeneralLlm, "invoke", new=AsyncMock(return_value=rationale)):
            pred = await bot._run_forecast_on_numeric(question, research="r", llm_to_use=GeneralLlm(model="test-model"))
    assert isinstance(pred.prediction_value, NumericDistribution)
    return pred.prediction_value


def _assert_metaculus_compliant_cdf(cdf: list[Percentile], question: NumericQuestion) -> None:
    """Assert the ensemble-aggregated CDF satisfies the Metaculus server-side
    constraints: 201 points, monotonic non-decreasing, min-step compliance,
    correct endpoint pinning."""
    assert len(cdf) == 201, f"expected 201 CDF points, got {len(cdf)}"

    probs = np.array([float(p.percentile) for p in cdf], dtype=float)
    values = np.array([float(p.value) for p in cdf], dtype=float)

    # Monotonic non-decreasing.
    diffs = np.diff(probs)
    assert (diffs >= 0).all(), f"CDF not monotonic; min diff {diffs.min()}"

    # Min-step compliance (server-side: 0.01 / 200 = 5e-5).
    min_required = 5e-5
    assert (diffs >= min_required - 1e-9).all(), f"min step violation; min diff {diffs.min()} < {min_required}"

    # Endpoint pinning (closed bounds).
    if not question.open_lower_bound:
        assert probs[0] == pytest.approx(0.0, abs=1e-9)
    else:
        assert probs[0] >= 0.001 - 1e-9
    if not question.open_upper_bound:
        assert probs[-1] == pytest.approx(1.0, abs=1e-9)
    else:
        assert probs[-1] <= 0.999 + 1e-9

    # Values must span [lower, upper].
    assert values[0] == pytest.approx(question.lower_bound, abs=1e-6)
    assert values[-1] == pytest.approx(question.upper_bound, abs=1e-6)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestEnsembleMixedFormats:
    """Two forecasters via mixture, one via percentiles. Both aggregation
    strategies must produce valid CDFs."""

    @pytest.mark.asyncio
    async def test_mean_aggregation_produces_compliant_cdf(self) -> None:
        bot = _make_bot()
        q = _make_numeric_question()

        mixture_a = [
            {"weight": 0.3, "mean": 25.0, "sd": 8.0},
            {"weight": 0.4, "mean": 50.0, "sd": 10.0},
            {"weight": 0.3, "mean": 75.0, "sd": 8.0},
        ]
        mixture_b = [
            {"weight": 0.5, "mean": 30.0, "sd": 12.0},
            {"weight": 0.5, "mean": 70.0, "sd": 12.0},
        ]

        pred_a = await _drive_forecast(bot, q, _mixture_only_rationale(_DECLARED_PERCENTILES_A, mixture_a), None)
        pred_b = await _drive_forecast(bot, q, _mixture_only_rationale(_DECLARED_PERCENTILES_B, mixture_b), None)
        pred_c = await _drive_forecast(
            bot,
            q,
            _percentiles_only_rationale(_DECLARED_PERCENTILES_C),
            _percentile_objs(_DECLARED_PERCENTILES_C),
        )

        aggregated = await combine_numeric_predictions([pred_a, pred_b, pred_c], q, AggregationStrategy.MEAN)
        _assert_metaculus_compliant_cdf(aggregated.cdf, q)

    @pytest.mark.asyncio
    async def test_median_aggregation_produces_compliant_cdf(self) -> None:
        bot = _make_bot()
        q = _make_numeric_question()

        mixture_a = [
            {"weight": 0.3, "mean": 25.0, "sd": 8.0},
            {"weight": 0.4, "mean": 50.0, "sd": 10.0},
            {"weight": 0.3, "mean": 75.0, "sd": 8.0},
        ]
        mixture_b = [
            {"weight": 0.5, "mean": 30.0, "sd": 12.0},
            {"weight": 0.5, "mean": 70.0, "sd": 12.0},
        ]

        pred_a = await _drive_forecast(bot, q, _mixture_only_rationale(_DECLARED_PERCENTILES_A, mixture_a), None)
        pred_b = await _drive_forecast(bot, q, _mixture_only_rationale(_DECLARED_PERCENTILES_B, mixture_b), None)
        pred_c = await _drive_forecast(
            bot,
            q,
            _percentiles_only_rationale(_DECLARED_PERCENTILES_C),
            _percentile_objs(_DECLARED_PERCENTILES_C),
        )

        aggregated = await combine_numeric_predictions([pred_a, pred_b, pred_c], q, AggregationStrategy.MEDIAN)
        _assert_metaculus_compliant_cdf(aggregated.cdf, q)


class TestEnsembleAllMixture:
    """All three forecasters use the mixture branch. The ensemble CDF must
    still satisfy server-side constraints — proves the mixture-derived 201-pt
    CDF flows through ``_postprocess_ensemble_cdf`` correctly."""

    @pytest.mark.asyncio
    async def test_three_mixtures_mean_aggregation(self) -> None:
        bot = _make_bot()
        q = _make_numeric_question()

        components_a = [
            {"weight": 0.5, "mean": 30.0, "sd": 10.0},
            {"weight": 0.5, "mean": 70.0, "sd": 10.0},
        ]
        components_b = [
            {"weight": 0.4, "mean": 20.0, "sd": 5.0},
            {"weight": 0.6, "mean": 60.0, "sd": 12.0},
        ]
        components_c = [
            {"weight": 0.33, "mean": 25.0, "sd": 7.0},
            {"weight": 0.34, "mean": 50.0, "sd": 8.0},
            {"weight": 0.33, "mean": 75.0, "sd": 7.0},
        ]

        pred_a = await _drive_forecast(bot, q, _mixture_only_rationale(_DECLARED_PERCENTILES_A, components_a), None)
        pred_b = await _drive_forecast(bot, q, _mixture_only_rationale(_DECLARED_PERCENTILES_B, components_b), None)
        pred_c = await _drive_forecast(bot, q, _mixture_only_rationale(_DECLARED_PERCENTILES_C, components_c), None)

        aggregated = await combine_numeric_predictions([pred_a, pred_b, pred_c], q, AggregationStrategy.MEAN)
        _assert_metaculus_compliant_cdf(aggregated.cdf, q)

    @pytest.mark.asyncio
    async def test_three_mixtures_median_aggregation(self) -> None:
        bot = _make_bot()
        q = _make_numeric_question()

        components_a = [
            {"weight": 0.5, "mean": 30.0, "sd": 10.0},
            {"weight": 0.5, "mean": 70.0, "sd": 10.0},
        ]
        components_b = [
            {"weight": 0.4, "mean": 20.0, "sd": 5.0},
            {"weight": 0.6, "mean": 60.0, "sd": 12.0},
        ]
        components_c = [
            {"weight": 0.33, "mean": 25.0, "sd": 7.0},
            {"weight": 0.34, "mean": 50.0, "sd": 8.0},
            {"weight": 0.33, "mean": 75.0, "sd": 7.0},
        ]

        pred_a = await _drive_forecast(bot, q, _mixture_only_rationale(_DECLARED_PERCENTILES_A, components_a), None)
        pred_b = await _drive_forecast(bot, q, _mixture_only_rationale(_DECLARED_PERCENTILES_B, components_b), None)
        pred_c = await _drive_forecast(bot, q, _mixture_only_rationale(_DECLARED_PERCENTILES_C, components_c), None)

        aggregated = await combine_numeric_predictions([pred_a, pred_b, pred_c], q, AggregationStrategy.MEDIAN)
        _assert_metaculus_compliant_cdf(aggregated.cdf, q)


class TestEnsembleAllPercentiles:
    """All three forecasters use the percentile branch. Regression check that
    activating the mixture wiring did not break the percentile-only path."""

    @pytest.mark.asyncio
    async def test_three_percentiles_mean_aggregation(self) -> None:
        bot = _make_bot()
        q = _make_numeric_question()

        pred_a = await _drive_forecast(
            bot,
            q,
            _percentiles_only_rationale(_DECLARED_PERCENTILES_A),
            _percentile_objs(_DECLARED_PERCENTILES_A),
        )
        pred_b = await _drive_forecast(
            bot,
            q,
            _percentiles_only_rationale(_DECLARED_PERCENTILES_B),
            _percentile_objs(_DECLARED_PERCENTILES_B),
        )
        pred_c = await _drive_forecast(
            bot,
            q,
            _percentiles_only_rationale(_DECLARED_PERCENTILES_C),
            _percentile_objs(_DECLARED_PERCENTILES_C),
        )

        aggregated = await combine_numeric_predictions([pred_a, pred_b, pred_c], q, AggregationStrategy.MEAN)
        _assert_metaculus_compliant_cdf(aggregated.cdf, q)

    @pytest.mark.asyncio
    async def test_three_percentiles_mean_diverges_from_individual(self) -> None:
        """Sanity: the aggregated CDF is not just one of the individual CDFs.
        Without this, an aggregator that silently picks the first model's
        prediction would still pass the compliance checks."""
        bot = _make_bot()
        q = _make_numeric_question()

        pred_a = await _drive_forecast(
            bot,
            q,
            _percentiles_only_rationale(_DECLARED_PERCENTILES_A),
            _percentile_objs(_DECLARED_PERCENTILES_A),
        )
        pred_b = await _drive_forecast(
            bot,
            q,
            _percentiles_only_rationale(_DECLARED_PERCENTILES_B),
            _percentile_objs(_DECLARED_PERCENTILES_B),
        )
        pred_c = await _drive_forecast(
            bot,
            q,
            _percentiles_only_rationale(_DECLARED_PERCENTILES_C),
            _percentile_objs(_DECLARED_PERCENTILES_C),
        )

        aggregated = await combine_numeric_predictions([pred_a, pred_b, pred_c], q, AggregationStrategy.MEAN)

        agg_probs = np.array([float(p.percentile) for p in aggregated.cdf])
        pred_a_probs = np.array([float(p.percentile) for p in pred_a.cdf])

        # The aggregated CDF must differ from any single source CDF (proves
        # the aggregator isn't a passthrough).
        assert not np.allclose(agg_probs, pred_a_probs, atol=1e-4), (
            "aggregated CDF identical to forecaster A's — aggregator may be a no-op"
        )
