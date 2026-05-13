"""End-to-end activation tests for build_cross_model_aggregation.

Gap 1: ``tests/test_tool_runner_activation.py`` covers the per-forecaster
hook (``run_tools_for_forecaster``). These tests cover the OTHER hook —
``build_cross_model_aggregation`` — which runs once per question after all
forecasters have produced predictions, and threads its output into the
stacker prompt as a "Cross-model aggregation (deterministic math)" block.

Drives the full STACKING / CONDITIONAL_STACKING path of
``_aggregate_predictions`` for binary, MC, numeric (with declared
percentiles) and numeric (with mixture components). Each test mocks the
stacker LLM's ``invoke`` so we can capture the resulting prompt and assert
the cross-model block landed in it.

Also covers the negative case: with ``aggregation_strategy=MEAN`` (no
stacker), the per-forecaster ``## Computed quantities`` blocks must STILL
be appended (per ``_make_prediction``), but no cross-model section is
emitted because no stacker prompt is built.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.tool_runner import FEATURE_FLAG_ENV

# ---------------------------------------------------------------------------
# Common fixtures: bots and questions
# ---------------------------------------------------------------------------


def _make_stacking_bot(strategy: AggregationStrategy = AggregationStrategy.STACKING) -> TemplateForecaster:
    """Bot configured with a stacker LLM so STACKING / CONDITIONAL_STACKING paths
    are reachable. Forecaster + stacker + analyzer are all the same fake LLM —
    we mock ``invoke`` later to capture prompts."""
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    llms: dict[str, Any] = {
        "forecasters": [test_llm, test_llm, test_llm],
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
        aggregation_strategy=strategy,
        llms=llms,
        is_benchmarking=True,
        min_forecasters_to_publish=2,  # 3 forecasters in the test, allow some failures.
    )


def _make_mean_bot() -> TemplateForecaster:
    """Bot with MEAN aggregation — no stacker. Used for the negative case in
    the cross-model section test."""
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    llms: dict[str, Any] = {
        "forecasters": [test_llm, test_llm, test_llm],
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
        llms=llms,
        is_benchmarking=True,
    )


_OPEN = datetime.now() - timedelta(days=30)
_RESOLVE = datetime.now() + timedelta(days=365)


def _make_binary_q(qid: int = 101) -> BinaryQuestion:
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = qid
    q.id_of_post = qid
    q.question_text = "Will it happen?"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.page_url = f"https://example.com/q/{qid}"
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


def _make_mc_q(qid: int = 102) -> MultipleChoiceQuestion:
    q = MagicMock(spec=MultipleChoiceQuestion)
    q.id_of_question = qid
    q.id_of_post = qid
    q.question_text = "Which color?"
    q.options = ["Red", "Blue", "Green"]
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.page_url = f"https://example.com/q/{qid}"
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


def _make_numeric_q(qid: int = 103) -> NumericQuestion:
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = qid
    q.id_of_post = qid
    q.question_text = "What value?"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.unit_of_measure = "USD"
    q.lower_bound = 0.0
    q.upper_bound = 100.0
    q.open_lower_bound = False
    q.open_upper_bound = False
    q.zero_point = None
    q.cdf_size = None
    q.nominal_lower_bound = None
    q.nominal_upper_bound = None
    q.page_url = f"https://example.com/q/{qid}"
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


# ---------------------------------------------------------------------------
# Structured-block payload builders
# ---------------------------------------------------------------------------


def _binary_rationale(model_name: str, posterior: float, prior: float, k: int, n: int) -> str:
    payload = {
        "question_type": "binary",
        "prior": {"prob": prior, "source": "history"},
        "base_rate": {"k": k, "n": n, "ref_class": "matching years"},
        "evidence": [{"summary": "policy change", "direction": "up", "strength": "moderate"}],
        "scenarios": [],
        "posterior_prob": posterior,
    }
    return f"Model: {model_name}\n\nAnalysis.\n\n```json\n{json.dumps(payload)}\n```\n\nProbability: {int(posterior * 100)}%"


def _mc_rationale(model_name: str, probs: dict[str, float]) -> str:
    payload = {
        "question_type": "multiple_choice",
        "option_probs": probs,
    }
    return f"Model: {model_name}\n\nAnalysis.\n\n```json\n{json.dumps(payload)}\n```\n"


def _numeric_percentiles_rationale(model_name: str, family: str, declared: dict[str, float]) -> str:
    payload = {
        "question_type": "numeric",
        "declared_percentiles": declared,
        "distribution_family_hint": family,
    }
    pct_lines = "\n".join(
        f"Percentile {int(float(k) * 100) if float(k) * 100 == int(float(k) * 100) else float(k) * 100}: {v}"
        for k, v in declared.items()
    )
    return f"Model: {model_name}\n\n```json\n{json.dumps(payload)}\n```\n\n{pct_lines}"


def _numeric_mixture_rationale(model_name: str, declared: dict[str, float], components: list[dict]) -> str:
    payload = {
        "question_type": "numeric",
        "declared_percentiles": declared,
        "mixture_components": components,
    }
    return f"Model: {model_name}\n\nAnalysis.\n\n```json\n{json.dumps(payload)}\n```\n"


# ---------------------------------------------------------------------------
# Helpers to drive _aggregate_predictions through the stacking path
# ---------------------------------------------------------------------------


def _make_binary_reasoned_predictions(
    posteriors: list[float], priors: list[float], k_n_pairs: list[tuple[int, int]]
) -> tuple[list[float], list[ReasonedPrediction[float]]]:
    rationales = [
        _binary_rationale(f"m{i + 1}", post, pri, kn[0], kn[1])
        for i, (post, pri, kn) in enumerate(zip(posteriors, priors, k_n_pairs))
    ]
    reasoned = [ReasonedPrediction(prediction_value=post, reasoning=r) for post, r in zip(posteriors, rationales)]
    return posteriors, reasoned


# ---------------------------------------------------------------------------
# Capture wrapper for the stacker's ``invoke`` call
# ---------------------------------------------------------------------------


class _PromptCapture:
    """Captures every prompt the stacker LLM sees and returns a canned response.

    The stacker's parser LLM separately decodes a structured prediction; we
    mock both: ``invoke`` returns ``stacker_response``, and
    ``structure_output`` returns ``parser_response``.
    """

    def __init__(self, stacker_response: str, parser_response: Any) -> None:
        self.stacker_response = stacker_response
        self.parser_response = parser_response
        self.prompts: list[str] = []

    async def stacker_invoke(self, prompt: str) -> str:
        await asyncio.sleep(0)
        self.prompts.append(prompt)
        return self.stacker_response

    async def parser_structure_output(self, *_args: Any, **_kwargs: Any) -> Any:
        await asyncio.sleep(0)
        return self.parser_response


# ---------------------------------------------------------------------------
# Gap 1A: Binary path — STACKING aggregation surfaces cross-model lines
# ---------------------------------------------------------------------------


class TestBinaryStackingCrossModelAggregation:
    @pytest.mark.asyncio
    async def test_binary_stacking_injects_pools_into_stacker_prompt(self, monkeypatch) -> None:
        monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
        bot = _make_stacking_bot(AggregationStrategy.STACKING)
        question = _make_binary_q()

        posteriors, reasoned = _make_binary_reasoned_predictions(
            posteriors=[0.30, 0.42, 0.55],
            priors=[0.20, 0.25, 0.40],
            k_n_pairs=[(2, 10), (3, 12), (4, 8)],
        )

        capture = _PromptCapture(
            stacker_response="Model agreement... Probability: 42%",
            parser_response=BinaryPrediction(prediction_in_decimal=0.42),
        )

        with patch.object(GeneralLlm, "invoke", new=capture.stacker_invoke):
            with patch("metaculus_bot.stacking.structure_output", new=capture.parser_structure_output):
                aggregated = await bot._aggregate_predictions(
                    posteriors,  # type: ignore[arg-type]
                    question,
                    research="research",
                    reasoned_predictions=reasoned,
                    aggregated_tool_output=__import__(
                        "metaculus_bot.tool_runner", fromlist=["build_cross_model_aggregation"]
                    ).build_cross_model_aggregation(
                        question=question,
                        rationales=[r.reasoning for r in reasoned],
                        prediction_values=posteriors,
                    )
                    or None,
                )

        assert isinstance(aggregated, float)
        assert len(capture.prompts) == 1
        prompt = capture.prompts[0]

        # Cross-model section must land before "Multiple Expert Analyses".
        assert "Cross-model aggregation (deterministic math)" in prompt
        agg_idx = prompt.find("Cross-model aggregation")
        analyses_idx = prompt.find("Multiple Expert Analyses")
        assert 0 <= agg_idx < analyses_idx

        # Specific tool outputs must appear:
        # - Pools line lists linear, log, Satopää
        assert "Pools over 3 forecasters" in prompt
        # The actual numeric values come from the tool — we don't pin them, but
        # the "linear ... log ... Satopää" anchor must be present.
        assert "linear" in prompt and "log" in prompt and "Satopää" in prompt

        # - Blended base rate from the 3 declared base_rates
        assert "Blended base rate across 3 forecasters" in prompt

        # - Prior/posterior snapshot from the 3 declared priors/posteriors
        assert "Prior/posterior snapshot" in prompt

    @pytest.mark.asyncio
    async def test_binary_conditional_stacking_injects_cross_model(self, monkeypatch) -> None:
        """CONDITIONAL_STACKING shares the same hook; verify it threads the
        aggregated_tool_output identically."""
        monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
        bot = _make_stacking_bot(AggregationStrategy.CONDITIONAL_STACKING)
        question = _make_binary_q(qid=104)

        posteriors, reasoned = _make_binary_reasoned_predictions(
            posteriors=[0.30, 0.50, 0.65],
            priors=[0.20, 0.30, 0.45],
            k_n_pairs=[(1, 8), (3, 12), (4, 9)],
        )

        capture = _PromptCapture(
            stacker_response="Probability: 47%",
            parser_response=BinaryPrediction(prediction_in_decimal=0.47),
        )
        from metaculus_bot.tool_runner import build_cross_model_aggregation

        with patch.object(GeneralLlm, "invoke", new=capture.stacker_invoke):
            with patch("metaculus_bot.stacking.structure_output", new=capture.parser_structure_output):
                await bot._aggregate_predictions(
                    posteriors,  # type: ignore[arg-type]
                    question,
                    research="research",
                    reasoned_predictions=reasoned,
                    aggregated_tool_output=build_cross_model_aggregation(
                        question=question,
                        rationales=[r.reasoning for r in reasoned],
                        prediction_values=posteriors,
                    )
                    or None,
                )

        assert "Cross-model aggregation (deterministic math)" in capture.prompts[0]
        assert "Pools over 3 forecasters" in capture.prompts[0]


# ---------------------------------------------------------------------------
# Gap 1B: Numeric (declared_percentiles) + Numeric (mixture) paths
# ---------------------------------------------------------------------------


_NUMERIC_DECLARED: dict[str, dict[str, float]] = {
    "m1": {
        "0.025": 5.0,
        "0.05": 10.0,
        "0.1": 15.0,
        "0.2": 25.0,
        "0.4": 35.0,
        "0.5": 45.0,
        "0.6": 55.0,
        "0.8": 70.0,
        "0.9": 80.0,
        "0.95": 88.0,
        "0.975": 95.0,
    },
    "m2": {
        "0.025": 8.0,
        "0.05": 14.0,
        "0.1": 20.0,
        "0.2": 27.0,
        "0.4": 38.0,
        "0.5": 50.0,
        "0.6": 60.0,
        "0.8": 75.0,
        "0.9": 83.0,
        "0.95": 90.0,
        "0.975": 96.0,
    },
    "m3": {
        "0.025": 3.0,
        "0.05": 7.0,
        "0.1": 12.0,
        "0.2": 22.0,
        "0.4": 32.0,
        "0.5": 40.0,
        "0.6": 50.0,
        "0.8": 65.0,
        "0.9": 78.0,
        "0.95": 86.0,
        "0.975": 94.0,
    },
}


def _percentile_objs_from(decl: dict[str, float]) -> list[Percentile]:
    return [
        Percentile(percentile=float(k), value=float(v)) for k, v in sorted(decl.items(), key=lambda kv: float(kv[0]))
    ]


class TestNumericStackingCrossModelAggregation:
    @pytest.mark.asyncio
    async def test_numeric_percentiles_only_injects_aggregation_section(self, monkeypatch) -> None:
        monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
        bot = _make_stacking_bot(AggregationStrategy.STACKING)
        question = _make_numeric_q()

        rationales = [
            _numeric_percentiles_rationale("m1", "normal", _NUMERIC_DECLARED["m1"]),
            _numeric_percentiles_rationale("m2", "lognormal", _NUMERIC_DECLARED["m2"]),
            _numeric_percentiles_rationale("m3", "normal", _NUMERIC_DECLARED["m3"]),
        ]

        # Build NumericDistributions to feed _aggregate_predictions. We
        # reuse the public router path for fidelity.
        from metaculus_bot.numeric_format_router import route_numeric_output
        from metaculus_bot.pchip_processing import create_pchip_numeric_distribution

        numeric_predictions = []
        for rationale, declared in zip(rationales, _NUMERIC_DECLARED.values()):
            pcts = _percentile_objs_from(declared)
            routed = route_numeric_output(rationale=rationale, declared_percentiles=pcts, question=question)
            assert routed.declared_percentiles is not None
            cdf_probs = [float(p.percentile) for p in routed.cdf_percentiles] or [
                float(p.percentile) for p in routed.declared_percentiles
            ]
            # For the percentile branch the router returns the raw declared list
            # — wrap in a NumericDistribution via the same builder ``main.py``
            # uses on the percentile path.
            from metaculus_bot.numeric_pipeline import build_numeric_distribution, sanitize_percentiles

            sanitized, zero_point = sanitize_percentiles(routed.declared_percentiles, question)
            pred = build_numeric_distribution(sanitized, question, zero_point)
            numeric_predictions.append(pred)
            _ = create_pchip_numeric_distribution  # silence unused-import lint
            _ = cdf_probs

        reasoned = [
            ReasonedPrediction(prediction_value=p, reasoning=r) for p, r in zip(numeric_predictions, rationales)
        ]

        capture = _PromptCapture(
            stacker_response="Percentile 50: 45.0",
            parser_response=_percentile_objs_from(_NUMERIC_DECLARED["m1"]),
        )
        from metaculus_bot.tool_runner import build_cross_model_aggregation

        with patch.object(GeneralLlm, "invoke", new=capture.stacker_invoke):
            with patch("metaculus_bot.stacking.structure_output", new=capture.parser_structure_output):
                await bot._aggregate_predictions(
                    numeric_predictions,  # type: ignore[arg-type]
                    question,
                    research="research",
                    reasoned_predictions=reasoned,
                    aggregated_tool_output=build_cross_model_aggregation(
                        question=question,
                        rationales=rationales,
                        prediction_values=[_percentile_objs_from(d) for d in _NUMERIC_DECLARED.values()],
                    )
                    or None,
                )

        prompt = capture.prompts[0]
        assert "Cross-model aggregation (deterministic math)" in prompt
        # Numeric aggregation should surface the medians line and family hints.
        assert "Forecaster medians" in prompt
        assert "Declared distribution families" in prompt

    @pytest.mark.asyncio
    async def test_numeric_with_mixture_components_injects_aggregation(self, monkeypatch) -> None:
        """Numeric structured aggregation for forecasters whose JSON blocks include
        ``mixture_components``. Cross-model aggregation still emits the medians +
        families lines because those are computed from declared_percentiles, which
        the schema requires alongside any mixture."""
        monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
        bot = _make_stacking_bot(AggregationStrategy.STACKING)
        question = _make_numeric_q()

        components_a = [
            {"weight": 0.3, "mean": 25.0, "sd": 8.0},
            {"weight": 0.4, "mean": 50.0, "sd": 10.0},
            {"weight": 0.3, "mean": 75.0, "sd": 8.0},
        ]
        components_b = [
            {"weight": 0.5, "mean": 30.0, "sd": 12.0},
            {"weight": 0.5, "mean": 70.0, "sd": 12.0},
        ]
        components_c = [
            {"weight": 0.4, "mean": 20.0, "sd": 7.0},
            {"weight": 0.6, "mean": 60.0, "sd": 11.0},
        ]

        rationales = [
            _numeric_mixture_rationale("m1", _NUMERIC_DECLARED["m1"], components_a),
            _numeric_mixture_rationale("m2", _NUMERIC_DECLARED["m2"], components_b),
            _numeric_mixture_rationale("m3", _NUMERIC_DECLARED["m3"], components_c),
        ]

        # Build numeric predictions via the router (mixture branch).
        from metaculus_bot.numeric_format_router import route_numeric_output
        from metaculus_bot.pchip_processing import create_pchip_numeric_distribution

        numeric_predictions = []
        for rationale, declared in zip(rationales, _NUMERIC_DECLARED.values()):
            pcts = _percentile_objs_from(declared)
            routed = route_numeric_output(rationale=rationale, declared_percentiles=pcts, question=question)
            # mixture or both branches set declared_percentiles to None and
            # cdf_percentiles to a 201-pt CDF.
            mixture_cdf_values: list[float] = [float(p.percentile) for p in routed.cdf_percentiles]
            from metaculus_bot.numeric_config import STANDARD_PERCENTILES

            mixture_declared: list[Percentile] = []
            for target_pct in STANDARD_PERCENTILES:
                hit = next(
                    (p for p in routed.cdf_percentiles if p.percentile >= target_pct),
                    routed.cdf_percentiles[-1],
                )
                mixture_declared.append(Percentile(percentile=target_pct, value=float(hit.value)))
            pred = create_pchip_numeric_distribution(mixture_cdf_values, mixture_declared, question, zero_point=None)
            numeric_predictions.append(pred)

        reasoned = [
            ReasonedPrediction(prediction_value=p, reasoning=r) for p, r in zip(numeric_predictions, rationales)
        ]

        capture = _PromptCapture(
            stacker_response="Percentile 50: 45.0",
            parser_response=_percentile_objs_from(_NUMERIC_DECLARED["m1"]),
        )
        from metaculus_bot.tool_runner import build_cross_model_aggregation

        with patch.object(GeneralLlm, "invoke", new=capture.stacker_invoke):
            with patch("metaculus_bot.stacking.structure_output", new=capture.parser_structure_output):
                await bot._aggregate_predictions(
                    numeric_predictions,  # type: ignore[arg-type]
                    question,
                    research="research",
                    reasoned_predictions=reasoned,
                    aggregated_tool_output=build_cross_model_aggregation(
                        question=question,
                        rationales=rationales,
                        prediction_values=[_percentile_objs_from(d) for d in _NUMERIC_DECLARED.values()],
                    )
                    or None,
                )

        prompt = capture.prompts[0]
        assert "Cross-model aggregation (deterministic math)" in prompt
        assert "Forecaster medians" in prompt


# ---------------------------------------------------------------------------
# Gap 1C: Multiple-choice path
# ---------------------------------------------------------------------------


class TestMcStackingCrossModelAggregation:
    @pytest.mark.asyncio
    async def test_mc_stacking_injects_dirichlet_aggregation(self, monkeypatch) -> None:
        monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
        bot = _make_stacking_bot(AggregationStrategy.STACKING)
        question = _make_mc_q()

        probs_per_forecaster = [
            {"Red": 0.5, "Blue": 0.3, "Green": 0.2},
            {"Red": 0.4, "Blue": 0.4, "Green": 0.2},
            {"Red": 0.6, "Blue": 0.2, "Green": 0.2},
        ]
        rationales = [_mc_rationale(f"m{i + 1}", p) for i, p in enumerate(probs_per_forecaster)]

        prediction_values = [
            PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name="Red", probability=p["Red"]),
                    PredictedOption(option_name="Blue", probability=p["Blue"]),
                    PredictedOption(option_name="Green", probability=p["Green"]),
                ]
            )
            for p in probs_per_forecaster
        ]
        reasoned = [ReasonedPrediction(prediction_value=v, reasoning=r) for v, r in zip(prediction_values, rationales)]

        capture = _PromptCapture(
            stacker_response="Red: 50%, Blue: 30%, Green: 20%",
            parser_response=PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name="Red", probability=0.5),
                    PredictedOption(option_name="Blue", probability=0.3),
                    PredictedOption(option_name="Green", probability=0.2),
                ]
            ),
        )
        from metaculus_bot.tool_runner import build_cross_model_aggregation

        with patch.object(GeneralLlm, "invoke", new=capture.stacker_invoke):
            with patch("metaculus_bot.stacking.structure_output", new=capture.parser_structure_output):
                await bot._aggregate_predictions(
                    prediction_values,  # type: ignore[arg-type]
                    question,
                    research="research",
                    reasoned_predictions=reasoned,
                    aggregated_tool_output=build_cross_model_aggregation(
                        question=question,
                        rationales=rationales,
                        prediction_values=prediction_values,
                    )
                    or None,
                )

        prompt = capture.prompts[0]
        assert "Cross-model aggregation (deterministic math)" in prompt
        # MC aggregation surfaces the linear-pool top-3.
        assert "Linear pool across 3 forecasters" in prompt


# ---------------------------------------------------------------------------
# Gap 1D: MEAN aggregation — flag ON keeps per-forecaster sections, no cross-model
# ---------------------------------------------------------------------------


class _FakeNotepad:
    """Minimal notepad stub — real _get_notepad touches the API."""

    def __init__(self) -> None:
        self.total_predictions_attempted = 0
        self.total_research_reports_attempted = 0


async def _fake_get_notepad(_self, _q):
    await asyncio.sleep(0)
    return _FakeNotepad()


class TestFlagOnNoStackerNoCrossModelSection:
    """When flag is ON but aggregation_strategy=MEAN, the per-forecaster
    ``## Computed quantities`` sections still get appended via
    ``run_tools_for_forecaster`` (in ``_make_prediction``), but no stacker
    prompt is built → no cross-model aggregation block goes anywhere."""

    @pytest.mark.asyncio
    async def test_per_forecaster_quantities_present_no_cross_model_section_in_reasoning(self, monkeypatch) -> None:
        monkeypatch.setenv(FEATURE_FLAG_ENV, "1")
        bot = _make_mean_bot()
        q = _make_binary_q(qid=205)

        rationale = _binary_rationale("m1", posterior=0.32, prior=0.20, k=2, n=10)

        async def fake_forecast(_q, _r, _llm):
            await asyncio.sleep(0)
            return ReasonedPrediction(prediction_value=0.32, reasoning=rationale)

        with (
            patch.object(TemplateForecaster, "_run_forecast_on_binary", side_effect=fake_forecast),
            patch.object(TemplateForecaster, "_get_notepad", _fake_get_notepad),
        ):
            llm = GeneralLlm(model="m1", temperature=0.0)
            out = await bot._make_prediction(q, "research", llm)

        # MEAN flow: per-forecaster Computed quantities section IS appended.
        assert "## Computed quantities" in out.reasoning
        # But NO cross-model aggregation section — that lives in the stacker
        # prompt, which is never constructed in MEAN aggregation.
        assert "Cross-model aggregation" not in out.reasoning


# Module-level imports kept for IDE / type-checker visibility despite local re-imports.
_ = pytest
