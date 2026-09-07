"""Ensemble scoring/simulation extracted from ``CorrelationAnalyzer``.

``EnsembleSimulator`` owns the "how would this set of models score as an ensemble"
concern: per-model performance/cost statistics, candidate evaluation, and the
question-by-question aggregation+scoring simulation. It reads benchmarks live off
the owning analyzer (so in-place filtering is reflected) and shares the safe-CDF
cache via a ``NumericCdfCache`` instance. ``CorrelationAnalyzer`` keeps thin
delegating wrappers for the analysis-script entry points.

Cache ownership / sharing
-------------------------
The simulator OWNS ``model_stats_cache`` and ``baseline_score_cache``. The analyzer
no longer holds its own copies; its ``add_benchmark_results`` / ``filter_models_inplace``
call ``invalidate_caches()`` here, and any external reader of
``analyzer._baseline_score_cache`` / ``analyzer._model_stats_cache`` goes through
properties that delegate to this object. This keeps a single source of truth and
sidesteps the stale-reference trap that a plain shared-dict ref would have (the old
code reassigned ``_model_stats_cache = None`` on the analyzer, which a borrowed ref
would not have observed).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.ensemble_analysis.benchmark_identity import extract_model_name, get_question_type
from metaculus_bot.ensemble_analysis.cdf_cache import NumericCdfCache
from metaculus_bot.ensemble_analysis.types import CorrelationMatrix, EnsembleCandidate
from metaculus_bot.scoring_patches import (
    calculate_multiple_choice_baseline_score,
    calculate_numeric_baseline_score,
)

if TYPE_CHECKING:
    from metaculus_bot.ensemble_analysis.correlation_analysis import CorrelationAnalyzer

logger = logging.getLogger(__name__)


def _normalize_strategy(strategy: AggregationStrategy | str) -> str:
    """Accept either an ``AggregationStrategy`` or a raw 'mean'/'median' string.

    External callers (``analyze_correlations.py``) pass raw strings, while internal
    code prefers the enum. We normalize to the string value so the rest of the
    simulation logic stays string-keyed without breaking the external contract.
    """
    return strategy.value if isinstance(strategy, AggregationStrategy) else strategy


def _option_probability(prediction: Any, option_name: str) -> float | None:
    """One prediction's probability for ``option_name``, or None if it doesn't name it."""
    for opt in prediction.predicted_options:
        if opt.option_name == option_name:
            return float(opt.probability)
    return None


def _member_option_probabilities(predictions: list[Any], option_names: list[str]) -> list[list[float]]:
    """Per option (positionally aligned with ``option_names``), every member's probability.

    An option no member quoted yields an empty list rather than being dropped, so the
    caller decides what an unquoted option contributes.
    """
    return [
        [prob for pred in predictions if (prob := _option_probability(pred, name)) is not None] for name in option_names
    ]


@dataclass(slots=True)
class _CdfPoint:
    """A single (value, cumulative-probability) point of a numeric CDF."""

    value: float
    percentile: float


class _AggregatedNumericPrediction:
    """Lightweight numeric prediction exposing only the ``.cdf`` downstream scoring reads."""

    def __init__(self, x: list[float], cdf_probs: list[float]) -> None:
        self._cdf = [_CdfPoint(v, p) for v, p in zip(x, cdf_probs, strict=False)]

    @property
    def cdf(self) -> list[_CdfPoint]:
        return self._cdf


class EnsembleSimulator:
    """Simulates ensemble performance by aggregating + scoring real model predictions."""

    def __init__(self, analyzer: CorrelationAnalyzer, cdf_cache: NumericCdfCache) -> None:
        self._analyzer = analyzer
        self._cdf_cache = cdf_cache
        self.model_stats_cache: dict[str, dict[str, float]] | None = None
        # (q_id, q_type) -> (score, diagnostics_logged)
        self.baseline_score_cache: dict[tuple[int, str], tuple[float, bool]] = {}

    @property
    def _benchmarks(self) -> list[BenchmarkForBot]:
        """Read benchmarks live off the analyzer so in-place filtering is reflected."""
        return self._analyzer.benchmarks

    def invalidate_caches(self) -> None:
        """Drop derived caches; called by the analyzer when its benchmark set changes."""
        self.model_stats_cache = None
        self.baseline_score_cache.clear()

    def calculate_model_statistics(self) -> dict[str, dict[str, float]]:
        """Calculate performance and cost statistics per model."""
        if self.model_stats_cache is not None:
            return self.model_stats_cache

        model_stats = {}

        for benchmark in self._benchmarks:
            model_name = extract_model_name(benchmark)
            total_cost: float = benchmark.total_cost if benchmark.total_cost is not None else 0.0
            num_questions = len(benchmark.forecast_reports)

            # Fix unrealistic costs for premium models and free models
            if model_name in ["gpt-5.1", "o3"] and total_cost < 0.10:
                # Estimate based on average reasoning length and known pricing
                avg_reasoning_length = self._estimate_avg_reasoning_length(benchmark)
                estimated_tokens = (avg_reasoning_length * 0.3) + 1000  # chars*0.3 + base prompt

                if model_name == "gpt-5.1":
                    total_cost = num_questions * (
                        estimated_tokens * 1.25 / 1_000_000
                    )  # $1.25 input + conservative output
                elif model_name == "o3":
                    total_cost = num_questions * (estimated_tokens * 2.0 / 1_000_000)  # $2 input + conservative output

                logger.info(
                    f"Adjusted {model_name} cost from ${benchmark.total_cost:.4f} to ${total_cost:.4f} "
                    f"(avg reasoning: {avg_reasoning_length} chars)"
                )
            elif total_cost == 0.0:
                # Apply minimum cost for free models to enable ensemble calculations
                total_cost = num_questions * 0.001  # $0.001 per question
                logger.info(
                    f"Applied minimum cost to free model {model_name}: ${total_cost:.3f} total (${0.001:.3f}/question)"
                )

            model_stats[model_name] = {
                "avg_performance": benchmark.average_expected_baseline_score,
                "avg_cost": total_cost / max(num_questions, 1),
                "total_cost": total_cost,
                "num_questions": num_questions,
                "efficiency_ratio": benchmark.average_expected_baseline_score / max(total_cost, 0.001),
            }

        self.model_stats_cache = model_stats
        return model_stats

    def _estimate_avg_reasoning_length(self, benchmark: BenchmarkForBot) -> float:
        """Estimate average reasoning text length for cost calculation."""
        total_reasoning_characters = 0
        reports_with_reasoning = 0

        for report in benchmark.forecast_reports:
            if report.explanation:
                total_reasoning_characters += len(report.explanation)
                reports_with_reasoning += 1

        return total_reasoning_characters / reports_with_reasoning if reports_with_reasoning else 2000

    def evaluate_ensemble(
        self,
        model_names: tuple[str, ...],
        model_stats: dict[str, dict[str, float]],
        corr_matrix: CorrelationMatrix,
        aggregation_strategy: AggregationStrategy | str = AggregationStrategy.MEAN,
    ) -> EnsembleCandidate:
        """Evaluate a specific ensemble configuration with a given aggregation strategy."""
        strategy = _normalize_strategy(aggregation_strategy)
        models = list(model_names)

        ensemble_performance = self.simulate_ensemble_performance(models, strategy)
        avg_cost = float(np.mean([model_stats[m]["avg_cost"] for m in models]))

        correlations = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                try:
                    corr = corr_matrix.get_correlation(models[i], models[j], "pearson")
                    correlations.append(abs(corr))
                except KeyError:
                    # Models might not have overlapping predictions
                    correlations.append(0.5)  # Neutral correlation

        avg_correlation = float(np.mean(correlations)) if correlations else 0.5
        diversity_score = 1.0 - avg_correlation
        efficiency_ratio = ensemble_performance / max(avg_cost, 0.001)

        return EnsembleCandidate(
            model_names=models,
            avg_performance=ensemble_performance,
            avg_cost=avg_cost,
            avg_correlation=avg_correlation,
            diversity_score=diversity_score,
            efficiency_ratio=efficiency_ratio,
            aggregation_strategy=strategy,
        )

    def _reports_by_question(self, models: list[str]) -> dict[Any, dict[str, Any]]:
        """Group the ensemble members' predictions per question, keyed by question id."""
        question_data: dict[Any, dict[str, Any]] = {}

        for benchmark in self._benchmarks:
            model_name = extract_model_name(benchmark)
            if model_name not in models:
                continue
            for report in benchmark.forecast_reports:
                q_id = report.question.id_of_question
                if q_id not in question_data:
                    q_type = get_question_type(report)
                    # DEPRECATED: community_prediction_at_access_time is always None for
                    # newly-fetched questions (Metaculus removed aggregations from list API).
                    # This field may still have values in historical benchmark data.
                    bin_cp = (
                        getattr(report.question, "community_prediction_at_access_time", None)
                        if q_type == "binary"
                        else None
                    )
                    question_data[q_id] = {
                        "individual_preds": {},
                        "community_pred": bin_cp,
                        "question": report.question,
                        "question_type": q_type,
                    }

                # Store actual prediction object (not just float)
                question_data[q_id]["individual_preds"][model_name] = report.prediction

        return question_data

    def _score_binary_question(self, question: Any, preds: list[Any], strategy: str) -> float | None:
        """Aggregate scalar probabilities and score with the binary baseline formula."""
        pred_vals = [float(p) for p in preds]
        if any(
            isinstance(pred, bool) or not np.isfinite(value) or not 0.0 <= value <= 1.0
            for pred, value in zip(preds, pred_vals, strict=True)
        ):
            raise ValueError("Binary prediction contains a non-finite or out-of-range probability")
        agg_p = float(np.mean(pred_vals)) if strategy == "mean" else float(np.median(pred_vals))
        community = getattr(question, "community_prediction_at_access_time", None)
        return self.calculate_baseline_score(agg_p, community, "binary")

    def _score_multiple_choice_question(self, question: Any, preds: list[Any], strategy: str) -> float | None:
        """Aggregate per-option probabilities (in the first prediction's option order) and score."""
        first_pred = preds[0]
        if not isinstance(first_pred, PredictedOptionList) or not first_pred.predicted_options:
            raise ValueError("Multiple choice prediction missing predicted_options")
        option_names = [opt.option_name for opt in first_pred.predicted_options]

        member_option_probabilities = _member_option_probabilities(preds, option_names)
        if any(len(probabilities) != len(preds) for probabilities in member_option_probabilities):
            raise ValueError("Multiple choice prediction is missing a declared option")
        if any(
            not np.isfinite(probability) or not 0.0 <= probability <= 1.0
            for probabilities in member_option_probabilities
            for probability in probabilities
        ):
            raise ValueError("Multiple choice prediction contains an invalid probability")
        if any(
            sum(option_probabilities[member_index] for option_probabilities in member_option_probabilities) <= 0.0
            for member_index in range(len(preds))
        ):
            raise ValueError("Multiple choice prediction has a member with no positive probability mass")
        aggregated = [
            float(np.mean(probabilities)) if strategy == "mean" else float(np.median(probabilities))
            for probabilities in member_option_probabilities
        ]
        total = sum(aggregated)
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("Multiple choice prediction has no positive probability mass")
        aggregated = [probability / total for probability in aggregated]

        pred_obj = SimpleNamespace(
            predicted_options=[
                SimpleNamespace(option_name=n, probability=p) for n, p in zip(option_names, aggregated, strict=True)
            ]
        )
        fake_report = SimpleNamespace(question=question, prediction=pred_obj)
        return calculate_multiple_choice_baseline_score(fake_report, self.baseline_score_cache)

    def _score_numeric_question(self, question: Any, members: list[tuple[str, Any]], strategy: str) -> float | None:
        """Aggregate member CDFs pointwise (via the safe-CDF ladder) and score the result.

        ``members`` pairs each prediction with its model name, which is the safe-CDF
        cache's key alongside the question id.
        """
        cdfs = []
        for model_name, pred in members:
            # Use safe CDF accessor that rebuilds from declared percentiles if needed
            cdf_list = self._cdf_cache.get_safe_numeric_cdf(
                model_name=model_name,
                question=question,
                prediction=pred,
            )
            if cdf_list is None:
                raise ValueError("Numeric prediction missing usable cdf after fallback")
            cdfs.append(cdf_list)

        # Use x-axis from first cdf, then stack cdf percentiles
        x_vals = [pt.value for pt in cdfs[0]]
        stacks = np.array([[float(pt.percentile) for pt in c] for c in cdfs])
        agg_cdf = stacks.mean(axis=0) if strategy == "mean" else np.median(stacks, axis=0)

        agg_pred = _AggregatedNumericPrediction(x_vals, list(agg_cdf))
        fake_report = SimpleNamespace(question=question, prediction=agg_pred)
        return calculate_numeric_baseline_score(fake_report, self.baseline_score_cache)

    def _score_aggregated_question(
        self, question: Any, members: list[tuple[str, Any]], *, q_type: str, strategy: str
    ) -> float | None:
        """Score one question's aggregate, dispatched on its type. None = not scoreable."""
        preds = [pred for _, pred in members]
        if q_type == "binary":
            return self._score_binary_question(question, preds, strategy)
        if q_type == "multiple_choice":
            return self._score_multiple_choice_question(question, preds, strategy)
        if q_type == "numeric":
            return self._score_numeric_question(question, members, strategy)
        return None

    def simulate_ensemble_performance(
        self, models: list[str], aggregation_strategy: AggregationStrategy | str
    ) -> float:
        """Simulate ensemble performance by aggregating actual model predictions and scoring them properly."""
        strategy = _normalize_strategy(aggregation_strategy)
        question_data = self._reports_by_question(models)

        ensemble_scores = []
        for q_id, data in question_data.items():
            # Only consider questions where all models in the ensemble made predictions
            if len(data["individual_preds"]) != len(models):
                continue
            # Aggregate in the ensemble's configured order, not the order reports arrived.
            members = [(m, data["individual_preds"][m]) for m in models]
            try:
                score = self._score_aggregated_question(
                    data["question"], members, q_type=data["question_type"], strategy=strategy
                )
            except Exception as e:  # noqa: BLE001  # soft-fail boundary: one unaggregatable question must not abort the simulation
                logger.warning(f"Failed to aggregate predictions for question {q_id}: {e}")
                continue
            if score is not None:
                ensemble_scores.append(score)

        # Return average ensemble performance across all questions
        result = float(np.mean(ensemble_scores)) if ensemble_scores else 0.0
        logger.debug(f"Ensemble {models} with {strategy}: {len(ensemble_scores)} questions, avg score {result:.2f}")
        return result

    def calculate_baseline_score(
        self, prediction_value: float, community_prediction: Any, question_type: str
    ) -> float | None:
        """Calculate a binary baseline score using the same logic as forecasting_tools."""
        if community_prediction is None:
            return None
        if question_type != "binary":
            raise ValueError(f"Baseline scoring is only implemented for binary questions, got {question_type!r}")

        # Use the exact formula from binary_report.py line 86.
        c = float(community_prediction)
        p = float(prediction_value)

        # Clamp prediction to avoid log errors (same as BinaryPrediction validation).
        p = max(0.001, min(0.999, p))
        return 100.0 * (c * (math.log2(p) + 1.0) + (1.0 - c) * (math.log2(1.0 - p) + 1.0))
