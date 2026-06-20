"""Ensemble scoring/simulation extracted from ``CorrelationAnalyzer``.

``EnsembleSimulator`` owns the "how would this set of models score as an ensemble"
concern: per-model performance/cost statistics, candidate evaluation, and the
question-by-question aggregation+scoring simulation. It reads benchmarks live off
the owning analyzer (so in-place filtering is reflected) and shares the safe-CDF
cache via a ``NumericCdfCache`` instance. ``CorrelationAnalyzer`` keeps thin
delegating wrappers for every externally-touched method.

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
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.ensemble_analysis.benchmark_identity import extract_model_name, get_question_type
from metaculus_bot.ensemble_analysis.cdf_cache import NumericCdfCache
from metaculus_bot.ensemble_analysis.types import CorrelationMatrix, EnsembleCandidate

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


@dataclass(slots=True)
class _CdfPoint:
    """A single (value, cumulative-probability) point of a numeric CDF."""

    value: float
    percentile: float


class _AggregatedNumericPrediction:
    """Lightweight numeric prediction exposing only the ``.cdf`` downstream scoring reads."""

    def __init__(self, x: list[float], cdf_probs: list[float]) -> None:
        self._cdf = [_CdfPoint(v, p) for v, p in zip(x, cdf_probs)]

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

        self.model_stats_cache = model_stats  # Cache the results
        return model_stats

    def _estimate_avg_reasoning_length(self, benchmark: BenchmarkForBot) -> float:
        """Estimate average reasoning text length for cost calculation."""
        total_chars = 0
        count = 0

        for report in benchmark.forecast_reports:
            if report.explanation:
                total_chars += len(report.explanation)
                count += 1

        return total_chars / max(count, 1) if count > 0 else 2000  # Default estimate

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

        # Calculate ensemble performance by simulating actual aggregation
        ensemble_performance = self.simulate_ensemble_performance(models, strategy)

        # Calculate average cost (same as before)
        avg_cost = float(np.mean([model_stats[m]["avg_cost"] for m in models]))

        # Calculate average pairwise correlation
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

    def simulate_ensemble_performance(
        self, models: list[str], aggregation_strategy: AggregationStrategy | str
    ) -> float:
        """Simulate ensemble performance by aggregating actual model predictions and scoring them properly."""
        from metaculus_bot.scoring_patches import (
            calculate_multiple_choice_baseline_score,
            calculate_numeric_baseline_score,
        )

        strategy = _normalize_strategy(aggregation_strategy)

        # Group data by question from benchmark reports
        question_data = {}

        for benchmark in self._benchmarks:
            model_name = extract_model_name(benchmark)
            if model_name in models:
                for report in benchmark.forecast_reports:
                    q_id = report.question.id_of_question
                    if q_id not in question_data:
                        # DEPRECATED: community_prediction_at_access_time is always None for
                        # newly-fetched questions (Metaculus removed aggregations from list API).
                        # This field may still have values in historical benchmark data.
                        q_type_tmp = get_question_type(report)
                        bin_cp = (
                            getattr(
                                report.question,
                                "community_prediction_at_access_time",
                                None,
                            )
                            if q_type_tmp == "binary"
                            else None
                        )
                        question_data[q_id] = {
                            "individual_preds": {},
                            "community_pred": bin_cp,
                            "question": report.question,
                            "question_type": q_type_tmp,
                        }

                    # Store actual prediction object (not just float)
                    question_data[q_id]["individual_preds"][model_name] = report.prediction

                    # Determine question type for proper aggregation
                    if question_data[q_id]["question_type"] is None:
                        question_data[q_id]["question_type"] = get_question_type(report)

        ensemble_scores = []

        for q_id, data in question_data.items():
            # Only consider questions where all models in the ensemble made predictions
            if len(data["individual_preds"]) != len(models):
                continue

            q = data["question"]
            q_type = data["question_type"]
            preds = [data["individual_preds"][m] for m in models]

            try:
                if q_type == "binary":
                    # Aggregate scalar prob and use binary baseline formula
                    pred_vals = [float(p) for p in preds]
                    if strategy == "mean":
                        agg_p = float(np.mean(pred_vals))
                    else:
                        agg_p = float(np.median(pred_vals))
                    c = getattr(q, "community_prediction_at_access_time", None)
                    score = self.calculate_baseline_score(agg_p, c, "binary")
                    if score is not None:
                        ensemble_scores.append(score)

                elif q_type == "multiple_choice":
                    # Aggregate per-option probabilities
                    # Build option name list from first prediction
                    first_pred = preds[0]
                    if not isinstance(first_pred, PredictedOptionList) or not first_pred.predicted_options:
                        raise ValueError("Multiple choice prediction missing predicted_options")
                    option_names = [getattr(opt, "option_name", str(opt)) for opt in first_pred.predicted_options]

                    aggregated = []
                    for name in option_names:
                        vals = []
                        for pred in preds:
                            for opt in pred.predicted_options:
                                if getattr(opt, "option_name", str(opt)) == name:
                                    vals.append(float(getattr(opt, "probability", 0)))
                                    break
                        if not vals:
                            aggregated.append(0.0)
                        else:
                            aggregated.append(float(np.mean(vals)) if strategy == "mean" else float(np.median(vals)))
                    # Normalize
                    s = sum(aggregated)
                    aggregated = [x / s for x in aggregated] if s > 0 else [1.0 / len(aggregated)] * len(aggregated)

                    # Build lightweight report-like object
                    pred_obj = SimpleNamespace(
                        predicted_options=[
                            SimpleNamespace(option_name=n, probability=p) for n, p in zip(option_names, aggregated)
                        ]
                    )
                    fake_report = SimpleNamespace(question=q, prediction=pred_obj)
                    score = calculate_multiple_choice_baseline_score(fake_report, self.baseline_score_cache)
                    if score is not None:
                        ensemble_scores.append(score)

                elif q_type == "numeric":
                    # Aggregate CDFs from predictions with safe-CDF fallback
                    # Extract CDF lists from each prediction (safe)
                    cdfs = []
                    for pred in preds:
                        # Use safe CDF accessor that rebuilds from declared percentiles if needed
                        cdf_list = self._cdf_cache.get_safe_numeric_cdf(
                            model_name=self.infer_model_name_from_prediction(q_id, pred),
                            question=q,
                            prediction=pred,
                        )
                        if cdf_list is None:
                            raise ValueError("Numeric prediction missing usable cdf after fallback")
                        cdfs.append(cdf_list)
                    # Use x-axis from first cdf
                    x_vals = [pt.value for pt in cdfs[0]]
                    # Stack cdf percentiles
                    stacks = np.array([[float(pt.percentile) for pt in c] for c in cdfs])
                    if strategy == "mean":
                        agg_cdf = stacks.mean(axis=0)
                    else:
                        agg_cdf = np.median(stacks, axis=0)

                    agg_pred = _AggregatedNumericPrediction(x_vals, list(agg_cdf))
                    fake_report = SimpleNamespace(question=q, prediction=agg_pred)
                    score = calculate_numeric_baseline_score(fake_report, self.baseline_score_cache)
                    if score is not None:
                        ensemble_scores.append(score)

                else:
                    continue

            except Exception as e:
                logger.warning(f"Failed to aggregate predictions for question {q_id}: {e}")
                continue

        # Return average ensemble performance across all questions
        result = float(np.mean(ensemble_scores)) if ensemble_scores else 0.0
        logger.debug(f"Ensemble {models} with {strategy}: {len(ensemble_scores)} questions, avg score {result:.2f}")
        return result

    def infer_model_name_from_prediction(self, q_id: int, pred: Any) -> str:
        """Best-effort resolve model name for stats when only prediction object is available.

        We search benchmarks mapping for a report with matching question id and same prediction object reference.
        Fallback to 'unknown' if not found. This is only used for logging counters.
        """
        try:
            for benchmark in self._benchmarks:
                name = extract_model_name(benchmark)
                for r in benchmark.forecast_reports:
                    if r.question.id_of_question == q_id and r.prediction is pred:
                        return name
        except Exception:
            logger.debug(f"Failed to infer model name for question {q_id}")
        return "unknown"

    def aggregate_predictions(
        self,
        individual_preds: dict[str, Any],
        models: list[str],
        question_type: str,
        aggregation_strategy: AggregationStrategy | str,
    ) -> float:
        """Aggregate individual model predictions based on question type and strategy.

        NOTE: dead in production — only exercised by ``test_real_ensemble_aggregation``.
        Retained (with a delegating wrapper on the analyzer) to preserve the test contract.
        """
        strategy = _normalize_strategy(aggregation_strategy)
        if question_type == "binary":
            # Direct aggregation of probabilities
            predictions = [individual_preds[model] for model in models]
            if strategy == "mean":
                return float(np.mean(predictions))
            elif strategy == "median":
                return float(np.median(predictions))
            else:
                raise ValueError(f"Unknown aggregation strategy: {strategy}")

        elif question_type == "multiple_choice":
            # Aggregate probability distributions
            predictions = [individual_preds[model] for model in models]

            # Extract options from first prediction for consistency
            first_pred = predictions[0]
            if not isinstance(first_pred, PredictedOptionList) or not first_pred.predicted_options:
                raise ValueError("Multiple choice prediction missing predicted_options")

            sorted_options = sorted(
                first_pred.predicted_options,
                key=lambda opt: opt.option_name,
            )
            option_names = [opt.option_name for opt in sorted_options]

            # Aggregate probabilities for each option
            aggregated_probs = []
            for option_name in option_names:
                option_probs = []
                for pred in predictions:
                    for opt in pred.predicted_options:
                        if opt.option_name == option_name:
                            option_probs.append(opt.probability)
                            break

                if option_probs:
                    if strategy == "mean":
                        aggregated_probs.append(np.mean(option_probs))
                    elif strategy == "median":
                        aggregated_probs.append(np.median(option_probs))
                    else:
                        raise ValueError(f"Unknown aggregation strategy: {strategy}")
                else:
                    aggregated_probs.append(0.0)

            # Normalize to sum to 1
            total_prob = sum(aggregated_probs)
            if total_prob > 0:
                aggregated_probs = [p / total_prob for p in aggregated_probs]

            # Return max probability as representative value for scoring
            return max(aggregated_probs) if aggregated_probs else 0.5

        elif question_type == "numeric":
            # Use median values for numeric questions
            median_values = []
            for model in models:
                pred = individual_preds[model]
                if isinstance(pred, NumericDistribution) and pred.declared_percentiles:
                    # Find 50th percentile or use mean of available percentiles
                    percentiles = pred.declared_percentiles
                    median_percentile = next((p for p in percentiles if p.percentile == 50), None)
                    if median_percentile:
                        median_values.append(float(median_percentile.value))
                    else:
                        median_values.append(float(np.mean([p.value for p in percentiles])))
                else:
                    # Fallback: treat as binary
                    median_values.append(0.5)

            if strategy == "mean":
                return float(np.mean(median_values))
            elif strategy == "median":
                return float(np.median(median_values))
            else:
                raise ValueError(f"Unknown aggregation strategy: {strategy}")

        else:
            raise ValueError(f"Unknown question type: {question_type}")

    def calculate_baseline_score(
        self, prediction_value: float, community_prediction: Any, question_type: str
    ) -> float | None:
        """Calculate baseline score using the same logic as forecasting_tools."""
        import math

        if community_prediction is None:
            return None

        try:
            if question_type == "binary":
                # Use the exact formula from binary_report.py line 86
                c = float(community_prediction)
                p = float(prediction_value)

                # Clamp prediction to avoid log errors (same as BinaryPrediction validation)
                p = max(0.001, min(0.999, p))

                return 100.0 * (c * (math.log2(p) + 1.0) + (1.0 - c) * (math.log2(1.0 - p) + 1.0))

            elif question_type in ["multiple_choice", "numeric"]:
                # For now, use a simplified scoring approach
                # This could be improved by implementing full PDF-based scoring for numeric
                # and log scoring for multiple choice, but this provides a reasonable proxy

                # Use a neutral baseline score for non-binary questions
                # This ensures ensemble comparison still works while avoiding complex scoring
                return 15.0  # Approximate average score

            else:
                return None

        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating baseline score: {e}")
            return None
