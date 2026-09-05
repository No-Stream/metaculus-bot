"""
Correlation analysis utilities for ensemble optimization.

Tracks inter-model correlations to optimize ensemble composition by balancing
performance with diversity.

``CorrelationAnalyzer`` owns the correlation-math, ingestion, and reporting
concerns. The identity helpers, safe-CDF cache, and ensemble simulation were
extracted into ``benchmark_identity``, ``cdf_cache``, and
``ensemble_simulator`` respectively. The analyzer keeps only the simulator and
cache methods used by the analysis scripts as delegating wrappers.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution
from scipy.stats import pearsonr

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.ensemble_analysis.benchmark_identity import (
    extract_model_name,
    get_question_type,
    identifiers_for_benchmark,
    is_stacking_benchmark,
)

# Re-exported for back-compat: callers/tests still do
# ``from metaculus_bot.ensemble_analysis.correlation_analysis import CorrelationMatrix, EnsembleCandidate``.
from metaculus_bot.ensemble_analysis.cdf_cache import NumericCdfCache
from metaculus_bot.ensemble_analysis.ensemble_simulator import EnsembleSimulator
from metaculus_bot.ensemble_analysis.types import CorrelationMatrix, EnsembleCandidate, ModelPrediction
from metaculus_bot.numeric.config import STANDARD_PERCENTILES

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_COMPONENT_PERCENTILES: tuple[float, ...] = tuple(
    percentile for percentile in STANDARD_PERCENTILES if 0.1 <= percentile <= 0.9 and percentile != 0.5
)


def _any_token_in_idents(tokens: list[str], idents: list[str]) -> bool:
    """Case-insensitive substring match of any token against any identifier."""
    lowered = [s.lower() for s in idents]
    return any(token.lower() in ident for token in tokens for ident in lowered)


def _models_matching_tokens(
    tokens: list[str],
    name_to_idents: dict[str, list[str]],
) -> tuple[set[str], dict[str, list[str]]]:
    """Models any token matched, plus the per-token hit lists the summary reports.

    A token with an empty hit list is what ``unmatched_includes`` /
    ``unmatched_excludes`` surface, so a typo'd token reads as "matched nothing"
    rather than silently filtering nothing.
    """
    matched_models: set[str] = set()
    hits_by_token: dict[str, list[str]] = {token: [] for token in tokens}
    for name, idents in name_to_idents.items():
        if not _any_token_in_idents(tokens, idents):
            continue
        matched_models.add(name)
        for token in tokens:
            if _any_token_in_idents([token], idents):
                hits_by_token[token].append(name)
    return matched_models, hits_by_token


def _filter_summary_lines(
    include_hits: dict[str, list[str]],
    exclude_hits: dict[str, list[str]],
    final_allowed: list[str],
) -> list[str]:
    """Human-readable filter disclosure, rendered under "Filters Applied" in the report."""
    lines: list[str] = []
    for label, hits in (("Included", include_hits), ("Excluded", exclude_hits)):
        if not hits:
            continue
        lines.append(f"{label} by tokens:")
        lines.extend(f"- {t}: {', '.join(names) if names else '(no match)'}" for t, names in hits.items())
    lines.append(f"Remaining models: {', '.join(final_allowed) if final_allowed else '(none)'}")
    return lines


def _component_correlation(
    q_type: str,
    components1: list[float],
    components2: list[float],
    *,
    q_id: int,
    model1: str,
    model2: str,
) -> float:
    """Correlation between two models' component vectors on one question.

    Binary is a 1/0 agreement indicator (a single scalar has no Pearson r); numeric
    and multiple choice correlate their component vectors, with a constant vector
    reported as 0.0 rather than NaN. Any shape we cannot correlate is 0.0.
    """
    if q_type == "binary":
        if len(components1) == 1 and len(components2) == 1:
            return 1.0 if components1[0] == components2[0] else 0.0
        return 0.0

    if q_type in ("numeric", "multiple_choice"):
        if len(components1) != len(components2) or len(components1) <= 1:
            return 0.0
        try:
            # Guard against constant vectors to avoid warnings and NaNs
            if np.std(components1) < 1e-12 or np.std(components2) < 1e-12:
                return 0.0
            corr_val, _ = pearsonr(components1, components2)
        except (ValueError, TypeError) as e:
            logger.debug(f"Pearson correlation failed for q={q_id} {model1} vs {model2}: {e}")
            return 0.0
        return float(corr_val) if not np.isnan(corr_val) else 0.0

    return 0.0


def _accumulate_question_correlations(
    q_id: int,
    model_data: dict[str, tuple[str, list[float]]],
    *,
    model_indices: dict[str, int],
    correlation_sums: np.ndarray,
    correlation_counts: np.ndarray,
) -> None:
    """Add one question's pairwise correlations into the running sum/count matrices.

    Contributes nothing when fewer than two models answered the question, or when
    the models disagree about its TYPE (a comparison across types is meaningless,
    and the disagreement itself is worth a warning).
    """
    available_models = list(model_data.keys())
    if len(available_models) < 2:
        return

    q_types = {data[0] for data in model_data.values()}
    if len(q_types) > 1:
        logger.warning(f"Question {q_id} has mixed types across models: {q_types}")
        return
    q_type = next(iter(q_types))

    for i, model1 in enumerate(available_models):
        for model2 in available_models[i + 1 :]:
            corr = _component_correlation(
                q_type,
                model_data[model1][1],
                model_data[model2][1],
                q_id=q_id,
                model1=model1,
                model2=model2,
            )
            idx1 = model_indices[model1]
            idx2 = model_indices[model2]
            correlation_sums[idx1, idx2] += corr
            correlation_sums[idx2, idx1] += corr  # Symmetric
            correlation_counts[idx1, idx2] += 1
            correlation_counts[idx2, idx1] += 1


def _average_pairwise_correlations(
    correlation_sums: np.ndarray,
    correlation_counts: np.ndarray,
    n_models: int,
) -> np.ndarray:
    """Mean correlation per model pair; self-correlation is 1.0, unobserved pairs 0.0."""
    matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        matrix[i, i] = 1.0  # Self-correlation is 1
        for j in range(i + 1, n_models):
            avg_corr = correlation_sums[i, j] / correlation_counts[i, j] if correlation_counts[i, j] > 0 else 0.0
            matrix[i, j] = avg_corr
            matrix[j, i] = avg_corr
    return matrix


class CorrelationAnalyzer:
    """Analyzes correlations between forecasting models for ensemble optimization."""

    def __init__(self):
        self.predictions: list[ModelPrediction] = []
        self.benchmarks: list[BenchmarkForBot] = []
        # Map cleaned model names to benchmark objects for later filtering (e.g., exclude stacking bots)
        self._model_name_to_benchmark: dict[str, BenchmarkForBot] = {}
        # Human-readable notes about any applied filters
        self._filter_summary_lines: list[str] = []
        # Safe-CDF machinery and ensemble simulation are owned by extracted helpers.
        # The simulator reads `self.benchmarks` live and owns the model-stats / baseline-score caches.
        self._cdf_cache = NumericCdfCache()
        self._simulator = EnsembleSimulator(self, self._cdf_cache)

    # --- cache-sharing shims -------------------------------------------------
    # External callers/tests historically read these caches off the analyzer; the
    # simulator now owns them. These properties keep the old attribute access working.
    @property
    def _model_stats_cache(self) -> dict[str, dict[str, float]] | None:
        return self._simulator.model_stats_cache

    @_model_stats_cache.setter
    def _model_stats_cache(self, value: dict[str, dict[str, float]] | None) -> None:
        self._simulator.model_stats_cache = value

    @property
    def _baseline_score_cache(self) -> dict[tuple[int, str], tuple[float, bool]]:
        return self._simulator.baseline_score_cache

    def add_benchmark_results(self, benchmarks: list[BenchmarkForBot]) -> None:
        """Extract predictions from benchmark results."""
        self.benchmarks = benchmarks
        self.predictions.clear()
        self._model_name_to_benchmark.clear()
        self._filter_summary_lines = []
        self._cdf_cache.clear()
        self._simulator.invalidate_caches()  # Clear derived caches when data changes

        for benchmark in benchmarks:
            model_name = extract_model_name(benchmark)
            # Track the mapping for later filtering
            self._model_name_to_benchmark[model_name] = benchmark

            for report in benchmark.forecast_reports:
                # Unsupported report shapes have no meaningful scalar representation and
                # must not enter the pivot as fabricated data.
                pred_value = self._extract_prediction_value(report)
                if pred_value is None:
                    logger.warning(
                        "Skipping unsupported prediction for model=%s question=%s",
                        model_name,
                        report.question.id_of_question,
                    )
                    continue

                prediction = ModelPrediction(
                    model_name=model_name,
                    # An absent id collapses to 0, which pools every id-less question into
                    # ONE pivot row. Kept because question_id is the pivot index and group
                    # key throughout this module, so it cannot be None; the collapse is
                    # noted here rather than surfacing as a pandas duplicate-index error.
                    question_id=report.question.id_of_question or 0,
                    question_url=report.question.page_url or "",
                    # No `or 0.0` on either: an unscored report or a run with no price
                    # estimate must not enter an analysis as a real 0.0 (the worst-possible
                    # baseline score). Both fields are None-able on ModelPrediction.
                    prediction_value=pred_value,
                    baseline_score=report.expected_baseline_score,
                    cost=report.price_estimate,
                )
                self.predictions.append(prediction)

        logger.info(f"Loaded {len(self.predictions)} predictions from {len(benchmarks)} models")

    # --- delegating wrappers: identity helpers (see benchmark_identity) ------
    def _extract_model_name(self, benchmark: BenchmarkForBot) -> str:
        """Delegates to ``benchmark_identity.extract_model_name``."""
        return extract_model_name(benchmark)

    def _is_stacking_benchmark(self, benchmark: BenchmarkForBot | None) -> bool:
        """Delegates to ``benchmark_identity.is_stacking_benchmark``."""
        return is_stacking_benchmark(benchmark)

    def _get_question_type(self, report) -> str:
        """Delegates to ``benchmark_identity.get_question_type``."""
        return get_question_type(report)

    def get_model_names(self) -> list[str]:
        """Return sorted unique model names present in current predictions/benchmarks."""
        if self._model_name_to_benchmark:
            names = list(self._model_name_to_benchmark.keys())
        else:
            names = sorted({p.model_name for p in self.predictions})
        return sorted(names)

    def filter_models_inplace(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """Filter benchmarks/predictions by substring-matched include/exclude lists.

        Matching is case-insensitive substring on several identifiers per model.
        If `include` is provided, only included models remain. Then `exclude` is
        applied to drop any matched models. Returns a dict summarizing matches.
        """
        # Clear previous summary
        self._filter_summary_lines = []

        tokens_inc = [t for t in (include or []) if isinstance(t, str) and t.strip()]
        tokens_exc = [t for t in (exclude or []) if isinstance(t, str) and t.strip()]
        if not tokens_inc and not tokens_exc:
            return {"included": [], "excluded": [], "unmatched_includes": [], "unmatched_excludes": []}

        # Compute model name -> identifiers map
        name_to_idents: dict[str, list[str]] = {}
        for b in self.benchmarks:
            name = extract_model_name(b)
            name_to_idents[name] = identifiers_for_benchmark(b, name)

        matched_include_models, include_hits = _models_matching_tokens(tokens_inc, name_to_idents)
        matched_exclude_models, exclude_hits = _models_matching_tokens(tokens_exc, name_to_idents)

        # Absent include tokens, every model is included; excludes then subtract.
        all_models: list[str] = list(name_to_idents.keys())
        included_set = matched_include_models if tokens_inc else set(all_models)
        final_allowed = [n for n in all_models if n in included_set and n not in matched_exclude_models]

        self._filter_summary_lines = _filter_summary_lines(include_hits, exclude_hits, final_allowed)
        self._restrict_to_models(set(final_allowed))

        return {
            "included": final_allowed if tokens_inc else [],
            "excluded": sorted(matched_exclude_models),
            "unmatched_includes": [t for t, hits in include_hits.items() if not hits],
            "unmatched_excludes": [t for t, hits in exclude_hits.items() if not hits],
        }

    def _restrict_to_models(self, allowed: set[str]) -> None:
        """Drop every benchmark/prediction outside ``allowed`` and invalidate derived caches."""
        before_bench = len(self.benchmarks)
        before_preds = len(self.predictions)
        self.benchmarks = [b for b in self.benchmarks if extract_model_name(b) in allowed]
        self.predictions = [p for p in self.predictions if p.model_name in allowed]
        self._model_name_to_benchmark = {k: v for k, v in self._model_name_to_benchmark.items() if k in allowed}
        self._simulator.invalidate_caches()

        logger.info(
            f"Model filtering applied: {before_bench}→{len(self.benchmarks)} benchmarks, {before_preds}→{len(self.predictions)} predictions"
        )

    def calculate_correlation_matrix(self) -> CorrelationMatrix:
        """Calculate Pearson and Spearman correlations between all model pairs."""
        # Create pivot table: questions × models
        df = pd.DataFrame(
            [
                {
                    "question_id": pred.question_id,
                    "model": pred.model_name,
                    "prediction": pred.prediction_value,
                }
                for pred in self.predictions
            ]
        )

        pivot_df = df.pivot(index="question_id", columns="model", values="prediction")

        # Remove questions where any model failed to predict
        pivot_df = pivot_df.dropna()

        logger.info(f"Correlation analysis using {len(pivot_df)} questions and {len(pivot_df.columns)} models")

        # Calculate correlation matrices
        pearson_corr = pivot_df.corr(method="pearson")
        spearman_corr = pivot_df.corr(method="spearman")

        return CorrelationMatrix(
            pearson_matrix=pearson_corr,
            spearman_matrix=spearman_corr,
            model_names=list(pivot_df.columns),
            num_questions=len(pivot_df),
        )

    def _find_report(self, question_id: int, model_name: str) -> Any | None:
        """The forecast report ``model_name`` produced for ``question_id``, if any."""
        for benchmark in self.benchmarks:
            if extract_model_name(benchmark) != model_name:
                continue
            for report in benchmark.forecast_reports:
                if (report.question.id_of_question or 0) == question_id:
                    return report
        return None

    def _components_by_question(self) -> dict[int, dict[str, tuple[str, list[float]]]]:
        """Per question, each model's ``(question_type, component vector)``.

        A question whose report cannot be located still gets an (empty) entry, so it
        counts toward the matrix's ``num_questions`` even though it contributes no
        correlation — the same accounting the pre-extraction loop produced.
        """
        question_data: dict[int, dict[str, tuple[str, list[float]]]] = {}
        for pred in self.predictions:
            models_for_question = question_data.setdefault(pred.question_id, {})
            report = self._find_report(pred.question_id, pred.model_name)
            if report is not None:
                components = self._extract_prediction_components(report)
                if components is not None:
                    models_for_question[pred.model_name] = components
        return question_data

    def calculate_correlation_matrix_by_components(self) -> CorrelationMatrix:
        """Calculate correlations using component-wise analysis for mixed question types.

        For each question, extracts prediction components and calculates correlations:
        - Binary: Direct correlation on probabilities
        - Numeric: Average correlation across percentiles (10, 20, 40, 60, 80, 90)
        - Multiple Choice: Average correlation across option probabilities
        """
        question_data = self._components_by_question()

        model_names = sorted({pred.model_name for pred in self.predictions})
        n_models = len(model_names)
        model_indices = {name: i for i, name in enumerate(model_names)}

        correlation_sums = np.zeros((n_models, n_models))
        correlation_counts = np.zeros((n_models, n_models))
        for q_id, model_data in question_data.items():
            _accumulate_question_correlations(
                q_id,
                model_data,
                model_indices=model_indices,
                correlation_sums=correlation_sums,
                correlation_counts=correlation_counts,
            )

        correlation_matrix = _average_pairwise_correlations(correlation_sums, correlation_counts, n_models)
        corr_df = pd.DataFrame(correlation_matrix, index=model_names, columns=model_names)

        logger.info(
            f"Component-wise correlation analysis using {len(question_data)} questions and {len(model_names)} models"
        )

        return CorrelationMatrix(
            pearson_matrix=corr_df,
            spearman_matrix=corr_df,  # For now, use same matrix for both
            model_names=model_names,
            num_questions=len(question_data),
        )

    def find_optimal_ensembles(
        self,
        max_ensemble_size: int = 5,
        max_cost_per_question: float = 1.0,
        min_performance: float = -100.0,
        use_component_analysis: bool = True,
    ) -> list[EnsembleCandidate]:
        """Find optimal ensemble configurations using performance + correlation data."""
        model_stats = self._simulator.calculate_model_statistics()

        # Exclude stacking bots from ensemble candidates using a single detection path
        if self._model_name_to_benchmark:
            model_stats = {
                name: stats
                for name, stats in model_stats.items()
                if not is_stacking_benchmark(self._model_name_to_benchmark.get(name))
            }

        # Use component-wise analysis for mixed question types if available
        if use_component_analysis and self._has_mixed_question_types():
            correlation_matrix = self.calculate_correlation_matrix_by_components()
            logger.info("Using component-wise correlation analysis for mixed question types")
        else:
            correlation_matrix = self.calculate_correlation_matrix()
            logger.info("Using traditional correlation analysis")

        candidates = []

        # Generate all possible ensemble combinations up to max_ensemble_size
        # Test both MEAN and MEDIAN aggregation strategies for each combination
        for size in range(2, max_ensemble_size + 1):
            for model_combo in combinations(model_stats.keys(), size):
                # Test both aggregation strategies for each model combination
                for agg_strategy in (AggregationStrategy.MEAN, AggregationStrategy.MEDIAN):
                    candidate = self._simulator.evaluate_ensemble(
                        model_combo, model_stats, correlation_matrix, agg_strategy
                    )

                    # Filter by constraints
                    if candidate.avg_cost <= max_cost_per_question and candidate.avg_performance >= min_performance:
                        candidates.append(candidate)

        # Sort by ensemble score (descending)
        candidates.sort(key=lambda x: x.ensemble_score, reverse=True)

        logger.info(f"Generated {len(candidates)} viable ensemble candidates")
        # Log numeric CDF fallback summary once per search to detect systemic issues.
        self._cdf_cache.log_numeric_cdf_summary()
        return candidates

    def _extract_prediction_value(self, report: Any) -> float | None:
        """Convert prediction to float for correlation analysis.

        This method is used for backward compatibility. For mixed question types,
        use _extract_prediction_components() instead.
        """
        prediction = report.prediction

        # Binary questions: return probability directly
        if isinstance(prediction, (int, float)) and not isinstance(prediction, bool):
            value = float(prediction)
            return value if np.isfinite(value) else None

        # Numeric questions: use median or mean of distribution
        if isinstance(prediction, NumericDistribution) and prediction.declared_percentiles:
            try:
                values = [(float(p.percentile), float(p.value)) for p in prediction.declared_percentiles]
            except (TypeError, ValueError):
                return None
            if not values or not all(np.isfinite(label) and np.isfinite(value) for label, value in values):
                return None
            median_value = next((value for label, value in values if label == 0.5), None)
            return median_value if median_value is not None else float(np.mean([value for _, value in values]))

        # Multiple choice: convert to single numeric score (entropy or max probability)
        if isinstance(prediction, PredictedOptionList):
            if not prediction.predicted_options:
                return None
            try:
                probabilities = [float(option.probability) for option in prediction.predicted_options]
            except (TypeError, ValueError):
                return None
            return max(probabilities) if all(np.isfinite(probability) for probability in probabilities) else None

        return None

    def _extract_prediction_components(self, report: Any) -> tuple[str, list[float]] | None:
        """Extract prediction components for improved correlation analysis.

        Returns:
            Tuple of (question_type, component_values)
            - Binary: ("binary", [probability])
            - Numeric: ("numeric", [p10, p20, p40, p60, p80, p90])
            - Multiple Choice: ("multiple_choice", [prob_option1, prob_option2, ...])
        """
        prediction = report.prediction

        # Binary questions: return probability directly
        if isinstance(prediction, (int, float)) and not isinstance(prediction, bool):
            value = float(prediction)
            return ("binary", [value]) if np.isfinite(value) else None

        # Multiple choice: extract all option probabilities (check this first to avoid median conflicts)
        if isinstance(prediction, PredictedOptionList) and prediction.predicted_options:
            try:
                sorted_options = sorted(prediction.predicted_options, key=lambda opt: opt.option_name)
                option_probs = [float(opt.probability) for opt in sorted_options]
            except (TypeError, ValueError, AttributeError):
                return None
            return ("multiple_choice", option_probs) if all(np.isfinite(option_probs)) else None

        # Numeric questions: extract all percentiles
        if isinstance(prediction, NumericDistribution) and prediction.declared_percentiles:
            try:
                percentile_pairs = [(float(p.percentile), float(p.value)) for p in prediction.declared_percentiles]
            except (TypeError, ValueError):
                return None
            if not percentile_pairs or not all(
                np.isfinite(label) and np.isfinite(value) for label, value in percentile_pairs
            ):
                return None
            percentile_dict = dict(percentile_pairs)
            if len(percentile_dict) != len(percentile_pairs) or any(
                label not in percentile_dict for label in _COMPONENT_PERCENTILES
            ):
                return None
            return ("numeric", [percentile_dict[label] for label in _COMPONENT_PERCENTILES])

        return None

    def _has_mixed_question_types(self) -> bool:
        """Check if the benchmarks contain mixed question types."""
        question_types = set()

        for benchmark in self.benchmarks:
            for report in benchmark.forecast_reports:
                components = self._extract_prediction_components(report)
                if components is not None:
                    question_types.add(components[0])

        return len(question_types) > 1

    def _get_question_type_breakdown(self) -> dict[str, int]:
        """Get count of each question type in the benchmarks."""
        type_counts = {}

        for benchmark in self.benchmarks:
            for report in benchmark.forecast_reports:
                components = self._extract_prediction_components(report)
                if components is not None:
                    q_type = components[0]
                    type_counts[q_type] = type_counts.get(q_type, 0) + 1

        return type_counts

    # --- delegating wrappers: ensemble simulation (see ensemble_simulator) ---
    def _calculate_model_statistics(self) -> dict[str, dict[str, float]]:
        """Delegates to ``EnsembleSimulator.calculate_model_statistics``."""
        return self._simulator.calculate_model_statistics()

    def _estimate_avg_reasoning_length(self, benchmark: BenchmarkForBot) -> float:
        """Delegates to ``EnsembleSimulator._estimate_avg_reasoning_length``."""
        return self._simulator._estimate_avg_reasoning_length(benchmark)

    def _simulate_ensemble_performance(
        self, models: list[str], aggregation_strategy: AggregationStrategy | str
    ) -> float:
        """Delegates to ``EnsembleSimulator.simulate_ensemble_performance``."""
        return self._simulator.simulate_ensemble_performance(models, aggregation_strategy)

    # --- delegating wrappers: numeric CDF cache (see numeric_cdf_cache) -------
    def _get_safe_numeric_cdf(self, model_name: str, question: Any, prediction: Any) -> list[Any] | None:
        """Delegates to ``NumericCdfCache.get_safe_numeric_cdf``."""
        return self._cdf_cache.get_safe_numeric_cdf(model_name, question, prediction)

    def _question_type_section(self) -> list[str]:
        """Type mix + the note that correlations came off component vectors."""
        type_counts = self._get_question_type_breakdown()
        lines = ["## Question Type Distribution"]
        lines.extend(f"- **{q_type.title()}**: {count} questions" for q_type, count in sorted(type_counts.items()))
        lines.append("- **Analysis Method**: Component-wise correlation\n")
        return lines

    @staticmethod
    def _model_performance_section(model_stats: dict[str, dict[str, float]]) -> list[str]:
        """Per-model score/cost/efficiency, best-performing first."""
        lines = ["## Individual Model Performance"]
        lines.extend(
            f"- **{model}**: Score {stats['avg_performance']:.2f}, "
            f"Cost ${stats['avg_cost']:.3f}/question, "
            f"Efficiency {stats['efficiency_ratio']:.1f}"
            for model, stats in sorted(model_stats.items(), key=lambda x: x[1]["avg_performance"], reverse=True)
        )
        return lines

    @staticmethod
    def _correlation_highlights_section(correlation_matrix: CorrelationMatrix) -> list[str]:
        """The five least-correlated model pairs — the diversity candidates."""
        lines = ["\n## Model Correlations (Pearson)", "**Most Independent Model Pairs:**"]
        least_correlated = correlation_matrix.get_least_correlated_pairs(threshold=0.8)
        lines.extend(f"- {model1} ↔ {model2}: r = {corr:.3f}" for model1, model2, corr in least_correlated[:5])
        return lines

    @staticmethod
    def _recommended_ensembles_section(optimal_ensembles: list[EnsembleCandidate]) -> list[str]:
        """Top five model combinations, each showing every aggregation strategy tried.

        Grouping by model set (rather than listing candidates flat) is the point: it
        puts mean and median for the same models side by side.
        """
        lines = ["\n## Recommended Ensembles (Both Aggregation Strategies)"]

        ensemble_groups: dict[tuple[str, ...], list[EnsembleCandidate]] = {}
        for ensemble in optimal_ensembles:
            ensemble_groups.setdefault(tuple(sorted(ensemble.model_names)), []).append(ensemble)

        ranked_groups = sorted(
            ensemble_groups.items(),
            key=lambda x: max(e.ensemble_score for e in x[1]),
            reverse=True,
        )
        for combination_count, (models_key, ensembles) in enumerate(ranked_groups[:5]):
            lines.append(f"\n**{combination_count + 1}. {' + '.join(models_key)}**")

            # Sort by aggregation strategy for consistent ordering (mean first, then median)
            ensembles.sort(key=lambda x: x.aggregation_strategy)
            lines.extend(
                f"   - **{ensemble.aggregation_strategy.upper()}**: "
                f"Score {ensemble.avg_performance:.2f}, "
                f"Cost ${ensemble.avg_cost:.3f}, "
                f"Diversity {ensemble.diversity_score:.3f}, "
                f"Overall {ensemble.ensemble_score:.3f}"
                for ensemble in ensembles
            )
        return lines

    def generate_correlation_report(self, output_path: str | None = None) -> str:
        """Generate human-readable correlation analysis report."""
        if not self.predictions:
            return "No prediction data available for correlation analysis."

        # Use component-wise analysis for mixed question types
        use_component_analysis = self._has_mixed_question_types()
        if use_component_analysis:
            correlation_matrix = self.calculate_correlation_matrix_by_components()
        else:
            correlation_matrix = self.calculate_correlation_matrix()

        model_stats = self._simulator.calculate_model_statistics()
        optimal_ensembles = self.find_optimal_ensembles(use_component_analysis=use_component_analysis)

        report = [
            "# Model Correlation Analysis Report",
            f"Based on {correlation_matrix.num_questions} questions across {len(correlation_matrix.model_names)} models\n",
        ]

        # Note any filters applied
        if self._filter_summary_lines:
            report.append("## Filters Applied")
            report.extend(self._filter_summary_lines)
            report.append("")

        # Add question type breakdown if mixed
        if use_component_analysis:
            report.extend(self._question_type_section())

        report.extend(self._model_performance_section(model_stats))
        report.extend(self._correlation_highlights_section(correlation_matrix))
        report.extend(self._recommended_ensembles_section(optimal_ensembles))

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info(f"Correlation report saved to {output_path}")

        return report_text
