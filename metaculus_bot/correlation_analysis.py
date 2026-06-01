"""
Correlation analysis utilities for ensemble optimization.

Tracks inter-model correlations to optimize ensemble composition by balancing
performance with diversity.

``CorrelationAnalyzer`` owns the correlation-math, ingestion, and reporting
concerns. The identity helpers, safe-CDF cache, and ensemble simulation were
extracted into ``benchmark_identity``, ``numeric_cdf_cache``, and
``ensemble_simulator`` respectively. Thin delegating wrappers are kept on the
analyzer for every method that external callers (``analyze_correlations.py``,
``community_benchmark.py``) and the test suite reach into, so the split is
non-breaking for the public + private API surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution
from scipy.stats import pearsonr

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.benchmark_identity import (
    extract_clean_model_name,
    extract_model_name,
    get_question_type,
    identifiers_for_benchmark,
    is_stacking_benchmark,
)
from metaculus_bot.ensemble_simulator import EnsembleSimulator
from metaculus_bot.numeric_cdf_cache import NumericCdfCache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ModelPrediction:
    """Single model's prediction on a question."""

    model_name: str
    question_id: int
    question_url: str
    prediction_value: float  # For binary: probability, for numeric: aggregated value
    baseline_score: float
    cost: float


@dataclass
class CorrelationMatrix:
    """Inter-model correlation analysis results."""

    pearson_matrix: pd.DataFrame
    spearman_matrix: pd.DataFrame
    model_names: list[str]
    num_questions: int

    def get_correlation(self, model1: str, model2: str, method: str = "pearson") -> float:
        """Get correlation coefficient between two models."""
        matrix = self.pearson_matrix if method == "pearson" else self.spearman_matrix
        return matrix.loc[model1, model2]

    def get_least_correlated_pairs(
        self, threshold: float = 0.7, method: str = "pearson"
    ) -> list[tuple[str, str, float]]:
        """Find model pairs with correlation below threshold."""
        matrix = self.pearson_matrix if method == "pearson" else self.spearman_matrix
        pairs = []

        for i in range(len(self.model_names)):
            for j in range(i + 1, len(self.model_names)):
                model1, model2 = self.model_names[i], self.model_names[j]
                corr = matrix.iloc[i, j]
                if abs(corr) < threshold:
                    pairs.append((model1, model2, corr))

        return sorted(pairs, key=lambda x: abs(x[2]))  # Sort by absolute correlation


@dataclass
class EnsembleCandidate:
    """Potential ensemble configuration with performance metrics."""

    model_names: list[str]
    avg_performance: float  # Average baseline score
    avg_cost: float
    avg_correlation: float  # Average pairwise correlation
    diversity_score: float  # Lower correlation = higher diversity
    efficiency_ratio: float  # Performance per dollar
    aggregation_strategy: str  # "mean" or "median"

    @property
    def ensemble_score(self) -> float:
        """Combined score balancing performance, cost, and diversity."""
        normalized_perf = self.avg_performance / 20.0  # Normalize typical scores
        normalized_efficiency = min(self.efficiency_ratio / 1500.0, 1.0)  # Cap at 1000
        diversity_bonus = 1.0 - self.avg_correlation
        PERF_WT, EFFIC_WT, DIVERS_WT = 0.9, 0.025, 0.075
        perf_score, effic_score, divers_score = (
            PERF_WT * normalized_perf,
            EFFIC_WT * normalized_efficiency,
            DIVERS_WT * diversity_bonus,
        )
        logger.debug(
            f"Score components: performance/accuracy {perf_score:.4f}, efficiency {effic_score:.4f}, diversity {divers_score:.4f}"
        )

        return perf_score + effic_score + divers_score


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
        # NOTE: the safe-CDF cache (`self._cdf_cache`) is intentionally NOT cleared here.
        # A fresh CorrelationAnalyzer is constructed per analysis run, so its (model, qid)
        # keys never collide across benchmark sets in practice. Clear it explicitly if that
        # assumption changes (see NumericCdfCache.clear).
        self._simulator.invalidate_caches()  # Clear derived caches when data changes

        for benchmark in benchmarks:
            model_name = extract_model_name(benchmark)
            # Track the mapping for later filtering
            self._model_name_to_benchmark[model_name] = benchmark

            for report in benchmark.forecast_reports:
                # Convert prediction to float for correlation analysis
                pred_value = self._extract_prediction_value(report)

                prediction = ModelPrediction(
                    model_name=model_name,
                    question_id=report.question.id_of_question or 0,
                    question_url=report.question.page_url,
                    prediction_value=pred_value,
                    baseline_score=report.expected_baseline_score or 0.0,
                    cost=report.price_estimate or 0.0,
                )
                self.predictions.append(prediction)

        logger.info(f"Loaded {len(self.predictions)} predictions from {len(benchmarks)} models")

    # --- delegating wrappers: identity helpers (see benchmark_identity) ------
    def _extract_model_name(self, benchmark: BenchmarkForBot) -> str:
        """Delegates to ``benchmark_identity.extract_model_name``."""
        return extract_model_name(benchmark)

    def _extract_clean_model_name(self, model_path: str) -> str:
        """Delegates to ``benchmark_identity.extract_clean_model_name``."""
        return extract_clean_model_name(model_path)

    def _identifiers_for_benchmark(self, benchmark: BenchmarkForBot, model_name: str) -> list[str]:
        """Delegates to ``benchmark_identity.identifiers_for_benchmark``."""
        return identifiers_for_benchmark(benchmark, model_name)

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

        # Helpers for case-insensitive substring matching
        def _any_token_in_idents(tokens: list[str], idents: list[str]) -> bool:
            if not tokens:
                return False
            lowers = [s.lower() for s in idents]
            for tok in tokens:
                lt = tok.lower()
                for s in lowers:
                    if lt in s:
                        return True
            return False

        # Determine included set
        all_models: list[str] = list(name_to_idents.keys())
        included_set = set(all_models)
        matched_includes: dict[str, list[str]] = {t: [] for t in tokens_inc}
        matched_excludes: dict[str, list[str]] = {t: [] for t in tokens_exc}

        if tokens_inc:
            included_set = set()
            for name, idents in name_to_idents.items():
                if _any_token_in_idents(tokens_inc, idents):
                    included_set.add(name)
                    # Track which tokens matched this name
                    for t in tokens_inc:
                        if _any_token_in_idents([t], idents):
                            matched_includes[t].append(name)

        # Apply excludes
        to_exclude: set[str] = set()
        if tokens_exc:
            for name, idents in name_to_idents.items():
                if _any_token_in_idents(tokens_exc, idents):
                    to_exclude.add(name)
                    for t in tokens_exc:
                        if _any_token_in_idents([t], idents):
                            matched_excludes[t].append(name)

        final_allowed = (
            [n for n in all_models if (n in included_set) and (n not in to_exclude)]
            if tokens_inc
            else [n for n in all_models if n not in to_exclude]
        )

        # Build summaries
        unmatched_includes = [t for t, hits in matched_includes.items() if not hits]
        unmatched_excludes = [t for t, hits in matched_excludes.items() if not hits]

        if tokens_inc:
            inc_lines = ["Included by tokens:"] + [
                f"- {t}: {', '.join(hits) if hits else '(no match)'}" for t, hits in matched_includes.items()
            ]
            self._filter_summary_lines.extend(inc_lines)
        if tokens_exc:
            exc_lines = ["Excluded by tokens:"] + [
                f"- {t}: {', '.join(hits) if hits else '(no match)'}" for t, hits in matched_excludes.items()
            ]
            self._filter_summary_lines.extend(exc_lines)
        self._filter_summary_lines.append(
            f"Remaining models: {', '.join(final_allowed) if final_allowed else '(none)'}"
        )

        # Apply filter to internal state
        allowed_set = set(final_allowed)
        before_bench = len(self.benchmarks)
        before_preds = len(self.predictions)
        self.benchmarks = [b for b in self.benchmarks if extract_model_name(b) in allowed_set]
        self.predictions = [p for p in self.predictions if p.model_name in allowed_set]
        self._model_name_to_benchmark = {k: v for k, v in self._model_name_to_benchmark.items() if k in allowed_set}
        self._simulator.invalidate_caches()

        logger.info(
            f"Model filtering applied: {before_bench}→{len(self.benchmarks)} benchmarks, {before_preds}→{len(self.predictions)} predictions"
        )

        return {
            "included": final_allowed if tokens_inc else [],
            "excluded": sorted(to_exclude),
            "unmatched_includes": unmatched_includes,
            "unmatched_excludes": unmatched_excludes,
        }

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

    def calculate_correlation_matrix_by_components(self) -> CorrelationMatrix:
        """Calculate correlations using component-wise analysis for mixed question types.

        For each question, extracts prediction components and calculates correlations:
        - Binary: Direct correlation on probabilities
        - Numeric: Average correlation across percentiles (10, 20, 40, 60, 80, 90)
        - Multiple Choice: Average correlation across option probabilities
        """
        # Group predictions by question and extract components
        question_data = {}

        for pred in self.predictions:
            q_id = pred.question_id
            if q_id not in question_data:
                question_data[q_id] = {}

            # Get the full report to extract components
            report = None
            for benchmark in self.benchmarks:
                for report_candidate in benchmark.forecast_reports:
                    if (report_candidate.question.id_of_question or 0) == q_id:
                        if extract_model_name(benchmark) == pred.model_name:
                            report = report_candidate
                            break
                if report:
                    break

            if report:
                q_type, components = self._extract_prediction_components(report)
                question_data[q_id][pred.model_name] = (q_type, components)

        # Calculate correlations for each question, then average
        model_names = list(set(pred.model_name for pred in self.predictions))
        n_models = len(model_names)

        # Initialize correlation matrices
        correlation_sums = np.zeros((n_models, n_models))
        correlation_counts = np.zeros((n_models, n_models))

        for q_id, model_data in question_data.items():
            # Only process questions where we have data for multiple models
            available_models = list(model_data.keys())
            if len(available_models) < 2:
                continue

            # Group by question type
            q_types = set(data[0] for data in model_data.values())
            if len(q_types) > 1:
                logger.warning(f"Question {q_id} has mixed types across models: {q_types}")
                continue

            q_type = list(q_types)[0]

            # Calculate correlation for this question
            model_indices = {name: i for i, name in enumerate(model_names)}

            for i, model1 in enumerate(available_models):
                for j, model2 in enumerate(available_models):
                    if i >= j:  # Skip duplicates and self-correlation
                        continue

                    # Get components for both models
                    _, components1 = model_data[model1]
                    _, components2 = model_data[model2]

                    # Calculate component-wise correlation
                    if q_type == "binary":
                        # Direct correlation for binary
                        if len(components1) == 1 and len(components2) == 1:
                            corr = 1.0 if components1[0] == components2[0] else 0.0
                        else:
                            corr = 0.0

                    elif q_type in ["numeric", "multiple_choice"]:
                        # Average correlation across components
                        if len(components1) == len(components2) and len(components1) > 1:
                            # Use scipy.stats.pearsonr for component pairs
                            try:
                                # Guard against constant vectors to avoid warnings and NaNs
                                if np.std(components1) < 1e-12 or np.std(components2) < 1e-12:
                                    corr_val = 0.0
                                else:
                                    corr_val, _ = pearsonr(components1, components2)
                                corr = corr_val if not np.isnan(corr_val) else 0.0
                            except (ValueError, TypeError) as e:
                                logger.debug(f"Pearson correlation failed for q={q_id} {model1} vs {model2}: {e}")
                                corr = 0.0
                        else:
                            corr = 0.0
                    else:
                        corr = 0.0

                    # Add to correlation matrix
                    idx1 = model_indices[model1]
                    idx2 = model_indices[model2]
                    correlation_sums[idx1, idx2] += corr
                    correlation_sums[idx2, idx1] += corr  # Symmetric
                    correlation_counts[idx1, idx2] += 1
                    correlation_counts[idx2, idx1] += 1

        # Calculate average correlations
        correlation_matrix = np.zeros((n_models, n_models))
        for i in range(n_models):
            correlation_matrix[i, i] = 1.0  # Self-correlation is 1
            for j in range(i + 1, n_models):
                if correlation_counts[i, j] > 0:
                    avg_corr = correlation_sums[i, j] / correlation_counts[i, j]
                    correlation_matrix[i, j] = avg_corr
                    correlation_matrix[j, i] = avg_corr
                else:
                    correlation_matrix[i, j] = 0.0
                    correlation_matrix[j, i] = 0.0

        # Convert to DataFrame
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
        from itertools import combinations

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
        # Log numeric CDF fallback summary once per search to detect systemic issues
        try:
            self.log_numeric_cdf_summary()
        except Exception:
            logger.debug("Failed to log numeric CDF summary")
        return candidates

    def _extract_prediction_value(self, report) -> float:
        """Convert prediction to float for correlation analysis.

        This method is used for backward compatibility. For mixed question types,
        use _extract_prediction_components() instead.
        """
        prediction = report.prediction

        # Binary questions: return probability directly
        if isinstance(prediction, (int, float)):
            return float(prediction)

        # Numeric questions: use median or mean of distribution
        if isinstance(prediction, NumericDistribution):
            if prediction.declared_percentiles:
                percentiles = prediction.declared_percentiles
                median_percentile = next((p for p in percentiles if p.percentile == 50), None)
                if median_percentile:
                    return float(median_percentile.value)
                return float(np.mean([p.value for p in percentiles]))

        # Multiple choice: convert to single numeric score (entropy or max probability)
        if isinstance(prediction, PredictedOptionList):
            return max(opt.probability for opt in prediction.predicted_options)

        # Last resort: hash the prediction for some numeric value
        return float(hash(str(prediction)) % 1000) / 1000.0

    def _extract_prediction_components(self, report) -> tuple[str, list[float]]:
        """Extract prediction components for improved correlation analysis.

        Returns:
            Tuple of (question_type, component_values)
            - Binary: ("binary", [probability])
            - Numeric: ("numeric", [p10, p20, p40, p60, p80, p90])
            - Multiple Choice: ("multiple_choice", [prob_option1, prob_option2, ...])
        """
        prediction = report.prediction

        # Binary questions: return probability directly
        if isinstance(prediction, (int, float)):
            return ("binary", [float(prediction)])

        # Multiple choice: extract all option probabilities (check this first to avoid median conflicts)
        if isinstance(prediction, PredictedOptionList) and prediction.predicted_options:
            try:
                sorted_options = sorted(
                    prediction.predicted_options,
                    key=lambda opt: opt.option_name,
                )
                option_probs = [float(opt.probability) for opt in sorted_options]
                return ("multiple_choice", option_probs)
            except (TypeError, AttributeError):
                return ("multiple_choice", [0.5, 0.5])

        # Numeric questions: extract all percentiles
        if isinstance(prediction, NumericDistribution) and prediction.declared_percentiles:
            target_percentiles = [10, 20, 40, 60, 80, 90]
            percentile_values = []

            try:
                percentile_dict = {p.percentile: p.value for p in prediction.declared_percentiles}
            except (TypeError, AttributeError):
                percentile_dict = {}

            for target_p in target_percentiles:
                if target_p in percentile_dict:
                    percentile_values.append(float(percentile_dict[target_p]))
                else:
                    available_values = list(percentile_dict.values())
                    percentile_values.append(float(np.mean(available_values)) if available_values else 0.0)

            return ("numeric", percentile_values)

        # Fallback: treat as binary with neutral prediction
        return ("binary", [0.5])

    def _has_mixed_question_types(self) -> bool:
        """Check if the benchmarks contain mixed question types."""
        question_types = set()

        for benchmark in self.benchmarks:
            for report in benchmark.forecast_reports:
                q_type, _ = self._extract_prediction_components(report)
                question_types.add(q_type)

        return len(question_types) > 1

    def _get_question_type_breakdown(self) -> dict[str, int]:
        """Get count of each question type in the benchmarks."""
        type_counts = {}

        for benchmark in self.benchmarks:
            for report in benchmark.forecast_reports:
                q_type, _ = self._extract_prediction_components(report)
                type_counts[q_type] = type_counts.get(q_type, 0) + 1

        return type_counts

    # --- delegating wrappers: ensemble simulation (see ensemble_simulator) ---
    def _calculate_model_statistics(self) -> dict[str, dict[str, float]]:
        """Delegates to ``EnsembleSimulator.calculate_model_statistics``."""
        return self._simulator.calculate_model_statistics()

    def _estimate_avg_reasoning_length(self, benchmark: BenchmarkForBot) -> float:
        """Delegates to ``EnsembleSimulator._estimate_avg_reasoning_length``."""
        return self._simulator._estimate_avg_reasoning_length(benchmark)

    def _evaluate_ensemble(
        self,
        model_names: tuple[str, ...],
        model_stats: dict[str, dict[str, float]],
        corr_matrix: CorrelationMatrix,
        aggregation_strategy: AggregationStrategy | str = "mean",
    ) -> EnsembleCandidate:
        """Delegates to ``EnsembleSimulator.evaluate_ensemble``."""
        return self._simulator.evaluate_ensemble(model_names, model_stats, corr_matrix, aggregation_strategy)

    def _simulate_ensemble_performance(
        self, models: list[str], aggregation_strategy: AggregationStrategy | str
    ) -> float:
        """Delegates to ``EnsembleSimulator.simulate_ensemble_performance``."""
        return self._simulator.simulate_ensemble_performance(models, aggregation_strategy)

    def _infer_model_name_from_prediction(self, q_id: int, pred: Any) -> str:
        """Delegates to ``EnsembleSimulator.infer_model_name_from_prediction``."""
        return self._simulator.infer_model_name_from_prediction(q_id, pred)

    def _aggregate_predictions(
        self,
        individual_preds: dict[str, Any],
        models: list[str],
        question_type: str,
        aggregation_strategy: AggregationStrategy | str,
    ) -> float:
        """Delegates to ``EnsembleSimulator.aggregate_predictions``."""
        return self._simulator.aggregate_predictions(individual_preds, models, question_type, aggregation_strategy)

    def _calculate_baseline_score(
        self, prediction_value: float, community_prediction: Any, question_type: str
    ) -> float | None:
        """Delegates to ``EnsembleSimulator.calculate_baseline_score``."""
        return self._simulator.calculate_baseline_score(prediction_value, community_prediction, question_type)

    # --- delegating wrappers: numeric CDF cache (see numeric_cdf_cache) -------
    def _get_safe_numeric_cdf(self, model_name: str, question: Any, prediction: Any) -> list[Any] | None:
        """Delegates to ``NumericCdfCache.get_safe_numeric_cdf``."""
        return self._cdf_cache.get_safe_numeric_cdf(model_name, question, prediction)

    def log_numeric_cdf_summary(self) -> None:
        """Delegates to ``NumericCdfCache.log_numeric_cdf_summary``."""
        self._cdf_cache.log_numeric_cdf_summary()

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

        report = []
        report.append("# Model Correlation Analysis Report")
        report.append(
            f"Based on {correlation_matrix.num_questions} questions across {len(correlation_matrix.model_names)} models\n"
        )

        # Note any filters applied
        if self._filter_summary_lines:
            report.append("## Filters Applied")
            report.extend(self._filter_summary_lines)
            report.append("")

        # Add question type breakdown if mixed
        if use_component_analysis:
            type_counts = self._get_question_type_breakdown()
            report.append("## Question Type Distribution")
            for q_type, count in sorted(type_counts.items()):
                report.append(f"- **{q_type.title()}**: {count} questions")
            report.append("- **Analysis Method**: Component-wise correlation\n")

        # Model Performance Summary
        report.append("## Individual Model Performance")
        for model, stats in sorted(model_stats.items(), key=lambda x: x[1]["avg_performance"], reverse=True):
            report.append(
                f"- **{model}**: Score {stats['avg_performance']:.2f}, "
                f"Cost ${stats['avg_cost']:.3f}/question, "
                f"Efficiency {stats['efficiency_ratio']:.1f}"
            )

        # Correlation Highlights
        report.append("\n## Model Correlations (Pearson)")
        least_correlated = correlation_matrix.get_least_correlated_pairs(threshold=0.8)
        report.append("**Most Independent Model Pairs:**")
        for model1, model2, corr in least_correlated[:5]:
            report.append(f"- {model1} ↔ {model2}: r = {corr:.3f}")

        # Optimal Ensembles with Aggregation Strategy Comparison
        report.append("\n## Recommended Ensembles (Both Aggregation Strategies)")

        # Group ensembles by model combination to show mean vs median comparison
        ensemble_groups = {}
        for ensemble in optimal_ensembles:
            models_key = tuple(sorted(ensemble.model_names))
            if models_key not in ensemble_groups:
                ensemble_groups[models_key] = []
            ensemble_groups[models_key].append(ensemble)

        # Show top 5 model combinations with both aggregation strategies
        combination_count = 0
        for models_key, ensembles in sorted(
            ensemble_groups.items(),
            key=lambda x: max(e.ensemble_score for e in x[1]),
            reverse=True,
        ):
            if combination_count >= 5:
                break

            models_str = " + ".join(models_key)
            report.append(f"\n**{combination_count + 1}. {models_str}**")

            # Sort by aggregation strategy for consistent ordering (mean first, then median)
            ensembles.sort(key=lambda x: x.aggregation_strategy)

            for ensemble in ensembles:
                report.append(
                    f"   - **{ensemble.aggregation_strategy.upper()}**: "
                    f"Score {ensemble.avg_performance:.2f}, "
                    f"Cost ${ensemble.avg_cost:.3f}, "
                    f"Diversity {ensemble.diversity_score:.3f}, "
                    f"Overall {ensemble.ensemble_score:.3f}"
                )

            combination_count += 1

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info(f"Correlation report saved to {output_path}")

        return report_text
