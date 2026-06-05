"""Shared dataclasses for ensemble correlation analysis.

Leaf module: imports only stdlib + pandas. ``correlation_analysis`` and
``ensemble_simulator`` both depend on these types; defining them here (rather
than in either module) breaks the import cycle those two modules used to form.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


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
    # Raw 'mean'/'median' string rather than AggregationStrategy: external callers
    # (analyze_correlations.py) pass raw strings, and report grouping reads .value-style
    # strings off this field, so the str boundary is intentional.
    aggregation_strategy: str

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
