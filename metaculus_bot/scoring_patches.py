"""
DEPRECATED: Scoring patches for mixed question types in forecasting-tools.

This module monkey patches the forecasting-tools library to add scoring support
for numeric and multiple choice questions that currently have NotImplementedError.

NOTE: Metaculus removed the ``aggregations`` field from their list API, so
``community_prediction_at_access_time`` and all community-prediction-based
baseline scoring (``expected_baseline_score``) is broken for newly-fetched questions.
This module is used by ``community_benchmark.py`` AND the active
``backtest.py`` / ``analyze_correlations.py`` scoring paths.
Prefer ``backtest.py`` + ``metaculus_bot/backtest/scoring.py`` which score against
actual question resolutions.

Scope: the monkey-patch installers, the baseline-score math, and the run-wide diagnostic
counters (the ``_MC_*`` / ``_NUMERIC_*`` globals that ``reset_scoring_path_stats`` clears
and ``get_scoring_path_stats`` reports). The parsing layer that reads community forecasts
off a question's ``api_json`` lives in ``metaculus_bot/scoring_extraction.py`` and is
re-exported below; its MC extractors bump the counters here by name.
"""

import logging
import math
from typing import Any

import numpy as np

from metaculus_bot.scoring_extraction import (
    _extract_mc_community_probs,
    _extract_numeric_community_cdf,
    extract_multiple_choice_probabilities,
    extract_numeric_percentiles,
    log_mc_vector_mismatch,
    validate_community_prediction_count,
)

# Re-exported for callers that have always imported them from this module path
# (``community_benchmark``, ``analyze_correlations``, ``ensemble_simulator``, the test
# suite). The parsing itself now lives in ``scoring_extraction``.
__all__ = [
    "apply_scoring_patches",
    "calculate_multiple_choice_baseline_score",
    "calculate_numeric_baseline_score",
    "extract_multiple_choice_probabilities",
    "extract_numeric_percentiles",
    "get_scoring_path_stats",
    "log_mc_vector_mismatch",
    "log_score_scale_validation",
    "log_scoring_path_stats",
    "patch_error_handling",
    "patch_multiple_choice_scoring",
    "patch_numeric_scoring",
    "reset_scoring_path_stats",
    "validate_community_prediction_count",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Scoring path counters for diagnostics across a run
_NUMERIC_PMF_ATTEMPTS = 0
_NUMERIC_PMF_SUCCESSES = 0
_NUMERIC_FALLBACK_ATTEMPTS = 0
_NUMERIC_FALLBACK_SUCCESSES = 0
_MC_ATTEMPTS = 0
_MC_MISSING_COMMUNITY = 0
_MC_SUCCESSES = 0
# MC diagnostics breakdown
_MC_MISSING_API_JSON = 0
_MC_MISSING_QUESTION_NODE = 0
_MC_MISSING_AGGREGATIONS = 0
_MC_MISSING_PROB_YES_PER_CATEGORY = 0


def reset_scoring_path_stats() -> None:
    global _NUMERIC_PMF_ATTEMPTS, _NUMERIC_PMF_SUCCESSES
    global _NUMERIC_FALLBACK_ATTEMPTS, _NUMERIC_FALLBACK_SUCCESSES
    global _MC_ATTEMPTS, _MC_MISSING_COMMUNITY, _MC_SUCCESSES
    global _MC_MISSING_API_JSON, _MC_MISSING_QUESTION_NODE, _MC_MISSING_AGGREGATIONS, _MC_MISSING_PROB_YES_PER_CATEGORY
    _NUMERIC_PMF_ATTEMPTS = 0
    _NUMERIC_PMF_SUCCESSES = 0
    _NUMERIC_FALLBACK_ATTEMPTS = 0
    _NUMERIC_FALLBACK_SUCCESSES = 0
    _MC_ATTEMPTS = 0
    _MC_MISSING_COMMUNITY = 0
    _MC_SUCCESSES = 0
    _MC_MISSING_API_JSON = 0
    _MC_MISSING_QUESTION_NODE = 0
    _MC_MISSING_AGGREGATIONS = 0
    _MC_MISSING_PROB_YES_PER_CATEGORY = 0


def get_scoring_path_stats() -> dict[str, float | int]:
    total_numeric = _NUMERIC_PMF_ATTEMPTS + _NUMERIC_FALLBACK_ATTEMPTS
    total_mc = _MC_ATTEMPTS
    return {
        "numeric_pmf_attempts": _NUMERIC_PMF_ATTEMPTS,
        "numeric_pmf_successes": _NUMERIC_PMF_SUCCESSES,
        "numeric_fallback_attempts": _NUMERIC_FALLBACK_ATTEMPTS,
        "numeric_fallback_successes": _NUMERIC_FALLBACK_SUCCESSES,
        "numeric_total": total_numeric,
        "numeric_fallback_rate": ((_NUMERIC_FALLBACK_ATTEMPTS / total_numeric) if total_numeric > 0 else 0.0),
        "mc_attempts": total_mc,
        "mc_successes": _MC_SUCCESSES,
        "mc_missing_community": _MC_MISSING_COMMUNITY,
        "mc_missing_rate": ((_MC_MISSING_COMMUNITY / total_mc) if total_mc > 0 else 0.0),
        # MC breakdown
        "mc_missing_api_json": _MC_MISSING_API_JSON,
        "mc_missing_question_node": _MC_MISSING_QUESTION_NODE,
        "mc_missing_aggregations": _MC_MISSING_AGGREGATIONS,
        "mc_missing_prob_yes_per_category": _MC_MISSING_PROB_YES_PER_CATEGORY,
    }


def log_scoring_path_stats() -> None:
    stats = get_scoring_path_stats()
    logger.info("=== SCORING PATH SUMMARY ===")
    logger.info(
        "Numeric: pmf_attempts=%d pmf_successes=%d fallback_attempts=%d fallback_successes=%d total=%d fallback_rate=%.2f",
        stats["numeric_pmf_attempts"],
        stats["numeric_pmf_successes"],
        stats["numeric_fallback_attempts"],
        stats["numeric_fallback_successes"],
        stats["numeric_total"],
        stats["numeric_fallback_rate"],
    )
    logger.info(
        "MC: attempts=%d successes=%d missing_community=%d missing_rate=%.2f",
        stats["mc_attempts"],
        stats["mc_successes"],
        stats["mc_missing_community"],
        stats["mc_missing_rate"],
    )
    logger.info(
        "MC missing breakdown: api_json=%d question_node=%d aggregations=%d prob_yes_per_category=%d",
        stats["mc_missing_api_json"],
        stats["mc_missing_question_node"],
        stats["mc_missing_aggregations"],
        stats["mc_missing_prob_yes_per_category"],
    )

    # Bright warnings when fallbacks dominate
    if stats["numeric_total"] and stats["numeric_fallback_rate"] >= 0.8:
        logger.warning(
            "⚠️  ALERT: Numeric scoring fallback used for %.0f%% of items. Check that model predictions expose CDFs.",
            100 * stats["numeric_fallback_rate"],
        )
    logger.info("=== END SCORING SUMMARY ===")


_CACHE_MISS = object()


def _cached_baseline_score(cache: dict | None, q_id: Any, q_type: str, log_label: str) -> Any:
    """The cached score for this question, or ``_CACHE_MISS`` when nothing is cached.

    A cached None means "previously failed"; the caller returns it without re-logging.
    ``log_label`` is passed rather than derived from ``q_type`` so each caller keeps its
    own long-standing log prefix ("MC" vs "Numeric") verbatim.
    """
    if cache is None or q_id is None:
        return _CACHE_MISS
    entry = cache.get((q_id, q_type))
    if entry is None:
        return _CACHE_MISS
    cached_score, _diagnostics_logged = entry
    if cached_score is not None:
        logger.debug(f"{log_label} Question {q_id}: using cached baseline score {cached_score:.2f}")
    return cached_score


def _report_mc_vector_mismatch(
    report: Any,
    *,
    bot_probs: list[float],
    community_probs: list[float],
    community_source: str,
    bot_option_names: list[str],
    q_id: Any,
    cache: dict | None,
) -> None:
    """Log the vector-length mismatch diagnostics once per question, then cache the failure."""
    diagnostics_logged = False
    if cache is not None and q_id is not None:
        entry = cache.get((q_id, "multiple_choice"))
        if entry is not None:
            _, diagnostics_logged = entry

    if diagnostics_logged:
        logger.debug(f"MC Question {q_id}: vector mismatch (diagnostics already logged)")
        return

    log_mc_vector_mismatch(
        report.question,
        bot_probs,
        community_probs,
        community_source=community_source,
        bot_option_names=bot_option_names,
    )
    # Cache the failed result with diagnostics logged
    if cache is not None and q_id is not None:
        cache[(q_id, "multiple_choice")] = (None, True)


def _clamp_and_renormalize(probs: list[float]) -> list[float]:
    """Clamp into [0.001, 0.999] then renormalize, falling back to uniform on a zero sum."""
    clamped = [max(min(float(p), 0.999), 0.001) for p in probs]
    total = sum(clamped)
    if total > 0:
        return [p / total for p in clamped]
    return [1.0 / len(clamped)] * len(clamped)


def _mc_expected_baseline_score(bot_probs: list[float], community_probs: list[float]) -> float:
    """100 * (E_c[ln p] / ln K + 1) over clamp-and-renormalized vectors."""
    eps = 1e-9
    bot_probs = _clamp_and_renormalize(bot_probs)
    community_probs = _clamp_and_renormalize(community_probs)

    K = max(1, len(bot_probs))
    lnK = math.log(K) if K > 1 else 1.0
    sum_ln = 0.0
    for c_i, p_i in zip(community_probs, bot_probs, strict=True):
        sum_ln += c_i * math.log(max(p_i, eps))
    return 100.0 * (sum_ln / lnK + 1.0)


def calculate_multiple_choice_baseline_score(report: Any, cache: dict | None = None) -> float | None:
    """
    Calculate baseline score for multiple choice questions.

    Uses the same log scoring pattern as binary questions:
    100.0 * sum(c_i * (log2(p_i) + 1.0)) for each option i

    Args:
        report: MultipleChoiceReport object
        cache: Optional cache dict to avoid duplicate calculations and logging
               Format: {(q_id, q_type): (score, diagnostics_logged)}

    Returns:
        Baseline score or None if cannot be calculated
    """
    global _MC_ATTEMPTS, _MC_SUCCESSES

    # Check cache first to avoid duplicate calculations
    q_id = getattr(report.question, "id_of_question", None)
    cached = _cached_baseline_score(cache, q_id, "multiple_choice", "MC")
    if cached is not _CACHE_MISS:
        return cached

    try:
        _MC_ATTEMPTS += 1
        # Extract bot prediction probabilities
        bot_probs, bot_option_names = extract_multiple_choice_probabilities(report.prediction)
        if not bot_probs:
            logger.warning(
                f"MC Question {getattr(report.question, 'id_of_question', 'unknown')}: cannot extract bot probabilities"
            )
            return None

        # Extract community probabilities (extractor logs causes and increments counters)
        community_probs, community_source = _extract_mc_community_probs(report.question)
        if not community_probs:
            logger.info(
                f"MC Question {getattr(report.question, 'id_of_question', 'unknown')}: missing community probabilities"
            )
            return None
        if len(community_probs) != len(bot_probs):
            _report_mc_vector_mismatch(
                report,
                bot_probs=bot_probs,
                community_probs=community_probs,
                community_source=community_source,
                bot_option_names=bot_option_names,
                q_id=q_id,
                cache=cache,
            )
            return None

        final_score = _mc_expected_baseline_score(bot_probs, community_probs)
        _MC_SUCCESSES += 1

        # Cache the result and log appropriately
        if cache is not None and q_id is not None:
            cache[(q_id, "multiple_choice")] = (
                final_score,
                False,
            )  # Score calculated, no diagnostics needed
            logger.debug(f"MC Question {q_id}: baseline score {final_score:.2f} (cached for future use)")
        else:
            logger.debug(
                f"MC Question {getattr(report.question, 'id_of_question', 'unknown')}: baseline score {final_score:.2f}"
            )

        return final_score

    except Exception:
        logger.exception(
            f"Error calculating MC baseline score for question {getattr(report.question, 'id_of_question', 'unknown')}"
        )
        return None


def calculate_numeric_baseline_score(report: Any, cache: dict | None = None) -> float | None:
    """
    Calculate baseline score for numeric questions relative to community distribution.

    Uses the same pattern as binary/MC scoring: computes expected log score of bot's PMF
    under community distribution, then scales to comparable range with binary/MC.

    Formula: 100.0 * (E_community[ln(bot_pmf)] / ln(10) + 1.0)

    The fixed normalization ln(10) ≈ 2.3 ensures numeric scores have similar benchmark
    impact as MC questions (~[-100, +20] range) rather than being over-compressed by
    the large number of bins typical in numeric questions.

    Args:
        report: NumericReport-like object with `.prediction.cdf` and question `api_json`.
        cache: Optional cache dict to avoid duplicate calculations and logging
               Format: {(q_id, q_type): (score, diagnostics_logged)}

    Returns:
        Baseline score or None if cannot be calculated (expected range: ~[-100, +20])
    """
    # Check cache first to avoid duplicate calculations
    q_id = getattr(report.question, "id_of_question", None)
    cached = _cached_baseline_score(cache, q_id, "numeric", "Numeric")
    if cached is not _CACHE_MISS:
        return cached

    try:
        model_cdf_percentiles = _model_cdf_percentiles(report)
        community_cdf = _extract_numeric_community_cdf(report.question)

        # If either CDF is missing or too short, fall back to percentile-based approximation
        if not community_cdf or len(community_cdf) < 2 or model_cdf_percentiles is None:
            logger.info(
                f"Numeric Question {getattr(report.question, 'id_of_question', 'unknown')}: missing community/model CDF; using percentile fallback"
            )
            return _score_numeric_from_declared_percentiles(report, model_cdf_percentiles, q_id=q_id, cache=cache)

        return _score_numeric_from_cdfs(report, model_cdf_percentiles, community_cdf, q_id=q_id, cache=cache)

    except Exception:
        logger.exception(
            f"Error calculating numeric baseline score for question {getattr(report.question, 'id_of_question', 'unknown')}"
        )
        return None


def _model_cdf_percentiles(report: Any) -> Any:
    """The model's CDF if it looks like a sequence of >= 2 percentile-like objects, else None."""
    try:
        candidate_cdf = getattr(report.prediction, "cdf", None)
        # Validate the CDF looks like a sequence of percentile-like objects
        if isinstance(candidate_cdf, (list, tuple)) and len(candidate_cdf) >= 2:
            return candidate_cdf
    # Boundary: ``cdf`` is a computed property on the prediction, so ANY failure to build it
    # degrades to the declared-percentile fallback below rather than losing the score.
    except Exception as e:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except
        logger.warning(
            f"Numeric Question {getattr(report.question, 'id_of_question', 'unknown')}: cannot compute model CDF: {e}"
        )
    return None


def _numeric_bounds(report: Any, missing_note: str) -> tuple[float, float] | None:
    """Read the question's bounds, warning with ``missing_note`` when either is absent."""
    lower_bound = getattr(report.question, "lower_bound", None)
    upper_bound = getattr(report.question, "upper_bound", None)
    if lower_bound is None or upper_bound is None:
        logger.warning(f"Numeric Question {getattr(report.question, 'id_of_question', 'unknown')}: {missing_note}")
        return None
    return lower_bound, upper_bound


def _score_numeric_from_declared_percentiles(
    report: Any, model_cdf_percentiles: Any, *, q_id: Any, cache: dict | None
) -> float | None:
    """Fallback path: approximate the bot PMF from declared percentiles, score vs uniform."""
    global _NUMERIC_FALLBACK_ATTEMPTS
    try:
        _NUMERIC_FALLBACK_ATTEMPTS += 1
        bounds = _numeric_bounds(report, "missing bounds, cannot calculate bins")
        if bounds is None:
            return None
        lower_bound, upper_bound = bounds

        # Use declared percentiles to approximate PMF
        declared = getattr(report.prediction, "declared_percentiles", None)
        if not declared and model_cdf_percentiles:
            declared = model_cdf_percentiles[::20]  # Take every 20th for approximation

        if not declared or len(declared) < 3:
            return None

        # Convert percentiles to CDF approximation
        percs = [
            float(getattr(p, "percentile", 0)) / (100.0 if getattr(p, "percentile", 1) > 1 else 1.0) for p in declared
        ]

        # Create approximate PMF from percentile differences
        bot_pmf = np.diff(percs)
        bot_pmf = np.maximum(bot_pmf, 0.0)
        if bot_pmf.sum() <= 0:
            return None
        bot_pmf = bot_pmf / bot_pmf.sum()

        # Create uniform community PMF (fallback when no community CDF)
        community_pmf = np.ones(len(bot_pmf)) / len(bot_pmf)

        return _calculate_relative_numeric_score(
            bot_pmf, community_pmf, total_range=upper_bound - lower_bound, q_id=q_id, cache=cache
        )

    # Boundary: the fallback is already the degraded rung, so its own failure means
    # "no score for this question", never a failed benchmark run.
    except Exception as e:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except
        logger.warning(f"Numeric Question {q_id}: percentile fallback scoring failed: {e}")
        return None


def _pmf_from_cdf(cdf_values: Any, degenerate_note: str, report: Any) -> np.ndarray | None:
    """Clip a CDF into [0, 1], difference it into a PMF, and renormalize. None if degenerate."""
    clipped = np.clip(np.array(cdf_values, dtype=float), 0.0, 1.0)
    pmf = np.maximum(np.diff(clipped), 0.0)
    if pmf.sum() <= 0:
        logger.warning(f"Numeric Question {getattr(report.question, 'id_of_question', 'unknown')}: {degenerate_note}")
        return None
    return pmf / pmf.sum()


def _score_numeric_from_cdfs(
    report: Any, model_cdf_percentiles: Any, community_cdf: list[float], *, q_id: Any, cache: dict | None
) -> float | None:
    """Primary path: PMF-based relative scoring against the community distribution."""
    global _NUMERIC_PMF_ATTEMPTS
    _NUMERIC_PMF_ATTEMPTS += 1

    bounds = _numeric_bounds(report, "missing bounds for PMF scoring")
    if bounds is None:
        return None
    lower_bound, upper_bound = bounds

    bot_pmf = _pmf_from_cdf([float(p.percentile) for p in model_cdf_percentiles], "model PMF degenerate", report)
    if bot_pmf is None:
        return None

    community_pmf = _pmf_from_cdf(community_cdf, "community PMF degenerate", report)
    if community_pmf is None:
        return None

    # Align lengths (guard, though both should be same length)
    m = min(len(bot_pmf), len(community_pmf))

    return _calculate_relative_numeric_score(
        bot_pmf[:m], community_pmf[:m], total_range=upper_bound - lower_bound, q_id=q_id, cache=cache
    )


def _calculate_relative_numeric_score(
    bot_pmf: np.ndarray, community_pmf: np.ndarray, *, total_range: float, q_id: int | None, cache: dict | None
) -> float | None:
    """
    Calculate relative numeric score using community PMF as expectation weights.

    Follows same pattern as binary/MC: 100.0 * (E_community[ln(bot_pmf)] / normalization + 1.0)

    Uses a bin-aware normalization to put numeric scores in a range comparable to MC/binary.
    Specifically, normalization = a + b * ln(num_bins) with a≈1.46, b≈0.06, calibrated so that:
    - Identical 11-bin uniform distributions score around -50
    - 201-bin uniform remains above ~-200

    Args:
        bot_pmf: Bot's probability mass function
        community_pmf: Community's probability mass function (used as weights)
        total_range: Upper bound - lower bound (for bin width calculation)
        q_id: Question ID for logging
        cache: Cache for results

    Returns:
        Relative baseline score (expected range: roughly [-200, +100])
    """
    try:
        # Apply 1% uniform mixture to PMF (like MC applies eps to probabilities)
        uniform_pmf = np.ones(len(bot_pmf)) / len(bot_pmf)
        bot_pmf_scored = 0.99 * bot_pmf + 0.01 * uniform_pmf

        # Calculate expected log score: E_community[ln(bot_pmf)]
        eps = 1e-9  # Same as MC scoring
        expected_log_score = sum(
            community_pmf[i] * math.log(max(bot_pmf_scored[i], eps)) for i in range(len(community_pmf))
        )

        # Bin-aware normalization so uniform vs uniform anchors near -50 for any bin count:
        # Solve 100 * (-ln(n)/norm + 1) = -50  =>  norm = ln(n)/1.5
        num_bins = max(2, len(bot_pmf))
        normalization = math.log(num_bins) / 1.5
        final_score = 100.0 * (expected_log_score / normalization + 1.0)

        global _NUMERIC_PMF_SUCCESSES
        _NUMERIC_PMF_SUCCESSES += 1

        # Cache the result and log appropriately
        if cache is not None and q_id is not None:
            cache[(q_id, "numeric")] = (
                final_score,
                False,
            )  # Score calculated, no diagnostics needed
            logger.debug(f"Numeric Question {q_id}: baseline score {final_score:.2f} (relative to community, cached)")
        else:
            logger.debug(f"Numeric Question: baseline score {final_score:.2f} (relative to community)")

        return final_score

    except Exception:
        logger.exception("Error in relative numeric scoring")
        return None


def patch_multiple_choice_scoring():
    """Monkey patch MultipleChoiceReport.expected_baseline_score"""
    try:
        from forecasting_tools.data_models.multiple_choice_report import (  # noqa: PLC0415  # late import: ImportError is handled to degrade the monkey-patch
            MultipleChoiceReport,
        )

        def expected_baseline_score_mc(self) -> float | None:
            return calculate_multiple_choice_baseline_score(self)

        # Monkey-patch: replace the read-only property descriptor at runtime.
        MultipleChoiceReport.expected_baseline_score = property(expected_baseline_score_mc)  # pyright: ignore[reportAttributeAccessIssue]  # monkey-patch over property
        logger.info("Successfully patched MultipleChoiceReport.expected_baseline_score")

    except ImportError as e:
        logger.error(f"Could not import MultipleChoiceReport for patching: {e}")
    # Boundary: patching a third-party class is best-effort. A failure here leaves ft's own
    # NotImplementedError property in place; it must not abort the importing benchmark.
    except Exception as e:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except
        logger.error(f"Error patching MultipleChoiceReport: {e}")


def patch_numeric_scoring():
    """Monkey patch NumericReport.expected_baseline_score"""
    try:
        from forecasting_tools.data_models.numeric_report import (  # noqa: PLC0415  # late import: ImportError is handled to degrade the monkey-patch
            NumericReport,
        )

        def expected_baseline_score_numeric(self) -> float | None:
            return calculate_numeric_baseline_score(self)

        # Monkey-patch: replace the read-only property descriptor at runtime.
        NumericReport.expected_baseline_score = property(expected_baseline_score_numeric)  # pyright: ignore[reportAttributeAccessIssue]  # monkey-patch over property
        logger.info("Successfully patched NumericReport.expected_baseline_score")

    except ImportError as e:
        logger.error(f"Could not import NumericReport for patching: {e}")
    # Boundary: same as the MC patch above — best-effort monkey-patch, never a hard failure.
    except Exception as e:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except
        logger.error(f"Error patching NumericReport: {e}")


def patch_error_handling():
    """Monkey patch ForecastReport.calculate_average_expected_baseline_score to fix UnboundLocalError"""
    try:
        from collections.abc import (  # noqa: PLC0415  # late import: ImportError is handled to degrade the monkey-patch
            Sequence,
        )

        import typeguard  # noqa: PLC0415  # late import: ImportError is handled to degrade the monkey-patch
        from forecasting_tools.data_models.forecast_report import (  # noqa: PLC0415  # late import: ImportError is handled to degrade the monkey-patch
            ForecastReport,
        )

        @staticmethod
        def calculate_average_expected_baseline_score_fixed(
            reports: Sequence[Any],
        ) -> float:
            assert len(reports) > 0, "Must have at least one report to calculate average expected baseline score"

            try:
                scores: list[float | None] = [report.expected_baseline_score for report in reports]
                # Filter out None scores
                valid_scores = [score for score in scores if score is not None]

                if not valid_scores:
                    logger.warning("All baseline scores are None, cannot calculate average")
                    return 0.0

                validated_scores: list[float] = typeguard.check_type(valid_scores, list[float])
                average_score = sum(validated_scores) / len(validated_scores)

                none_count = len([score for score in scores if score is None])
                if none_count > 0:
                    logger.warning(f"Calculated average from {len(valid_scores)} scores, {none_count} were None")

                return average_score

            except Exception as e:
                # Fix the UnboundLocalError by ensuring scores is always defined
                scores = [report.expected_baseline_score for report in reports]
                none_count = len([score for score in scores if score is None])
                raise ValueError(
                    f"Error calculating average expected baseline score. {len(reports)} reports. "
                    f"There were {none_count} None scores. Error: {e}"
                ) from e

        ForecastReport.calculate_average_expected_baseline_score = calculate_average_expected_baseline_score_fixed
        logger.info("Successfully patched ForecastReport.calculate_average_expected_baseline_score")

    except ImportError as e:
        logger.error(f"Could not import ForecastReport for patching: {e}")
    # Boundary: same as the two patches above — best-effort monkey-patch, never a hard failure.
    except Exception as e:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except
        logger.error(f"Error patching ForecastReport: {e}")


def log_score_scale_validation(benchmarks: list[Any]) -> None:
    """
    Log score distributions by question type to verify consistent scaling.

    Args:
        benchmarks: List of BenchmarkForBot objects
    """
    try:
        scores_by_label = _scores_by_question_type(benchmarks)

        logger.info("=== SCORE SCALE VALIDATION ===")
        for label, scores in scores_by_label.items():
            _log_score_distribution(label, scores)
        logger.info("=== END SCORE VALIDATION ===")

    # Boundary: this is diagnostics-only, so a report that cannot score must not take down
    # the benchmark run that called it.
    except Exception as e:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except
        logger.error(f"Error in score scale validation: {e}")


def _scores_by_question_type(benchmarks: list[Any]) -> dict[str, list[float]]:
    """Bucket every non-None ``expected_baseline_score`` by question type, in report order."""
    from forecasting_tools.data_models.questions import (  # noqa: PLC0415  # late import: ImportError is handled by the caller to degrade the diagnostics
        BinaryQuestion,
        MultipleChoiceQuestion,
        NumericQuestion,
    )

    label_by_type = {BinaryQuestion: "Binary", NumericQuestion: "Numeric", MultipleChoiceQuestion: "MC"}
    scores: dict[str, list[float]] = {label: [] for label in label_by_type.values()}

    for benchmark in benchmarks:
        for report in benchmark.forecast_reports:
            score = report.expected_baseline_score
            if score is None:
                continue
            for question_type, label in label_by_type.items():
                if isinstance(report.question, question_type):
                    scores[label].append(score)
                    break

    return scores


def _log_score_distribution(label: str, scores: list[float]) -> None:
    """One count/range/mean/mean_abs line per question type, or a "no data" line."""
    if not scores:
        logger.info(f"{label} scores: no data")
        return
    logger.info(
        f"{label} scores: count={len(scores)}, range=[{min(scores):.1f}, {max(scores):.1f}], "
        f"mean={np.mean(scores):.1f}, mean_abs={np.mean([abs(s) for s in scores]):.1f}"
    )


def apply_scoring_patches() -> None:
    """
    Apply all scoring patches to the forecasting-tools library.

    This function should be called before running benchmarks with mixed question types.
    """
    logger.info("Applying scoring patches for mixed question types...")

    patch_multiple_choice_scoring()
    patch_numeric_scoring()
    patch_error_handling()

    logger.info("Scoring patches applied successfully")
