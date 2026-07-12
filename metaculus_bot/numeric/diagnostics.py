"""Log diagnostic information for numeric prediction processing."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from forecasting_tools import NumericDistribution
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.numeric.config import OPEN_BOUND_PILING_THRESHOLD

logger = logging.getLogger(__name__)


def log_cdf_diagnostics_on_error(prediction: NumericDistribution, question: NumericQuestion, error: Exception) -> None:
    """Log rich diagnostics when CDF construction fails."""
    try:
        declared = getattr(prediction, "declared_percentiles", [])
        bounds = {
            "lower_bound": question.lower_bound,
            "upper_bound": question.upper_bound,
            "open_lower_bound": question.open_lower_bound,
            "open_upper_bound": question.open_upper_bound,
            "zero_point": question.zero_point,
            "cdf_size": getattr(question, "cdf_size", None),
        }
        vals = [float(p.value) for p in declared]
        prcs = [float(p.percentile) for p in declared]
        deltas_val = [b - a for a, b in zip(vals, vals[1:])]
        deltas_pct = [b - a for a, b in zip(prcs, prcs[1:])]

        logger.error(
            "Numeric CDF spacing assertion for Q %s | URL %s | error=%s\n"
            "Bounds=%s\n"
            "Declared percentiles (p%% -> v): %s\n"
            "Value deltas: %s | Percentile deltas: %s",
            getattr(question, "id_of_question", None),
            getattr(question, "page_url", None),
            error,
            bounds,
            [(p, v) for p, v in zip(prcs, vals)],
            deltas_val,
            deltas_pct,
        )
    # Caller re-raises the ORIGINAL error; a crash in diagnostics logging must not mask it.
    except Exception as log_e:  # HARNESS-SCAN-EXEMPT-broad-except
        logger.error("Failed logging numeric CDF diagnostics: %s", log_e)


def validate_cdf_construction(prediction: NumericDistribution, question: NumericQuestion) -> None:
    """Validate CDF construction for non-PCHIP distributions."""
    # Skip CDF validation for PCHIP distributions since they enforce constraints internally.
    # PchipNumericDistribution is defined locally in pchip_processing.py and can't be imported,
    # so we check for the distinguishing attribute directly.
    if getattr(prediction, "_pchip_cdf_values", None) is not None:
        logger.debug(f"Question {question.id_of_question}: Skipping CDF validation for PCHIP distribution")
        return

    try:
        # Force CDF construction to surface any issues
        _ = prediction.cdf
    except (AssertionError, ZeroDivisionError) as e:
        log_cdf_diagnostics_on_error(prediction, question, e)
        raise


def log_final_prediction(prediction: NumericDistribution, question: NumericQuestion) -> None:
    """Log the final prediction for debugging purposes."""
    logger.info(
        f"Forecasted URL {getattr(question, 'page_url', '<unknown>')} as {getattr(prediction, 'declared_percentiles', [])}"
    )


def log_open_bound_piling_diagnostics(
    prediction: NumericDistribution,
    question: NumericQuestion,
    model_name: str,
    declared_percentiles: Sequence[Percentile],
    *,
    threshold: float = OPEN_BOUND_PILING_THRESHOLD,
) -> None:
    """WARN when probability mass piles onto the terminal bin of an OPEN bound.

    On an open bound, the displayed edge is not a hard cap: mass beyond it is expressed
    by placing percentiles past the edge. When a model instead crams the terminal bin and
    keeps every declared percentile inside the range, it is treating the open edge as a
    hard limit — the prompt-contradiction bug this detector surfaces.

    ``declared_percentiles`` must be the MODEL-DECLARED (sanitized, pre-CDF-build) values,
    not ``prediction.declared_percentiles``: on discrete questions the resample in
    ``build_numeric_distribution`` overwrites that field with a grid on
    ``[lower_bound, upper_bound]``, pinning the max declared value at exactly the raw
    bound and false-firing on models that correctly placed percentiles above the ceiling.
    Terminal-bin mass is still read from the built ``prediction.cdf``.

    Diagnostics only: never raises, never mutates the prediction.
    """
    if not question.open_upper_bound and not question.open_lower_bound:
        return

    cdf = getattr(prediction, "cdf", None)
    if not cdf or len(cdf) < 2 or not declared_percentiles:
        return

    declared_values = [p.value for p in declared_percentiles]

    def _warn(bound: str, bin_mass: float, declared_edge: float, bound_value: float) -> None:
        logger.warning(
            "OPEN_BOUND_PILING: question=%s model=%s bound=%s bin_mass=%.3f declared_edge=%.6g bound_value=%.6g",
            getattr(question, "id_of_question", None),
            model_name,
            bound,
            bin_mass,
            declared_edge,
            bound_value,
        )

    if question.open_upper_bound:
        top_bin_mass = cdf[-1].percentile - cdf[-2].percentile
        max_declared = max(declared_values)
        if top_bin_mass >= threshold and max_declared <= question.upper_bound:
            _warn("upper", top_bin_mass, max_declared, question.upper_bound)

    if question.open_lower_bound:
        bottom_bin_mass = cdf[1].percentile - cdf[0].percentile
        min_declared = min(declared_values)
        if bottom_bin_mass >= threshold and min_declared >= question.lower_bound:
            _warn("lower", bottom_bin_mass, min_declared, question.lower_bound)


def log_pchip_fallback(question: NumericQuestion, error: Exception) -> None:
    """Log when PCHIP CDF construction fails and fallback is used."""
    logger.warning(
        f"Question {getattr(question, 'id_of_question', 'N/A')}: PCHIP CDF construction failed ({str(error)}), "
        "falling back to forecasting-tools default"
    )
