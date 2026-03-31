"""Log diagnostic information for numeric prediction processing."""

import logging

from forecasting_tools import NumericDistribution
from forecasting_tools.data_models.questions import NumericQuestion

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
    except Exception as log_e:
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


def log_pchip_fallback(question: NumericQuestion, error: Exception) -> None:
    """Log when PCHIP CDF construction fails and fallback is used."""
    logger.warning(
        f"Question {getattr(question, 'id_of_question', 'N/A')}: PCHIP CDF construction failed ({str(error)}), "
        "falling back to forecasting-tools default"
    )
