"""Post-processing steps applied after aggregation: Platt calibration, discrete integer
snapping, and the single-survivor binary publish floor."""

import logging

from forecasting_tools import BinaryQuestion, NumericQuestion
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.calibration import PlattParams, apply_binary_platt, apply_mc_platt
from metaculus_bot.constants import (
    PLATT_BINARY_MAX_ABS_DEVIATION,
    PLATT_CALIBRATION_ENABLED_ENV,
    PLATT_MC_MAX_ABS_DEVIATION,
    THIN_PUBLISH_BINARY_CEIL,
    THIN_PUBLISH_BINARY_FLOOR,
    env_flag_enabled,
)
from metaculus_bot.numeric.discrete_snap import majority_votes_discrete, snap_distribution_to_integers

logger = logging.getLogger(__name__)


def apply_thin_publish_floor(value: float, survivors: int) -> float:
    """Clamp a PUBLISHED binary probability to [THIN_PUBLISH_BINARY_FLOOR, _CEIL] when
    exactly one forecaster survived; any other survivor count passes ``value`` through.

    Pure, and keyed on the survivor COUNT rather than the value, because the floor
    prices a missing aggregation step: median-of-1 has no variance reduction, so the range
    the published value may occupy is NARROWED in exactly that state — the band sits
    strictly inside the per-model ``[BINARY_PROB_MIN, BINARY_PROB_MAX]`` clamp the member
    already passed. The receipt behind the constants discusses a ``k <= 2`` generalisation
    (at k=2 the median is the midpoint, so a lone outlier is averaged in at half weight
    instead of outvoted); a k=2 publish has never happened, so there is nothing to fit it
    on and the rule stays at ``== 1``. Do not loosen that trigger without data.
    """
    if survivors != 1:
        return value
    return max(THIN_PUBLISH_BINARY_FLOOR, min(THIN_PUBLISH_BINARY_CEIL, value))


def apply_platt_calibration(
    prediction: PredictionTypes,
    question: MetaculusQuestion,
    binary_platt_params: PlattParams,
    mc_platt_params: PlattParams,
) -> PredictionTypes:
    """Apply post-hoc Platt scaling to a binary or MC probability.

    Returns the prediction unchanged when calibration is disabled, both param
    sets are identity, or the prediction is a NumericDistribution.
    """
    if not env_flag_enabled(PLATT_CALIBRATION_ENABLED_ENV):
        return prediction

    if binary_platt_params.is_identity() and mc_platt_params.is_identity():
        return prediction

    if isinstance(question, BinaryQuestion) and isinstance(prediction, (int, float)):
        calibrated = apply_binary_platt(
            float(prediction),
            binary_platt_params,
            max_abs_deviation=PLATT_BINARY_MAX_ABS_DEVIATION,
        )
        if calibrated != float(prediction):
            logger.info(
                "PLATT_BINARY: q=%s raw=%.4f calibrated=%.4f bias=%.4f slope=%.4f cap=%.3f",
                question.id_of_question,
                float(prediction),
                calibrated,
                binary_platt_params.bias,
                binary_platt_params.slope,
                PLATT_BINARY_MAX_ABS_DEVIATION,
            )
        return calibrated  # type: ignore[return-value]

    if isinstance(prediction, PredictedOptionList):
        raw_summary = {o.option_name: round(o.probability, 4) for o in prediction.predicted_options}
        apply_mc_platt(
            prediction,
            mc_platt_params,
            max_abs_deviation=PLATT_MC_MAX_ABS_DEVIATION,
        )
        calibrated_summary = {o.option_name: round(o.probability, 4) for o in prediction.predicted_options}
        if raw_summary != calibrated_summary:
            logger.info(
                "PLATT_MC: q=%s raw=%s calibrated=%s bias=%.4f slope=%.4f cap=%.3f",
                question.id_of_question,
                raw_summary,
                calibrated_summary,
                mc_platt_params.bias,
                mc_platt_params.slope,
                PLATT_MC_MAX_ABS_DEVIATION,
            )
        return prediction

    return prediction


def maybe_snap_to_integers(
    prediction: PredictionTypes,
    question: MetaculusQuestion,
    discrete_votes: list[bool],
) -> PredictionTypes:
    """Apply discrete integer CDF snapping if a majority of votes are DISCRETE.

    The caller is responsible for popping the votes from the per-question dict
    and passing them in; this function is pure.
    """
    if not isinstance(prediction, NumericDistribution) or not isinstance(question, NumericQuestion):
        return prediction

    if not majority_votes_discrete(discrete_votes):
        if discrete_votes:
            logger.info(
                "Discrete snap skipped for Q %s: votes=%s (majority=CONTINUOUS)",
                question.id_of_question,
                discrete_votes,
            )
        return prediction

    logger.info("Discrete snap: majority voted DISCRETE for Q %s | votes=%s", question.id_of_question, discrete_votes)
    snapped = snap_distribution_to_integers(prediction, question)
    if snapped is None:
        logger.info("Discrete snap returned None for Q %s (guard condition), keeping original", question.id_of_question)
        return prediction
    return snapped  # type: ignore[return-value]
