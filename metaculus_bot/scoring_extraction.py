"""Read forecast values off a Metaculus question's ``api_json`` and off a bot prediction.

Split out of ``scoring_patches.py``, which had grown to hold three unrelated jobs: the
monkey-patch installation into forecasting-tools, the baseline-score math, and this — the
parsing layer that walks ``aggregations.recency_weighted.latest`` for community data and
pulls probabilities/percentiles off a prediction object. Only the parsing lives here; the
scoring formulas and the patch installers stay in ``scoring_patches``, which re-exports
every name below that outside callers import.

The MC extractors report each community-data miss through
``scoring_diagnostics.record_mc_missing``. Those counters live in their own leaf module
rather than in ``scoring_patches`` so that this module never has to import the module that
imports it.
"""

import logging
from typing import Any

from metaculus_bot import scoring_diagnostics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_multiple_choice_probabilities(
    prediction: Any,
) -> tuple[list[float], list[str]]:
    """
    Safely extracts probabilities from a PredictedOptionList, sorting by option name.

    Note: forecasting_tools PredictedOption uses the field `option_name`.

    Returns:
        Tuple of (probabilities, option_names) both in sorted order
    """
    if not prediction or not hasattr(prediction, "predicted_options") or prediction.predicted_options is None:
        return [], []
    # Sort by option name to ensure consistent order
    try:
        sorted_options = sorted(prediction.predicted_options, key=lambda o: o.option_name)
        option_names = [opt.option_name for opt in sorted_options]
    except AttributeError:
        # Fallback if mocks used a different attribute during tests
        sorted_options = sorted(prediction.predicted_options, key=lambda o: getattr(o, "option", ""))
        option_names = [getattr(opt, "option", f"option_{i}") for i, opt in enumerate(sorted_options)]
    return [opt.probability for opt in sorted_options], option_names


def extract_numeric_percentiles(prediction: Any) -> list[tuple[float, float]]:
    """
    Extract (percentile, value) pairs from a numeric prediction.

    Args:
        prediction: NumericDistribution or similar object

    Returns:
        List of (percentile, value) tuples
    """
    try:
        if hasattr(prediction, "declared_percentiles") and prediction.declared_percentiles:
            return [(float(p.percentile), float(p.value)) for p in prediction.declared_percentiles]
    except (TypeError, AttributeError, ValueError) as e:
        logger.warning(f"Failed to extract numeric percentiles: {e}")

    return []


def log_mc_vector_mismatch(
    question: Any,
    bot_probs: list[float],
    community_probs: list[float],
    *,
    community_source: str,
    bot_option_names: list[str],
) -> None:
    """
    Log detailed diagnostics for MC vector length mismatches.

    Args:
        question: MetaculusQuestion object
        bot_probs: Bot prediction probabilities
        community_probs: Community prediction probabilities
        community_source: Source of community data (e.g., "forecast_values", "probability_yes_per_category")
        bot_option_names: Option names from bot prediction (sorted)
    """
    qid = getattr(question, "id_of_question", "unknown")
    question_options = getattr(question, "options", None)

    logger.warning(f"MC Question {qid} VECTOR MISMATCH:")
    logger.warning(f"  Bot prediction: {len(bot_probs)} options {bot_option_names}")
    logger.warning(f"  Community data: {len(community_probs)} options (source: {community_source})")

    if question_options and isinstance(question_options, list):
        logger.warning(f"  Question options: {len(question_options)} options {question_options}")

        # Analyze potential causes
        if len(bot_probs) == len(question_options) and len(community_probs) != len(question_options):
            logger.warning(
                f"  → Likely cause: Community data missing {len(question_options) - len(community_probs)} options"
            )
        elif len(community_probs) == len(question_options) and len(bot_probs) != len(question_options):
            logger.warning(f"  → Likely cause: Bot prediction missing {len(question_options) - len(bot_probs)} options")
        else:
            logger.warning("  → Complex mismatch: bot≠question≠community")
    else:
        logger.warning(f"  Question options: unavailable (type={type(question_options)})")
        logger.warning("  → Cannot determine root cause without question.options")


def _locate_mc_rw_latest(question: Any) -> tuple[dict | None, list | None, Any, str]:
    """Walk ``api_json`` down to ``aggregations.recency_weighted.latest`` for an MC question.

    Returns ``(rw_latest, options, qid, reason)``. On success ``rw_latest`` is the dict and
    ``reason`` is ``""``; on failure ``rw_latest`` is None, ``reason`` names the miss, and the
    miss has been recorded against its breakdown counter plus the ``mc_missing_community``
    rollup.
    """
    # Basic fingerprint
    post_id = getattr(question, "id_of_post", None)
    qid = getattr(question, "id_of_question", None)
    api_json = getattr(question, "api_json", None)
    if not isinstance(api_json, dict):
        logger.warning(
            "MC q=%s post=%s: api_json missing or not dict (type=%s)",
            qid,
            post_id,
            type(api_json).__name__,
        )
        scoring_diagnostics.record_mc_missing("api_json")
        return None, None, qid, "missing_api_json"

    # Detect the question node
    api_has_question = isinstance(api_json.get("question"), dict)
    question_obj = api_json.get("question") if api_has_question else api_json
    if not isinstance(question_obj, dict):
        logger.warning(
            "MC q=%s post=%s: missing question object (api_has_question=%s, type=%s)",
            qid,
            post_id,
            api_has_question,
            type(question_obj).__name__,
        )
        scoring_diagnostics.record_mc_missing("question_node")
        return None, None, qid, "missing_question_node"

    qtype = question_obj.get("type")
    options = getattr(question, "options", None)
    if options is None and isinstance(question_obj.get("options"), list):
        options = question_obj.get("options")

    aggregations = question_obj.get("aggregations")
    if not isinstance(aggregations, dict):
        logger.info(
            "MC q=%s: aggregations missing (question.type=%s). keys=%s",
            qid,
            qtype,
            list(question_obj.keys()),
        )
        scoring_diagnostics.record_mc_missing("aggregations")
        return None, options, qid, "missing_aggregations"

    rw = aggregations.get("recency_weighted")
    rw_latest = rw.get("latest") if isinstance(rw, dict) else None
    logger.debug(
        "MC q=%s: rw.latest keys=%s (agg.keys=%s)",
        qid,
        list(rw_latest.keys()) if isinstance(rw_latest, dict) else None,
        list(aggregations.keys()),
    )

    if not isinstance(rw_latest, dict):
        logger.info("MC q=%s: recency_weighted.latest missing", qid)
        scoring_diagnostics.record_mc_missing("prob_yes_per_category")
        return None, options, qid, "missing_rw_latest"

    return rw_latest, options, qid, ""


def _mc_probs_from_forecast_values(fv: list, options: Any, qid: Any) -> tuple[list[float] | None, str]:
    """Read ``rw.latest.forecast_values``, which is aligned by index with ``options``."""
    if not options or not isinstance(options, list):
        logger.info("MC q=%s: options unavailable; cannot align forecast_values", qid)
        scoring_diagnostics.record_mc_missing()
        return None, "forecast_values_no_options"
    if len(fv) != len(options):
        logger.warning(
            "MC q=%s: forecast_values length %d != options length %d",
            qid,
            len(fv),
            len(options),
        )
        scoring_diagnostics.record_mc_missing("prob_yes_per_category")
        return None, "forecast_values_length_mismatch"
    try:
        probs = [float(x) for x in fv]
    except (TypeError, ValueError) as e:
        logger.warning("MC q=%s: forecast_values cast error: %s", qid, e)
        scoring_diagnostics.record_mc_missing("prob_yes_per_category")
        return None, "forecast_values_cast_error"
    within = all(0.0 <= p <= 1.0 for p in probs)
    total = sum(probs)
    if not within:
        logger.warning("MC q=%s: forecast_values contain out-of-range probabilities", qid)
        scoring_diagnostics.record_mc_missing("prob_yes_per_category")
        return None, "forecast_values_out_of_range"
    if abs(total - 1.0) > 1e-3:
        logger.warning("MC q=%s: forecast_values sum %.6f far from 1.0", qid, total)
        scoring_diagnostics.record_mc_missing("prob_yes_per_category")
        return None, "forecast_values_bad_sum"
    if abs(total - 1.0) > 1e-6:
        logger.info("MC q=%s: normalizing forecast_values (sum=%.6f)", qid, total)
        probs = [p / total for p in probs]
    logger.debug("MC q=%s: using rw.latest.forecast_values aligned to options", qid)
    return probs, "forecast_values"


def _mc_probs_from_pyc(pyc: dict, options: Any, qid: Any) -> tuple[list[float] | None, str]:
    """Read ``rw.latest.probability_yes_per_category``, aligning it to ``options`` order."""
    if not (options and isinstance(options, list)):
        logger.info("MC q=%s: options unavailable; cannot align pyc", qid)
        scoring_diagnostics.record_mc_missing()
        return None, "pyc_no_options"

    keys = sorted(pyc.keys())
    missing = [opt for opt in options if opt not in pyc]
    extra = [k for k in keys if k not in options]
    if missing or extra:
        logger.warning(
            "MC q=%s: option mismatch vs pyc. missing=%s extra=%s",
            qid,
            missing,
            extra,
        )
    # `.get(opt, 0.0)` is deliberate HERE, unlike the backtest scorer's version:
    # the mismatch is already warned about above, and the sum gate below rejects
    # the record outright when a materially-missing option pulls the total off
    # 1.0 — so a fabricated zero never reaches a score. A rounding-level gap
    # (< 1e-3) is renormalized away.
    probs = [float(pyc.get(opt, 0.0)) for opt in options]
    total = sum(probs)
    if abs(total - 1.0) > 1e-6 and abs(total - 1.0) <= 1e-3:
        logger.info("MC q=%s: normalizing pyc (sum=%.6f)", qid, total)
        probs = [p / total for p in probs]
    elif abs(total - 1.0) > 1e-3:
        logger.warning("MC q=%s: pyc sum %.6f far from 1.0", qid, total)
        scoring_diagnostics.record_mc_missing("prob_yes_per_category")
        return None, "pyc_bad_sum"
    logger.debug("MC q=%s: using rw.latest.probability_yes_per_category", qid)
    return probs, "probability_yes_per_category"


def _extract_mc_community_probs(question: Any) -> tuple[list[float] | None, str]:
    """Extract community option probabilities for an MC question from api_json.

    According to the Metaculus API, community MC aggregations expose
    `probability_yes_per_category` under `aggregations.recency_weighted.latest`.
    We align the resulting vector to `question.options` order.
    """
    try:
        rw_latest, options, qid, reason = _locate_mc_rw_latest(question)
        if rw_latest is None:
            return None, reason

        # Prefer forecast_values (index-aligned) over the pyc dict.
        fv = rw_latest.get("forecast_values")
        if isinstance(fv, list):
            return _mc_probs_from_forecast_values(fv, options, qid)

        pyc = rw_latest.get("probability_yes_per_category")
        if isinstance(pyc, dict):
            return _mc_probs_from_pyc(pyc, options, qid)

        logger.info(
            "MC q=%s: neither forecast_values nor pyc available in rw.latest (keys=%s)",
            qid,
            list(rw_latest.keys()),
        )
        scoring_diagnostics.record_mc_missing("prob_yes_per_category")
        return None, "no_forecast_data"

    # Boundary: this extractor's contract is "never raise, report a reason" — the MC scorer
    # treats every reason as "no community data" and skips the question.
    except Exception as e:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except
        logger.warning(f"Failed to extract MC community probabilities: {e}")
        scoring_diagnostics.record_mc_missing()
    return None, "exception"


def _extract_numeric_community_cdf(question: Any) -> list[float] | None:
    """Extract community CDF (forecast_values) from api_json with structured logging; no fallback."""
    try:
        post_id = getattr(question, "id_of_post", None)
        qid = getattr(question, "id_of_question", None)
        api_json = getattr(question, "api_json", None)
        if not isinstance(api_json, dict):
            logger.warning(
                "Numeric q=%s post=%s: api_json missing or not dict (type=%s)",
                qid,
                post_id,
                type(api_json).__name__,
            )
            return None

        api_has_question = isinstance(api_json.get("question"), dict)
        question_obj = api_json.get("question") if api_has_question else api_json
        if not isinstance(question_obj, dict):
            logger.warning(
                "Numeric q=%s post=%s: missing question object (api_has_question=%s, type=%s)",
                qid,
                post_id,
                api_has_question,
                type(question_obj).__name__,
            )
            return None

        expected_len = None
        try:
            scaling = question_obj.get("scaling", {})
            inbound = scaling.get("inbound_outcome_count")
            if inbound is not None:
                expected_len = int(inbound) + 1
        except (ValueError, TypeError):
            expected_len = None

        aggregations = question_obj.get("aggregations")
        if not isinstance(aggregations, dict):
            logger.info(
                "Numeric q=%s: aggregations missing. keys=%s",
                qid,
                list(question_obj.keys()),
            )
            return None

        rw = aggregations.get("recency_weighted")
        rw_latest = rw.get("latest") if isinstance(rw, dict) else None
        rw_keys = list(rw_latest.keys()) if isinstance(rw_latest, dict) else None
        logger.debug(
            "Numeric q=%s: rw.latest keys=%s (agg.keys=%s)",
            qid,
            rw_keys,
            list(aggregations.keys()),
        )
        if not isinstance(rw_latest, dict):
            logger.info("Numeric q=%s: recency_weighted.latest missing", qid)
            return None

        fv = rw_latest.get("forecast_values")
        if isinstance(fv, list) and len(fv) >= 2:
            if expected_len and len(fv) != expected_len:
                logger.warning(
                    "Numeric q=%s: forecast_values length %d != expected %d",
                    qid,
                    len(fv),
                    expected_len,
                )
            logger.debug(
                "Numeric q=%s: using rw.latest.forecast_values len=%d first=%.5f last=%.5f",
                qid,
                len(fv),
                float(fv[0]),
                float(fv[-1]),
            )
            return [float(x) for x in fv]

        logger.info("Numeric q=%s: forecast_values missing in rw.latest (keys=%s)", qid, rw_keys)
    # Boundary: this extractor's contract is "never raise, return None" — the numeric scorer
    # degrades to the declared-percentile fallback when community data is unavailable.
    except Exception as e:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except
        logger.warning(f"Failed to extract numeric community CDF: {e}")
    return None
