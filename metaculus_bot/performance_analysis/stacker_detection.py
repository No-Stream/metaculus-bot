"""Multi-signal stacker detection for historical performance records.

Combines explicit flags, comment-body markers, spread thresholds, and
production-vs-median deltas to classify whether an LLM stacker fired on a
given resolved-question record. Designed to be importable from notebooks,
scripts, and ablation analyses.

Signal hierarchy (strongest first):
1. Explicit ``was_stacked`` field (authoritative log).
2. ``stacker_outcome`` field present and non-null.
3. Bot comment body marker (STACKER_OUTCOME, STACKED, historical body signature).
4. Spread > production threshold AND production differs materially from median.
5. Spread <= production threshold → likely_median.
6. None of the above → unknown.
"""

from __future__ import annotations

import logging
import statistics
from typing import Literal

from metaculus_bot.comment_markers import (
    HISTORICAL_STACKER_SIGNATURE_RE,
    STACKED_MARKER_RE,
    STACKER_OUTCOME_RE,
)
from metaculus_bot.constants import (
    CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD,
    CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD,
    CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD,
)
from metaculus_bot.performance_analysis.parsing import _parse_probability

logger: logging.Logger = logging.getLogger(__name__)

DetectorVerdict = Literal[
    "confirmed_stacker",
    "confirmed_median",
    "likely_stacker",
    "likely_median",
    "unknown",
]

# Stacker outcomes that indicate the stacker LLM actually produced a value
_STACKER_FIRED_OUTCOMES: frozenset[str] = frozenset({"primary", "fallback_llm"})

# Stacker outcomes that indicate the stacker did NOT produce a value
_STACKER_MEDIAN_OUTCOMES: frozenset[str] = frozenset({"skipped", "fallback_median", "fallback_mean"})


# ---------------------------------------------------------------------------
# Signal extractors
# ---------------------------------------------------------------------------


def has_was_stacked_flag(record: dict) -> bool | None:
    """Check if the record has an explicit ``was_stacked`` boolean field.

    Returns True/False if the field is a bool, None if missing or null.
    """
    value = record.get("was_stacked")
    if isinstance(value, bool):
        return value
    return None


def get_stacker_outcome_field(record: dict) -> str | None:
    """Return the ``stacker_outcome`` field value if present and non-empty.

    Returns None if the field is missing, null, or empty string.
    """
    value = record.get("stacker_outcome")
    if value is None or value == "":
        return None
    return str(value)


def has_stacker_body_marker(record: dict) -> bool | None:
    """Check comment body for stacker markers.

    Returns:
        True if a marker indicates the stacker fired (primary, fallback_llm, STACKED=true, historical signature).
        False if a marker indicates the stacker did NOT fire (skipped, fallback_median, STACKED=false).
        None if no relevant marker is found or comment_text is absent.
    """
    comment_text = record.get("comment_text")
    if not comment_text:
        return None

    # Check STACKER_OUTCOME marker first (most specific)
    outcome_match = STACKER_OUTCOME_RE.search(comment_text)
    if outcome_match is not None:
        outcome = outcome_match.group(1).lower()
        if outcome in _STACKER_FIRED_OUTCOMES:
            return True
        if outcome in _STACKER_MEDIAN_OUTCOMES:
            return False

    # Check legacy STACKED marker
    stacked_match = STACKED_MARKER_RE.search(comment_text)
    if stacked_match is not None:
        return stacked_match.group(1).lower() == "true"

    # Check historical body signature (pre-marker era)
    if HISTORICAL_STACKER_SIGNATURE_RE.search(comment_text) is not None:
        return True

    return None


def compute_production_vs_median_delta(record: dict) -> float | None:
    """Compute |production_probability - median_of_per_model_probabilities|.

    Only works for binary records currently. Returns None if insufficient data.

    When ``per_base_model_forecasts`` is present (populated for stacked records
    by the collector), prefer it for the median computation since
    ``per_model_forecasts`` on stacked records collapses to the stacker's single
    aggregated value.
    """
    question_type = record.get("type", "")

    if question_type == "binary":
        prod_prob = record.get("our_prob_yes")
        if prod_prob is None:
            return None

        per_base_model = record.get("per_base_model_forecasts") or {}
        per_model = per_base_model if per_base_model else (record.get("per_model_forecasts") or {})
        probs = [_parse_probability(v) for v in per_model.values()]
        probs = [p for p in probs if p is not None]
        if len(probs) < 2:
            return None

        median_prob = statistics.median(probs)
        return abs(float(prod_prob) - median_prob)

    # Numeric/MC delta computation is more complex (CDF comparison);
    # return None for now — the spread threshold alone is the primary signal
    # for those types.
    return None


def exceeded_spread_threshold(record: dict) -> bool | None:
    """Check whether per-model spread exceeds the production trigger threshold.

    Uses the same logic as ``compute_binary_spread_from_record`` and
    ``compute_numeric_spread_from_record`` from the historical residual script,
    but rewritten to operate on the raw record dict directly.

    Returns True if spread > threshold, False if spread <= threshold,
    None if spread cannot be computed.
    """
    question_type = record.get("type", "")

    if question_type == "binary":
        per_base_model = record.get("per_base_model_forecasts") or {}
        per_model = per_base_model if per_base_model else (record.get("per_model_forecasts") or {})
        probs = [_parse_probability(v) for v in per_model.values()]
        probs = [p for p in probs if p is not None]
        if len(probs) < 2:
            return None
        spread = max(probs) - min(probs)
        return spread > CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD

    if question_type == "multiple_choice":
        # TODO: MC likely has the same collapse-on-stacking issue as binary — once
        # per_base_model_forecasts is populated for MC records, prefer it here too.
        per_model_mc = record.get("per_model_mc_forecasts") or {}
        if len(per_model_mc) < 2:
            return None

        # Collect per-option probabilities across models
        option_probs: dict[str, list[float]] = {}
        for model_options in per_model_mc.values():
            for opt in model_options:
                name = opt.get("option_name", "")
                prob = opt.get("probability")
                if prob is not None:
                    option_probs.setdefault(name, []).append(float(prob))

        if not option_probs:
            return None

        max_spread = max((max(probs) - min(probs)) for probs in option_probs.values() if len(probs) >= 2)
        return max_spread > CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD

    if question_type in ("numeric", "discrete"):
        # Reuse the normalized percentile spread approach from the residual script
        pnp = record.get("per_model_numeric_percentiles") or {}
        if len(pnp) < 2:
            return None

        key_indices = [2, 5, 8]  # 10th, 50th, 90th in standard 11-percentile list
        all_pcts: list[list[list[float]]] = []
        for model_pcts in pnp.values():
            if len(model_pcts) >= 9:
                all_pcts.append(model_pcts)
        if len(all_pcts) < 2:
            return None

        import math

        scaling = record.get("scaling") or {}
        open_lower = record.get("open_lower_bound", scaling.get("open_lower_bound", False))
        open_upper = record.get("open_upper_bound", scaling.get("open_upper_bound", False))
        range_min = scaling.get("range_min")
        range_max = scaling.get("range_max")

        has_finite_range = (
            not open_lower
            and not open_upper
            and range_min is not None
            and range_max is not None
            and math.isfinite(range_min)
            and math.isfinite(range_max)
        )

        if has_finite_range and range_max is not None and range_min is not None:
            denominator = range_max - range_min
        else:
            p10_values = [pcts[key_indices[0]][1] for pcts in all_pcts]
            p90_values = [pcts[key_indices[2]][1] for pcts in all_pcts]
            denominator = statistics.median(p90_values) - statistics.median(p10_values)

        if denominator <= 0:
            return None

        max_normalized_spread = 0.0
        for idx in key_indices:
            values_at_pct = [pcts[idx][1] for pcts in all_pcts]
            raw_spread = max(values_at_pct) - min(values_at_pct)
            normalized = raw_spread / denominator
            max_normalized_spread = max(max_normalized_spread, normalized)

        return max_normalized_spread > CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD

    return None


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------


def detect_stacker_fired(record: dict, *, default_threshold: float = 0.05) -> DetectorVerdict:
    """Detect whether an LLM stacker fired on this resolved-question record.

    Combines multiple signals in priority order to return a confidence-weighted
    verdict. See module docstring for the full signal hierarchy.

    Parameters
    ----------
    record : dict
        A single performance record (as loaded from cached JSON datasets).
    default_threshold : float
        Material-difference threshold for signal #4: |production - median| must
        exceed this to indicate the stacker produced a different value. Default 0.05.

    Returns
    -------
    DetectorVerdict
        One of: confirmed_stacker, confirmed_median, likely_stacker, likely_median, unknown.
    """
    # Signal 1: explicit was_stacked flag (authoritative)
    was_stacked = has_was_stacked_flag(record)
    if was_stacked is True:
        return "confirmed_stacker"

    # Signal 2: stacker_outcome field
    stacker_outcome = get_stacker_outcome_field(record)
    if stacker_outcome is not None:
        if stacker_outcome in _STACKER_FIRED_OUTCOMES:
            return "confirmed_stacker"
        if stacker_outcome in _STACKER_MEDIAN_OUTCOMES:
            return "confirmed_median"

    # Signal 3: body markers
    body_marker = has_stacker_body_marker(record)
    if body_marker is True:
        return "confirmed_stacker"
    if body_marker is False:
        # Body says no stacker — but check if was_stacked was explicitly False
        # (already handled above if True). If was_stacked is False, confirmed_median.
        if was_stacked is False:
            return "confirmed_median"
        # Body marker says no stacker, treat as confirmed_median
        return "confirmed_median"

    # If was_stacked is explicitly False but no body marker found, still factor it in
    # Combined with spread signal below for the verdict.

    # Signals 4 & 5: spread threshold + production-vs-median delta
    spread_exceeded = exceeded_spread_threshold(record)
    delta = compute_production_vs_median_delta(record)

    if was_stacked is False:
        # Explicit flag says no stacking. But check if evidence contradicts.
        if spread_exceeded is True and delta is not None and delta > default_threshold:
            # Flag says no, but evidence suggests stacker actually fired
            return "likely_stacker"
        return "confirmed_median"

    # No explicit flags or body markers from here on — rely on spread + delta
    if spread_exceeded is False:
        return "likely_median"

    if spread_exceeded is True:
        if delta is not None and delta > default_threshold:
            return "likely_stacker"
        # High spread but production matches median — stacker may have failed/fallen back
        return "likely_median"

    # spread_exceeded is None (cannot compute) and no other signals
    return "unknown"


__all__ = [
    "DetectorVerdict",
    "compute_production_vs_median_delta",
    "detect_stacker_fired",
    "exceeded_spread_threshold",
    "get_stacker_outcome_field",
    "has_stacker_body_marker",
    "has_was_stacked_flag",
]
