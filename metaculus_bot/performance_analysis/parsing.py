"""Comment text parsing and resolution parsing utilities."""

import logging
import re

logger: logging.Logger = logging.getLogger(__name__)

# Current model roster (from llm_configs.py). Index matches the Forecaster N label.
MODEL_NAMES: list[str] = [
    "gpt-5.2",
    "gpt-5.1",
    "claude-4.6-opus",
    "claude-opus-4.5",
    "gemini-3.1-pro-preview",
]

# Index-to-name mapping (Forecaster labels are 1-indexed)
_MODEL_MAP: dict[int, str] = {i + 1: name for i, name in enumerate(MODEL_NAMES)}

# Matches lines like: *Forecaster 3*: 72.0%
_FORECASTER_RE: re.Pattern[str] = re.compile(r"\*Forecaster\s+(\d+)\*\s*:\s*(.+)")

SKIP_RESOLUTIONS: frozenset[str] = frozenset({"annulled", "ambiguous"})


def parse_per_model_forecasts(
    comment_text: str,
    model_names: list[str] | None = None,
) -> dict[str, str]:
    """Extract per-model predictions from the forecaster section of a comment.

    Returns dict mapping model name to raw value string (e.g. "72.0%").
    """
    model_map = _MODEL_MAP
    if model_names is not None:
        model_map = {i + 1: name for i, name in enumerate(model_names)}

    result: dict[str, str] = {}
    for match in _FORECASTER_RE.finditer(comment_text):
        idx = int(match.group(1))
        value = match.group(2).strip()
        model_name = model_map.get(idx)
        if model_name:
            result[model_name] = value
    return result


def parse_resolution(
    resolution_raw: str,
    question_type: str,
    options: list[str] | None = None,
) -> tuple[bool | float | str | None, bool]:
    """Parse raw resolution string into a typed value.

    Returns (parsed_value, should_skip). should_skip=True means this question
    should be excluded from analysis.
    """
    if resolution_raw in SKIP_RESOLUTIONS:
        return None, True

    if question_type == "binary":
        if resolution_raw == "yes":
            return True, False
        if resolution_raw == "no":
            return False, False
        logger.warning(f"Unexpected binary resolution: {resolution_raw!r}")
        return None, True

    if question_type in ("numeric", "discrete"):
        if resolution_raw == "above_upper_bound":
            return "above_upper_bound", False
        if resolution_raw == "below_lower_bound":
            return "below_lower_bound", False
        try:
            return float(resolution_raw), False
        except (ValueError, TypeError):
            logger.warning(f"Unparseable numeric resolution: {resolution_raw!r}")
            return None, True

    if question_type == "multiple_choice":
        return resolution_raw, False

    logger.warning(f"Unknown question type: {question_type!r}")
    return None, True
