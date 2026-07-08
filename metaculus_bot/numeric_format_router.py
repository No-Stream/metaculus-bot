"""Route numeric LLM output through the percentile pipeline.

After Workstream C1 (2026-07-07) the mixture branch was deleted - zero prod
fires in the 90-day window, and benchmarks showed mixtures don't beat
percentiles+PCHIP. The router's remaining job is:

1. If the parser extracted percentiles (``declared_percentiles`` arg), use them.
2. F5 fallback: if the parser missed the trailing lines but the JSON block
   carries ``declared_percentiles``, lift them.
3. Otherwise raise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.structured_output_schema import (
    NumericStructured,
    extract_json_block,
    parse_structured_block,
)

logger = logging.getLogger(__name__)


NumericFormat = Literal["percentiles"]


@dataclass(frozen=True)
class RoutedNumericForecast:
    """Result of routing an LLM numeric output. After C1, format is always
    percentiles."""

    format: NumericFormat
    cdf_percentiles: list[Percentile]
    declared_percentiles: list[Percentile] | None


def detect_numeric_format(rationale: str) -> NumericFormat | None:
    """Inspect a rationale for whether it carries usable percentiles.

    Returns ``"percentiles"`` when the structured JSON block has valid
    ``declared_percentiles``, or ``None`` when no usable block is present.
    """
    raw = extract_json_block(rationale)
    if raw is None:
        return None

    structured = parse_structured_block(rationale, "numeric")
    if structured is None or not isinstance(structured, NumericStructured):
        return None

    if bool(structured.declared_percentiles):
        return "percentiles"
    return None


def route_numeric_output(
    rationale: str,
    declared_percentiles: list[Percentile] | None,
    question: NumericQuestion,
) -> RoutedNumericForecast:
    """Route numeric output through the percentile pipeline.

    Parameters
    ----------
    rationale:
        Full LLM text. Inspected for a structured JSON block.
    declared_percentiles:
        The list[Percentile] already extracted by the parser. May be None.
    question:
        NumericQuestion (unused after C1, retained for interface stability).

    Raises
    ------
    ValueError
        If no percentiles are available from either the parser or the block.
    """
    structured = parse_structured_block(rationale, "numeric")
    structured_percentiles_fallback: list[Percentile] | None = None
    if structured is not None and isinstance(structured, NumericStructured):
        declared = structured.declared_percentiles
        if declared:
            structured_percentiles_fallback = [
                Percentile(percentile=float(k), value=float(v)) for k, v in sorted(declared.items())
            ]

    has_percentiles = declared_percentiles is not None and len(declared_percentiles) > 0
    # F5 fallback: if the percentile parser missed the trailing
    # "Percentile X.X" lines but the structured block carries
    # declared_percentiles, lift them as a backup.
    effective_percentiles: list[Percentile] | None
    if has_percentiles:
        effective_percentiles = list(declared_percentiles or [])
    elif structured_percentiles_fallback is not None:
        effective_percentiles = structured_percentiles_fallback
    else:
        effective_percentiles = None
    has_effective_percentiles = effective_percentiles is not None and len(effective_percentiles) > 0

    if has_effective_percentiles:
        return RoutedNumericForecast(
            format="percentiles",
            cdf_percentiles=list(effective_percentiles or []),
            declared_percentiles=list(effective_percentiles or []),
        )

    raise ValueError(
        "numeric_format_router: no declared_percentiles available from parser or "
        "structured block; cannot produce a numeric forecast."
    )


__all__ = [
    "NumericFormat",
    "RoutedNumericForecast",
    "detect_numeric_format",
    "route_numeric_output",
]
