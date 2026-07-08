"""Route numeric LLM output through the percentile pipeline.

After Workstream C1 (2026-07-07) the mixture branch was deleted — zero prod
fires in the 90-day window, and benchmarks showed mixtures don't beat
percentiles+PCHIP. Post-F8 the router is a thin fallback shim:

1. If the parser extracted percentiles (``declared_percentiles`` arg), use them.
2. F5 fallback: if the parser missed the trailing lines but the JSON block
   carries ``declared_percentiles``, lift them.
3. Otherwise raise ``ValueError``.
"""

from __future__ import annotations

import logging

from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.structured_output_schema import (
    NumericStructured,
    parse_structured_block,
)

logger = logging.getLogger(__name__)


def route_numeric_output(
    rationale: str,
    declared_percentiles: list[Percentile] | None,
) -> list[Percentile]:
    """Return the effective ``list[Percentile]`` for a numeric forecast.

    Parameters
    ----------
    rationale:
        Full LLM text. Inspected for a structured JSON block on the F5 fallback.
    declared_percentiles:
        The list[Percentile] already extracted by the parser. May be None.

    Returns
    -------
    list[Percentile]
        Either the parser's percentiles or, on the F5 fallback, the block's
        ``declared_percentiles`` re-lifted as ``Percentile`` objects.

    Raises
    ------
    ValueError
        If no percentiles are available from either source.
    """
    if declared_percentiles:
        return list(declared_percentiles)

    structured = parse_structured_block(rationale, "numeric")
    if isinstance(structured, NumericStructured) and structured.declared_percentiles:
        # F5 fallback: parser missed the trailing lines but the block carries
        # declared_percentiles — lift them so we still get a forecast.
        return [
            Percentile(percentile=float(k), value=float(v)) for k, v in sorted(structured.declared_percentiles.items())
        ]

    raise ValueError(
        "numeric_format_router: no declared_percentiles available from parser or "
        "structured block; cannot produce a numeric forecast."
    )


__all__ = ["route_numeric_output"]
