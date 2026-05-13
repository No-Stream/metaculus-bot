"""Detect which numeric output format the LLM emitted (percentiles vs.
mixture-of-normals) and route to the matching CDF builder.

Single-decisive routing per user steer (atlas_inspired_improvements.md
Workstream E, 2026-05-12): if a valid mixture is present in the structured
block, use it; otherwise use percentiles. NO consistency check between the
two — if the LLM emits both, the mixture wins and a WARNING is logged so the
frequency is auditable for future calibration work.

The router always returns a ``RoutedNumericForecast`` carrying:

- ``format`` — which branch produced the CDF; recorded for residual analysis.
- ``cdf_percentiles`` — for the mixture path, a 201-point Metaculus-compliant
  CDF already constraint-enforced via
  ``percentiles_to_metaculus_cdf_via_mixture``. For the percentile path, the
  raw declared percentiles passed straight through (the existing
  ``sanitize_percentiles`` + ``build_numeric_distribution`` pipeline runs
  downstream in ``main.py``).
- ``declared_percentiles`` — the raw LLM percentile list when the percentile
  branch fired; ``None`` on the mixture branch.
- ``mixture`` — the constructed ``MixtureOfNormals`` when the mixture branch
  fired; ``None`` on the percentile branch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.probabilistic_tools.mixtures import (
    MixtureComponent,
    MixtureOfNormals,
    percentiles_to_metaculus_cdf_via_mixture,
)
from metaculus_bot.structured_output_schema import (
    NumericStructured,
    extract_json_block,
    parse_structured_block,
)

logger = logging.getLogger(__name__)


NumericFormat = Literal["percentiles", "mixture", "both"]


@dataclass(frozen=True)
class RoutedNumericForecast:
    """Result of routing an LLM's numeric output. ``format`` records which
    branch produced the result; the CDF is always present when the call
    succeeded."""

    format: NumericFormat
    cdf_percentiles: list[Percentile]
    declared_percentiles: list[Percentile] | None
    mixture: MixtureOfNormals | None


def detect_numeric_format(rationale: str) -> NumericFormat | None:
    """Inspect a rationale for which format the LLM emitted.

    Looks at the structured JSON block:

    - If both ``mixture_components`` (populated, non-null) and
      ``declared_percentiles`` (always required by the schema) are present:
      ``"both"``.
    - If only ``declared_percentiles`` is present (mixture absent or null):
      ``"percentiles"``.
    - If only ``mixture_components`` is somehow present without
      ``declared_percentiles``, ``"mixture"``. The current schema validator
      requires ``declared_percentiles``, so this branch is unreachable in
      practice but kept for future schema relaxation.
    - Returns ``None`` when no JSON block is present, the JSON is malformed,
      or the JSON does not parse as ``NumericStructured`` — caller falls
      back to the trailing ``Percentile X.X:`` lines.
    """
    raw = extract_json_block(rationale)
    if raw is None:
        return None

    structured = parse_structured_block(rationale, "numeric")
    if structured is None or not isinstance(structured, NumericStructured):
        return None

    has_mixture = structured.mixture_components is not None and len(structured.mixture_components) > 0
    has_percentiles = bool(structured.declared_percentiles)

    if has_mixture and has_percentiles:
        return "both"
    if has_mixture:
        return "mixture"
    if has_percentiles:
        return "percentiles"
    return None


def _build_mixture_from_structured(structured: NumericStructured) -> MixtureOfNormals | None:
    """Convert a NumericStructured.mixture_components list to a
    MixtureOfNormals. Pydantic has already validated weights ≥ 0, sd > 0,
    and weight-sum ≈ 1.0; MixtureOfNormals.__post_init__ normalizes anyway."""
    if structured.mixture_components is None or len(structured.mixture_components) < 2:
        return None
    components = tuple(
        MixtureComponent(weight=mc.weight, mean=mc.mean, sd=mc.sd) for mc in structured.mixture_components
    )
    return MixtureOfNormals(components=components)


def route_numeric_output(
    rationale: str,
    declared_percentiles: list[Percentile] | None,
    question: NumericQuestion,
) -> RoutedNumericForecast:
    """Single-decisive routing — mixture wins when present, else percentiles.

    Parameters
    ----------
    rationale:
        Full LLM text. Inspected for a structured JSON block via
        ``parse_structured_block``.
    declared_percentiles:
        The list[Percentile] already pulled from the trailing
        ``Percentile X.X: ...`` lines by ``main.py``. May be ``None`` if the
        LLM emitted only a mixture.
    question:
        ``NumericQuestion`` whose bounds drive the mixture-CDF grid.

    Returns
    -------
    RoutedNumericForecast — see module docstring.

    Raises
    ------
    ValueError
        If neither the mixture nor declared_percentiles can produce a CDF.
    """
    structured = parse_structured_block(rationale, "numeric")
    mixture: MixtureOfNormals | None = None
    structured_has_percentiles: bool = False
    structured_percentiles_fallback: list[Percentile] | None = None
    if structured is not None and isinstance(structured, NumericStructured):
        mixture = _build_mixture_from_structured(structured)
        structured_has_percentiles = bool(structured.declared_percentiles)
        if structured_has_percentiles:
            structured_percentiles_fallback = [
                Percentile(percentile=float(k), value=float(v))
                for k, v in sorted(structured.declared_percentiles.items())
            ]

    has_percentiles = declared_percentiles is not None and len(declared_percentiles) > 0
    # F5 fallback: if the percentile parser missed the trailing
    # "Percentile X.X" lines but the structured block carries
    # declared_percentiles, lift them as a backup so the percentile path
    # stays reachable.
    effective_percentiles: list[Percentile] | None
    if has_percentiles:
        effective_percentiles = list(declared_percentiles or [])
    elif structured_percentiles_fallback is not None:
        effective_percentiles = structured_percentiles_fallback
    else:
        effective_percentiles = None
    has_effective_percentiles = effective_percentiles is not None and len(effective_percentiles) > 0
    # "both" means the rationale's JSON contained both shapes — either the
    # LLM literally emitted both, or the schema mandates declared_percentiles
    # alongside an opted-in mixture. Either way, residual analysis cares about
    # how often the LLM put both shapes on the wire, not just how many made
    # it through main.py's percentile parser.
    rationale_has_both = (mixture is not None) and (structured_has_percentiles or has_percentiles)

    # Mixture path — when the LLM emitted a valid mixture, use it. If
    # percentiles also came along, log so frequency is auditable.
    if mixture is not None:
        if rationale_has_both:
            logger.warning(
                "numeric_format_router: LLM emitted both percentiles and mixture; "
                "using mixture (cdf_size=%d, mixture_components=%d)",
                201,
                len(mixture.components),
            )
        try:
            cdf = percentiles_to_metaculus_cdf_via_mixture(mixture, question)
        except Exception as exc:
            # Soft-fall to percentiles when the mixture builder blows up. We
            # log loudly because this is a real failure mode worth auditing.
            logger.warning(
                "numeric_format_router: mixture CDF build failed (%s); falling back to percentiles",
                exc,
            )
            if not has_effective_percentiles:
                raise ValueError(
                    "Mixture CDF build failed and no fallback declared_percentiles "
                    "available; cannot produce a numeric forecast."
                ) from exc
            return RoutedNumericForecast(
                format="percentiles",
                cdf_percentiles=list(effective_percentiles or []),
                declared_percentiles=list(effective_percentiles or []),
                mixture=None,
            )

        return RoutedNumericForecast(
            format="both" if rationale_has_both else "mixture",
            cdf_percentiles=cdf,
            declared_percentiles=None,
            mixture=mixture,
        )

    # Percentile path — pass declared_percentiles straight through so the
    # existing main.py pipeline can run sanitize_percentiles +
    # build_numeric_distribution.
    if has_effective_percentiles:
        return RoutedNumericForecast(
            format="percentiles",
            cdf_percentiles=list(effective_percentiles or []),
            declared_percentiles=list(effective_percentiles or []),
            mixture=None,
        )

    raise ValueError(
        "numeric_format_router: neither mixture_components nor declared_percentiles "
        "available; cannot produce a numeric forecast."
    )


__all__ = [
    "NumericFormat",
    "RoutedNumericForecast",
    "detect_numeric_format",
    "route_numeric_output",
]
