"""Readers for the trailer markers a published bot comment carries.

One responsibility: turning the ``STACKED=``, ``STACKER_OUTCOME=``,
``STACKER_SKIP_REASON=`` and ``FORECASTERS_USED=`` markers (plus the pre-marker
stacker body signature that stands in for them on 2026-04-era comments) into
typed values, so a residual cut can tell a stacked publish from a skipped one.
The marker REGEXES live in ``metaculus_bot.comment.markers`` beside the writers
that emit them; only the reading is here. Split out of ``parsing`` so the
marker vocabulary can be read without the per-model attribution machinery.
"""

from metaculus_bot.comment.markers import (
    FORECASTERS_USED_MARKER_RE,
    HISTORICAL_STACKER_SIGNATURE_RE,
    STACKED_MARKER_RE,
    STACKER_OUTCOME_RE,
    STACKER_SKIP_REASON_RE,
)


def parse_stacker_outcome_marker(comment_text: str) -> str | None:
    """Return the STACKER_OUTCOME literal in ``comment_text``, else None.

    Returns one of ``"primary"``, ``"fallback_llm"``, ``"fallback_median"``,
    ``"fallback_mean"``, ``"skipped"``, ``"skipped_config_off"`` (always
    lower-cased), or ``None`` if no marker is present. Older comments
    predating the tri-state marker return ``None``; comments published before
    ``skipped_config_off`` shipped (2026-07-19) collapse both skip reasons
    into ``"skipped"``.
    """
    match = STACKER_OUTCOME_RE.search(comment_text)
    if match is None:
        return None
    return match.group(1).lower()


def parse_stacker_skip_reason_marker(comment_text: str) -> str | None:
    """Return the STACKER_SKIP_REASON literal in ``comment_text``, else None.

    One of the ``comment.markers.STACKER_SKIP_REASONS`` literals (always lower-cased). The
    marker is additive alongside STACKER_OUTCOME: a plain ``skipped`` outcome alone cannot
    distinguish a below-threshold skip from the single-forecaster short-circuit (q44870), and
    comments predating the marker return None. ``spread_undefined`` (added 2026-08-25) is the
    one a calibration cut must NOT pool with ``spread_below_threshold``: it means no spread
    was measurable, not that the models agreed.
    """
    match = STACKER_SKIP_REASON_RE.search(comment_text)
    if match is None:
        return None
    return match.group(1).lower()


def detect_historical_stacker_signature(comment_text: str) -> bool:
    """Return True if the comment carries the pre-marker stacker body signature.

    The stacking commit at 2026-04-02 (`c6d1ab3`) collapsed base predictions
    into a single Forecaster 1 whose reasoning began with `## Meta-Analysis`
    (later renamed to `## Stacker Meta-Analysis` on 2026-04-27, `95c4fff`).
    Comments published in that ~25-day window AND any earlier code variants
    that emitted the same shape carry no explicit `STACKED=` or
    `STACKER_OUTCOME=` marker, but the body alone is recognizable.

    Match conditions: the FIRST `## R1: Forecaster 1 Reasoning` block must
    open with `## (Stacker )?Meta-Analysis` (modulo a possible `Model:` line
    and whitespace). A bare meta-analysis header inside an ordinary forecaster
    body is NOT signal — that's a model's own reasoning structure.

    Returns False on comments that don't match the pattern (including all
    non-stacked comments, all post-marker comments, and the very oldest
    pre-stacking-commit comments).
    """
    return HISTORICAL_STACKER_SIGNATURE_RE.search(comment_text) is not None


def parse_inferred_stacker_outcome(comment_text: str) -> tuple[str | None, str]:
    """Return ``(outcome, source)`` combining marker and historical signature.

    Source is a string explaining how the outcome was determined:

    * ``"marker_outcome"`` — explicit ``STACKER_OUTCOME=...`` marker present.
    * ``"marker_legacy"`` — explicit ``STACKED=true|false`` marker only;
      outcome inferred to ``"primary"`` (true) or ``"skipped"`` (false). The
      legacy marker can't distinguish primary from fallback_llm or skipped
      from fallback_median, so this is a lossy mapping kept for back-compat.
    * ``"historical_body"`` — no marker, but the comment body carries the
      pre-marker stacker signature (`## R1: Forecaster 1 Reasoning` opening
      with `## (Stacker )?Meta-Analysis`). Outcome inferred to ``"primary"``
      since the body shape was only produced when the stacker LLM ran
      successfully — failed-stacker / median-fallback paths in old code did
      NOT collapse to a single Forecaster-1-with-Meta-Analysis shape.
    * ``"none"`` — neither marker nor historical signature present. Returns
      outcome=None, leaving downstream interpretation to the caller (it could
      be a non-stacking strategy, a skipped trigger, or an old comment from
      pre-stacking days).

    Use this when analyzing a dataset that spans multiple code versions —
    e.g., the spring-aib-2026 closing dataset where all forecasts predate
    the explicit markers and the only signal is body shape.
    """
    marker_outcome = parse_stacker_outcome_marker(comment_text)
    if marker_outcome is not None:
        return marker_outcome, "marker_outcome"
    legacy = parse_stacked_marker(comment_text)
    if legacy is True:
        return "primary", "marker_legacy"
    if legacy is False:
        return "skipped", "marker_legacy"
    if detect_historical_stacker_signature(comment_text):
        return "primary", "historical_body"
    return None, "none"


def parse_stacked_marker(comment_text: str) -> bool | None:
    """Return True/False if a STACKED=true/false marker is present, else None.

    Older comments without the marker return None. Collectors can use the
    tri-state return to distinguish "known stacked", "known not stacked",
    and "unknown".
    """
    match = STACKED_MARKER_RE.search(comment_text)
    if match is None:
        return None
    return match.group(1).lower() == "true"


def parse_forecasters_used_marker(comment_text: str) -> tuple[int, int] | None:
    """Return ``(n_used, n_configured)`` from a FORECASTERS_USED marker, else None.

    ``n_used`` is how many forecasters contributed to the published aggregate (==
    the number of per-model summary bullets); ``n_configured`` is the roster size
    that run. When ``n_used < n_configured`` the question published on a degraded
    ensemble (a model dropped), which is what disambiguates a missing bullet from
    a genuine roster change. Older comments predating the marker return None
    (unknown ensemble size), so callers can distinguish "known degraded",
    "known full", and "unknown".
    """
    match = FORECASTERS_USED_MARKER_RE.search(comment_text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))
