"""Shared factory for the provider-health observation store.

``record_venue_observation`` takes a frozen dataclass with five fields, so every suite that needs
one venue's per-question outcome in the store used to build it by hand: the cli exit-status suite,
the provider-health rules and the run-level degradation counters each had their own copy, and one
of them drifted to a different default shape.
"""

from __future__ import annotations

from metaculus_bot.research.provider_health import (
    VENUE_EXPECTED_LIQUIDITY_FIELDS,
    VenueObservation,
    record_venue_observation,
)


def observe_venue(
    venue: str,
    *,
    qid: int = 1,
    candidates: int = 3,
    rows: int = 3,
    fields: frozenset[str] | None = None,
) -> None:
    """Record one venue observation, defaulting to a healthy shape: every declared field present."""
    record_venue_observation(
        VenueObservation(
            qid=qid,
            venue=venue,
            candidates_pre_filter=candidates,
            rows_post_filter=rows,
            liquidity_fields_present=frozenset(VENUE_EXPECTED_LIQUIDITY_FIELDS[venue]) if fields is None else fields,
        )
    )
