"""Reading Metaculus's own platform scores off a performance record.

**The tournament ranks on SPOT PEER.** Verified against the live API 2026-08-31: the
project carries ``score_type=spot_peer_tournament``, every question's
``default_score_type`` is ``spot_peer``, and ``spot_scoring_time`` equals
``actual_close_time`` on all 158 posts pulled.

``peer_score`` on the same record is the COVERAGE-SCALED quantity. Measured on the
2026-08-31 residual round's 30 new records, ``spot_peer_score x coverage`` reproduces the
platform's own ``peer_score`` to a median residual of 0.69 points (max 13.05), and that
residual is crowd movement in the 1.5-3h window between our submit and the close rather
than anything the bot decided. Those figures are that round's measurement, not a standing
repo constant — re-derive with
``scratch/residual_2026-08-31/dossiers/44798_peer_vs_spot.py``.

The bot submits exactly once per question and never revises (forecast history length 1 on
157/158 posts), so its coverage is just whatever fraction of the open window that single
submission happened to span. Coverage scaling therefore FLATTERS misses and dulls hits
(q44872: peer -15.0 against spot peer -38.8), which makes peer unusable as a ranking
key — two records ordered by it are ordered partly by how early each was submitted.

So: rank and aggregate on ``spot_peer_score``; carry ``peer_score`` alongside as a
labelled secondary. And never sort a mixed set on whichever field happens to be present:
:func:`ranking_score` returns the field it read so callers can tier spot-scored records
separately from peer-only ones instead of interleaving two different quantities.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

logger: logging.Logger = logging.getLogger(__name__)

SPOT_PEER_FIELD = "spot_peer_score"
PEER_FIELD = "peer_score"
SPOT_BASELINE_FIELD = "spot_baseline_score"
BASELINE_FIELD = "baseline_score"
COVERAGE_FIELD = "coverage"

# Ranking preference, leaderboard-true field first. Coverage-scaled peer is only ever a
# last resort, for records pulled before spot peer was captured.
RANKING_FIELDS: tuple[str, ...] = (SPOT_PEER_FIELD, PEER_FIELD)

# Sort tiers past the platform scores, so a caller's type-specific fallback (Brier, log
# score) never interleaves with a platform score and "no score at all" sorts last.
FALLBACK_TIER: int = len(RANKING_FIELDS)
NO_SCORE_TIER: int = FALLBACK_TIER + 1


def platform_scores(record: dict) -> dict:
    """The record's ``metaculus_scores`` block, or an empty dict when it carries none."""
    return record.get("metaculus_scores") or {}


def score_field(record: dict, field: str) -> float | None:
    """One named platform score as a float, or None when absent/null."""
    value = platform_scores(record).get(field)
    return None if value is None else float(value)


def spot_peer_score(record: dict) -> float | None:
    """The leaderboard-true peer score — our log score against the crowd's, unscaled."""
    return score_field(record, SPOT_PEER_FIELD)


def peer_score(record: dict) -> float | None:
    """The COVERAGE-SCALED peer score. Secondary/diagnostic only, never a ranking key."""
    return score_field(record, PEER_FIELD)


def spot_baseline_score(record: dict) -> float | None:
    """The leaderboard-analogous baseline score (unscaled), the spot sibling of baseline."""
    return score_field(record, SPOT_BASELINE_FIELD)


def baseline_score(record: dict) -> float | None:
    """The COVERAGE-SCALED baseline score. Secondary, for the same reason peer is."""
    return score_field(record, BASELINE_FIELD)


def coverage(record: dict) -> float | None:
    """The fraction of the question's open window our forecast spanned, per Metaculus."""
    return score_field(record, COVERAGE_FIELD)


@dataclass(frozen=True, slots=True)
class RankingScore:
    """A record's ranking score plus WHICH field produced it.

    ``field`` is load-bearing rather than informational: a spot-peer value and a
    coverage-scaled peer value are different quantities, so a caller must sort them in
    separate tiers instead of interleaving them into one order.
    """

    value: float
    field: str

    @property
    def tier(self) -> int:
        """Sort tier: 0 for spot peer, 1 for coverage-scaled peer."""
        return RANKING_FIELDS.index(self.field)


def ranking_score(record: dict) -> RankingScore | None:
    """Spot peer when the record carries it, else coverage-scaled peer, else None."""
    for field in RANKING_FIELDS:
        value = score_field(record, field)
        if value is not None:
            return RankingScore(value=value, field=field)
    return None


def log_ranking_score_sources(records: Iterable[dict], *, cut: str) -> dict[str, int]:
    """Count which field each record would rank on, and say so — loudly for peer-only.

    A cohort that fell back to coverage-scaled peer is still ranked, but its order is
    contaminated by submission timing, so it gets a WARN rather than being silently
    reported as a peer ranking.
    """
    counts: dict[str, int] = dict.fromkeys(RANKING_FIELDS, 0)
    counts["none"] = 0
    for record in records:
        ranked = ranking_score(record)
        counts["none" if ranked is None else ranked.field] += 1

    detail = " ".join(f"{name}={count}" for name, count in counts.items())
    logger.info(f"PLATFORM_RANKING_SOURCE: cut={cut} {detail}")
    if counts[PEER_FIELD]:
        logger.warning(
            f"PLATFORM_RANKING_SOURCE: cut={cut} {counts[PEER_FIELD]} record(s) carry no "
            f"{SPOT_PEER_FIELD}, so ranking or banding them can only use coverage-scaled "
            f"{PEER_FIELD}, whose order is partly submission timing rather than skill. "
            "Each cut states what it did with them."
        )
    return counts


def platform_score_fragments(record: dict) -> list[str]:
    """Markdown fragments for a record's platform scores, spot peer first.

    Peer is rendered with its scaling named so a reader can never mistake it for the
    leaderboard number.
    """
    fragments: list[str] = []
    spot = spot_peer_score(record)
    if spot is not None:
        fragments.append(f"spot peer **{spot:+.1f}**")
    peer = peer_score(record)
    if peer is not None:
        fragments.append(f"peer {peer:+.1f} (coverage-scaled, secondary)")
    return fragments
