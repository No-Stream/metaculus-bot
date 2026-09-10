"""The per-caller knobs one run of the ladder is bounded by, and the resolution-source preset.

A knob lives here only where the two callers (the resolution-source fetcher and the gap-fill v2
loop) genuinely differ. Everything they share stays a plain module constant where it is read —
the two byte caps, the per-hop HTTP timeout, every per-rung wall floor, the chrome and
JavaScript-wall thresholds, the robots pre-check, the platform self-reference refusal, the
per-question Wayback and paid-read caps — because a constant read off a frozen dataclass is a
constant a test can no longer patch on the module that reads it, and that failure is silent.

The policy rides :class:`~metaculus_bot.research.fetch_ladder.context.LadderContext` rather than
being threaded through the twenty rung functions that already take one, so a rung reads
``ctx.policy.<knob>`` and no rung signature changes. :func:`ladder.fetch_url` is the one place
that pairs a policy with a context.

Every knob's per-caller value and the reason it is a knob: ``docs/architecture.md``, "The shared
fetch ladder".
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from metaculus_bot.constants import (
    RESOLUTION_SOURCE_PER_URL_MAX_CHARS,
    RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S,
    RESOLUTION_SOURCE_WALL_TIMEOUT,
    RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS,
)
from metaculus_bot.research.fetch_ladder.digest import DigestFn
from metaculus_bot.research.rendered_fetch import MemoScope
from metaculus_bot.research.resolution_fetch_result import FetchResult

LadderCaller = Literal["resolution_source", "gap_fill_v2"]

# Rung 0, ahead of every fetch: a URL a public API answers exactly, or None to fall through.
KnownApiFn = Callable[[str], Awaitable[FetchResult | None]]


@dataclass(frozen=True, slots=True)
class LadderPolicy:
    """What one caller's run of the ladder is allowed to spend, disclose and collect.

    Read-only by construction: a rung reads a knob and never writes one, and a caller that needs
    a variant builds it with :func:`dataclasses.replace`. ``total_wall_s`` less
    ``rung_wall_margin_s`` is the wall every rung bounds itself against
    (:meth:`LadderContext.rung_budget_s`), and it is the reason this class exists: the fetcher has
    45 s per question where the loop's ``fetch`` tool has 90 s and its document ladder 25 s.
    ``known_api`` and ``digest`` are seats whose sibling implementations land separately.
    """

    caller: LadderCaller
    render_memo_scope: MemoScope
    total_wall_s: float
    rung_wall_margin_s: float
    per_url_max_chars: int | None
    wayback_max_age_days: float | None
    disclose_unreadable_embeds: bool
    thin_content_escalation_chars: int | None
    collect_links: bool
    known_api: KnownApiFn | None = None
    digest: DigestFn | None = None


RESOLUTION_SOURCE_POLICY = LadderPolicy(
    caller="resolution_source",
    render_memo_scope="resolution_source",
    total_wall_s=RESOLUTION_SOURCE_WALL_TIMEOUT,
    rung_wall_margin_s=RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S,
    per_url_max_chars=RESOLUTION_SOURCE_PER_URL_MAX_CHARS,
    wayback_max_age_days=RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS,
    disclose_unreadable_embeds=True,
    thin_content_escalation_chars=None,
    collect_links=False,
)
