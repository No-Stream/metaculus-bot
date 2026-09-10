"""The per-caller knobs one run of the ladder is bounded by, and the two callers' presets.

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
from dataclasses import dataclass, replace
from typing import Literal

from metaculus_bot.constants import (
    GAP_FILL_V2_MIN_CONTENT_CHARS,
    RESOLUTION_SOURCE_HTTP_TIMEOUT,
    RESOLUTION_SOURCE_PER_URL_MAX_CHARS,
    RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S,
    RESOLUTION_SOURCE_WALL_TIMEOUT,
    RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS,
)
from metaculus_bot.research.fetch_ladder.digest import DigestFn
from metaculus_bot.research.fetch_ladder.verdict import (
    GAP_FILL_VERDICT,
    RESOLUTION_SOURCE_VERDICT,
    LadderVerdict,
)
from metaculus_bot.research.rendered_fetch import MemoScope
from metaculus_bot.research.resolution_fetch_result import FetchResult, FetchRoute, FetchStatus

# Named rather than spelled at each site: both ride the `caller=` field of the two fetch markers.
LADDER_CALLER_RESOLUTION_SOURCE = "resolution_source"
LADDER_CALLER_GAP_FILL_V2 = "gap_fill_v2"
LadderCaller = Literal["resolution_source", "gap_fill_v2"]

# Rung 0, ahead of every fetch: a URL a public API answers exactly, or None to fall through.
KnownApiFn = Callable[[str], Awaitable[FetchResult | None]]

# The five rungs `rungs_enabled` gates; the two inside the direct fetch are not gateable.
_ESCALATION_RUNGS: frozenset[FetchRoute] = frozenset(
    {"impersonate", "derived_api", "rendered", "wayback", "url_context"}
)


@dataclass(frozen=True, slots=True)
class LadderPolicy:
    """What one caller's run of the ladder is allowed to spend, disclose and collect.

    Read-only by construction: a rung reads a knob and never writes one, and a caller that needs
    a variant builds it with :func:`dataclasses.replace`. ``total_wall_s`` less
    ``rung_wall_margin_s`` is the wall every rung bounds itself against
    (:meth:`LadderContext.rung_budget_s`), and it is the reason this class exists: the fetcher has
    45 s per question where the loop's ``fetch`` tool has 90 s and its document ladder 25 s.
    ``verdict`` is the one seat the two callers' READINGS differ in (:mod:`verdict`);
    ``known_api`` and ``digest`` are seats whose sibling implementations land separately.
    """

    caller: LadderCaller
    render_memo_scope: MemoScope
    verdict: LadderVerdict
    rungs_enabled: frozenset[FetchRoute]
    total_wall_s: float
    rung_wall_margin_s: float
    per_url_max_chars: int | None
    wayback_max_age_days: float | None
    wayback_extra_trigger_statuses: frozenset[FetchStatus]
    wayback_needs_host_refusal: bool
    impersonate_dial_wall_s: float | None
    disclose_unreadable_embeds: bool
    thin_content_escalation_chars: int | None
    collect_links: bool
    known_api: KnownApiFn | None = None
    digest: DigestFn | None = None


RESOLUTION_SOURCE_POLICY = LadderPolicy(
    caller=LADDER_CALLER_RESOLUTION_SOURCE,
    render_memo_scope="resolution_source",
    verdict=RESOLUTION_SOURCE_VERDICT,
    rungs_enabled=_ESCALATION_RUNGS,
    total_wall_s=RESOLUTION_SOURCE_WALL_TIMEOUT,
    rung_wall_margin_s=RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S,
    per_url_max_chars=RESOLUTION_SOURCE_PER_URL_MAX_CHARS,
    wayback_max_age_days=RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS,
    wayback_extra_trigger_statuses=frozenset(),
    wayback_needs_host_refusal=False,
    impersonate_dial_wall_s=None,
    disclose_unreadable_embeds=True,
    thin_content_escalation_chars=None,
    collect_links=False,
)

# The `fetch` tool's own ladder; every value's reason: docs/architecture.md, the knob table.
GAP_FILL_FETCH_POLICY = LadderPolicy(
    caller=LADDER_CALLER_GAP_FILL_V2,
    render_memo_scope="gap_fill_v2",
    verdict=GAP_FILL_VERDICT,
    rungs_enabled=frozenset({"impersonate", "derived_api", "rendered", "wayback"}),
    total_wall_s=90.0,
    # No margin: the loop's own outer bounds are the cut (see the doc).
    rung_wall_margin_s=0.0,
    per_url_max_chars=None,
    wayback_max_age_days=None,
    # The loop's archive rung also substitutes for a body it could not read at all.
    wayback_extra_trigger_statuses=frozenset({"unsupported_type"}),
    wayback_needs_host_refusal=True,
    impersonate_dial_wall_s=RESOLUTION_SOURCE_HTTP_TIMEOUT,
    disclose_unreadable_embeds=False,
    thin_content_escalation_chars=GAP_FILL_V2_MIN_CONTENT_CHARS,
    collect_links=True,
)

# `read_document`'s free acquisition ladder: 25 s, and no archive rung (see the doc).
GAP_FILL_DOCUMENT_POLICY = replace(
    GAP_FILL_FETCH_POLICY, total_wall_s=25.0, rungs_enabled=frozenset({"impersonate", "rendered"})
)

# One direct fetch and nothing else, for the robots.txt pre-check (see the doc).
GAP_FILL_DIRECT_POLICY = replace(GAP_FILL_FETCH_POLICY, rungs_enabled=frozenset())
