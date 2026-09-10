"""The dispatcher: one URL's direct fetch, then the rungs its outcome earns, in one order.

``fetch_url`` is the entry point both callers use, ``_fetch_one`` is the whole ladder for one
URL, ``_escalate_unresolved`` is the rung order and the browser gate, and ``_run_rung`` is the
bracket that closes each rung's attempts on that rung's own wall and outcome rather than the
ladder's. Nothing here dials anything itself: :mod:`direct_fetch` and :mod:`rungs` own every
request. Why the rungs sit in this order: ``docs/architecture.md``, "The shared fetch ladder".
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import replace
from typing import Any

from metaculus_bot.research.fetch_ladder import context, direct_fetch, guard, rungs
from metaculus_bot.research.fetch_ladder.policy import LadderPolicy
from metaculus_bot.research.http_fetch import host_semaphores
from metaculus_bot.research.resolution_fetch_result import FetchResult, FetchStatus


async def _run_rung(
    ctx: context.LadderContext, fallback: FetchStatus, rung: Awaitable[FetchResult | None]
) -> FetchResult | None:
    """Await one rung and close the attempts it opened with that rung's own wall and outcome.

    The one home for the bracket every dispatcher site used to copy by hand: read
    ``len(ctx.rungs)`` before the rung runs, await it, then close every attempt opened since
    with the status that stood once it was over — its result's, or ``fallback`` (the status it
    left standing) when it declined (:meth:`LadderContext.close_rungs`). Structural rather than
    stylistic: a rung awaited without the bracket still returned its result, and its attempt fell
    through to :func:`_stamped_with_route`'s last-resort close, which stamps the ladder's FINAL
    status and the whole-ladder wall — the two figures the per-rung close exists to keep apart,
    with the marker parsing either way. ``rung`` is the coroutine created at the call site, which
    runs none of its code until it is awaited here, so the length is read first.
    """
    first_new = len(ctx.rungs)
    result = await rung
    ctx.close_rungs(first_new, fallback if result is None else result.status)
    return result


async def _escalate_unresolved(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult:
    """Run the escalation rungs a direct fetch's outcome earns, cheapest first.

    Returns the FIRST rung's rescue, or ``direct`` unchanged when every rung declines or fails; a
    rung that fired and produced nothing still leaves its attempt on the context, and each rung is
    closed the moment its result is known (:func:`_run_rung`) so its attempt carries its own wall
    and outcome rather than the ladder's. The Wayback withhold as fallback, the archive-readable
    convention behind an attempt with no rescue, and which rungs use ``session`` at all:
    ``docs/architecture.md`` "What the dispatcher returns, and what it carries forward".
    """
    if direct.status == "success":
        return direct
    # First because it is free and its triggers are disjoint from the browser's (see the doc).
    impersonated = await _run_rung(
        ctx, direct.status, rungs._impersonate_rung(url, direct, host_sems=host_sems, ctx=ctx)
    )
    if impersonated is not None:
        return impersonated
    if rungs._rendered_rung_applies(direct):
        # One per-host gate across the feed-then-browser pair (`browser_escalation_gate`).
        async with ctx.shared.browser_escalation_gate(url):
            derived = await _run_rung(
                ctx, direct.status, rungs._derived_api_rung(session, url, direct, host_sems=host_sems, ctx=ctx)
            )
            if derived is not None:
                return derived
            # Declined here rather than inside the rung, whose own gates all cost something.
            if ctx.fast_path:
                context._skip_for_fast_path(ctx, "rendered", direct, url)
            else:
                rendered = await _run_rung(ctx, direct.status, rungs._rendered_rung(url, direct, host_sems, ctx))
                if rendered is not None:
                    return rendered
    # Reached only for the statuses the browser rungs do not claim (`_WAYBACK_TRIGGER_STATUSES`).
    wayback = await _run_rung(
        ctx, direct.status, rungs._wayback_rung(session, url, direct, host_sems=host_sems, ctx=ctx)
    )
    if wayback is not None and wayback.status == "success":
        return wayback
    # Last (the only paid rung), asked about the DIRECT outcome so an archive withhold does not close it.
    read = await _run_rung(
        ctx, direct.status, rungs._url_context_rung(session, url, direct, host_sems=host_sems, ctx=ctx)
    )
    if read is not None:
        return read
    return wayback if wayback is not None else direct


async def _fetch_one(
    session: Any, url: str, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext | None = None
) -> FetchResult:
    """Fetch a single URL directly, then escalate what the direct route could not read.

    ``ctx`` carries the question text a PDF digest ranks passages against, the wall-clock
    origin each rung bounds itself with, and the rung attempts stamped onto the returned
    result. It defaults to a fresh one so the fetch surface can still be driven with three
    arguments, which is what every existing caller and test does.
    """
    ctx = context.LadderContext() if ctx is None else ctx
    direct = await direct_fetch._fetch_direct(session, url, host_sems, ctx)
    # The direct fetch's own rungs are over, and its status is what they left standing.
    ctx.close_rungs(0, direct.status)
    escalated = await _escalate_unresolved(session, url, direct, host_sems=host_sems, ctx=ctx)
    if direct.chrome_metric_withheld:
        # A fact about this URL's ladder rather than about one result (see the doc).
        escalated.chrome_metric_withheld = True
    return context._stamped_with_route(escalated, ctx)


async def fetch_url(url: str, *, policy: LadderPolicy, ctx: context.LadderContext) -> FetchResult:
    """Fetch one URL through the whole ladder under ``policy``: the entry point both callers use.

    Rung 0 first (``policy.known_api``, a public API that answers this URL exactly, with no page
    fetch at all), then the ladder proper. The policy is bound onto the context here rather than
    threaded through the rungs, so every rung reads ``ctx.policy`` and no rung signature carries
    a second argument.

    ``ctx.session`` and ``ctx.host_sems`` are how a caller that already holds a session and the
    process-wide politeness map keeps them: with a session per URL the fetcher's connector limits
    would change. A context naming neither gets a session opened and closed for this URL alone
    and the process-wide map, which is what a caller with one fetch in hand wants.
    """
    if policy.known_api is not None:
        translated = await policy.known_api(url)
        if translated is not None:
            return translated
    host_sems = ctx.host_sems if ctx.host_sems is not None else host_semaphores()
    bound = replace(ctx, policy=policy, host_sems=host_sems)
    if bound.session is not None:
        return await _fetch_one(bound.session, url, host_sems, bound)
    async with guard._get_session() as session:
        return await _fetch_one(session, url, host_sems, replace(bound, session=session))
