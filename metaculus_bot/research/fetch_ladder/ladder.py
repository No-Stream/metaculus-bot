"""The dispatcher: one URL's direct fetch, then the rungs its outcome earns, in one order.

``_fetch_one`` is the whole ladder for one cited URL, ``_escalate_unresolved`` is the rung order
and the browser gate, and ``_run_rung`` is the bracket that closes each rung's attempts on that
rung's own wall and outcome rather than the ladder's. Nothing here dials anything itself:
:mod:`direct` and :mod:`rungs` own every request.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from metaculus_bot.research.fetch_ladder import context, direct_fetch, rungs
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

    Returns the FIRST rung's rescue, or ``direct`` unchanged when every rung declines or fails.
    A rung that fired and produced nothing still leaves its attempt on the context, which is
    what makes ``route=rendered status=js_wall`` readable in the archive as "we tried the
    browser and this is still the answer" — the same convention the meta-refresh hop already
    follows. The Wayback rung is the one rung whose non-rescue is a VERDICT rather than None
    (``stale_data``, a capture we read and will not serve); it is kept as the fallback rather
    than as an early return, so the paid rung below is still reachable for that page.

    Each rung is closed the moment its result is known (:func:`_run_rung`), so the attempts it
    opened carry that rung's own wall and outcome rather than the ladder's: the status it
    returned, or the direct status it left standing when it declined.

    ``session`` is the aiohttp session the rungs that issue an ordinary GET use; the browser
    rung and the impersonated retry ignore it, because Chromium and libcurl each bring their own
    transport.
    """
    if direct.status == "success":
        return direct
    # First among the rungs, and the position is a reading choice rather than a functional one:
    # the trigger sets are disjoint (`_rendered_rung_applies` fires only on `js_wall` and the
    # `thin_page` shape of `no_resolving_content`, never on `blocked`), so this rung never
    # contends for the browser escalation gate. It matches the `FetchRoute` Literal's own ladder
    # order, meets a reader with the cheap free retry before the expensive ones, sits before the
    # archive so a live page beats a stale capture, and before the paid reader so a rescue saves
    # the read on that URL entirely.
    impersonated = await _run_rung(
        ctx, direct.status, rungs._impersonate_rung(url, direct, host_sems=host_sems, ctx=ctx)
    )
    if impersonated is not None:
        return impersonated
    if rungs._rendered_rung_applies(direct):
        # The two browser-family rungs run under one per-host gate for this question, so a
        # same-host sibling asks `endpoint_for` only after this escalation has recorded (or
        # failed to record) an endpoint — see `QuestionRungBudget.browser_escalation_gate`.
        async with ctx.shared.browser_escalation_gate(url):
            derived = await _run_rung(
                ctx, direct.status, rungs._derived_api_rung(session, url, direct, host_sems=host_sems, ctx=ctx)
            )
            if derived is not None:
                return derived
            # Declined HERE rather than inside the rung: the rung's own gates all cost something
            # (a budget read, a memo lookup, a launch), and the fast path is a fact about the
            # question the dispatcher already holds.
            if ctx.fast_path:
                context._skip_for_fast_path(ctx, "rendered", direct, url)
            else:
                rendered = await _run_rung(ctx, direct.status, rungs._rendered_rung(url, direct, host_sems, ctx))
                if rendered is not None:
                    return rendered
    # Reached only for the statuses the browser rungs do not claim — the two trigger sets are
    # disjoint by construction (see `_WAYBACK_TRIGGER_STATUSES`), so the order between them is a
    # reading choice: free-and-local first, then the route whose egress is not ours.
    wayback = await _run_rung(
        ctx, direct.status, rungs._wayback_rung(session, url, direct, host_sems=host_sems, ctx=ctx)
    )
    if wayback is not None and wayback.status == "success":
        return wayback
    # Last, because it is the only rung that spends money and the only one whose product is a
    # model's answer rather than the host's bytes. Off by default in code, on in every bot workflow.
    # It is asked about the DIRECT outcome, and an archive WITHHOLD does not stand in its way: a
    # capture too old to serve is still a page we could not read fresh, which is exactly the
    # population this rung exists for. The withhold stays the fallback below, so with the flag
    # off (or the reader declining) a stale capture still reports `stale_data`. The paid rung's
    # own attempt closes on the DIRECT status when it declines, like every other rung: the
    # archive's verdict is not an outcome a model read can produce, and on the escalation line
    # `rung=url_context outcome=stale_data` read as if it had.
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
    # The rungs inside the direct fetch (the meta-refresh hop, the local PDF read) are over
    # once it returns, and its status is what they left standing.
    ctx.close_rungs(0, direct.status)
    escalated = await _escalate_unresolved(session, url, direct, host_sems=host_sems, ctx=ctx)
    if direct.chrome_metric_withheld:
        # The withhold is a fact about this URL's ladder, carried onto whatever the ladder
        # returns for it: a rung's rescue has its own extraction, which the metric never
        # withheld, so summed off final results alone the withholds the ladder then paid off
        # — the policy's whole point — reached no count at all (`chrome_metric_withholds`).
        escalated.chrome_metric_withheld = True
    return context._stamped_with_route(escalated, ctx)
