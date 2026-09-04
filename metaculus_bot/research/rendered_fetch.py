"""The headless-Chromium page render, shared by both SSRF-guarded fetch paths.

One browser rung, two callers. The gap-fill v2 ``fetch`` tool has escalated a JavaScript
wall to Chromium since 2026-07; the Tier-1 resolution-source fetcher gained the same rung on
2026-09-03, after a replay of 47 archived fetch failures measured Chromium rescuing 6 of the
8 pages that still failed from a residential address. Keeping the transport here is what
makes that one rung rather than two: the DNS pin, the per-request route guard, the
process-global launch cap and the wait-condition arithmetic are all SSRF- or
memory-load-bearing, and a second copy of any of them would drift.

What this module owns is the TRANSPORT — launch, navigate, read the DOM, optionally record
the JSON the page fetched for itself — and it hands back a :class:`RenderedPage` rather than
either caller's result type. Classification stays with the caller, because the two disagree
about what a rendered page MEANS: v2 wants extracted text plus outbound links for a driver
model, while Tier-1 re-enters its own chart-config / ARIA-rewrite / chrome-floor path and can
still withhold the page. That split is also why the "rendered to nothing" memo is written by
the caller (:func:`note_rendered_no_text`) and only READ here, and why both render memos are
keyed by the caller's ``memo_scope`` as well as the URL: only the caller knows whether its
own extraction found anything, and the two callers' answers to that question are not
interchangeable (see :func:`rendered_to_nothing`). The timed-out memo is the one this module
writes itself, because only the transport knows whether a browser actually ran before the
clock cut it off (see :class:`RenderTimeout`).

The SSRF guard is resolved through ``resolution_source`` at call time, from inside the two
functions that need it, and both halves of that are deliberate — see :func:`_resolve_pinned_host`.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from metaculus_bot.constants import RENDERED_DOM_MAX_CHARS
from metaculus_bot.research.http_fetch import BROWSER_HEADERS
from metaculus_bot.research.public_suffix import registrable_domain

logger = logging.getLogger(__name__)

# Hard ceiling on one render, and the fixed settle after the DOM is ready that stands in for
# waiting on network idle. Measured 2026-09-03: 4 of the replay's 10 render rescues came from
# pages where ``page.goto`` raised TimeoutError with the DOM fully rendered (both ballotpedia
# questions, both fts.unocha.org summaries) — network idle never arrives on a page carrying a
# long-poll widget or an analytics beacon, so waiting for it discarded content Chromium already
# had. The navigation's worst case is the cap: the goto budget is the cap MINUS the settle, and
# the one call after the settle, the DOM read, carries its own bound below.
RENDER_TIMEOUT_MS: int = 35_000
RENDER_SETTLE_MS: int = 2_000
# Bound on ``page.content()``. On a settled DOM it is a sub-second round trip to the browser; it
# runs long only when the page keeps navigating after the settle, which Playwright retries
# internally before giving up ("the page is navigating and changing the content"). Measured
# 2026-09-03 on ogimet.com: the goto timed out at 33 s as designed, the settle ran, and the
# unbounded read then blocked for a further 40 s, so the render ran 76 s against a Tier-1
# provider wall of 45 s and every page that question had already fetched was discarded. A DOM
# that has not answered in this long is one that will not settle. Fixed rather than derived
# from the remaining goto budget, because the read that matters most is the salvage AFTER a
# goto timeout, when that remainder is zero by construction.
RENDER_DOM_READ_TIMEOUT_MS: int = 5_000
# What a render spends AFTER its goto returns or times out, at worst: the settle, then the DOM
# read at its bound. A caller-derived deadline has to leave this much room past the goto, or the
# salvage-after-goto-timeout population above is cut off by the caller's own bound the instant
# the settle ends — which is what happened at every Tier-1 budget under 40 s before the
# navigation budget was recomputed here (see :func:`_goto_budget_ms`). Two costs are NOT in this
# tail and are the CALLER's to reserve, in the deadline it hands the transport
# (:data:`RENDER_EXIT_RESERVE_MS`): the launch, which runs AFTER the recompute (0.3 s warm,
# seconds cold), and the shared teardown bound. So when the goto runs its budget out, the DOM
# read's fixed bound lands past the transport's deadline by the launch time, inside that reserve,
# and the harvest drain is clamped to the deadline itself in :func:`_navigate_and_read_dom` so a
# body still in flight cannot spend the reserve on the harvest's behalf.
RENDER_POST_GOTO_TAIL_MS: int = RENDER_SETTLE_MS + RENDER_DOM_READ_TIMEOUT_MS
# Floor under the navigation budget once the gates are held. Below it the render DECLINES rather
# than making a token attempt: a launch is 100-300 MB and one of the process-wide slots, and a
# page that could not answer in five seconds after a queue this long is not one we would have
# read in the seconds that were left. Live only through :func:`_goto_budget_ms`, which is the one
# place the budget is computed after both gates are held.
RENDER_MIN_GOTO_MS: int = 5_000
# ONE bound on the whole teardown (unroute, context close, browser close together), shared by
# the three steps through :class:`_TeardownBudget`. A healthy browser answers each in well under
# a second; a wedged one never answers `BrowserContext.close`, which awaits its closed-future with
# no timeout of its own, and a render whose DOM was already read would then sit in teardown until
# the caller's bound cut it — discarding the page it had. Shared rather than per step because
# three separate 2 s bounds let a wedged browser run 6 s past the caller's cut (`asyncio.wait_for`
# cancels the render and then AWAITS its unwinding finallys), which is what tripped the Tier-1
# provider's 45 s wall. Past the bound a step is abandoned and the driver stop at the end of the
# ``async_playwright()`` block kills the browser process anyway, so nothing is left running.
RENDER_TEARDOWN_TIMEOUT_MS: int = 2_000
# What a caller with a hard deadline must reserve for the render's EXIT, on top of the goto and
# the post-goto tail: the shared teardown bound plus one second for the costs that run between
# the budget recompute and the goto (the launch at 0.3 s warm, `new_context`, `new_page`) and for
# the driver stop after the closes. The Tier-1 rung hands the transport its wall budget LESS this
# reserve while keeping its own `wait_for` at the full budget, so a render that runs every bound
# out still hands its DOM back before the outer cut instead of being cancelled in teardown. It
# can only shorten the goto or decline earlier at `RENDER_MIN_GOTO_MS`, never lengthen a wait.
RENDER_EXIT_RESERVE_MS: int = RENDER_TEARDOWN_TIMEOUT_MS + 1_000

# Process-global cap on concurrent headless Chromium launches. Module-level, so the bound
# spans every question running under the orchestrator's Semaphore(6) AND both fetch paths:
# each Chromium is ~100-300MB, one v2 step's parallel_tool_calls can request many fetches, a
# Tier-1 question can escalate up to RESOLUTION_SOURCE_MAX_URLS pages, and an unbounded launch
# fan-out would OOM the runner (an escape try/except cannot catch that). Cap 2 covers real
# bursts of 1-3 rendered pages while bounding worst-case memory.
RENDER_LAUNCH_CAP: int = 2
_RENDERED_FETCH_GLOBAL_SEMAPHORE = asyncio.Semaphore(RENDER_LAUNCH_CAP)

# The two per-run render memos, one bounded map. ``no_text``: Chromium ran and the caller's
# extraction found nothing, so a second launch on the same URL in the same run is skipped —
# 100-300 MB and a whole slot out of the cap above, to learn what this run already knows.
# ``timed_out``: the render was cut off (the page kept navigating), and the transport re-raises
# that rather than launching again. Both deliberately record ONLY those outcomes — a `blocked` or
# `error` GET is never memoized, because 429 is retryable and a throttle interstitial has to
# stay re-requestable (the q45191 receipt). Keyed by (scope, url): the two callers mean
# different things by "rendered to nothing" (gap-fill v2 writes it on bare trafilatura
# emptiness; Tier-1 only after the ARIA rewrite, the inline-chart read and the XHR-harvest
# fallback have ALL failed), so one path's negative must never answer the other's question.
MemoScope = Literal["resolution_source", "gap_fill_v2"]
_MemoKind = Literal["no_text", "timed_out"]
_RENDER_MEMO_MAX_ENTRIES = 50
_RENDER_MEMO: OrderedDict[tuple[MemoScope, str], _MemoKind] = OrderedDict()
_PLAYWRIGHT_WARNED = False

# --- JSON the page fetched for itself (the derived-API rung's discovery half) -------------
#
# Measured 2026-09-03 across six JavaScript dashboards: grepping the served HTML for an API
# URL found ONE candidate (a Maps key), 3 of 4 hand-guessed endpoints were wrong, and XHR
# capture during a render found a working unauthenticated JSON endpoint for all six. So the
# endpoints are discoverable only by rendering, which is why this rides the browser rung
# instead of being a static registry.
#
# Every bound here is a memory bound: the bodies are buffered in the render task while the
# page is live, on a runner also holding a 100-300 MB browser.
HARVEST_MAX_RESPONSES: int = 5
# Below this a JSON body is a ping, a feature flag or an empty envelope, not a dataset.
HARVEST_MIN_BODY_BYTES: int = 200
HARVEST_MAX_BODY_BYTES: int = 256 * 1024
# Hosts other than the page's own publisher that may serve a harvestable dataset. Deliberately
# almost empty: an allow-list entry is a decision that a third-party host's JSON may be read as
# the cited page's content, and only Datawrapper's CDN has an already-trusted data route in this
# repo (the Tier-2 hop fetches from it by construction). Anything else waits for a measurement.
HARVEST_ALLOWED_CDN_HOSTS: frozenset[str] = frozenset({"static.dwcdn.net"})
_JSON_CONTENT_TYPE_TOKENS = ("application/json", "text/json", "+json")


class RenderTimeout(TimeoutError):
    """A render that ran and was CUT OFF, or a URL whose render already was in this run.

    A subclass of the builtin so both callers' ``except TimeoutError`` catch it unchanged, and a
    class of its own so the transport can tell its deliberate cut-off from a raw OS-level
    ``TimeoutError`` (also a builtin subclass, via ``OSError``) surfacing from somewhere under the
    render: the first is re-raised for the caller to record as ``render_timeout``, the second is
    not a fact about the page and lands in the logged boundary like any other unexpected error.

    The timed-out memo is written at the RAISE SITE of this exception, inside the transport, and
    nowhere else: a caller that bounds the whole call with its own ``wait_for`` cannot tell a
    render that ran out the clock from one that never left the queue behind the two gates, and
    memoising the second would switch the URL off for every later question in the run over a
    wait that was never about the page. At the raise site rather than in a handler because the
    exception unwinds through the bounded teardown before any handler sees it, and a caller's
    cancellation landing during that teardown replaces it, so a handler would never run.
    """


class RenderBudgetExpired(TimeoutError):
    """The caller's deadline left less than ``RENDER_MIN_GOTO_MS`` once both gates were held.

    Raised instead of returning ``None`` so the caller can record it as what it is — its wall
    budget ran out in the queue — rather than as the renderer being unavailable, the count the
    operator reads as the Chromium install having failed. A ``TimeoutError`` because that is what
    it is (a deadline passed while waiting) and because a caller that bounds the whole call with
    its own ``wait_for`` already handles that class; deliberately NOT a :class:`RenderTimeout`,
    because no browser ran and there is nothing to memoise. Only reachable with a deadline, so
    the gap-fill v2 path never sees it.
    """


class RenderDomOverCeiling(Exception):
    """Chromium rendered the page and its DOM is over ``RENDERED_DOM_MAX_CHARS``.

    Raised instead of returning ``None`` so the caller can record it as what it is, a fact about
    the page (it rendered, and it is too big to read safely), rather than as the renderer being
    unavailable, the count the operator reads as the Chromium install having failed. Not a
    ``TimeoutError``: no clock ran out. Not memoised: the DOM had content, so "rendered to
    nothing" would be false, and a size is not something a retry changes within a run either
    way. Both callers decline on it the way they decline on ``None``.
    """


@dataclass(frozen=True, slots=True)
class HarvestedJson:
    """One JSON response the rendered page fetched for itself."""

    url: str
    body: bytes


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """What a successful render hands back: the DOM, plus any JSON the page fetched.

    ``html`` is ``page.content()`` — the DOM as it stood after the settle, which is the
    whole point of the rung. ``content_type`` and ``http_status`` are the main-frame
    response's, empty and ``None`` when the goto timed out and the DOM was salvaged (no
    response object survives that path). The status is carried because a browser can be
    answered differently from the direct GET on the same URL — a 403 or 429 interstitial whose
    markup clears a chrome floor is not the page — and only the caller decides what to do
    about that.
    """

    url: str
    content_type: str
    html: str
    json_responses: tuple[HarvestedJson, ...] = ()
    http_status: int | None = None


def reset_render_state() -> None:
    """Drop the run-scoped render state: the memos, the warn latch, the launch gate.

    For tests. The launch gate is REBOUND rather than drained because an ``asyncio.Semaphore``
    binds to the loop that first blocks on it and raises from any other, so a contended
    acquire in one test's event loop would otherwise leak that binding into the next.
    """
    global _PLAYWRIGHT_WARNED, _RENDERED_FETCH_GLOBAL_SEMAPHORE  # noqa: PLW0603  # run-scoped render state: the warn latch and the loop-bound launch gate
    _RENDER_MEMO.clear()
    _PLAYWRIGHT_WARNED = False
    _RENDERED_FETCH_GLOBAL_SEMAPHORE = asyncio.Semaphore(RENDER_LAUNCH_CAP)


def rendered_to_nothing(url: str, *, memo_scope: MemoScope) -> bool:
    """True when a browser already read ``url`` to nothing in this run, by ``memo_scope``'s own test."""
    return _RENDER_MEMO.get((memo_scope, url)) == "no_text"


def render_timed_out(url: str, *, memo_scope: MemoScope) -> bool:
    """True when a render of ``url`` under ``memo_scope`` was already cut off in this run."""
    return _RENDER_MEMO.get((memo_scope, url)) == "timed_out"


def note_rendered_no_text(url: str, *, memo_scope: MemoScope) -> None:
    """Record that Chromium rendered ``url`` and the caller found no content in the result."""
    _note(url, memo_scope, "no_text")


def _note_render_timeout(url: str, *, memo_scope: MemoScope) -> None:
    """Record that a render of ``url`` RAN and was cut off, so this run does not pay for it again.

    Transport-private: only :func:`render_page` knows a browser actually ran (see
    :class:`RenderTimeout`), so no caller writes this one.
    """
    _note(url, memo_scope, "timed_out")


def _note(url: str, memo_scope: MemoScope, kind: _MemoKind) -> None:
    key = (memo_scope, url)
    _RENDER_MEMO[key] = kind
    _RENDER_MEMO.move_to_end(key)
    while len(_RENDER_MEMO) > _RENDER_MEMO_MAX_ENTRIES:
        _RENDER_MEMO.popitem(last=False)


def _warn_playwright_unavailable_once(exc: BaseException) -> None:
    global _PLAYWRIGHT_WARNED  # noqa: PLW0603  # one-shot process-wide warn latch so the rendered rung logs once per run
    if _PLAYWRIGHT_WARNED:
        return
    _PLAYWRIGHT_WARNED = True
    logger.warning("rendered fetch rung unavailable: %s: %s", type(exc).__name__, exc)


def _ip_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _host_resolver_rule(host: str, ip: str) -> str:
    """Build the Chromium ``--host-resolver-rules`` MAP value pinning ``host`` to ``ip``.

    IPv6 literals must be bracketed in the MAP target per Chromium's rule parser
    (``MAP host [dead::beef]``); IPv4 literals are bare. A malformed ``ip`` is
    passed through unbracketed — callers only ever feed this a value already
    vetted by :func:`_resolve_pinned_host`, so that branch is defensive only.
    """
    parsed_ip = _ip_literal(ip)
    target = f"[{ip}]" if parsed_ip is not None and parsed_ip.version == 6 else ip
    return f"--host-resolver-rules=MAP {host} {target}"


def _pinnable_url_host(url: str) -> str | None:
    """Hostname of a URL eligible for DNS pinning, or None when the URL itself disqualifies it.

    The pin is only as good as the MAP pattern's match. Chromium canonicalises a hostname
    (punycode for a unicode name, the trailing dot dropped) before it consults
    ``--host-resolver-rules``, and ``urlparse`` does neither, so a pattern built from a unicode
    or trailing-dot host is accepted and matches NOTHING: the main frame then resolves through
    Chromium's own resolver and the rebinding window the pin closes re-opens. Failing closed is
    the only safe answer — reproducing Chromium's canonicalisation is not, because the stdlib
    ``idna`` codec is IDNA2003 and diverges from Chromium's UTS#46 form on real names
    (``straße`` maps to ``strasse`` there and to ``xn--strae-oqa`` in Chromium), so it would emit
    a pattern that looks right and is inert.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme.lower() not in ("http", "https"):
        return None
    # Userinfo defeats hostname-based trust (`https://trusted@10.0.0.1/`).
    if parsed.username is not None or parsed.password is not None:
        return None
    host = parsed.hostname
    if not host or not host.isascii() or host.endswith("."):
        return None
    return host


async def _resolve_pinned_host(url: str) -> tuple[str, str] | None:
    """Vet ``url``'s host and resolve it to ONE public IP for Chromium DNS pinning.

    Returns ``(host, vetted_ip)`` — the ``--host-resolver-rules=MAP`` operands — or
    ``None`` when the URL is non-public, unresolvable, or ANY resolved address is
    disallowed. Mirrors :func:`resolution_source.is_public_http_url`'s classification
    (scheme, userinfo, and the shared :func:`resolution_source.resolve_vetted_public_ip`
    predicate) so Chromium can only dial an address the airtight aiohttp
    ``FilteringResolver`` path would also accept.

    This is what closes the DNS-rebinding TOCTOU on the rendered rung: the per-request
    route guard runs its own ``getaddrinfo`` independently of Chromium's socket connect, so
    a rebinding host (TTL 0) can pass the preflight and connect to a private IP. Pinning the
    main-frame host to a single pre-vetted IP removes that second resolution entirely —
    Chromium's connect can only reach the vetted address.

    Fails CLOSED: on any rejection the caller skips Chromium for that host and its fetch
    ladder degrades to whatever the plain rung already had.

    The ``resolution_source`` import is function-scoped for two reasons at once, and both
    have to hold for it to stay. It is a REAL circular import — that module imports this one
    at module scope for its own rendered rung — and it is the LATE BINDING the suites rely
    on: ``resolution_source.is_public_http_url`` / ``resolve_vetted_public_ip`` /
    ``_ip_is_disallowed`` are monkeypatched on that module by both fetch paths' tests, and
    the guard has exactly ONE patch surface precisely because every reader resolves it
    there. Hoisting the guard into a lower module would give it two, which is how a patch at
    the wrong module stays green while proving nothing.
    """
    from metaculus_bot.research import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # real cycle + the guard's single patch surface, per the docstring
        resolution_source,
    )

    host = _pinnable_url_host(url)
    if not host:
        return None

    # IP-literal host: no DNS to rebind. Vet directly and pin to itself.
    literal = _ip_literal(host)
    if literal is not None:
        if resolution_source._ip_is_disallowed(literal):
            return None
        return host, str(literal)

    vetted_ip = await resolution_source.resolve_vetted_public_ip(host)
    if vetted_ip is None:
        return None
    return host, vetted_ip


async def _vet_route(route: Any, request: Any, playwright_error: type[BaseException]) -> None:
    """Let one request Chromium is about to make through, or abort it.

    ``playwright_error`` is passed in because the class comes from the caller's function-scoped
    optional import. A request can still be in flight when the page/context tears down (typically
    after a goto timeout): ``continue_``/``abort`` then races the close and raises
    ``TargetClosedError`` in this detached event-listener task — the unhandled-error storm seen
    2026-07-25. It is swallowed because a closed target has no live socket, so an abort that
    "fails" because the target is already gone still lets nothing through and the SSRF guarantee
    is unaffected; ``unroute_all`` in the caller's ``finally`` is the primary drain and this is the
    residual-race backstop. Only Playwright's own Error is caught, so a genuine bug still
    propagates.

    The guard is resolved through ``resolution_source`` at call time for the reasons
    :func:`_resolve_pinned_host` states.
    """
    from metaculus_bot.research import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # real cycle + the guard's single patch surface, see _resolve_pinned_host
        resolution_source,
    )

    try:
        if await resolution_source.is_public_http_url(request.url):
            await route.continue_()
        else:
            await route.abort("blockedbyclient")
    except playwright_error as exc:  # the teardown race documented above, not a broad catch
        logger.debug("rendered fetch route guard race during teardown: %s", exc)


def _goto_budget_ms(goto_timeout_ms: int, deadline_monotonic_s: float | None) -> int | None:
    """The navigation budget once both gates are held, or None when it is under the floor.

    Computed AFTER the acquires, for the same reason ``_fetch_one_hop`` computes its hop timeout
    after the host semaphore: the caller measured its budget before two unbounded waits — the
    loop-wide per-host gate shared by every concurrent question, and the process-global launch
    cap shared with the other fetch path — and a render admitted after queueing behind both
    would otherwise navigate on a figure that expired in the queue. The caller's figure stays a
    CEILING (it carries the transport's own 35 s cap); the deadline, when given, tightens it to
    what is actually left less the post-goto tail, so a goto that runs its budget out can still
    be settled and read inside the caller's bound (:data:`RENDER_POST_GOTO_TAIL_MS`). Without a
    deadline the caller's figure stands: gap-fill v2 bounds the whole call with its own ceilings.
    """
    budget_ms = goto_timeout_ms
    if deadline_monotonic_s is not None:
        remaining_ms = int((deadline_monotonic_s - time.monotonic()) * 1000) - RENDER_POST_GOTO_TAIL_MS
        budget_ms = min(budget_ms, remaining_ms)
    return budget_ms if budget_ms >= RENDER_MIN_GOTO_MS else None


async def _in_own_task(call: Any, *, timeout_ms: int) -> Any:
    """Await one Playwright call under a bound, in a task of its own.

    The task is the point, not a convenience. Playwright registers the cancellation of a protocol
    call on the TASK that sent it: when that task finishes cancelled, the pending reply future is
    cancelled with it. ``asyncio.wait_for`` on a bare coroutine runs it in the current task and
    converts the cancellation into ``TimeoutError``, so the current task never finishes
    cancelled, the reply future stays pending, and the late reply (or the ``TargetClosedError``
    teardown sets on it) has nobody to retrieve it — one "Future exception was never retrieved"
    ERROR per cut-off. Wrapping the call in its own task makes the cancellation land where
    Playwright looks for it.
    """
    return await asyncio.wait_for(asyncio.ensure_future(call), timeout=timeout_ms / 1000)


class _TeardownBudget:
    """One ``RENDER_TEARDOWN_TIMEOUT_MS`` shared by every teardown step of one render.

    Started by the FIRST step that asks, not when the render begins: computed eagerly before the
    launch it would already be spent by the time teardown runs, and every step would be abandoned
    to the driver stop. Each later step gets what the earlier ones left, floored at zero, so the
    steps together hold a render for at most one bound past its last read whatever the browser
    does. A step handed zero is scheduled and cancelled before it runs, which the driver stop then
    covers, rather than skipped: the one code path keeps the one log line.
    """

    def __init__(self) -> None:
        self._deadline_s: float | None = None

    def remaining_ms(self) -> int:
        now = time.monotonic()
        if self._deadline_s is None:
            self._deadline_s = now + RENDER_TEARDOWN_TIMEOUT_MS / 1000
        return max(0, int((self._deadline_s - now) * 1000))


async def _teardown_step(name: str, call: Any, playwright_error: type[BaseException], budget: _TeardownBudget) -> None:
    """Run one teardown call so that it can neither wedge the render nor replace its exception.

    Bounded by what is left of the render's shared teardown ``budget``; past it the call is
    abandoned and the driver stop kills the browser. Playwright's own errors are swallowed at
    DEBUG because teardown races them by construction: ``BrowserContext.close`` does not swallow
    ``TargetClosedError`` the way ``Browser.close`` does, and the next protocol call re-raises any
    error a detached listener stored on the connection, so an unguarded close in a ``finally``
    would replace the :class:`RenderTimeout` (or the caller's cancellation) unwinding through it
    and turn a cut-off render into "the renderer is unavailable". Anything else is a bug and
    propagates.
    """
    remaining_ms = budget.remaining_ms()
    try:
        await _in_own_task(call, timeout_ms=remaining_ms)
    except TimeoutError:
        logger.warning(
            "rendered fetch teardown: %s did not finish in the %dms left of the %dms teardown budget; "
            "leaving it to the driver stop",
            name,
            remaining_ms,
            RENDER_TEARDOWN_TIMEOUT_MS,
        )
    except playwright_error as exc:
        logger.debug("rendered fetch teardown: %s raced the close: %s", name, exc)


async def _navigate_and_read_dom(
    page: Any,
    url: str,
    playwright_error: type[BaseException],
    *,
    memo_scope: MemoScope,
    goto_timeout_ms: int,
    deadline_monotonic_s: float | None,
    harvest: _JsonHarvest | None,
) -> RenderedPage:
    """Navigate to ``url``, let it settle, and return the rendered page.

    Two departures from a plain ``networkidle`` navigation, both measured 2026-09-03. The wait
    condition is ``domcontentloaded`` plus a fixed settle, because network idle never arrives on
    a page carrying a long-poll widget or an analytics beacon. And a goto failure is SALVAGED
    rather than treated as a dead rung: Playwright's ``TimeoutError`` subclasses ``Error``, and a
    timed-out goto routinely leaves a fully rendered DOM behind — 4 of the replay's 10 render
    rescues came from exactly that. A genuine navigation error lands here too and salvages an
    empty ``about:blank``, which reaches the ladder as the same "rendered read nothing".

    The DOM read is bounded by ``RENDER_DOM_READ_TIMEOUT_MS`` and raises :class:`RenderTimeout`
    when it fires — deliberately NOT swallowed into the salvage, because a page that keeps
    navigating has no DOM to salvage and the caller needs to tell this from a browser that is
    missing or broken (see :func:`render_page`). The timed-out memo for ``memo_scope`` is written
    here, immediately before the raise, so it lands even when the caller's own cut arrives while
    the exception is still unwinding through teardown. The harvest's in-flight body reads are
    drained INSIDE what is left of that same bound, never after it, and no later than the
    caller's ``deadline_monotonic_s`` when a deadline was given: the harvest is opportunistic and
    may not lengthen a render whose real product is the DOM. The second clamp is there because
    the read bound is fixed and the launch runs after the budget recompute, so in the salvage
    shape the read bound lands past the transport's deadline by the launch time, and a body still
    in flight there used to hold the drain into the caller's exit reserve and past its outer cut,
    which then discarded a DOM this function had already read. The deadline already excludes
    that reserve (:data:`RENDER_EXIT_RESERVE_MS`), so nothing is subtracted from it again here.
    The DOM read itself keeps its fixed bound (see :data:`RENDER_DOM_READ_TIMEOUT_MS` for why).

    The DOM is measured in characters against ``RENDERED_DOM_MAX_CHARS`` before anything copies
    it, because the copies are the hazard: the Tier-1 caller encodes it, decodes it back,
    rewrites it and hands trafilatura a tree several times its size, all while the browser is
    still resident. A DOM over the ceiling raises :class:`RenderDomOverCeiling`, its own decline,
    so the caller can count it apart from a browser that is missing.
    """
    try:
        response = await page.goto(url, wait_until="domcontentloaded", timeout=goto_timeout_ms)
    except playwright_error as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # the salvage above: Playwright's own Error class, passed in from the optional import
        logger.debug("rendered fetch goto failed, salvaging DOM: %s: %s", type(exc).__name__, exc)
        response = None
    await page.wait_for_timeout(RENDER_SETTLE_MS)
    # ``goto`` returns None for an about:blank or same-document navigation, and the salvage
    # above leaves no response either; both read as "no main-frame response".
    content_type = (response.headers.get("content-type") or "").lower() if response is not None else ""
    http_status: int | None = response.status if response is not None else None
    read_deadline_s = time.monotonic() + RENDER_DOM_READ_TIMEOUT_MS / 1000
    try:
        html = await _in_own_task(page.content(), timeout_ms=RENDER_DOM_READ_TIMEOUT_MS)
    except TimeoutError as exc:
        _note_render_timeout(url, memo_scope=memo_scope)
        raise RenderTimeout(
            f"the DOM read of {urlparse(url).netloc} outlived {RENDER_DOM_READ_TIMEOUT_MS}ms: the page kept navigating"
        ) from exc
    if len(html) > RENDERED_DOM_MAX_CHARS:
        logger.warning(
            "rendered fetch declined the DOM of %s: %d chars is over the %d-char ceiling",
            urlparse(url).netloc,
            len(html),
            RENDERED_DOM_MAX_CHARS,
        )
        raise RenderDomOverCeiling(
            f"the rendered DOM of {urlparse(url).netloc} is {len(html)} chars, over the {RENDERED_DOM_MAX_CHARS}-char ceiling"
        )
    json_responses: tuple[HarvestedJson, ...] = ()
    if harvest is not None:
        drain_until_s = read_deadline_s if deadline_monotonic_s is None else min(read_deadline_s, deadline_monotonic_s)
        await harvest.drain(until_monotonic_s=drain_until_s)
        json_responses = tuple(harvest.bodies)
    return RenderedPage(
        url=url, content_type=content_type, html=html, json_responses=json_responses, http_status=http_status
    )


def _harvestable_json_host(response_host: str, page_host: str) -> bool:
    """True when ``response_host`` may serve harvestable JSON for a page on ``page_host``.

    Same publisher, by registrable domain: the public suffix plus one label, from the
    public-suffix list the market-retrieval join already vendors (``registrable_domain``). A page
    on ``www.x.gov`` whose data endpoint is ``api.x.gov`` or ``data.x.gov`` is the ordinary
    dashboard shape, and a page on ``forest-fire.emergency.copernicus.eu`` fed by
    ``api2.effis.emergency.copernicus.eu`` is the sibling-subdomain shape live QA found
    (2026-09-03); a bare ``endswith`` relation matched neither. The PSL is also what makes
    ``a.co.uk`` and ``b.co.uk`` strangers and a stranger's site on a shared suffix
    (``x.github.io`` against ``github.io``) no match, which is exactly where a wrong answer reads
    someone else's JSON as the cited page's content. IP literals have no registrable domain and
    the PSL algorithm would collapse two of them to their last two octets, so they are compared
    exactly and never otherwise. The explicitly allow-listed CDNs are the one exception to
    same-publisher.
    """
    if not response_host or not page_host:
        return False
    if response_host == page_host:
        return True
    if response_host in HARVEST_ALLOWED_CDN_HOSTS:
        return True
    if _ip_literal(response_host) is not None or _ip_literal(page_host) is not None:
        return False
    publisher = registrable_domain(page_host)
    return publisher is not None and registrable_domain(response_host) == publisher


def is_json_content_type(content_type: str) -> bool:
    """Whether a (lower-cased) Content-Type names a JSON body: ``application/json``, ``text/json``
    or any ``+json`` structured suffix (``application/geo+json``, ``application/vnd.api+json``).

    The ONE JSON vocabulary for the fetch paths. The harvest applies it at discovery, the
    derived-feed rung at reuse, and the resolution-source 200-response router when it decides
    what a directly cited URL is; a narrower spelling at any one of them strands a feed the
    others accept, and a remembered endpoint is first-find-wins for the run.
    """
    return any(token in content_type for token in _JSON_CONTENT_TYPE_TOKENS)


def _declared_length_over_cap(content_length: str | None) -> bool:
    """Whether the response DECLARES a body over the harvest cap.

    The pre-screen: ``Response.body()`` materialises the whole body (base64 over the driver pipe,
    then decoded, so ~2.3x the body at peak) before any post-read size test can run, and a
    dashboard's 60 MB GeoJSON would buffer beside a 100-300 MB browser on the strength of its
    content type alone. A declared length over the cap is enough to refuse the read: for a
    compressed response it is the compressed size, and the decoded body is only bigger. Absent,
    unparseable or lying headers fall through to the post-read test, which stays as the backstop.
    """
    if content_length is None:
        return False
    try:
        return int(content_length) > HARVEST_MAX_BODY_BYTES
    except ValueError:
        return False


async def _harvest_json_response(
    response: Any, *, page_host: str, into: list[HarvestedJson], playwright_error: type[BaseException]
) -> None:
    """Record one JSON response the page fetched, if it clears every bound.

    Reads the body inside the response event so it is still available — after the page
    navigates or closes, Playwright discards it. Every rejection is silent: this is
    opportunistic discovery attached to a render whose real product is the DOM, and a body we
    could not read must never be able to fail the render.

    ``response.url`` and ``response.headers`` (lower-cased names) are Playwright's documented
    ``Response`` contract; the fake in the tests honours the same one.
    """
    if len(into) >= HARVEST_MAX_RESPONSES:
        return
    url = response.url
    if not _harvestable_json_host(urlparse(url).hostname or "", page_host):
        return
    headers = response.headers
    if not is_json_content_type((headers.get("content-type") or "").lower()):
        return
    if _declared_length_over_cap(headers.get("content-length")):
        return
    try:
        body = await response.body()
    except playwright_error as exc:
        logger.debug("rendered fetch could not read a harvested response body: %s", exc)
        return
    # Re-checked AFTER the read: every handler that passed the check above was suspended in
    # ``body()`` together, so the check before it bounds nothing on its own.
    if len(into) >= HARVEST_MAX_RESPONSES:
        return
    if not (HARVEST_MIN_BODY_BYTES <= len(body) <= HARVEST_MAX_BODY_BYTES):
        return
    into.append(HarvestedJson(url=url, body=body))


class _JsonHarvest:
    """The JSON a page fetches for itself, collected from its response events during one render.

    Owns the tasks the events spawn. Playwright's ``Page`` is a pyee emitter, and pyee wraps a
    coroutine listener in ``ensure_future`` and forgets it, so every ``response`` firing used to
    be a detached task nobody joined: the snapshot was taken while reads were still pending and
    teardown then made their ``body()`` raise into a swallowed path, which dropped exactly the
    derived-API rung's payload — and the miss stuck, because the caller memoised the empty render.
    The listener is therefore SYNC and creates the tasks itself, so pyee neither tracks them nor
    re-emits their exceptions on the page's ``error`` event, and :meth:`drain` joins them before
    the snapshot, inside the DOM read's own bound.
    """

    def __init__(self, *, page_host: str, playwright_error: type[BaseException]) -> None:
        self._page_host = page_host
        self._playwright_error = playwright_error
        self.bodies: list[HarvestedJson] = []
        self._pending: set[asyncio.Task[None]] = set()

    def on_response(self, response: Any) -> None:
        task = asyncio.create_task(
            _harvest_json_response(
                response, page_host=self._page_host, into=self.bodies, playwright_error=self._playwright_error
            )
        )
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def drain(self, *, until_monotonic_s: float) -> None:
        """Wait for the in-flight body reads, but no later than ``until_monotonic_s``; cancel the rest."""
        if self._pending:
            await asyncio.wait(set(self._pending), timeout=max(0.0, until_monotonic_s - time.monotonic()))
        self.cancel_pending()

    def cancel_pending(self) -> None:
        for task in self._pending:
            task.cancel()


async def render_page(
    url: str,
    *,
    memo_scope: MemoScope,
    host_gate: asyncio.Semaphore,
    goto_timeout_ms: int = RENDER_TIMEOUT_MS - RENDER_SETTLE_MS,
    deadline_monotonic_s: float | None = None,
    harvest_json: bool = False,
) -> RenderedPage | None:
    """Render ``url`` in headless Chromium and return its DOM, or None when the rung declines.

    ``None`` is the one graceful-failure signal both callers already handle, and it covers
    every way this rung can DECLINE with nothing rendered: a URL a browser already read to
    nothing in this run under the caller's own ``memo_scope``, Playwright missing or broken, a
    host that cannot be pinned to a public IP, and any error out of the browser. Two declines are
    NOT ``None``, because each is a fact the caller counts apart from a missing browser: a
    navigation budget under the floor once the gates are held raises :class:`RenderBudgetExpired`
    (the caller's wall budget ran out in the queue), and a rendered DOM over
    ``RENDERED_DOM_MAX_CHARS`` raises :class:`RenderDomOverCeiling` (the page rendered and is too
    big to read safely). A caller that wants to tell the ``None`` causes apart reads
    :func:`rendered_to_nothing` and :func:`_resolve_pinned_host` itself.

    A render that ran and was CUT OFF is different, and raises :class:`RenderTimeout` (a
    builtin ``TimeoutError``) instead: the DOM read outlived ``RENDER_DOM_READ_TIMEOUT_MS``
    because the page kept navigating. It is not folded into ``None`` because the two mean
    different things to a caller — a timeout says nothing about whether Chromium works, so it
    must neither trip the once-per-process unavailable warning nor be recorded as the browser
    being absent — and it is its own class so the Tier-1 rung, which also bounds this whole call
    with a ``wait_for`` of its own, can tell a render that ran out the clock (this exception) from
    one its outer bound cut while it was still queued behind the two gates (a bare
    ``TimeoutError``, which says nothing about the page). The transport memoises the timed-out
    URL for the run at the raise site — and only there, for that same reason — and a memoised
    timeout is RE-RAISED here rather than declined, so a second question citing the same hostile
    page records the same reason again instead of reading as a missing browser.

    What a deadline bounds, and what it does not. ``deadline_monotonic_s`` is the instant by
    which the transport must have its DOM (and any harvested bodies) in hand: the goto is sized
    off it less the post-goto tail, the harvest drain is clamped to it, and the salvage DOM read
    can land at most the launch time past it. Teardown then runs inside the shared
    ``RENDER_TEARDOWN_TIMEOUT_MS`` bound, and the driver stop at the end of the
    ``async_playwright()`` block runs after that, so a caller with a hard wall has to hand in a
    deadline that reserves both (:data:`RENDER_EXIT_RESERVE_MS`). The driver stop is the one
    step left UNBOUNDED, deliberately: it is what actually kills the Chromium process, and
    abandoning it on a wedged browser would leak 100-300 MB past ``RENDER_LAUNCH_CAP``, an OOM
    nothing can catch. A healthy stop is milliseconds and fits inside the reserve's spare second.

    The failure boundary around the browser is split three ways, and the split is the point.
    Playwright's own ``Error`` keeps the once-per-process latch: a launch that cannot find the
    executable is "the renderer is unavailable" and one line per run is the right volume.
    Anything else is a bug in this module or a dependency behaving unexpectedly, and it is
    logged with a full traceback EVERY time — one line per run and then silence was how a
    programming error in the render path hid behind the latch (2026-09-03). It is still
    swallowed rather than raised, because a raise here propagates out of the Tier-1 provider's
    ``gather`` and cancels the question's OTHER pages, which is the one thing a rung failure
    must never do.

    ``host_gate`` is the caller's own per-host politeness semaphore for this URL — passed in
    rather than derived, because the two fetch paths keep separate maps (Tier-1's is
    loop-wide across concurrent questions; v2's is its own module global). It is acquired
    OUTSIDE the launch cap, in that order, and HELD across the launch-cap queue, the launch,
    the navigation, the settle and the teardown, because Chromium dials that host itself. Both
    acquires are unbounded by design (bounding them is the operator's reserved call, FUTURE.md
    item 5), which is why the navigation budget is recomputed only once both are held: pass
    ``deadline_monotonic_s`` and the goto gets what is left of it less the post-goto tail, or
    the render declines under ``RENDER_MIN_GOTO_MS`` before a browser is launched
    (:func:`_goto_budget_ms`).

    ``harvest_json`` turns on recording the JSON the page fetches for itself. Off by default:
    the bodies are buffered alongside a 100-300 MB browser, and only the caller that has a
    use for a derived endpoint should pay that.
    """
    if rendered_to_nothing(url, memo_scope=memo_scope):
        logger.debug("rendered fetch skipped (already rendered to nothing): %s", urlparse(url).netloc)
        return None
    if render_timed_out(url, memo_scope=memo_scope):
        logger.info("rendered fetch skipped (a render already timed out this run): %s", urlparse(url).netloc)
        raise RenderTimeout(f"a render of {urlparse(url).netloc} already timed out this run")
    try:
        from playwright.async_api import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import
            Error as PlaywrightError,
        )
        from playwright.async_api import async_playwright  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # optional-dep import boundary: playwright missing/broken degrades the rendered rung, never the run
        _warn_playwright_unavailable_once(exc)
        return None

    # Pin Chromium's DNS to a single pre-vetted public IP BEFORE launch. If the host can't be
    # resolved to a public address, fail closed: skip Chromium and let the caller's ladder fall
    # back (same graceful-failure signal as a playwright-unavailable / render-error return).
    pinned = await _resolve_pinned_host(url)
    if pinned is None:
        logger.warning("rendered fetch skipped (host not pinnable to a public IP): %s", urlparse(url).netloc)
        return None
    host, vetted_ip = pinned

    try:
        async with host_gate, _RENDERED_FETCH_GLOBAL_SEMAPHORE, async_playwright() as playwright:
            return await _render_in_browser(
                playwright,
                url,
                PlaywrightError,
                memo_scope=memo_scope,
                host=host,
                vetted_ip=vetted_ip,
                goto_timeout_ms=goto_timeout_ms,
                deadline_monotonic_s=deadline_monotonic_s,
                harvest_json=harvest_json,
            )
    # The two declines the caller records under their own tokens: nothing is memoised for either
    # (nothing ran for the first; the second rendered content, just too much of it).
    except (RenderBudgetExpired, RenderDomOverCeiling):
        raise
    # Ordered before Playwright's Error deliberately, though the order is not load-bearing:
    # Playwright's TimeoutError derives from its own Error, not the builtin, so this clause never
    # sees one, and a raw OS-level TimeoutError is not a RenderTimeout and falls through to the
    # logged boundary below. The memo was already written at the raise site; only the log is here.
    except RenderTimeout:
        logger.warning(
            "rendered fetch timed out reading the DOM of %s after %dms: the page kept navigating",
            urlparse(url).netloc,
            RENDER_DOM_READ_TIMEOUT_MS,
        )
        raise
    except PlaywrightError as exc:
        _warn_playwright_unavailable_once(exc)
        return None
    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except  # soft-fail boundary for the rendered rung: a raise here cancels the question's sibling pages (see the docstring); logged in full every time so it can never be silent
        logger.exception(
            "rendered fetch failed unexpectedly for %s; leaving the plain result standing", urlparse(url).netloc
        )
        return None


async def _render_in_browser(
    playwright: Any,
    url: str,
    playwright_error: type[BaseException],
    *,
    memo_scope: MemoScope,
    host: str,
    vetted_ip: str,
    goto_timeout_ms: int,
    deadline_monotonic_s: float | None,
    harvest_json: bool,
) -> RenderedPage:
    """Recompute the budget, launch, render inside a guarded context, tear both down bounded.

    Split out of :func:`render_page` so the gates it runs under, and the soft-fail boundary
    around them, read as one statement each there. The three teardown steps (the unroute inside
    the context, then the two closes here) share ONE :class:`_TeardownBudget`, so whatever the
    browser does the exit costs at most ``RENDER_TEARDOWN_TIMEOUT_MS`` before the driver stop.
    """
    teardown = _TeardownBudget()
    goto_budget_ms = _goto_budget_ms(goto_timeout_ms, deadline_monotonic_s)
    if goto_budget_ms is None:
        logger.warning(
            "rendered fetch declined after the gates for %s: under %dms of navigation budget left",
            urlparse(url).netloc,
            RENDER_MIN_GOTO_MS,
        )
        raise RenderBudgetExpired(
            f"under {RENDER_MIN_GOTO_MS}ms of navigation budget left for {urlparse(url).netloc} once the gates were held"
        )
    # --host-resolver-rules pins the browser's own resolution to the IP vetted above, so
    # Chromium's socket connect cannot independently re-resolve `host` to a private address
    # (the DNS-rebinding TOCTOU the per-request preflight alone cannot close). A fresh browser
    # is launched per call, so per-launch host-resolver-rules is clean.
    browser = await playwright.chromium.launch(headless=True, args=[_host_resolver_rule(host, vetted_ip)])
    # Two nested try/finally blocks, one per resource, so a failure at ANY step after the launch
    # — context creation, the route registration, the page, the navigation — still closes what
    # exists rather than leaving the driver stop as the only thing that kills Chromium.
    try:
        context = await browser.new_context(
            user_agent=BROWSER_HEADERS["User-Agent"],
            extra_http_headers={key: value for key, value in BROWSER_HEADERS.items() if key != "User-Agent"},
            # Service workers are the one request channel `browser_context.route` does not
            # intercept (Playwright 1.61 documents it and recommends exactly this whenever
            # interception is in use), so a worker could dial past the route guard below. Nothing
            # is lost: a fresh context has no installed worker, so no first-load content is ever
            # served from one.
            service_workers="block",
        )
        try:
            return await _render_in_context(
                context,
                url,
                playwright_error,
                memo_scope=memo_scope,
                goto_budget_ms=goto_budget_ms,
                deadline_monotonic_s=deadline_monotonic_s,
                harvest_json=harvest_json,
                teardown=teardown,
            )
        finally:
            await _teardown_step("context.close", context.close(), playwright_error, teardown)
    finally:
        await _teardown_step("browser.close", browser.close(), playwright_error, teardown)


async def _render_in_context(
    context: Any,
    url: str,
    playwright_error: type[BaseException],
    *,
    memo_scope: MemoScope,
    goto_budget_ms: int,
    deadline_monotonic_s: float | None,
    harvest_json: bool,
    teardown: _TeardownBudget,
) -> RenderedPage:
    """Guard the context, open the page, navigate and read; leave the closes to the caller."""

    # Defense-in-depth on top of the main-frame pin above. The route guard re-checks every HTTP
    # request Chromium makes (main-frame goto, server and client-side redirects, subresources)
    # against is_public_http_url. Two channels are outside it: service-worker traffic, which the
    # context was created with blocked, and WebSocket connections, which Playwright routes through a separate
    # `route_web_socket` API and which nothing here opens on the page's behalf — a page's own
    # WebSocket to a private address would be neither pinned (the pin covers one hostname, and an
    # IP-literal target skips the resolver) nor guarded, and is a documented residual alongside
    # the subresource one below. Threat model: these fetches run on GitHub-hosted Azure runners,
    # where a request to a link-local / RFC1918 host (Azure IMDS at 169.254.169.254, localhost
    # services, the internal runner network) would exfiltrate internal content into the research
    # prompt AND the public Metaculus comment. The main-frame host is pinned, so its rebinding
    # TOCTOU is closed; subresource / redirect hosts remain guarded only by this per-request
    # preflight (whose getaddrinfo resolves independently of Chromium's connect), so their
    # rebinding TOCTOU is a documented residual — a filtering forward proxy would close it,
    # deferred as its own change. Harvested JSON therefore also passes this guard: a harvestable
    # response is one Chromium was allowed to dial.
    async def _guard_route(route: Any, request: Any) -> None:
        # A thin closure so the registration keeps Playwright's expected handler shape while
        # the vetting itself stays module-level and directly testable.
        await _vet_route(route, request, playwright_error)

    await context.route("**/*", _guard_route)
    page = await context.new_page()
    harvest: _JsonHarvest | None = None
    if harvest_json:
        harvest = _JsonHarvest(page_host=urlparse(url).hostname or "", playwright_error=playwright_error)
        page.on("response", harvest.on_response)
    try:
        return await _navigate_and_read_dom(
            page,
            url,
            playwright_error,
            memo_scope=memo_scope,
            goto_timeout_ms=goto_budget_ms,
            deadline_monotonic_s=deadline_monotonic_s,
            harvest=harvest,
        )
    finally:
        # A body read still pending here (the DOM read timed out, or the DOM was over the
        # ceiling) would otherwise race the close below and raise into its own swallowed path.
        if harvest is not None:
            harvest.cancel_pending()
        # Drain in-flight route handlers BEFORE teardown. Without this, a request still in
        # flight when we close (common after a goto timeout) fires _guard_route against the
        # closing context and raises TargetClosedError in a detached event listener — the
        # unhandled traceback storm seen 2026-07-25 that buries real fetch failures in the
        # logs. unroute_all(ignoreErrors) removes the handlers and silently swallows any still
        # mid-flight (Playwright's own remedy for this exact message). SSRF is unaffected: the
        # guard already ran for every request dialed while the page was live, and a request
        # racing teardown has no live target to exfiltrate through. Bounded and guarded like the
        # two closes that follow it, on the same shared budget, so neither a race nor a wedged
        # browser can skip them or hold the render past one teardown bound.
        await _teardown_step("unroute_all", context.unroute_all(behavior="ignoreErrors"), playwright_error, teardown)
