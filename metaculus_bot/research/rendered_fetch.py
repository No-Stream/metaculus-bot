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
the caller (:func:`note_rendered_no_text`) and only READ here: only the caller knows whether
its own extraction found anything.

The SSRF guard is resolved through ``resolution_source`` at call time, from inside the two
functions that need it, and both halves of that are deliberate — see :func:`_resolve_pinned_host`.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from metaculus_bot.research.http_fetch import BROWSER_HEADERS

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
# Floor under a caller-derived goto budget. A caller bounding the render against its own wall
# can arrive with very little left; a token attempt is still worth making, because nothing
# downstream distinguishes "timed out at 0 s" from "timed out at 5 s" and a fast page is one we
# would otherwise refuse for free. Same argument as the Tier-1 hop timeout's own floor.
RENDER_MIN_GOTO_MS: int = 5_000

# Process-global cap on concurrent headless Chromium launches. Module-level, so the bound
# spans every question running under the orchestrator's Semaphore(6) AND both fetch paths:
# each Chromium is ~100-300MB, one v2 step's parallel_tool_calls can request many fetches, a
# Tier-1 question can escalate up to RESOLUTION_SOURCE_MAX_URLS pages, and an unbounded launch
# fan-out would OOM the runner (an escape try/except cannot catch that). Cap 2 covers real
# bursts of 1-3 rendered pages while bounding worst-case memory.
_RENDERED_FETCH_GLOBAL_SEMAPHORE = asyncio.Semaphore(2)

# URLs where Chromium ran and the caller's extraction found nothing, so a second launch on the
# same URL in the same run is skipped: 100-300 MB and a whole slot out of the cap above, to
# learn what this run already knows. Deliberately records ONLY that outcome — a `blocked` or
# `error` GET is never memoized, because 429 is retryable and a throttle interstitial has to
# stay re-requestable (the q45191 receipt).
_RENDERED_NO_TEXT_MAX_ENTRIES = 50
_RENDERED_NO_TEXT: OrderedDict[str, None] = OrderedDict()
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
# Hosts other than the page's own that may serve a harvestable dataset. Deliberately almost
# empty: an allow-list entry is a decision that a third-party host's JSON may be read as the
# cited page's content, and only Datawrapper's CDN has an already-trusted data route in this
# repo (the Tier-2 hop fetches from it by construction). Anything else waits for a measurement.
HARVEST_ALLOWED_CDN_HOSTS: frozenset[str] = frozenset({"static.dwcdn.net"})
_JSON_CONTENT_TYPE_TOKENS = ("application/json", "text/json", "+json")


@dataclass(frozen=True, slots=True)
class HarvestedJson:
    """One JSON response the rendered page fetched for itself."""

    url: str
    body: bytes


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """What a successful render hands back: the DOM, plus any JSON the page fetched.

    ``html`` is ``page.content()`` — the DOM as it stood after the settle, which is the
    whole point of the rung. ``content_type`` is the main-frame response's, empty when the
    goto timed out and the DOM was salvaged (no response object survives that path).
    """

    url: str
    content_type: str
    html: str
    json_responses: tuple[HarvestedJson, ...] = ()


def reset_render_state() -> None:
    """Drop the run-scoped render state: the no-text memo, the warn latch, the launch gate.

    For tests. The launch gate is REBOUND rather than drained because an ``asyncio.Semaphore``
    binds to the loop that first blocks on it and raises from any other, so a contended
    acquire in one test's event loop would otherwise leak that binding into the next.
    """
    global _PLAYWRIGHT_WARNED, _RENDERED_FETCH_GLOBAL_SEMAPHORE  # noqa: PLW0603  # run-scoped render state: the warn latch and the loop-bound launch gate
    _RENDERED_NO_TEXT.clear()
    _PLAYWRIGHT_WARNED = False
    _RENDERED_FETCH_GLOBAL_SEMAPHORE = asyncio.Semaphore(2)


def rendered_to_nothing(url: str) -> bool:
    """True when a browser already read ``url`` to nothing in this run."""
    return url in _RENDERED_NO_TEXT


def note_rendered_no_text(url: str) -> None:
    """Record that Chromium rendered ``url`` and the caller found no content in the result."""
    _RENDERED_NO_TEXT[url] = None
    _RENDERED_NO_TEXT.move_to_end(url)
    while len(_RENDERED_NO_TEXT) > _RENDERED_NO_TEXT_MAX_ENTRIES:
        _RENDERED_NO_TEXT.popitem(last=False)


def _warn_playwright_unavailable_once(exc: BaseException) -> None:
    global _PLAYWRIGHT_WARNED  # noqa: PLW0603  # one-shot process-wide warn latch so the rendered rung logs once per run
    if _PLAYWRIGHT_WARNED:
        return
    _PLAYWRIGHT_WARNED = True
    logger.warning("rendered fetch rung unavailable: %s: %s", type(exc).__name__, exc)


def _host_resolver_rule(host: str, ip: str) -> str:
    """Build the Chromium ``--host-resolver-rules`` MAP value pinning ``host`` to ``ip``.

    IPv6 literals must be bracketed in the MAP target per Chromium's rule parser
    (``MAP host [dead::beef]``); IPv4 literals are bare. A malformed ``ip`` is
    passed through unbracketed — callers only ever feed this a value already
    vetted by :func:`_resolve_pinned_host`, so that branch is defensive only.
    """
    try:
        parsed_ip = ipaddress.ip_address(ip)
    except ValueError:
        target = ip
    else:
        target = f"[{ip}]" if parsed_ip.version == 6 else ip
    return f"--host-resolver-rules=MAP {host} {target}"


def _pinnable_url_host(url: str) -> str | None:
    """Hostname of a URL eligible for DNS pinning, or None when the URL itself disqualifies it."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme.lower() not in ("http", "https"):
        return None
    # Userinfo defeats hostname-based trust (`https://trusted@10.0.0.1/`).
    if parsed.username is not None or parsed.password is not None:
        return None
    return parsed.hostname or None


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
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
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


async def _navigate_and_read_dom(
    page: Any, url: str, playwright_error: type[BaseException], *, goto_timeout_ms: int
) -> tuple[str, str]:
    """Navigate to ``url``, let it settle, and return ``(content_type, html)``.

    Two departures from a plain ``networkidle`` navigation, both measured 2026-09-03. The wait
    condition is ``domcontentloaded`` plus a fixed settle, because network idle never arrives on
    a page carrying a long-poll widget or an analytics beacon. And a goto failure is SALVAGED
    rather than treated as a dead rung: Playwright's ``TimeoutError`` subclasses ``Error``, and a
    timed-out goto routinely leaves a fully rendered DOM behind — 4 of the replay's 10 render
    rescues came from exactly that. A genuine navigation error lands here too and salvages an
    empty ``about:blank``, which reaches the ladder as the same "rendered read nothing".

    The DOM read itself is bounded by ``RENDER_DOM_READ_TIMEOUT_MS`` and raises the builtin
    ``TimeoutError`` when it fires — deliberately NOT swallowed into the salvage, because a page
    that keeps navigating has no DOM to salvage and the caller needs to tell this from a browser
    that is missing or broken (see :func:`render_page`).
    """
    try:
        response = await page.goto(url, wait_until="domcontentloaded", timeout=goto_timeout_ms)
    except playwright_error as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # the salvage above: Playwright's own Error class, passed in from the optional import
        logger.debug("rendered fetch goto failed, salvaging DOM: %s: %s", type(exc).__name__, exc)
        response = None
    await page.wait_for_timeout(RENDER_SETTLE_MS)
    content_type = (
        (response.headers.get("content-type") or "").lower()
        if response is not None and hasattr(response, "headers")
        else ""
    )
    html = await asyncio.wait_for(page.content(), timeout=RENDER_DOM_READ_TIMEOUT_MS / 1000)
    return content_type, html


def _harvestable_json_host(response_host: str, page_host: str) -> bool:
    """True when ``response_host`` may serve harvestable JSON for a page on ``page_host``.

    Same origin, either direction of the subdomain relation (a page on ``www.x.gov`` whose
    data endpoint is ``x.gov`` or ``api.x.gov`` is the ordinary shape), or an explicitly
    allow-listed CDN. Deliberately no public-suffix reasoning: the suffix cases that a
    two-label heuristic gets wrong (``a.co.uk`` vs ``b.co.uk``) are exactly the ones where a
    wrong answer reads a stranger's JSON as the cited page's content.
    """
    if not response_host or not page_host:
        return False
    if response_host == page_host:
        return True
    if response_host.endswith(f".{page_host}") or page_host.endswith(f".{response_host}"):
        return True
    return response_host in HARVEST_ALLOWED_CDN_HOSTS


def _is_json_content_type(content_type: str) -> bool:
    return any(token in content_type for token in _JSON_CONTENT_TYPE_TOKENS)


def _response_content_type(response: Any) -> str:
    headers = getattr(response, "headers", None)
    if not isinstance(headers, dict):
        return ""
    return (headers.get("content-type") or headers.get("Content-Type") or "").lower()


async def _harvest_json_response(
    response: Any, *, page_host: str, into: list[HarvestedJson], playwright_error: type[BaseException]
) -> None:
    """Record one JSON response the page fetched, if it clears every bound.

    Reads the body inside the response event so it is still available — after the page
    navigates or closes, Playwright discards it. Every rejection is silent: this is
    opportunistic discovery attached to a render whose real product is the DOM, and a body we
    could not read must never be able to fail the render.
    """
    if len(into) >= HARVEST_MAX_RESPONSES:
        return
    url = getattr(response, "url", "") or ""
    if not _harvestable_json_host(urlparse(url).hostname or "", page_host):
        return
    if not _is_json_content_type(_response_content_type(response)):
        return
    try:
        body = await response.body()
    except playwright_error as exc:
        logger.debug("rendered fetch could not read a harvested response body: %s", exc)
        return
    if not isinstance(body, bytes) or not (HARVEST_MIN_BODY_BYTES <= len(body) <= HARVEST_MAX_BODY_BYTES):
        return
    into.append(HarvestedJson(url=url, body=body))


async def render_page(
    url: str,
    *,
    host_gate: asyncio.Semaphore,
    goto_timeout_ms: int = RENDER_TIMEOUT_MS - RENDER_SETTLE_MS,
    harvest_json: bool = False,
) -> RenderedPage | None:
    """Render ``url`` in headless Chromium and return its DOM, or None when the rung is unavailable.

    ``None`` is the one graceful-failure signal both callers already handle, and it covers
    every way this rung can DECLINE: a URL a browser already read to nothing in this run,
    Playwright missing or broken, a host that cannot be pinned to a public IP, and any error
    out of the browser. A caller that wants to tell those apart reads
    :func:`rendered_to_nothing` and :func:`_resolve_pinned_host` itself.

    A render that ran and was CUT OFF is different, and raises the builtin ``TimeoutError``
    instead: the DOM read outlived ``RENDER_DOM_READ_TIMEOUT_MS`` because the page kept
    navigating. It is not folded into ``None`` because the two mean different things to a
    caller — a timeout says nothing about whether Chromium works, so it must neither trip the
    once-per-process unavailable warning nor be recorded as the browser being absent — and
    because the Tier-1 rung bounds this whole call against its own wall budget with the same
    exception, so one ``except TimeoutError`` there covers both bounds. Both callers memoise a
    timed-out URL for the run (:func:`note_rendered_no_text`), so a second question citing the
    same hostile page does not pay for it again.

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
    OUTSIDE the launch cap, in that order, exactly as it was before this module existed.

    ``harvest_json`` turns on recording the JSON the page fetches for itself. Off by default:
    the bodies are buffered alongside a 100-300 MB browser, and only the caller that has a
    use for a derived endpoint should pay that.
    """
    if rendered_to_nothing(url):
        logger.debug("rendered fetch skipped (already rendered to nothing): %s", urlparse(url).netloc)
        return None
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
                host=host,
                vetted_ip=vetted_ip,
                goto_timeout_ms=goto_timeout_ms,
                harvest_json=harvest_json,
            )
    except TimeoutError:
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
    host: str,
    vetted_ip: str,
    goto_timeout_ms: int,
    harvest_json: bool,
) -> RenderedPage:
    """Launch, guard, navigate, read. Split out of :func:`render_page` so the gates it runs
    under, and the soft-fail boundary around them, read as one statement each."""
    # --host-resolver-rules pins the browser's own resolution to the IP vetted above, so
    # Chromium's socket connect cannot independently re-resolve `host` to a private address
    # (the DNS-rebinding TOCTOU the per-request preflight alone cannot close). A fresh browser
    # is launched per call, so per-launch host-resolver-rules is clean.
    browser = await playwright.chromium.launch(headless=True, args=[_host_resolver_rule(host, vetted_ip)])
    context = await browser.new_context(
        user_agent=BROWSER_HEADERS["User-Agent"],
        extra_http_headers={key: value for key, value in BROWSER_HEADERS.items() if key != "User-Agent"},
    )

    # Defense-in-depth on top of the main-frame pin above. The route guard re-checks EVERY
    # request Chromium makes (main-frame goto, server and client-side redirects, subresources)
    # against is_public_http_url. Threat model: these fetches run on GitHub-hosted Azure
    # runners, where a request to a link-local / RFC1918 host (Azure IMDS at 169.254.169.254,
    # localhost services, the internal runner network) would exfiltrate internal content into
    # the research prompt AND the public Metaculus comment. The main-frame host is pinned, so
    # its rebinding TOCTOU is closed; subresource / redirect hosts remain guarded only by this
    # per-request preflight (whose getaddrinfo resolves independently of Chromium's connect),
    # so their rebinding TOCTOU is a documented residual — a filtering forward proxy would
    # close it, deferred as its own change. Harvested JSON therefore also passes this guard:
    # a harvestable response is one Chromium was allowed to dial.
    async def _guard_route(route: Any, request: Any) -> None:
        # A thin closure so the registration keeps Playwright's expected handler shape while
        # the vetting itself stays module-level and directly testable.
        await _vet_route(route, request, playwright_error)

    await context.route("**/*", _guard_route)
    page = await context.new_page()
    harvested: list[HarvestedJson] = []
    if harvest_json:
        page_host = urlparse(url).hostname or ""

        async def _on_response(response: Any) -> None:
            await _harvest_json_response(
                response, page_host=page_host, into=harvested, playwright_error=playwright_error
            )

        page.on("response", _on_response)
    try:
        content_type, html = await _navigate_and_read_dom(
            page, url, playwright_error, goto_timeout_ms=max(RENDER_MIN_GOTO_MS, goto_timeout_ms)
        )
        return RenderedPage(url=url, content_type=content_type, html=html, json_responses=tuple(harvested))
    finally:
        # Drain in-flight route handlers BEFORE teardown. Without this, a request still in
        # flight when we close (common after a goto timeout) fires _guard_route against the
        # closing context and raises TargetClosedError in a detached event listener — the
        # unhandled traceback storm seen 2026-07-25 that buries real fetch failures in the
        # logs. unroute_all(ignoreErrors) removes the handlers and silently swallows any still
        # mid-flight (Playwright's own remedy for this exact message). SSRF is unaffected: the
        # guard already ran for every request dialed while the page was live, and a request
        # racing teardown has no live target to exfiltrate through. Guarded so a teardown-race
        # error here can't skip context/browser close (leak).
        try:
            await context.unroute_all(behavior="ignoreErrors")
        except playwright_error as exc:
            logger.debug("rendered fetch unroute_all race: %s", exc)
        await context.close()
        await browser.close()
