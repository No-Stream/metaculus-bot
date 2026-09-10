"""The one classification path for a fetched body, whichever rung obtained it.

A response arrives here as bytes plus a content type and leaves as a :class:`FetchResult`:
the redirect and non-200 verdicts, the HTML extractor policy with its chrome and JS-wall
floors, the raw text and CSV branches, and the local document read. Sharing it is what makes
a rescued page indistinguishable downstream from a directly fetched one.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import aiohttp
import trafilatura

from metaculus_bot.constants import (
    DOCUMENT_DIGEST_TOP_K,
    DOCUMENT_TEXT_MAX_PAGES,
    DOCUMENT_TEXT_MAX_SECONDS,
    DOCUMENT_TEXT_PDF_MAX_BYTES,
    RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS,
    RESOLUTION_SOURCE_CONTENT_SHARE_MIN,
    RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS,
    RESOLUTION_SOURCE_JS_WALL_MIN_CHARS,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
    RESOLUTION_SOURCE_META_REFRESH_MIN_BUDGET_S,
    RESOLUTION_SOURCE_PDF_MIN_BUDGET_S,
    RESOLUTION_SOURCE_PER_URL_MAX_CHARS,
    RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S,
)
from metaculus_bot.research import resolution_presentation
from metaculus_bot.research.document_text import (
    DocumentDigest,
    PdfText,
    digest_pdf,
    extract_pdf_text,
    has_text_layer,
    is_pdf_body,
)
from metaculus_bot.research.fetch_ladder import context, guard
from metaculus_bot.research.http_fetch import (
    REDIRECT_STATUSES,
    decode_text_body,
    extract_datawrapper_charts,
    meta_refresh_target,
    pdf_parse_semaphore,
    read_body_capped,
    rewrite_aria_tables,
    unreadable_data_embed_providers,
)
from metaculus_bot.research.rendered_fetch import is_json_content_type
from metaculus_bot.research.resolution_body_text import _truncate_with_marker, strip_html_tags
from metaculus_bot.research.resolution_chart_data import render_inline_chart_data
from metaculus_bot.research.resolution_fetch_result import (
    _NON_OK_FETCH_STATUS,
    PDF_CONTENT_TYPES,
    FetchResult,
    FetchStatus,
    FetchStatusReason,
    http_failure_class,
    server_header_token,
    vacuous_body_status,
)

logger = logging.getLogger(__name__)
_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")
_RAW_TEXT_CONTENT_TYPES = ("text/plain", "text/csv")


def looks_like_js_wall(text: str) -> bool:
    """A 200 OK whose extracted text is shorter than the JS-wall threshold is a
    strong signal the page needs JS to render — Tier-2 candidate."""
    return len(text.strip()) < RESOLUTION_SOURCE_JS_WALL_MIN_CHARS


def looks_like_page_chrome(text: str) -> bool:
    """True when an extraction is too thin to be anything but chrome around the content.

    The floor is what the ``no_resolving_content`` verdict rests on; a named embed
    provider only says WHERE the content went (`embed_shell` vs `thin_page`). It
    was gated on a named provider when it shipped, which withheld one shape of
    chrome and published the other: the 2026-09-01 round's five content-free
    `success` renders named no provider between them.

    Calibration re-checked for the ungated rule against the same census (89
    `resolution_source` archive records, 68 cited successes, 2026-09-02): 8 sit
    under 400 chars and all 8 are chrome — a 127-char SPA tab list
    (data.wastewaterscan.org, twice), 385 chars of Kazakh region names
    (election.gov.kz), AP's org boilerplate (355), an ABS release-date list with no
    figure (344), a mass-shooting tracker's "about the data" note (262), a
    Portuguese feedback-form blurb (157), and a clinicaltrials.gov data-element
    pointer (111). The shortest carrying the resolving content is still exactly
    401 (myfloridaelections.com's election-date table), so 400 remains the observed
    elbow and the floor stays deliberately below it: a page above it keeps its text
    and, where an embed hid figures, gets the disclosure note instead.

    Trafilatura's own precision filter drops most embed credit blocks ("Created
    with Infogram" and friends), so the char floor carries this on its own and no
    boilerplate-pattern list is needed.
    """
    return len(text.strip()) < RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS


def content_share(text: str) -> float:
    """Share of an extraction's characters that sit in content-shaped lines.

    Content is table rows (lines starting with ``|``, whatever their length: a price-history
    table is rows of 10-char cells) and lines of at least
    ``RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS``; every other line is chrome-shaped. Lines are
    stripped and blank ones dropped first. One pass over the extracted text, no second parse.

    A line-shape rule separates navigation trees from content and nothing more: prose-shaped
    boilerplate (a cookie-consent wall, a glossary) is sentences and passes, and a news ticker
    made of short headlines is withheld with the menu around it. Both are the deliberate
    trade; the calibration numbers sit on ``RESOLUTION_SOURCE_CONTENT_SHARE_MIN``.
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    total = sum(len(line) for line in lines)
    if total == 0:
        return 0.0
    content = sum(
        len(line) for line in lines if line.startswith("|") or len(line) >= RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS
    )
    return content / total


# Extraction wrapper (isolated for tests; offloads trafilatura's sync API)
# ---------------------------------------------------------------------------


def _extract_main_text(body: bytes | str, url: str, *, favor_precision: bool = False) -> str | None:
    """Trafilatura extraction. Callers wrap in ``await asyncio.to_thread(...)``.

    Takes bytes (the response body, letting trafilatura detect the encoding) or text
    (a body this module already decoded and rewrote — see :func:`_extract_page_text`).

    Default recall is the primary extraction and ``favor_precision=True`` the fallback, under
    the policy :func:`_extract_page_text` applies, because each setting alone loses pages the
    other reads. Precision alone withholds readable pages (kasa.go.kr pruned to 78 chars, two
    tracxn funding tables, manifold's market body). Default alone publishes chrome: on
    congress.gov it swaps the 2,411-char bill-status card for 54,393 chars of a member-name
    dropdown (trafilatura's readability fallback replaces the main extraction when
    readability's text is over twice as long, and only precision prunes the dropdown out of
    that backup tree first), and menu trees (abs.gov.au, kasa.go.kr) clear the chrome floor as
    `success` — and character count, the metric that once picked it, is the wrong one under a
    head-preserving cap. The receipt for running both is the 2026-09-03 calibration
    (`scratch/fetch_ladder_2026-09-03/chrome_calibration.md`: 118 bodies, five extractor
    variants on identical bytes, texts labelled by hand). ``include_comments=False`` stays
    at both settings.

    Returns None on empty/failed extraction so callers can classify.
    """
    try:
        out = trafilatura.extract(
            body,
            url=url,
            include_comments=False,
            include_tables=True,
            output_format="txt",
            favor_precision=favor_precision,
        )
    except (ValueError, TypeError, RuntimeError) as e:
        # Trafilatura occasionally raises on truly malformed input. We soft-fail
        # here so a single broken page doesn't take down the provider.
        logger.warning(f"trafilatura extraction failed for {url}: {e}")
        return None
    if not out or not out.strip():
        return None
    return out


async def _resolution_redirect_outcome(resp: Any, current_url: str, content_type: str) -> FetchResult | str:
    """Vet a 3xx hop: the next URL to follow, or a terminal error/blocked result."""
    status = resp.status
    location = resp.headers.get("Location")
    if not location:
        # Malformed redirect — no Location header.
        logger.info(f"resolution_source {urlparse(current_url).netloc}: {status} redirect with no Location header")
        return FetchResult(
            url=current_url,
            status="error",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    return await guard._vetted_hop_target(
        location, current_url, http_status=status, content_type=content_type, kind="redirect"
    )


def _network_failure_class(exc: BaseException) -> str:
    """Bucket a transport exception for the fetch marker's ``failure_class``.

    The specific subclasses come FIRST because aiohttp's TLS and DNS connector errors both
    subclass ``ClientConnectorError``, so the general connection bucket would otherwise swallow
    them — and the whole point of the field is to tell a host that refused our TLS from one our
    egress IP could not resolve. ``exc`` on the same line keeps the exact class name for anything
    this coarse vocabulary lumps together.

    ``malformed_response`` is a response aiohttp's parser refused before the body was ours: a
    ``Content-Encoding`` it cannot decode (the trueup.io zstd failure that had the brotli and zstd
    decoders added, 2026-09-03), a header past the session's size caps, a bad status line. The
    parser raises those as ``HttpProcessingError`` and the client re-raises them as
    ``ClientResponseError(status=400)``, a SIBLING of ``ClientPayloadError`` under ``ClientError``
    rather than a subclass, so ``decode`` cannot claim them and ``connection`` used to. Its own
    token rather than a wider ``decode`` because the two say different things: ``decode`` is a
    body that arrived and could not be read, this is a response that never got that far. Nothing
    on this path calls ``raise_for_status`` or follows redirects through aiohttp, so a
    ``ClientResponseError`` here is always the parser's.
    """
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(
        exc, aiohttp.ClientConnectorCertificateError | aiohttp.ClientConnectorSSLError | aiohttp.ClientSSLError
    ):
        return "tls"
    if isinstance(exc, aiohttp.ClientConnectorDNSError):
        return "dns"
    if isinstance(exc, aiohttp.ClientPayloadError):
        return "decode"
    if isinstance(exc, aiohttp.ClientResponseError):
        return "malformed_response"
    return "connection"


def _resolution_status_outcome(
    status: int, current_url: str, content_type: str, *, server: str | None = None
) -> FetchResult | None:
    """Terminal result for a non-200 status, or None when the body should be read."""
    if status == 200:
        return None
    fetch_status = _NON_OK_FETCH_STATUS.get(status, "error")
    return FetchResult(
        url=current_url,
        status=fetch_status,
        text="",
        http_status=status,
        content_type=content_type or None,
        failure_class=http_failure_class(status),
        server=server_header_token(server),
    )


@dataclass(frozen=True, slots=True)
class _PageExtraction:
    """What the extractor policy decided for one HTML body (:func:`_extract_page_text`).

    ``text`` is what the classifier sees: the precision re-extraction when it rescued the
    page, else the default one (None when nothing extracted). ``chrome_metric_withheld``
    marks a default extraction that cleared the chrome floor and failed the line-shape
    metric with no rescue, so the classifier withholds that text (the page itself still
    publishes when a chart block carries its numbers); ``precision_rescued`` marks a text
    that came from the fallback. Both ride the result into ``details["counts"]``.
    """

    text: str | None
    chrome_metric_withheld: bool = False
    precision_rescued: bool = False


def _extract_page_text(
    html_text: str, body: bytes, url: str, undecodable_ratio: float, *, remaining_wall_s: float | None = None
) -> _PageExtraction:
    """The publishable extraction of an HTML body: ARIA tables rewritten first, default
    recall as the primary extractor, precision as the fallback, both scored by line shape.

    All of it is CPU-bound sync work over a body up to the response cap, so it runs in one
    ``asyncio.to_thread`` hop rather than several.

    The precision pass is the one part of it that is skippable, and it is skipped when
    ``remaining_wall_s`` — the provider wall the caller had left when it handed the body over,
    less what the default pass has since spent — is under
    ``RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S``. Nothing budgets this work otherwise: the
    rendered rung gives the browser its whole remaining budget and classifies the DOM afterwards,
    and a 5 MiB navigation tree costs seconds per pass against the 2 s margin the rung leaves the
    outer ``wait_for``, which discards every page the question already fetched when it fires. A
    skipped pass takes the exit a FAILED pass takes, the default text withheld under the metric,
    so the wall can only ever withhold a page here, never publish one the metric refused. None,
    the default, means unbounded, which is what the direct-path tests drive it with.

    The policy, calibrated 2026-09-03 (receipt on ``RESOLUTION_SOURCE_CONTENT_SHARE_MIN``):
    the default extraction publishes when it clears the chrome floor and
    :func:`content_share` is at least the threshold. When it clears the floor on chrome
    alone, the same input is re-extracted under ``favor_precision=True``, which is the one
    setting that prunes the tree readability's fallback swapped in, and that text publishes
    only if it clears the floor AND the metric. Otherwise the page is chrome and the
    classifier withholds it under ``thin_page``, so the rendered rung still fires
    (uk.finance.yahoo's direct body is a menu plus one quote line; its render is the whole
    price table). On the calibration corpus this publishes every labelled content text (46
    of 46, three of them the congress.gov status card the default alone loses) and blocks the
    navigation-tree chrome (abs, ocearch, portwatch, copernicus, the congress dropdown), at
    the cost of one extra trafilatura pass on the pages that fail the metric. What it gives
    up: prose-shaped boilerplate (a cookie-consent wall, a glossary) passes any line-shape
    metric, and kasa.go.kr's news ticker is withheld with its menu. An extraction under the
    floor skips the metric, because precision only ever shortens.

    Trafilatura gets the ORIGINAL BYTES in two cases, and in both its extraction is
    byte-identical to what it was before this rung existed: a page with no ARIA role at
    all, and a page our own decode mangled. The second is the one that matters —
    ``decode_text_body`` honours a BOM and the HTTP header's ``charset``, but a page that
    declares its encoding only in a ``<meta charset>`` decodes as UTF-8 here and comes
    back as mojibake, while trafilatura reading the bytes would have found the meta
    declaration. Handing it the rewritten mojibake instead would lose a page we can read
    today, so the rewrite is trusted ONLY on a body that decoded cleanly.

    That is why the gate is ``== 0.0`` and not ``MAX_UNDECODABLE_CHAR_RATIO``: the shared
    bound is the refuse-the-whole-body threshold and is far too loose for this decision. A
    mostly-ASCII cp1252 page whose only non-UTF-8 bytes are accented characters scores
    around 0.01 against a bound of 0.10, so under the old gate it took the rewrite and
    reached forecasters as ``R<?>sum<?> ... Qu<?>bec`` where the bytes path returns the
    accents. Any U+FFFD at all means this decode lost information trafilatura might not
    have, and the rewrite is the only thing that forecloses its own encoding detection.
    """
    started = time.monotonic()
    rewritten = rewrite_aria_tables(html_text) if undecodable_ratio == 0.0 else None
    source = body if rewritten is None else rewritten
    default = _extract_main_text(source, url)
    if (
        default is None
        or looks_like_page_chrome(default)
        or content_share(default) >= RESOLUTION_SOURCE_CONTENT_SHARE_MIN
    ):
        return _PageExtraction(text=default)
    if remaining_wall_s is not None:
        wall_left_s = remaining_wall_s - (time.monotonic() - started)
        if wall_left_s < RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S:
            logger.info(
                "resolution_source: skipping the precision re-extraction for %s — %.1fs of wall budget left; "
                "withholding the default text",
                urlparse(url).netloc,
                wall_left_s,
            )
            return _PageExtraction(text=default, chrome_metric_withheld=True)
    precision = _extract_main_text(source, url, favor_precision=True)
    if (
        precision is not None
        and not looks_like_page_chrome(precision)
        and content_share(precision) >= RESOLUTION_SOURCE_CONTENT_SHARE_MIN
    ):
        return _PageExtraction(text=precision, precision_rescued=True)
    return _PageExtraction(text=default, chrome_metric_withheld=True)


def _no_content_verdict(
    extracted: str | None, unreadable_embeds: list[str]
) -> tuple[FetchStatus, FetchStatusReason | None]:
    """Which withhold a 200 with no readable content earns, and why.

    Order is load-bearing and unchanged since the chrome floor generalised: a named
    routeless embed is the most specific thing we can say (`embed_shell` — the numbers
    exist and we have no route to them), the JS-wall floor keeps its own much lower
    threshold and its position in the middle so the chrome floor cannot swallow that
    population, and `thin_page` is everything else: under the floor, or over it on chrome
    alone (the line-shape metric in :func:`_extract_page_text`).
    """
    if unreadable_embeds:
        # Datawrapper is exempt from the embed scan (it has the Tier-2 hop), so a
        # walled tracker still comes back `js_wall` below and still hops.
        return "no_resolving_content", "embed_shell"
    if extracted is None or looks_like_js_wall(extracted):
        # An empty extraction on a 200 OK is a JS-wall (SPA that rendered client-side,
        # cookie/consent gate, etc.) — exactly the Tier-2 candidate signal. Treated
        # identically to short-but-nonempty extractions.
        return "js_wall", None
    return "no_resolving_content", "thin_page"


async def _meta_refresh_hop(
    html_text: str,
    current_url: str,
    ctx: context.LadderContext,
    *,
    from_status: FetchStatus,
    http_status: int,
    content_type: str,
) -> FetchResult | str | None:
    """Follow a ``<meta http-equiv="refresh">`` stub, or None when there is nothing to follow.

    A hop rather than a terminal result on purpose: the target re-enters the same
    classification path (chrome floor, JS-wall floor, chart rung, PDF read) and consumes
    one of ``MAX_REDIRECTS``, so a refresh chain is bounded exactly like a 3xx chain and
    the meta-refresh check itself works on a body only a later hop could obtain.

    Only reached with no readable content, which is what keeps it off the pages that
    already worked: a real page that ALSO carries a refresh tag (some CMSs emit one for
    a canonical URL) is served as-is rather than re-fetched.
    """
    target = meta_refresh_target(html_text)
    if target is None:
        return None
    if (
        ctx.claim_rung_budget("meta_refresh", from_status, current_url, RESOLUTION_SOURCE_META_REFRESH_MIN_BUDGET_S)
        is None
    ):
        return None
    ctx.start_rung("meta_refresh", from_status, current_url)
    logger.info(
        f"resolution_source meta_refresh: {urlparse(current_url).netloc} -> {target} (direct read was {from_status})"
    )
    return await guard._vetted_hop_target(
        target, current_url, http_status=http_status, content_type=content_type, kind="meta_refresh"
    )


@dataclass(frozen=True, slots=True)
class _HtmlClassification:
    """One classified HTML body, plus the decoded text the meta-refresh rung still needs.

    ``html_text`` rides along because the two callers want different things from the same
    decode: :func:`_resolution_html_outcome` looks for a refresh stub in it, while the
    rendered rung has already followed every hop a browser follows and only wants the verdict.
    Decoding twice would double the CPU on a body up to the 5 MiB response cap, or a rendered
    DOM up to ``RENDERED_DOM_MAX_CHARS`` (sized to it).
    """

    result: FetchResult
    html_text: str


async def _classify_html_body(
    body: bytes, current_url: str, content_type: str, *, http_status: int, remaining_wall_s: float | None = None
) -> _HtmlClassification:
    """Trafilatura extraction plus the inline-chart rung and the chrome / JS-wall checks.

    The ONE classification path for an HTML body, whichever rung obtained it: the direct
    fetch, a meta-refresh hop, or a headless-Chromium render. That is what makes a rescued
    page indistinguishable from a directly-fetched one downstream — same chart read, same ARIA
    rewrite, same floors, same disclosure leads.

    ``remaining_wall_s`` is the provider wall the caller has left (``LadderContext.rung_budget_s``),
    handed through to :func:`_extract_page_text` so its optional second pass can decline under
    the floor; both production callers pass it, and None keeps the pass unbounded.

    Order of the three verdicts, and why:

    1. CONTENT is extracted text that clears the chrome floor and the line-shape metric
       (:func:`_extract_page_text`, which retries under precision when the default
       extraction is chrome-shaped) OR chart data read out of the raw HTML. The chart
       rung runs on every HTML page, not only thin ones, because q43949's page
       extracted ~80k chars of prose with none of the resolving figures in it — a
       thin-only gate would miss the record the rung exists for.
    2. With no content, a named routeless embed makes it `embed_shell`, an
       extraction under the JS-wall floor makes it `js_wall`, and anything else,
       under the chrome floor or over it on short lines alone, makes it `thin_page`.
       The `js_wall` check keeps its exact old meaning and its position between the
       two, so the generalised chrome floor cannot swallow the JS-wall population.
    3. Chart data therefore rescues a page the chrome floor would have withheld —
       including a JS-walled one, where the config in the raw HTML is precisely the
       data the wall was hiding. That is the one place the `js_wall` outcome moves,
       and it moves only when we actually recovered the numbers. A body the line-shape
       metric withheld does not ride along under the chart block: the metric's verdict is
       that the text is chrome, the same text is withheld one branch up when no chart
       block is present, and published it filled the per-URL cap with up to 6,000 chars
       of navigation. The chart block publishes alone, and the withhold is still counted.
       The under-floor rider (`looks_like_page_chrome`, under 400 chars) is unchanged.
    """
    # Both embed scans are only possible on the RAW HTML —
    # trafilatura drops iframes and embed scripts at every
    # setting — so they run on the raw body, before (and
    # regardless of) main-text extraction. Decoded through the
    # shared helper so a BOM'd / non-UTF-8 page's embeds are
    # still findable; the page's main text is trafilatura's to
    # decode, which is why no vacuity check runs on this branch
    # (a thin extraction is classified below instead).
    html_text, undecodable_ratio = decode_text_body(body, content_type)
    charts = extract_datawrapper_charts(html_text)
    unreadable_embeds = unreadable_data_embed_providers(html_text)
    extraction = await asyncio.to_thread(
        _extract_page_text, html_text, body, current_url, undecodable_ratio, remaining_wall_s=remaining_wall_s
    )
    extracted = extraction.text
    # In a thread for the same reason the extraction is: it is sync CPU work (one
    # regex sweep plus a `json.loads` per config) over a body up to the 5 MiB
    # response cap — or a rendered DOM up to RENDERED_DOM_MAX_CHARS, the browser rung's
    # ceiling — and blocking the loop here would stall every sibling fetch.
    # Measured 22 ms on the 1.1 MB q43949 page, but the bound is the page, not that
    # sample. The Datawrapper / embed scans above are single regex searches and stay
    # inline.
    chart_block = await asyncio.to_thread(render_inline_chart_data, html_text)
    if (extraction.chrome_metric_withheld or looks_like_page_chrome(extracted or "")) and not chart_block:
        # No content anywhere. Which of the three withholds applies is a disclosure
        # question, not a routing one — all three retain the result as the escalation
        # seam and none of them render. A walled page still exposes its
        # embeds, so the charts ride along on every one of them.
        verdict, reason = _no_content_verdict(extracted, unreadable_embeds)
        return _HtmlClassification(
            result=FetchResult(
                url=current_url,
                status=verdict,
                text="",
                http_status=http_status,
                content_type=content_type or None,
                status_reason=reason,
                datawrapper_charts=charts,
                unreadable_embeds=unreadable_embeds,
                chrome_metric_withheld=extraction.chrome_metric_withheld,
            ),
            html_text=html_text,
        )
    # Reachable only with a non-empty chart block, so a blank body still renders the lead alone
    # (`resolution_presentation._lead_then_capped_body`) and the blank-success guard on `FetchResult` cannot trip.
    published_text = "" if extraction.chrome_metric_withheld else (extracted or "")
    return _HtmlClassification(
        result=FetchResult(
            url=current_url,
            status="success",
            text=resolution_presentation._page_text_with_leads(
                published_text, current_url, unreadable_embeds, chart_block
            ),
            http_status=http_status,
            content_type=content_type or None,
            datawrapper_charts=charts,
            unreadable_embeds=unreadable_embeds,
            chrome_metric_withheld=extraction.chrome_metric_withheld,
            precision_rescued=extraction.precision_rescued,
        ),
        html_text=html_text,
    )


async def _resolution_html_outcome(
    resp: Any, current_url: str, content_type: str, ctx: context.LadderContext
) -> FetchResult | str:
    """Classify the HTML body, then let the meta-refresh rung look for a hop no status announced.

    Only once there is no content anywhere does the meta-refresh rung run. It returns the
    target as the next hop, so this function's return type is ``FetchResult | str`` exactly
    like the redirect dispatcher's, and a refresh chain is bounded by the same
    ``MAX_REDIRECTS`` cap with the same per-hop SSRF re-guard.
    """
    status = resp.status
    netloc = urlparse(current_url).netloc
    body = await read_body_capped(
        resp,
        max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
        label=f"resolution_source {netloc}",
    )
    if body is None:
        return FetchResult(
            url=current_url,
            status="error",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    classified = await _classify_html_body(
        body, current_url, content_type, http_status=status, remaining_wall_s=ctx.rung_budget_s()
    )
    if classified.result.status == "success":
        return classified.result
    hop = await _meta_refresh_hop(
        classified.html_text,
        current_url,
        ctx,
        from_status=classified.result.status,
        http_status=status,
        content_type=content_type,
    )
    if hop is not None:
        return hop
    return classified.result


async def _resolution_text_outcome(resp: Any, current_url: str, content_type: str) -> FetchResult:
    """Capped raw body for a JSON / plain-text / CSV response, refusing a vacuous one."""
    status = resp.status
    netloc = urlparse(current_url).netloc
    body = await read_body_capped(
        resp,
        max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
        label=f"resolution_source {netloc}",
    )
    if body is None:
        return FetchResult(
            url=current_url,
            status="error",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    return _raw_body_outcome(body, current_url, content_type, http_status=status)


def _raw_body_outcome(body: bytes, current_url: str, content_type: str, *, http_status: int) -> FetchResult:
    """Classify a raw JSON / plain-text / CSV body we already hold: the one copy of the rule.

    The bytes-level tail of :func:`_resolution_text_outcome`, split from the read so the
    impersonated retry's body (:func:`_impersonated_body_outcome`) goes through the same
    charset-honouring decode, the same markup strip and the same vacuity refusal as a directly
    fetched one, rather than a second partial copy of them.
    """
    netloc = urlparse(current_url).netloc
    raw, undecodable_ratio = decode_text_body(body, content_type)
    # Markup stripping on the text branches only: a CSV or
    # plain-text body carrying `<a href=…>` per row spends the
    # per-URL budget on tags (see `strip_html_tags`), while a
    # JSON body's angle brackets sit inside string values that
    # are the data. Both text types get it because the labels
    # are demonstrably unreliable here — Datawrapper's own
    # versioned route serves CSV as application/octet-stream.
    if any(ct in content_type for ct in _RAW_TEXT_CONTENT_TYPES):
        raw = strip_html_tags(raw)
    vacuous = vacuous_body_status(raw, undecodable_ratio, require_csv_rows=False)
    if vacuous is not None:
        # Reason line, not an outcome line: the marker carries the status, this
        # carries the body size and decode score that explain it.
        logger.info(
            f"resolution_source {netloc}: 200 body carries no usable content "
            f"({vacuous}, {len(body)} bytes, undecodable={undecodable_ratio:.2f})"
        )
        return FetchResult(
            url=current_url,
            status=vacuous,
            text="",
            http_status=http_status,
            content_type=content_type or None,
        )
    return FetchResult(
        url=current_url,
        status="success",
        text=_truncate_with_marker(raw, RESOLUTION_SOURCE_PER_URL_MAX_CHARS, current_url),
        http_status=http_status,
        content_type=content_type or None,
    )


def _pdf_unreadable_reason(pdf: PdfText) -> FetchStatusReason:
    """Why a document we read the bytes of yielded no text.

    ``encrypted`` / ``malformed`` come from the parse; ``no_text_layer`` is a document
    that parsed fine and carries images instead of text, which is the ONE shape a paid
    document read could still rescue.
    """
    if pdf.unreadable_reason == "encrypted":
        return "encrypted"
    if pdf.unreadable_reason == "malformed":
        return "malformed"
    return "no_text_layer"


@dataclass(frozen=True, slots=True)
class _PendingDocument:
    """A PDF whose bytes we hold and whose parse has not started yet.

    Exists so the parse happens OUTSIDE the per-host politeness semaphore. That map is
    loop-wide, so a 20 s parse held inside it blocked every other concurrent question's
    fetch of any URL on that host — and this population is concentrated on a handful of
    government hosts, so same-host collisions across questions in one round are the
    expected case. Two questions queued behind one parse of a shared host exhaust their own
    ``RESOLUTION_SOURCE_WALL_TIMEOUT``, and the outer ``wait_for`` then discards every page
    they had already fetched.
    """

    url: str
    body: bytes
    http_status: int
    content_type: str
    from_status: FetchStatus


def _parse_and_digest(
    body: bytes, *, max_seconds: float, query: str, source_url: str
) -> tuple[PdfText, DocumentDigest | None]:
    """pypdf parse plus BM25 passage selection: both CPU-bound, so ONE thread hop, never two.

    The digest is as CPU-bound as the parse and was running inline on the loop two lines
    below a call carefully threaded for exactly that reason: ``select_passages`` tokenises
    every window of the joined document and holds a ``Counter`` per window alive at once,
    measured at 96-235 ms per 400-page document and additive across the six concurrent
    questions — a stall that lands inside the 2 s ``RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S``
    and delays every sibling provider's I/O, not just this fetch.

    ``None`` for the digest means the document carried no text layer, which the caller
    reports rather than digests.
    """
    pdf = extract_pdf_text(body, max_pages=DOCUMENT_TEXT_MAX_PAGES, max_seconds=max_seconds)
    if not has_text_layer(pdf):
        return pdf, None
    return pdf, digest_pdf(
        pdf,
        query=query,
        top_k=DOCUMENT_DIGEST_TOP_K,
        max_chars=RESOLUTION_SOURCE_PER_URL_MAX_CHARS,
        source_url=source_url,
    )


async def _resolution_pdf_outcome(
    resp: Any, current_url: str, content_type: str, ctx: context.LadderContext, *, from_status: FetchStatus
) -> FetchResult | _PendingDocument:
    """Read a PDF we are already holding, locally, and render the query-relevant passages.

    Free and deterministic: pypdf plus BM25 passage selection (``research/document_text``),
    no model call and no second request. Before this rung a cited PDF was the one
    resolution source we dropped unread — measured at 833,450 chars in 5.3 s out of the
    6.7 MB 220-page document behind the constants, with the passage the reader wanted in
    it, while the paid alternative returned nothing for the same file.

    Byte cap depends on whether the server DECLARED a PDF. A declared one gets
    ``DOCUMENT_TEXT_PDF_MAX_BYTES``, not the 5 MiB response cap the text branches use,
    because the receipt file is 6.7 MB and the general cap would refuse exactly the
    document that motivated the rung. An UNDECLARED body — the sniffed case — keeps the
    5 MiB cap: it is far more likely to be an image or an archive than a document, and
    buffering 40 MiB of it per URL across every concurrent question is a memory cost
    with nothing on the other side. An undeclared PDF above 5 MiB is therefore still
    lost, which is a deliberate trade rather than an oversight.

    Self-bounding twice over. The parse is skipped outright below
    ``RESOLUTION_SOURCE_PDF_MIN_BUDGET_S`` of remaining wall (the bytes are still read —
    that already happened — but the CPU is not spent), and ``max_seconds`` is the
    remaining budget capped at ``DOCUMENT_TEXT_MAX_SECONDS``, so a 900-page document
    comes back partial-and-labelled rather than taking the outer wall down with every
    sibling page that already succeeded.

    This half runs INSIDE the response context, so it does only what needs the open
    response: the capped read, then :func:`_document_outcome` for the ``%PDF-`` check and the
    budget-floor skip. A real document comes back as a :class:`_PendingDocument` and
    :func:`_finish_document` parses it once the host semaphore has been released.
    """
    status = resp.status
    netloc = urlparse(current_url).netloc
    declared_pdf = any(ct in content_type for ct in PDF_CONTENT_TYPES)
    body = await read_body_capped(
        resp,
        max_bytes=DOCUMENT_TEXT_PDF_MAX_BYTES if declared_pdf else RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
        label=f"resolution_source pdf {netloc}",
    )
    if body is None:
        return FetchResult(
            url=current_url,
            status="error",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    return _document_outcome(body, current_url, content_type, ctx, http_status=status, from_status=from_status)


def _document_outcome(
    body: bytes,
    current_url: str,
    content_type: str,
    ctx: context.LadderContext,
    *,
    http_status: int,
    from_status: FetchStatus,
) -> FetchResult | _PendingDocument:
    """Hold a document body we already read, or refuse a body that is not one: the one copy of the rule.

    The bytes-level tail of :func:`_resolution_pdf_outcome`: the ``%PDF-`` magic check, the
    :class:`_PendingDocument` construction and the ``pdf_local`` budget gate, split from the read
    so the impersonated retry's body (:func:`_impersonated_body_outcome`) goes through the same
    rule as a directly fetched one.
    """
    netloc = urlparse(current_url).netloc
    if not is_pdf_body(body):
        # Declared a PDF and is not one (or carried no content type and is not one):
        # unchanged behaviour, minus the assumption that the label was right.
        logger.info(f"resolution_source {netloc}: body is not a document we can read, ct={content_type!r}")
        return FetchResult(
            url=current_url,
            status="unsupported_type",
            text="",
            http_status=http_status,
            content_type=content_type or None,
        )
    pending = _PendingDocument(
        url=current_url,
        body=body,
        http_status=http_status,
        content_type=content_type,
        from_status=from_status,
    )
    # Checked here, before the response context closes, so a question with no budget left never
    # even queues for a parse slot it would have to give back.
    if ctx.claim_rung_budget("pdf_local", from_status, current_url, RESOLUTION_SOURCE_PDF_MIN_BUDGET_S) is None:
        return _document_not_parsed(pending, "budget_skipped")
    return pending


async def _finish_document(pending: _PendingDocument, ctx: context.LadderContext) -> FetchResult:
    """Parse a held PDF and render its digest, with no host semaphore and no response held.

    Runs after :func:`_fetch_one_hop` has left both the ``session.get`` context and the
    per-host gate, which is the whole point: the parse is up to
    ``min(DOCUMENT_TEXT_MAX_SECONDS, budget)`` of CPU, and holding a loop-wide
    ``Semaphore(1)`` for a host through it stalls every other concurrent question's fetch of
    that host (see :class:`_PendingDocument`).

    The parse contends instead for :func:`http_fetch.pdf_parse_semaphore`, the loop-wide
    2-slot gate this route shares with the gap-fill v2 local-document ladder — the bound has
    to hold across the two routes, not inside each, because a Tier-1 fan-out alone is up to
    ``RESOLUTION_SOURCE_MAX_URLS`` documents per question across
    ``DEFAULT_MAX_CONCURRENT_RESEARCH`` questions. The wait is bounded by the remaining
    budget less the floor and degrades to the same leave-it-unread skip, since queueing until
    the outer wall fires would discard every sibling page that already succeeded.

    Never raises: ``extract_pdf_text`` returns a ``PdfText`` carrying ``unreadable_reason``
    rather than throwing, and the digest is pure.
    """
    netloc = urlparse(pending.url).netloc
    gate = pdf_parse_semaphore()
    budget_s = ctx.rung_budget_s()
    try:
        # Bounded, not a bare acquire: queueing behind two other documents until the outer
        # wall fires would discard every sibling page this question already fetched, which
        # costs strictly more than leaving one document unread. Leaving the floor unspent
        # means a slot won at the last moment still has time to parse something.
        await asyncio.wait_for(gate.acquire(), timeout=max(0.0, budget_s - RESOLUTION_SOURCE_PDF_MIN_BUDGET_S))
    except TimeoutError:
        logger.warning(
            "resolution_source: skipping the local PDF read for %s — no parse slot within %.1fs of wall budget",
            netloc,
            budget_s,
        )
        ctx.skip_rung("pdf_local", pending.from_status, pending.url, "parse_contention")
        return _document_not_parsed(pending, "parse_contention")
    try:
        # Re-read after the wait: the queue itself consumed budget, and `max_seconds` is
        # wall-clock, so a stale figure would hand pypdf a bound that already expired.
        budget_s = ctx.claim_rung_budget(
            "pdf_local", pending.from_status, pending.url, RESOLUTION_SOURCE_PDF_MIN_BUDGET_S, note=" after queueing"
        )
        if budget_s is None:
            return _document_not_parsed(pending, "budget_skipped")
        attempt = ctx.start_rung("pdf_local", pending.from_status, pending.url)
        pdf, digest = await asyncio.to_thread(
            _parse_and_digest,
            pending.body,
            max_seconds=min(DOCUMENT_TEXT_MAX_SECONDS, budget_s),
            query=ctx.query,
            source_url=pending.url,
        )
        # Stamped inside the gate so wall_s measures the parse this rung actually did, not
        # the time it spent queueing for a slot.
        attempt.wall_s = max(0.0, time.monotonic() - attempt.started_at)
    finally:
        gate.release()
    if digest is None:
        reason = _pdf_unreadable_reason(pdf)
        logger.warning(
            f"resolution_source {netloc}: PDF carried no readable text ({reason}, "
            f"{pdf.page_count} pages, {pdf.pages_read} read)"
        )
        return FetchResult(
            url=pending.url,
            status="unreadable_document",
            text="",
            http_status=pending.http_status,
            content_type=pending.content_type or None,
            status_reason=reason,
        )
    if not digest.passages:
        # A document we read END TO END that does not discuss the ask. Its block is the header,
        # the outline and one sentence saying nothing matched, which under the "primary grading
        # evidence" caption is prose standing in for an absent section: it counted the provider
        # as succeeded, defeated every downstream empty guard, and read in the run log exactly
        # like a document that handed the forecasters the resolving paragraph. Withheld like any
        # other content-free 200, with `no_matching_passage` saying which rule withheld it —
        # a document we DID read, which is why it is excluded from the paid rung's population
        # (:func:`_url_context_rung_applies`) that every other `no_resolving_content` is in.
        return FetchResult(
            url=pending.url,
            status="no_resolving_content",
            text="",
            http_status=pending.http_status,
            content_type=pending.content_type or None,
            status_reason="no_matching_passage",
        )
    return FetchResult(
        url=pending.url,
        status="success",
        text=digest.block,
        http_status=pending.http_status,
        content_type=pending.content_type or None,
    )


def _document_not_parsed(pending: _PendingDocument, reason: FetchStatusReason) -> FetchResult:
    """The result for a document we held and chose not to parse.

    ``unsupported_type`` rather than ``unreadable_document``: nothing read the bytes, so
    nothing established they carry no text, and only the latter is worth a paid document
    read later. ``reason`` says which rule declined — the same token the rung attempt's
    ``skipped_reason`` carries, repeated here because the two ride different markers
    (``RESOLUTION_SOURCE_ESCALATION`` versus ``RESOLUTION_SOURCE_FETCH``) and a reader of
    the per-fetch line should not have to join to learn we were holding a document.
    """
    return FetchResult(
        url=pending.url,
        status="unsupported_type",
        text="",
        http_status=pending.http_status,
        content_type=pending.content_type or None,
        status_reason=reason,
    )


async def _resolution_response_outcome(
    resp: Any, current_url: str, ctx: context.LadderContext
) -> FetchResult | _PendingDocument | str:
    """Classify one response: a terminal FetchResult, a held document, or the next hop's URL.

    The :class:`_PendingDocument` case is the PDF branch handing its parse back to the
    caller to run outside the host semaphore; every other branch is terminal or a hop.
    """
    status = resp.status
    content_type = (resp.headers.get("Content-Type") or "").lower()

    if status in REDIRECT_STATUSES:
        return await _resolution_redirect_outcome(resp, current_url, content_type)

    # Non-redirect response — same status routing as before. The `Server` header rides the
    # non-200 result so a 403 can be attributed to the CDN that served it (Akamai / Cloudflare).
    server = resp.headers.get("Server")
    non_ok = _resolution_status_outcome(status, current_url, content_type, server=server)
    if non_ok is not None:
        return non_ok

    # 200 OK: route on content type. JSON is recognised by the one vocabulary the harvest and
    # the derived-feed reuse gate use (`text/json` and `+json` feeds included), so a feed one
    # half of the ladder discovers is not `unsupported_type` to the other.
    if any(ct in content_type for ct in _HTML_CONTENT_TYPES):
        return await _resolution_html_outcome(resp, current_url, content_type, ctx)
    if is_json_content_type(content_type) or any(ct in content_type for ct in _RAW_TEXT_CONTENT_TYPES):
        return await _resolution_text_outcome(resp, current_url, content_type)

    # Everything else routes through the PDF rung, which reads the body and checks the
    # `%PDF-` magic before deciding anything. That covers a declared `application/pdf`
    # and the sniffed case: a missing/empty Content-Type header (ct=''), or a document
    # served as `application/octet-stream`, which is how several government hosts ship
    # theirs. A body that is not a PDF comes back `unsupported_type` exactly as before —
    # so the cost of sniffing is one capped read, and the benefit is that a cited PDF is
    # no longer dropped unread on the strength of a header we cannot rely on.
    return await _resolution_pdf_outcome(resp, current_url, content_type, ctx, from_status="unsupported_type")
