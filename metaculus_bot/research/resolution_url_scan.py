"""Cited-source URL scanning for the resolution-source provider.

One responsibility: pull the http(s) URLs a Metaculus question cites in its
resolution criteria / fine print out of that markdown, and classify the ones
another provider already covers (Metaculus self-refs, FRED series, Yahoo ticker
quote pages). No I/O and no caps here — ``resolution_source.select_fetchable_urls``
composes these into the capped fetch list (the cap lives there because the test
suites patch it on that module), and ``market_retrieval.settlement_join`` reuses
the extractor plus the self-ref predicate for the Kalshi settlement-source join.

Split out of ``research.resolution_source`` so the markdown-escape, trailing-
punctuation and dedup rules — each of which came from a measured live failure —
can be read and tested without the fetch machinery around them.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

# Metaculus-injected markdown escapes: `\_`, `\.`, `\&`, `\-`, `\#`, `\(`, `\)`.
# FINDINGS: 3.4% of URLs carry these; one flips 404→success once unescaped.
_MARKDOWN_ESCAPE_RE = re.compile(r"\\([_&.\-#()])")

# Markdown link: [label](https://...) — capture only the URL.
_MARKDOWN_LINK_URL_RE = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)")

# Bare URL — stops at whitespace and common closers.
_BARE_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+")

# Trailing punctuation to strip from an extracted URL.
_TRAILING_PUNCT = ".,;:)]}>\"'"


def strip_markdown_escapes(url: str) -> str:
    """Remove markdown backslash escapes Metaculus injects into rendered URLs."""
    return _MARKDOWN_ESCAPE_RE.sub(r"\1", url)


def extract_source_urls(text: str) -> list[str]:
    """Extract http(s) URLs from ``text``.

    Handles markdown links ``[label](https://…)`` and bare URLs. Strips trailing
    punctuation, applies backslash-unescape, dedupes preserving order (case-
    insensitive scheme+host; exact path and query — query params stay in the
    key because we may need them, e.g. for FRED graph_id; fragments are
    excluded because they're never sent over HTTP). Returns the FULL deduped
    list — the ``RESOLUTION_SOURCE_MAX_URLS`` cap is applied downstream by
    :func:`select_fetchable_urls`, AFTER the self-ref/FRED/Yahoo skip filter,
    so a run of leading self-refs doesn't starve the real sources out of the
    fetch budget.
    """
    if not text:
        return []

    # Collect (start_pos, url) from both regex families so the final order
    # tracks appearance in the source text — not extraction order. Markdown
    # link URLs are typically ALSO matched by the bare-URL regex (its match
    # sits inside the `[label](URL)` parens); the earlier position wins after
    # sort, and dedup drops the duplicate.
    positioned: list[tuple[int, str]] = []
    for match in _MARKDOWN_LINK_URL_RE.finditer(text):
        # Anchor at the URL group's start, not the link's start, so a
        # markdown link and a same-position bare URL rank identically.
        positioned.append((match.start(1), match.group(1)))
    for match in _BARE_URL_RE.finditer(text):
        positioned.append((match.start(), match.group(0)))
    positioned.sort(key=lambda pair: pair[0])

    cleaned: list[str] = []
    for _pos, raw in positioned:
        u = raw
        # Strip trailing punctuation (may repeat: "foo.,").
        while u and u[-1] in _TRAILING_PUNCT:
            u = u[:-1]
        u = strip_markdown_escapes(u)
        if not u.lower().startswith(("http://", "https://")):
            continue
        cleaned.append(u)

    # Dedup preserving order (first-seen URL string wins). Case-insensitive
    # scheme+netloc; exact path/query. Fragments are excluded — they're never
    # sent over HTTP, so URLs differing only by fragment are the same fetch.
    # A bare host and a bare host + "/" collapse to one entry (real questions
    # cite both forms of the same root page — observed on childmortality.org
    # in the 2026-07-09 smoke test, burning a duplicate fetch slot).
    seen: set[str] = set()
    deduped: list[str] = []
    for u in cleaned:
        try:
            parsed = urlparse(u)
        except ValueError:
            continue
        key = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path or '/'}?{parsed.query}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(u)
    return deduped


def is_metaculus_self_ref(url: str) -> bool:
    """A URL that points back at Metaculus is a self-reference (no new info).

    Uses ``.hostname`` (not ``.netloc``) so a port or userinfo can't slip a
    metaculus URL past the check — ``.netloc`` keeps ``:443`` / ``user@``, which
    would defeat the exact-host and suffix comparisons below.
    """
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host == "metaculus.com" or host.endswith(".metaculus.com")


def is_fred_url(url: str) -> bool:
    """FRED series URLs are already served by the financial-data provider."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host == "fred.stlouisfed.org"


def is_yahoo_ticker_url(url: str) -> bool:
    """Yahoo Finance `/quote/…` URLs are yfinance-served; skip.

    Generic Yahoo article / news URLs remain fetchable — only the ticker
    quote pages overlap with the financial-data provider.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return (parsed.hostname or "").lower() == "finance.yahoo.com" and parsed.path.startswith("/quote/")
