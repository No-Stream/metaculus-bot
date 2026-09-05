"""Cited-source URL scanning for the resolution-source provider.

One responsibility: pull the http(s) URLs a Metaculus question cites in its
resolution criteria / fine print out of that markdown, and classify the ones
another provider already covers (Metaculus self-refs, FRED series, Yahoo ticker
quote pages). No I/O and no caps here — ``resolution_source.select_fetchable_urls``
composes these into the capped fetch list (the cap lives there because the test
suites patch it on that module), and ``market_retrieval.settlement_join`` reuses
the extractor plus the self-ref predicate for the Kalshi settlement-source join.

Split out of ``research.resolution_source`` so the markdown-escape, paren-balance,
trailing-punctuation and dedup rules — each of which came from a measured live
failure — can be read and tested without the fetch machinery around them.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from metaculus_bot.research.wayback import innermost_url

# Metaculus-injected markdown escapes: `\_`, `\.`, `\&`, `\-`, `\#`, `\(`, `\)`.
# FINDINGS: 3.4% of URLs carry these; one flips 404→success once unescaped.
_MARKDOWN_ESCAPED_CHARS = r"_&.\-#()"
_MARKDOWN_ESCAPE_RE = re.compile(rf"\\([{_MARKDOWN_ESCAPED_CHARS}])")

# A URL body is a run of atoms; both URL regexes below are built from these, so
# they agree about where a URL ends. Three atom kinds, tried in this order:
#   1. A markdown escape — exactly the set `strip_markdown_escapes` removes, so
#      the matcher and the unescaper can never disagree about what is an escape.
#      An escaped paren must NOT end the URL: Metaculus renders
#      `…/wiki/Nuri_\(rocket\)`, and stopping at the `\)` yielded
#      `…/wiki/Nuri_(rocket\` — a 404 whose trailing backslash survived both the
#      punctuation strip and the unescape (measured 2026-09-03; the repaired URL
#      returns 200 with 16,753 chars).
#   2. A balanced `(…)` pair — Wikipedia/Ballotpedia style `…_(rocket)`,
#      `…_(August_18_Republican_primary)`. That closing paren is part of the URL,
#      and dropping it 404s the same way (the second archived instance).
#   3. Any other character except whitespace, the common closers, and a LONE
#      `)` — a `)` that closes nothing opened inside the URL is prose
#      punctuation (`(see https://example.com/x)`), so it ends the match.
# `_URL_ATOM` adds a fourth, lowest-priority alternative for a lone `(`, so an
# unbalanced open paren doesn't truncate a bare URL that never had a closing one.
# The markdown-link form uses `_BALANCED_URL_ATOM` instead — see below.
_ESCAPE_ATOM = rf"\\[{_MARKDOWN_ESCAPED_CHARS}]"
_PLAIN_URL_CHAR = r"[^\s()\\<>\"'\]]"
_BALANCED_PARENS = rf"\((?:{_ESCAPE_ATOM}|{_PLAIN_URL_CHAR})*\)"
_BALANCED_URL_ATOM = rf"(?:{_ESCAPE_ATOM}|{_BALANCED_PARENS}|{_PLAIN_URL_CHAR})"
_URL_ATOM = rf"(?:{_BALANCED_URL_ATOM}|\()"

# Markdown link: [label](https://...) — capture only the URL. The atoms stop at
# a lone `)`, so the link's own closing paren is the one `\)` consumes while an
# inner `(x)` pair stays inside the capture. The lone-`(` atom is deliberately
# excluded here: inside `(…)` delimiters an unbalanced open paren is not a
# markdown link at all, and allowing it lets the regex backtrack into a
# truncated capture (`[a](https://x.test/p_(q)` → `…/p_(q`) that then rides
# alongside the bare-URL match as a second, broken fetch target.
_MARKDOWN_LINK_URL_RE = re.compile(rf"\[[^\]]*\]\((https?://{_BALANCED_URL_ATOM}*)\)")

# Bare URL — stops at whitespace, common closers and unbalanced parens.
_BARE_URL_RE = re.compile(rf"https?://{_URL_ATOM}*")

# Trailing punctuation to strip from an extracted URL. `)` is NOT here: whether a
# trailing paren belongs to the URL depends on balance, handled separately by
# `_trim_trailing_delimiters`.
_TRAILING_PUNCT = ".,;:]}>\"'"


def strip_markdown_escapes(url: str) -> str:
    """Remove markdown backslash escapes Metaculus injects into rendered URLs."""
    return _MARKDOWN_ESCAPE_RE.sub(r"\1", url)


def _closes_inner_paren(url: str) -> bool:
    """True when ``url``'s final ``)`` closes a ``(`` opened inside ``url``."""
    depth = 0
    for char in url[:-1]:
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
    return depth > 0


def _trim_trailing_delimiters(url: str) -> str:
    """Strip sentence punctuation the URL picked up from the surrounding prose.

    Runs AFTER the unescape so escaped and unescaped parens are judged by one
    rule: a trailing ``)`` is kept only when it closes a ``(`` from inside the
    URL, and dropped otherwise (a URL wrapped in escaped prose parens,
    ``\\(https://x.test/y\\)``, unescapes to a trailing unmatched paren). A
    trailing backslash is always dropped — it is never part of a live URL, and
    letting one survive is what produced the measured 404.
    """
    while url:
        if url[-1] == ")":
            if _closes_inner_paren(url):
                break
        elif url[-1] not in _TRAILING_PUNCT and url[-1] != "\\":
            break
        url = url[:-1]
    return url


def extract_source_urls(text: str) -> list[str]:
    r"""Extract http(s) URLs from ``text``.

    Handles markdown links ``[label](https://…)`` and bare URLs, including parens
    that belong to the URL — escaped (``…/Nuri_\(rocket\)``) or balanced
    (``…/Nuri_(rocket)``). Applies backslash-unescape, then strips trailing
    punctuation, then dedupes preserving order (case-insensitive scheme+host;
    exact path and query — query params stay in the
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
        # Unescape BEFORE trimming so one paren rule covers both `\)` and `)`,
        # and so an escaped trailing period (`…foo\.`) cannot leave a backslash
        # behind as the last character.
        u = _trim_trailing_delimiters(strip_markdown_escapes(raw))
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

    Judged on the INNERMOST URL of a Wayback capture, at any depth of nesting: an archived
    copy of a Metaculus page in front of a forecaster is still the question quoting itself,
    and the capture URL's own hostname is ``web.archive.org``, which is how a cited capture
    (or a capture of a capture) sailed past every self-reference filter in the pipeline.
    """
    try:
        host = (urlparse(innermost_url(url)).hostname or "").lower()
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
