"""The settlement-source provenance join: match questions to Kalshi events on who settles them.

A Metaculus question names the source that settles it (a BLS release page, an IMF PortWatch
dashboard, an EIA series). Kalshi ships the same fact structurally: every open event carries
``settlement_sources[].url``. So the two can be joined on provenance instead of on words —
which reaches the markets whose titles share almost no vocabulary with the question, the class
a fuzzy scorer cannot see. No scoring, no floor, no LLM, no HTTP request: it either fires or it
does not, and the caller re-ranks within the channel.

Ported from the bake-off's Arm A (``scratch/bakeoff_run_2026-08-03/arms/arm_join.py``), which
measured 12/17 near-identical rows and 0/36 no-bearing rows kept on the frozen cohort. Its two
dials are facts rather than tunings:

  1. ``SELF_REFERENCE_DOMAINS`` — Kalshi points 986 events' placeholder sources at
     ``https://kalshi.com/`` (the "local news outlets", "official social media accounts"
     entries), so joining on that domain unions a tenth of the exchange into every question
     that happens to link Kalshi. It names the venue, not a series.
  2. The **public suffix list**, used to compute the registrable domain. ``data.bls.gov`` and
     ``www.bls.gov`` are the same publisher and meet at ``bls.gov``. But naive
     "last two labels" collapses ``abs.gov.au`` to ``gov.au``, which would union every
     Australian government event into any question citing any Australian agency. The PSL is the
     published answer to exactly that question.

There is deliberately **no pool cap** on this channel: capping the per-host pool at 50 events
dropped near-identical from 12/17 to 2/17, because the BLS pool is 77 events and holds 10 of
the 17 rows.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

# Reuses the SHIPPED extractor and the SHIPPED Metaculus self-reference test rather than
# re-implementing either: `extract_source_urls` already handles markdown links, bare URLs,
# Metaculus's backslash escapes, trailing-punctuation stripping and order-preserving dedup, and
# prod's resolution-source fetcher is graded on it. Private copies here would be two more things
# to keep in sync.
from metaculus_bot.research.resolution_source import extract_source_urls, is_metaculus_self_ref

logger = logging.getLogger(__name__)

# Domains that name the venue or the question site rather than a settlement series. The
# Metaculus half is ALSO enforced through `is_metaculus_self_ref` so there is one shipped
# definition of "points back at Metaculus"; this set stays the declarative statement of both
# facts. Don't collapse the two — the helper is where a future Metaculus host change lands.
SELF_REFERENCE_DOMAINS: frozenset[str] = frozenset({"kalshi.com", "metaculus.com"})

# Frozen 2026-08-04 from https://publicsuffix.org/list/public_suffix_list.dat — PSL version
# 2026-07-25_14-20-03_UTC, 10,239 rules (281 wildcard, 8 exception), byte-identical to the copy
# the bake-off measured on. Vendored rather than fetched: a `tldextract`-style dependency
# downloads the list at runtime, which the tests' egress guard blocks and which would be a new
# prod failure mode. Refreshing it is a deliberate commit, so a suffix-list change can never
# silently move a measured pool.
_PUBLIC_SUFFIX_LIST_PATH = Path(__file__).parent / "data/public_suffix_list.dat"

_WWW_PREFIX_RE = re.compile(r"^www\d*\.")


@lru_cache(maxsize=1)
def _public_suffix_rules() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """The PSL split into (exact, wildcard-parent, exception) rule sets.

    The three kinds are the whole algorithm: `gov.au` is exact, `*.ck` means every direct
    child of `ck` is itself a suffix, and `!www.ck` carves one back out. Comments (`//`) and
    blank lines are dropped; the ICANN/PRIVATE section split is deliberately ignored, because
    a private registry (`s3.amazonaws.com`) is still not a publisher boundary we want to
    collapse across.
    """
    exact: set[str] = set()
    wildcard_parents: set[str] = set()
    exceptions: set[str] = set()
    for raw in _PUBLIC_SUFFIX_LIST_PATH.read_text(encoding="utf-8").splitlines():
        rule = raw.strip()
        if not rule or rule.startswith("//"):
            continue
        if rule.startswith("!"):
            exceptions.add(rule[1:].lower())
        elif rule.startswith("*."):
            wildcard_parents.add(rule[2:].lower())
        else:
            exact.add(rule.lower())
    return frozenset(exact), frozenset(wildcard_parents), frozenset(exceptions)


def normalize_host(url: str) -> str | None:
    """The lower-cased hostname of a URL, `www.`-stripped. None when there isn't one.

    `.hostname` rather than `.netloc` so a port or userinfo cannot smuggle a host past the
    self-reference check — the same reason `resolution_source.is_metaculus_self_ref` uses it.
    `www\\d*\\.` catches `www150.statcan.gc.ca`, which is real in the Kalshi payload.
    """
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return None
    host = _WWW_PREFIX_RE.sub("", host).strip(".")
    return host or None


def registrable_domain(host: str) -> str | None:
    """Collapse a host to its registrable domain: the public suffix plus one more label.

    `data.bls.gov` -> `bls.gov`, `abs.gov.au` -> `abs.gov.au` (because `gov.au` is itself a
    public suffix, so the registrable domain is the whole three-label host, NOT `gov.au`).
    Returns None for a host that IS a bare public suffix with nothing registered under it —
    there is no publisher there to join on.
    """
    exact, wildcard_parents, exceptions = _public_suffix_rules()
    labels = host.split(".")

    # Longest matching rule wins, per the PSL algorithm. An exception rule (`!www.ck`) means
    # the matched suffix is one label SHORTER than the wildcard would make it.
    suffix_len = 1  # An unknown TLD is treated as a suffix of one label, per the PSL default.
    for start in range(len(labels)):
        candidate = ".".join(labels[start:])
        if candidate in exceptions:
            suffix_len = len(labels) - start - 1
            break
        if candidate in exact:
            suffix_len = max(suffix_len, len(labels) - start)
        parent = ".".join(labels[start + 1 :])
        if parent and parent in wildcard_parents:
            suffix_len = max(suffix_len, len(labels) - start)

    if len(labels) <= suffix_len:
        return None
    return ".".join(labels[-(suffix_len + 1) :])


@lru_cache(maxsize=4096)
def _publisher_domain(url: str) -> str | None:
    """The registrable domain a URL publishes under, or None when there is nothing to join on.

    None covers all four uninteresting cases in one place: no host, a bare public suffix, and
    either self-reference. Both callers go through this so the question side and the venue side
    cannot drift on what counts as a publisher.

    Memoized because Kalshi's catalogue repeats a small set of settlement URLs across a large
    set of events: the frozen universe carries 21,721 ``settlement_sources[].url`` values over
    only 1,378 distinct strings, so `settlement_domain_index` spends 52ms re-deriving the same
    ~1,378 answers where 3ms would do. Cache safety is trivial — the PSL is vendored and frozen,
    so every input maps to one answer for the process's life. Caching HERE and not on the index
    (see `settlement_domain_index`, which deliberately isn't cached) because the entries are
    short strings rather than whole event dicts, and `maxsize` bounds them well above the
    distinct-URL count a 6h catalogue refresh can churn through.
    """
    if is_metaculus_self_ref(url):
        return None
    host = normalize_host(url)
    if host is None:
        return None
    domain = registrable_domain(host)
    if domain is None or domain in SELF_REFERENCE_DOMAINS:
        return None
    return domain


def question_domains(text: str) -> set[str]:
    """The registrable domains a question's resolution text points at, minus self-references.

    `text` is the question's `resolution_criteria` + `fine_print`. The FULL extracted URL list
    is used — `RESOLUTION_SOURCE_MAX_URLS` is the fetcher's budget for HTTP work, and this join
    does no HTTP, so capping it here would drop domains for free.
    """
    return {domain for url in extract_source_urls(text) if (domain := _publisher_domain(url)) is not None}


def settlement_domain_index(events: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """registrable domain -> the Kalshi events settling on it.

    `settlement_sources` lives at the EVENT level; every nested market's copy is null, so an
    index built off nested markets would be permanently empty. Events with no `event_ticker` are
    skipped (the ticker is the pool's dedup key, so a ticketless event is unusable downstream),
    and an event appears at most once per domain even when several of its sources reduce to the
    same publisher.

    Events come back in CATALOGUE order, which is not an evidential order — the caller must
    re-rank within the channel (`fuzzy_best` against the query set). The arm this is ported from
    returned alphabetical ticker order, and leaving that unranked would make the fail-open slate
    measure the alphabet.

    Not memoized: the caller owns the catalogue and its 6h cache, so caching here would pin a
    second copy of every event for the process's life. A caller that builds this per question
    should hold the result alongside the catalogue rather than rebuilding it (~10k events is
    real, if small, blocking CPU).
    """
    index: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for event in events:
        ticker = event.get("event_ticker")
        if not ticker:
            continue
        for source in event.get("settlement_sources") or []:
            if not isinstance(source, dict):
                continue
            domain = _publisher_domain(str(source.get("url") or ""))
            if domain is None:
                continue
            index[domain].setdefault(str(ticker), event)
    logger.info(f"settlement-source index: {len(index)} domains over {len(events)} open Kalshi events")
    return {domain: list(by_ticker.values()) for domain, by_ticker in index.items()}
