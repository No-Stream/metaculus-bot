"""Manifold: search plus a per-candidate detail GET, because a multi-outcome row is priceless.

``contractType=ALL``, and a multi-outcome row's price arrives from a SECOND request. ``BINARY``
was a measured ~30% recall ceiling (27 of 89 labeled-wanted markets are MULTIPLE_CHOICE /
MULTI_NUMERIC / NUMBER / DATE). Those rows come back with ``probability`` null and no ``answers``
key, so their whole price lives in the per-candidate detail GET the enrichment hook already fires.

That one array is read TWICE, for two surfaces with different room: ``manifold_top_answers`` takes
three leaders for the ranker's one-line candidate, and ``manifold_answer_children`` takes every
answer with its own volume for the rendered ``↳`` sub-rows.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from metaculus_bot.research.market_retrieval.http import http_get_with_backoff, safe_float, safe_int
from metaculus_bot.research.market_retrieval.types import MarketChild, MarketMatch
from metaculus_bot.research.market_retrieval.venues._shared import RULES_TEXT_MAX_CHARS, VENUE_SEARCH_LIMIT

logger = logging.getLogger(__name__)

MANIFOLD_SEARCH_URL = "https://api.manifold.markets/v0/search-markets"
MANIFOLD_MARKET_URL = "https://api.manifold.markets/v0/market"

MANIFOLD_MAX_ATTEMPTS = 2

# A multi-outcome Manifold market's leading answers, kept off its detail payload for the RANKER's
# candidate line. Three, because they become one pipe-separated segment of a one-line candidate and
# the leaders carry the shape of the distribution; the tail, where a threshold ladder's 17 rungs
# live, is what the rendered `↳` sub-rows keep instead (`manifold_answer_children`), since a table
# row has room a prompt line does not.
#
# 60 chars because measured answer texts are short (10-13 on the committed fixture: `Over $4.60`,
# `$3.80 - $4.19`, `Nov-Dec 2026`), so the cap only bounds a pathological answer. It applies to the
# sub-row titles too, under the renderer's own tighter `CHILD_TITLE_MAX_CHARS`.
MANIFOLD_TOP_ANSWERS_RENDERED = 3
MANIFOLD_ANSWER_TEXT_MAX_CHARS = 60


def _walk_tiptap_text(node: Any) -> list[str]:
    """Depth-first collect every leaf ``text`` string from a TipTap/ProseMirror doc."""
    out: list[str] = []
    if isinstance(node, dict):
        text = node.get("text")
        if isinstance(text, str) and text:
            out.append(text)
        content = node.get("content")
        if isinstance(content, list):
            for child in content:
                out.extend(_walk_tiptap_text(child))
    elif isinstance(node, list):
        for child in node:
            out.extend(_walk_tiptap_text(child))
    return out


def manifold_rules_text(market: dict[str, Any]) -> str:
    """A Manifold market's rules text: ``textDescription``, else the flattened doc, else "".

    Deliberately NO fall back to the question title: that renders a candidate line whose
    ``rules:`` segment repeats its own title, spending ranker tokens on nothing. The
    enrichment hook is what fills real description text in, and a blank here is the honest
    signal that it has not run or found nothing.
    """
    text_description = market.get("textDescription")
    if isinstance(text_description, str) and text_description.strip():
        return text_description[:RULES_TEXT_MAX_CHARS]

    description = market.get("description")
    if isinstance(description, dict):
        collected = _walk_tiptap_text(description)
        return " ".join(collected)[:RULES_TEXT_MAX_CHARS] if collected else ""
    if isinstance(description, str) and description.strip():
        return description[:RULES_TEXT_MAX_CHARS]
    return ""


def parse_manifold_matches(payload: Any, *, width: int) -> list[MarketMatch] | None:
    """Parse a Manifold search response into candidate rows, venue-rank order.

    Close time and every liquidity field are read off EACH SEARCH ROW. That is the fix for
    the blank close/liquidity cells the bake-off measured on 52 of 94 Manifold rows: those
    came from looking each row up in a cached universe it was never in.

    ``None`` when the TOP-LEVEL shape is not the documented array, honouring the package's
    ``None``-means-fetch-failed contract. ``http_get_with_backoff`` has already turned every
    non-200 and every undecodable body into ``None`` before this runs, so a shapeless payload
    here is a 200 whose contract changed — a loss, not a search that found nothing, and ``[]``
    published it as a benign ``none``. Per-ROW malformation still skips the row: one unreadable
    market among ten is not a venue-wide failure.
    """
    if not isinstance(payload, list):
        logger.warning(f"Manifold returned a {type(payload).__name__} payload, not the documented array")
        return None

    out: list[MarketMatch] = []
    for rank, market in enumerate(payload):
        if len(out) >= width:
            break
        if not isinstance(market, dict):
            continue
        slug = market.get("slug") or ""
        creator = market.get("creatorUsername") or ""
        url = f"https://manifold.markets/{creator}/{slug}" if slug and creator else (market.get("url") or "")

        close_ms = market.get("closeTime")
        close_time: datetime | None = None
        if isinstance(close_ms, (int, float)):
            try:
                close_time = datetime.fromtimestamp(float(close_ms) / 1000.0, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                close_time = None

        out.append(
            MarketMatch(
                platform="manifold",
                market_title=market.get("question") or "",
                market_url=url,
                implied_prob_yes=safe_float(market.get("probability")),
                bid=None,
                ask=None,
                spread=None,
                volume_24h=safe_float(market.get("volume24Hours")),
                close_time=close_time,
                is_resolved=bool(market.get("isResolved")),
                match_confidence=100.0 - rank,
                raw_rules=manifold_rules_text(market),
                total_volume=safe_float(market.get("volume")),
                liquidity=safe_float(market.get("totalLiquidity")),
                num_bettors=safe_int(market.get("uniqueBettorCount")),
                venue_market_id=str(market.get("id") or ""),
                retrieval_channel="venue_search",
            )
        )
    return out


async def manifold_search(session: Any, query: str, *, width: int) -> list[MarketMatch] | None:
    """Search Manifold for one query. ``None`` when the fetch failed OR the shape changed.

    ``contractType=ALL`` rather than ``BINARY``, which was a ~30% recall ceiling: 27 of the 89
    labeled-wanted Manifold markets are MULTIPLE_CHOICE / MULTI_NUMERIC / NUMBER / DATE, and no
    query, width or enrichment could reach them while the parameter was pinned. On this module's
    own fixture term the same search returns 6 BINARY under ``BINARY`` and 6 BINARY + 2
    MULTIPLE_CHOICE under ``ALL``.

    A multi-outcome row arrives with ``probability`` NULL and NO ``answers`` key — the search
    response carries no per-answer data at all — so it would reach the ranker priceless. The
    answers come from the per-candidate detail GET the enrichment hook already fires
    (``manifold_top_answers`` below); a row whose detail was lost stays title-only and renders
    ``-`` for its price, the same soft-fail the rules text has.
    """
    payload = await http_get_with_backoff(
        session,
        MANIFOLD_SEARCH_URL,
        {"term": query, "contractType": "ALL", "limit": str(VENUE_SEARCH_LIMIT)},
        max_attempts=MANIFOLD_MAX_ATTEMPTS,
        retryable_statuses=(429, 500, 502, 503, 504),
        label=f"Manifold q={query[:40]!r}",  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # log-label truncation
    )
    if payload is None:
        return None
    return parse_manifold_matches(payload, width=width)


async def manifold_market_detail(session: Any, market_id: str) -> dict[str, Any] | None:
    """One market's full record, for the ``textDescription`` and ``answers`` the listing omits.

    ``None`` on any failure, which the enrichment hook treats as "leave the row title-only":
    a lost detail GET costs no recall, only rules text and — on a multi-outcome row — its only
    price. One attempt, no retry — 60 of these fire per question under a 10s fan-out wall, and a
    row's description is not worth spending that wall on twice.
    """
    payload = await http_get_with_backoff(
        session,
        f"{MANIFOLD_MARKET_URL}/{market_id}",
        {},
        max_attempts=1,
        retryable_statuses=(429, 500, 502, 503, 504),
        label=f"Manifold detail id={market_id[:40]!r}",  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # log-label truncation
    )
    return payload if isinstance(payload, dict) else None


def manifold_top_answers(detail: dict[str, Any]) -> tuple[tuple[str, float], ...]:
    """A multi-outcome market's leading ``(answer_text, probability)`` pairs, highest first.

    The detail payload's ``answers`` array is where ALL of a non-BINARY market's price
    information lives: the search listing omits the array entirely and its market-level
    ``probability`` is null, so before this a multi-outcome row reached the forecaster with a
    title and nothing else.

    The array IS the outcome-type discriminator, which is why nothing here reads
    ``outcomeType``: a BINARY detail carries no ``answers`` key at all, while MULTIPLE_CHOICE,
    MULTI_NUMERIC, NUMBER and DATE all publish the same ``{text, probability}`` answer shape
    (live-verified across all five, 2026-08-05), so one read covers the whole ~30% of Manifold
    the ``contractType`` flip reaches without a second field to keep in sync.

    The sort's STABILITY is load-bearing rather than incidental. A threshold ladder resolves its
    crossed rungs to probability exactly 1.0 — 10 of 17 answers on the committed fixture — so the
    leaders are routinely a tie, and only keeping the array's own order among equals makes the
    pick deterministic across runs. ``reverse=True`` preserves that order rather than inverting
    it, per the sort's documented guarantee.

    An answer without usable text or a readable probability is dropped rather than carried blank,
    mirroring ``predictit_contract_names``: an empty label spends ranker tokens saying nothing, and a
    non-finite probability would defeat every comparison in the sort and then render as ``nan%``.
    """
    answers = detail.get("answers")
    if not isinstance(answers, list):
        return ()

    scored: list[tuple[str, float]] = []
    for answer in answers:
        if not isinstance(answer, dict):
            continue
        # Answer text is user-authored, so whitespace is collapsed here at the boundary — a
        # newline would break both the one-line candidate line and the one-line rules bullet.
        text = " ".join(str(answer.get("text") or "").split())
        probability = safe_float(answer.get("probability"))
        if not text or probability is None:
            continue
        scored.append((text[:MANIFOLD_ANSWER_TEXT_MAX_CHARS], probability))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return tuple(scored[:MANIFOLD_TOP_ANSWERS_RENDERED])


def manifold_answer_children(detail: dict[str, Any]) -> tuple[MarketChild, ...]:
    """Every answer as its own sub-row, MOST-PROBABLE FIRST. ``()`` on a BINARY market.

    The render-side counterpart to ``manifold_top_answers``, off the same ``answers`` array and with
    the same discriminator (a BINARY detail carries no such key), differing in two ways that follow
    from where each one goes. This keeps ALL the answers rather than the leading
    ``MANIFOLD_TOP_ANSWERS_RENDERED``, because the renderer has its own child budget and a
    three-answer cap tuned for a one-line ranker segment would throw away rungs the table has room
    for. And each answer keeps its own ``volume`` — the sub-row has a column for it, the ranker line
    does not.

    Ordered by OPEN-FIRST, then by PROBABILITY — not by volume like the real-money venues, because
    these answers are a distribution over one question's outcomes and reading it means seeing where
    the mass sits. Probability alone was the obvious choice and is measurably wrong on this module's
    own committed fixture: a threshold ladder settles its crossed rungs to exactly 1.0 while the
    market stays open, so 10 of that market's 17 answers sort to the front as a block of ``1.00
    RESOLVED`` rows and push all 7 rungs that still carry uncertainty past the render budget. The
    settled rungs are real evidence — they establish the floor the series has already crossed — but
    they are not the forecast, so they queue behind it.

    Within each group the sort is the same stable one ``manifold_top_answers`` documents: the tied
    1.0 rungs keep the array's own order, which is what makes the render deterministic across runs.

    ``num_bettors`` is the MARKET's ``uniqueBettorCount``, copied onto every child: Manifold scores
    participation on bettors and publishes no per-answer count, so this is the honest figure for each
    answer (they share one bettor pool) and it makes each sub-row's ``signal`` cell agree with its
    parent's instead of reading a false ``no-liquidity-data``.
    """
    answers = detail.get("answers")
    if not isinstance(answers, list):
        return ()

    num_bettors = safe_int(detail.get("uniqueBettorCount"))
    children: list[MarketChild] = []
    for answer in answers:
        if not isinstance(answer, dict):
            continue
        text = " ".join(str(answer.get("text") or "").split())
        probability = safe_float(answer.get("probability"))
        if not text or probability is None:
            continue
        children.append(
            MarketChild(
                title=text[:MANIFOLD_ANSWER_TEXT_MAX_CHARS],
                implied_prob_yes=probability,
                total_volume=safe_float(answer.get("volume")),
                num_bettors=num_bettors,
                # A resolved answer carries a `resolution` verdict ("YES"/"NO"); an open one carries
                # null. On a threshold ladder the crossed rungs settle individually while the market
                # stays open, so this is per-answer rather than the market's own flag.
                is_resolved=bool(answer.get("resolution")),
            )
        )
    children.sort(key=lambda child: (child.is_resolved, -(child.implied_prob_yes or 0.0)))
    return tuple(children)
