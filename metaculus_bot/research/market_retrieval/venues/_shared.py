"""The bounds and rules more than one venue parser reads.

Everything else in this package is venue-local by construction: a constant that only Kalshi's
catalogue pull or only Manifold's answers array cares about lives in that venue's module, so
this file stays the short list of what genuinely spans venues.
"""

from __future__ import annotations

from metaculus_bot.research.market_retrieval.types import MarketChild

# The venue-search endpoints' own `limit`. NOT a retrieval width: the pool's per-venue width
# is generation's business, and the parsers take theirs as an explicit argument, so no hard
# slice in a parser can silently cap a wider pool — a `payload[:10]` left in one would make
# "width 60" mean 10 per query with nothing to see at the call site.
VENUE_SEARCH_LIMIT = 10

# Rules-text retention at parse time. Generous here and tightened per venue in the ranker prompt:
# this bound only stops a pathological row from being carried around.
RULES_TEXT_MAX_CHARS = 2000


def child_render_order_key(child: MarketChild) -> tuple[bool, float]:
    """ONE render-order rule for every venue's child outcomes: open first, then price-descending.

    The renderer truncates a long child list from the end, so this order decides what survives
    the budget. Open before settled because a settled rung's price is a realized outcome, not a
    forecast — on a Manifold threshold ladder the crossed rungs settle to exactly 1.0 while the
    market stays open, so price alone would front a block of ``1.00 RESOLVED`` rows and push
    every rung still carrying uncertainty past the budget. Price-descending within each group
    because the priced rungs are the market's answer, whatever traded: traded-size ordering let
    near-zero-probability rungs with open interest evict the informative brackets (q45189: the
    six omitted rungs held 0.365 of price mass, all on one side). A child with no live quote
    sorts with the zero-priced rungs, so the priced rows are the ones that survive.

    Callers rely on Python's sort STABILITY: equal-priced (and quoteless) children keep the
    venue's own order — Kalshi's threshold order, Gamma's array order, Manifold's answer order —
    which is what keeps the render deterministic across runs. One definition here rather than
    three per-venue lambdas, so a change to the ordering rule cannot land on two venues out of
    three.
    """
    return (child.is_resolved, -(child.implied_prob_yes or 0.0))
