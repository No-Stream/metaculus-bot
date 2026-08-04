"""Ranked prediction-market retrieval: pool generation, one LLM ranking call, render.

The package holds the retrieval machinery; `metaculus_bot.research.prediction_market` stays the
seam module every consumer imports from (the provider factory, the caches and degradation
counters, the aiohttp session factory, and the row types it re-exports from here).

`types` imports nothing else in this repo, which is what keeps the graph acyclic: the venue and
pipeline modules depend on the row type, and the seam module depends on them.
"""

from metaculus_bot.research.market_retrieval.queries import (
    MANIFOLD_RELAXATION_MAX_TOKENS,
    MAX_FRAMINGS,
    MAX_QUERY_CHARS,
    MAX_SYNONYMS,
    dedupe_queries,
    deterministic_queries,
    fuzzy_best,
    manifold_relaxation_terms,
    parse_query_author,
    strip_dates_and_numbers,
)
from metaculus_bot.research.market_retrieval.settlement_join import (
    SELF_REFERENCE_DOMAINS,
    normalize_host,
    question_domains,
    registrable_domain,
    settlement_domain_index,
)
from metaculus_bot.research.market_retrieval.types import (
    LIQUIDITY_DEEP_USD,
    LIQUIDITY_THIN_USD,
    MANIFOLD_HIGH_BETTORS,
    MANIFOLD_THIN_BETTORS,
    MarketMatch,
    MarketSnapshot,
    SettlementSource,
)

__all__ = [
    "LIQUIDITY_DEEP_USD",
    "LIQUIDITY_THIN_USD",
    "MANIFOLD_HIGH_BETTORS",
    "MANIFOLD_RELAXATION_MAX_TOKENS",
    "MANIFOLD_THIN_BETTORS",
    "MAX_FRAMINGS",
    "MAX_QUERY_CHARS",
    "MAX_SYNONYMS",
    "MarketMatch",
    "MarketSnapshot",
    "SELF_REFERENCE_DOMAINS",
    "SettlementSource",
    "dedupe_queries",
    "deterministic_queries",
    "fuzzy_best",
    "manifold_relaxation_terms",
    "normalize_host",
    "parse_query_author",
    "question_domains",
    "registrable_domain",
    "settlement_domain_index",
    "strip_dates_and_numbers",
]
