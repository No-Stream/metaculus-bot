"""Ranked prediction-market retrieval: pool generation, one LLM ranking call, render.

Import the submodule you need — this package root deliberately re-exports nothing, so there is
one export surface per name instead of a second one drifting from `types`/`queries`. Every
consumer already imports the submodules directly:

- `types` — the `MarketMatch` / `MarketSnapshot` rows and the liquidity vocabulary. Imports
  nothing else in this repo, which is what keeps the graph acyclic.
- `http` — the bounded GET, the body caps, the field coercions.
- `queries` — the deterministic query set plus the LLM query author's prompt and parser.
- `venues` — the four venue fetch/parse paths.
- `settlement_join` — the who-settles-this provenance join onto Kalshi events.
- `generation` — the three retrieval channels unioned into one candidate pool.
- `ranking` — the ranker prompt, the parser, and the deterministic fail-open slate.
- `rendering` — the markdown snapshot in the ranker's order.

`metaculus_bot.research.prediction_market` stays the seam module every consumer OUTSIDE this
package imports from: it owns the provider factory, the caches and degradation counters, the
aiohttp session factory, and it re-exports the row types from `types`.
"""
