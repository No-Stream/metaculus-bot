"""The two bounds more than one venue parser reads.

Everything else in this package is venue-local by construction: a constant that only Kalshi's
catalogue pull or only Manifold's answers array cares about lives in that venue's module, so
this file stays the short list of what genuinely spans venues.
"""

from __future__ import annotations

# The venue-search endpoints' own `limit`. NOT a retrieval width: the pool's per-venue width
# is generation's business, and the parsers take theirs as an explicit argument, so no hard
# slice in a parser can silently cap a wider pool — a `payload[:10]` left in one would make
# "width 60" mean 10 per query with nothing to see at the call site.
VENUE_SEARCH_LIMIT = 10

# Rules-text retention at parse time. Generous here and tightened per venue in the ranker prompt:
# this bound only stops a pathological row from being carried around.
RULES_TEXT_MAX_CHARS = 2000
