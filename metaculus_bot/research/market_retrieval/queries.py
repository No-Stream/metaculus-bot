"""Query construction for prediction-market retrieval: the deterministic set, the LLM author's
additions, and the fuzzy scorer that orders the enumerable venues' catalogues.

Three properties are load-bearing here, each with a measurement behind it:

- **The deterministic set is never replaced, only added to.** A *replacing* query author's
  failure mode is an empty query set, which is indistinguishable from "no markets exist" — the
  silent failure that hid the Manifold breakage for 17+ days. So `parse_query_author` returns
  `()` on every unreadable shape and the caller unions onto `deterministic_queries`, which means
  the author's worst case is "no gain".
- **Digits are handled in code rather than by asking the model nicely, and the two channels
  handle them differently.** Manifold's `term` is a strict conjunction of content tokens, so one
  date token no market's text carries returns `[]`. Two mechanisms enforce that, scoped
  differently on purpose. The conjunctive venues strip EVERY digit-bearing token at their own call
  site (`strip_dates_and_numbers`), because theirs is the channel with the cliff. Author output is
  filtered at PARSE time by `has_date_like_token` instead: a synonym carrying a DATE (a year, a
  bare number, a month-and-day) is dropped whole rather than trimmed to its non-digit remnant,
  because a remnant like `"rate"` scores ~100 against thousands of off-topic events and
  `fuzzy_best` has no floor to stop it displacing a real hit. A synonym whose digits sit inside an
  identifier (`"U-3"`, `"S&P 500"`, `"10-K"`) is KEPT: series codes and tickers are the vocabulary
  the author exists to contribute, and the un-stripping fuzzy channel is where they pay off.
  `deterministic_queries` itself does NOT strip: a year is real signal when scoring against a
  catalogue of dated market titles, and the enumerable venues score on the raw set.
- **`fuzzy_best` has no score floor.** It ORDERS a catalogue so its top N fits in a prompt; it
  never drops anything. The retired `KALSHI_MIN_FUZZY_SCORE` / `PREDICTIT_MIN_FUZZY_SCORE`
  floors are what discarded the adjacent-cut markets that carry most of the evidential value,
  so the parameter is absent rather than defaulted — a floor cannot come back by argument.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Sequence

import numpy as np
from rapidfuzz import fuzz, process

from metaculus_bot.structured_output_schema import extract_first_balanced_braces, extract_json_block

logger = logging.getLogger(__name__)

# Caps on the query author's output: the prompt's own stated ceilings, enforced in code so a
# runaway completion cannot blow up the pool. A query longer than MAX_QUERY_CHARS is not a query,
# it is the model restating the question.
MAX_SYNONYMS = 8
MAX_FRAMINGS = 3
MAX_QUERY_CHARS = 80

# Longest relaxation rung, in content tokens. 3 is the measured ceiling for a satisfiable
# Manifold conjunction: recall collapses at ~4 (`US gas prices 2026` returns 10,
# `US gas prices high 2026` returns 0).
MANIFOLD_RELAXATION_MAX_TOKENS = 3

_DIGIT_RE = re.compile(r"\d")
_MANIFOLD_TERM_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'À-ɏ&.\-]*")

# `has_date_like_token`'s vocabulary. A day number carries an optional ordinal suffix so
# "June 30th" reads as a date rather than as an identifier ending in letters.
_TOKEN_EDGE_PUNCTUATION = "\"'“”‘’.,;:!?()[]{}"
_DIGIT_GROUP_RE = re.compile(r"\d+")
_DAY_NUMBER_RE = re.compile(r"\d{1,2}(?:st|nd|rd|th)?", re.IGNORECASE)
_PLAUSIBLE_YEARS = range(1900, 2100)
_MONTH_WORDS: frozenset[str] = frozenset(
    """january february march april may june july august september october november december
    jan feb mar apr jun jul aug sep sept oct nov dec
    """.split()
)

# Content-word stopwords for Manifold's conjunction-relaxation ladder. Kept byte-identical to
# the tuning script (scratch/new_analyses_2026-07-18/market_match_precision.py `_overlap`) it was
# graded on. The ladder uses it because Manifold's search constrains on the same notion of a
# content word: stopwords measurably do not narrow a `term` ("gas prices" and "gas prices in the"
# both return 10 results).
_RELEVANCE_STOPWORDS: frozenset[str] = frozenset(
    """a an the of in on at to for by with will be is are was were before after during between
    and or not no yes if then than as from into over under above below more less most least
    what which who whom whose when where why how this that these those there here it its
    do does did done have has had having get gets got question market resolve resolves resolved
    resolution against per any all each both other another same different new old first last
    2025 2026 2027 january february march april may june july august september october november december
    """.split()
)


def _strategy_s2(question_text: str) -> str:
    """Natural-language framing: question_text trimmed at the first '?'."""
    t = (question_text or "").strip()
    i = t.find("?")
    if i > 0:
        t = t[:i]
    return t.strip()


def manifold_relaxation_terms(title: str) -> list[str]:
    """Progressively shorter Manifold `term` candidates, most specific first.

    Manifold's `/v0/search-markets` treats `term` as a STRICT CONJUNCTION of content
    tokens: every token must appear in a market's text, and one absent token returns `[]`
    (measured 2026-08-03 — appending a nonsense token to a query with a known hit zeroes
    it, while reordering present tokens does not, which rules out a relevance floor).
    Its semantics are undocumented, so `tests/test_prediction_market_integration.py`
    carries a live tripwire for the day upstream switches to ranked search.

    Two consequences follow, and together they fix the query length rather than tune it:
    recall is monotone DECREASING in token count, and precision is monotone INCREASING.
    So the best query is the LONGEST SATISFIABLE conjunction — which is why every rung is
    issued as a first-class query rather than walked until one lands: the pool wants the
    precise rung's hits AND the general rung's, and the ranker sorts them out.

    Tokens are ranked by how much they narrow the search rather than by position:
    acronyms first (`VIX`), then proper nouns (`Australia`), then ordinary content words.
    Duplicates are dropped — a repeated entity would spend a rung slot without narrowing
    anything ("Sturgis Sturgis Motorcycle").
    """
    ranked: list[tuple[int, int, str]] = []
    seen_tokens: set[str] = set()
    for position, token in enumerate(_MANIFOLD_TERM_TOKEN_RE.findall(title or "")):
        normalized = token.lower().strip(".-'")
        if len(normalized) < 3 or normalized in _RELEVANCE_STOPWORDS or normalized in seen_tokens:
            continue
        seen_tokens.add(normalized)
        if token.isupper() and len(token) >= 2:
            specificity = 0  # acronym / ticker
        elif token[0].isupper() and position > 0:
            # Position 0 is the leading interrogative ("Will", "What"), not an entity.
            specificity = 1  # proper noun
        else:
            specificity = 2
        ranked.append((specificity, position, token))

    # Most-narrowing first; longer tokens break ties (they carry more signal than "the"-ish
    # short words that survived the stopword filter), then original order.
    ranked.sort(key=lambda t: (t[0], -len(t[2]), t[1]))

    terms: list[str] = []
    seen_terms: set[str] = set()
    for width in range(min(MANIFOLD_RELAXATION_MAX_TOKENS, len(ranked)), 0, -1):
        # Re-sort the chosen tokens into reading order: `term` matching is order-invariant,
        # but a readable query is what shows up in logs.
        chosen = sorted(ranked[:width], key=lambda t: t[1])
        term = " ".join(t[2] for t in chosen)
        if term and term.lower() not in seen_terms:
            seen_terms.add(term.lower())
            terms.append(term)
    return terms


def strip_dates_and_numbers(query: str) -> str:
    """Drop every token containing a digit, for the CONJUNCTIVE venues (Manifold, Polymarket).

    Manifold's `term` is a strict conjunction of content tokens, so one date token that no
    market's text carries returns `[]` — the measured cliff behind the 2.3% Manifold recall.
    Whole tokens go, not just the digits: `Q3` and `$90.50` are as unsatisfiable as `2026`.
    """
    kept = [tok for tok in query.split() if not _DIGIT_RE.search(tok)]
    return " ".join(kept).strip()


def has_date_like_token(query: str) -> bool:
    """True when a query carries a DATE, as opposed to digits sitting inside an identifier.

    The distinction exists because the two are worth opposite things to the fuzzy channel, which
    scores queries against a whole catalogue with no floor. A date token is high-frequency and
    low-information: a bare `"2026"` is a token-set subset of every dated market title, so it
    scores ~100 on thousands of them and displaces real hits. A series code is the reverse —
    `"U-3"`, `"S&P 500"`, `"10-K"` match almost nothing except the market that actually tracks the
    quantity, which is exactly why the query author is asked for them.

    A token is DATE-LIKE when, ignoring edge punctuation:

    - it is numeric (no letters) and one of its digit groups is a plausible calendar year, i.e.
      four digits in 1900-2099 — `"2026"`, `"2026-07-31"`, `"07/2026"`; or
    - it is numeric and the query names nothing else: a query with no alphabetic token at all is a
      number, not vocabulary — `"2026"`, `"44796"`, `"500"` standing alone; or
    - it is a 1-2 digit day number (with or without an ordinal suffix) next to a month word —
      `"June 30"`, `"30th June"`.

    Everything else digit-bearing is IDENTIFIER-LIKE and survives: digits attached to letters
    (`"U-3"`, `"CPI-U"`, `"10-K"`, `"50bp"`, `"ISO-8601"`) and a number qualifying a name in a
    query that has one (`"S&P 500"`, `"Nikkei 225"`, `"Brent above $90.50"`).

    Two boundary calls, both taken deliberately:

    - The year test fires on index names built from a year-shaped number, so `"Russell 2000"` and
      `"Nasdaq 100"` split — 2000 is in range, 100 is not. The year test wins: a bare in-range
      four-digit token is the measured ranking hazard, and the question's own words already reach
      the venues through `deterministic_queries`.
    - Fiscal shapes glued to letters (`"FY2026"`, `"H1"`, `"Q3"`) read as identifiers. They are
      low-frequency, so they cannot subset many titles, and the conjunctive venues strip them at
      their own call site regardless — the cliff this whole rule protects is not exposed to them.
    """
    cores = [token.strip(_TOKEN_EDGE_PUNCTUATION) for token in query.split()]
    query_names_something = any(any(char.isalpha() for char in core) for core in cores)

    for position, core in enumerate(cores):
        if not _DIGIT_RE.search(core):
            continue
        if _DAY_NUMBER_RE.fullmatch(core):
            neighbors = cores[max(0, position - 1) : position] + cores[position + 1 : position + 2]
            if any(neighbor.lower() in _MONTH_WORDS for neighbor in neighbors):
                return True
        if any(char.isalpha() for char in core):
            continue
        if not query_names_something:
            return True
        if any(len(group) == 4 and int(group) in _PLAUSIBLE_YEARS for group in _DIGIT_GROUP_RE.findall(core)):
            return True
    return False


def dedupe_queries(values: list[str]) -> list[str]:
    """Order-preserving, case-insensitive dedup, dropping blanks."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        norm = (value or "").strip()
        if not norm or norm.casefold() in seen:
            continue
        seen.add(norm.casefold())
        out.append(norm)
    return out


def deterministic_queries(title: str) -> list[str]:
    """The query set retrieval always issues, whatever the LLM author does.

    `[full title, natural-language framing, *relaxation rungs]`, deduped case-insensitively with
    the full title first. The full-length query is INCLUDED and unconditional: the validated
    design issues every rung in parallel rather than walking them until one lands, and a question
    that already matches at full length would otherwise lose its high-precision hits.
    """
    return dedupe_queries([title, _strategy_s2(title), *manifold_relaxation_terms(title)])


# The query author's prompt. It carries the recall framing, the additive framing ("your
# queries are ADDED to a set that already has the question's own words") and the no-dates
# rule. The date ban is ASKED for here and ENFORCED in `_authored_strings`; the code is the
# guarantee, since a model that ignores the instruction would otherwise zero a Manifold
# conjunction. The ban has to be scoped exactly as `has_date_like_token` scopes it: a prompt
# banning digits outright would suppress the series codes the parser now keeps, which is the
# same silent drift as a ceiling raised in code while the prompt still asks for the old one.
# Filled by `build_query_author_prompt`, never `.format`, so a brace in a question title can
# never raise.
QUERY_AUTHOR_PROMPT = """You are writing search queries to find prediction markets RELATED to a forecasting question.

Recall is the objective: a market on the same subject with a different resolution date or a different threshold is a WANTED hit, not a miss.

Your queries are ADDED to a deterministic query set that already contains the question's own words. So do not restate words already in the question -- they are already covered, and a repeat wastes a slot. Emit the vocabulary a token match on the question could never reach.

Return JSON with exactly two keys:
  "synonyms":  domain vocabulary absent from the question -- alternate names for the measured quantity, the agency or index that publishes it, the ticker, trader slang. Up to 8 short strings.
  "framings":  2-3 alternate short phrasings of the whole question, each at most 4 words.

Do NOT include dates, years, or standalone numbers in any string: some venues treat a query as a strict conjunction, so a date token zeroes the result set. Digits INSIDE a name are wanted -- "U-3", "S&P 500", "10-K" are series codes a token match on the question could never reach, so emit them when they name the quantity.

QUESTION
title: {title}
resolution criteria: {rc}

Return ONLY the JSON object.
JSON:"""

# The author only needs enough resolution criteria to name the publishing agency and the
# measured quantity; the ranker is the stage that reads the full clause.
QUERY_AUTHOR_RC_CHARS = 800

_QUERY_AUTHOR_PLACEHOLDER_RE = re.compile(r"\{title\}|\{rc\}")


def build_query_author_prompt(title: str, resolution_criteria: str) -> str:
    """Fill the author prompt's two placeholders in ONE pass over the template.

    Single-pass because sequential `.replace` calls let question data be re-substituted: a
    title containing the literal `"{rc}"` (Metaculus titles quote JSON and template syntax)
    would have its own text replaced by the resolution criteria on the second call, so the
    model reads a corrupted question. `re.sub` with a lookup visits each placeholder once and
    never looks at what it substituted.
    """
    substitutions = {
        "{title}": title or "",
        "{rc}": (resolution_criteria or "")[:QUERY_AUTHOR_RC_CHARS],
    }
    return _QUERY_AUTHOR_PLACEHOLDER_RE.sub(lambda m: substitutions[m.group(0)], QUERY_AUTHOR_PROMPT)


def _authored_strings(value: object) -> list[str]:
    """One of the author's two JSON arrays, cleaned into usable queries.

    Date-bearing synonyms are DROPPED whole here, not trimmed down to their non-digit tokens.
    Filtering has to happen at parse time — a date token in a synonym would otherwise survive into
    the Kalshi fuzzy channel, which does not strip — but keeping the REMNANT is worse than dropping
    the synonym. `"2026 rate print"` reduced to `"rate print"`, and `fuzzy_best` maxes with no
    floor, so one generic word scores ~100 via `token_set_ratio` against every catalogue event
    whose rules text contains it: measured on the real 9,762-event catalogue, bare `"rate"` scores
    >=99 on 52 events and pushed the first wanted row from pool rank 2 to rank 31, replacing an
    entire fail-open slate with Fed-funds markets.

    Identifier-shaped digits survive instead of being dropped with the dates (see
    `has_date_like_token` for where the line falls), because series codes and tickers are the whole
    reason this stage exists: `"U-3"` and `"S&P 500"` are precisely the vocabulary a token match on
    the question cannot reach. They reach the fuzzy channel verbatim, and the conjunctive venues
    still see them stripped, since that strip runs on the whole query set at the venues' own call
    site — the same treatment the question's own title already gets.
    """
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = " ".join(item.split())
        if not normalized or has_date_like_token(normalized):
            continue
        out.append(normalized[:MAX_QUERY_CHARS].strip())
    return out


def parse_query_author(text: str) -> tuple[str, ...]:
    """`{"synonyms": [...], "framings": [...]}` -> a deduped tuple of EXTRA queries.

    Returns `()` on every unusable shape — empty completion, no JSON object, malformed JSON, a
    non-object, neither key carrying usable strings. The caller treats `()` as "no extra
    queries", which is the deterministic set exactly, and records the additive stage as lost. So
    every `()` is a failure the caller may report, and no `()` costs recall.
    """
    blob = (text or "").strip()
    if not blob:
        logger.warning("market query author: empty completion")
        return ()
    # The canonical extractors rather than a local fence regex plus a widest-brace slice. The
    # slice spanned from the first `{` to the LAST `}` anywhere in the output, so a well-formed
    # object followed by ordinary prose containing a brace ("Note use {care}.") failed to parse
    # and reported the additive stage lost on a usable payload. `iter_balanced_braces` (behind
    # `extract_first_balanced_braces`) is string-literal-aware and stops at the object's own
    # closing brace.
    object_blob = extract_json_block(blob) or extract_first_balanced_braces(blob)
    if object_blob is None:
        logger.warning(f"market query author: no JSON object found in {blob[:160]!r}")  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # log truncation, not data sampling
        return ()
    try:
        parsed = json.loads(object_blob)
    except ValueError as exc:
        # ValueError, not json.JSONDecodeError: CPython's 4300-digit int-conversion cap raises a
        # BARE ValueError from `json.loads`, which JSONDecodeError does not cover — the same hole
        # `ranking._first_usable_array` closes on the ranker side.
        logger.warning(f"market query author: object did not parse: {exc}")
        return ()
    if not isinstance(parsed, dict):
        logger.warning(f"market query author: parsed to {type(parsed).__name__}, not an object")
        return ()

    synonyms = _authored_strings(parsed.get("synonyms"))[:MAX_SYNONYMS]
    framings = _authored_strings(parsed.get("framings"))[:MAX_FRAMINGS]
    if not synonyms and not framings:
        logger.warning(f"market query author: neither key carried usable strings: keys={sorted(parsed)}")
        return ()
    return tuple(dedupe_queries([*synonyms, *framings]))


def fuzzy_best(queries: list[str], title: str, rules: str) -> float:
    """`0.7 * title + 0.3 * rules` token_set_ratio, maxed over the query set.

    The incumbent's Kalshi scorer shape, kept because it is the MEASURED picker: you cannot put
    9,762 Kalshi events in a prompt, so something has to order them and take the top 100. It is
    a ranker only — there is no `min_score` parameter, deliberately, so the floors that dropped
    the adjacent-cut markets cannot return by way of a defaulted argument.
    """
    title_lower = title.lower()
    rules_lower = rules.lower()
    best = 0.0
    for query in queries:
        q = query.lower()
        title_score = fuzz.token_set_ratio(q, title_lower)
        rules_score = fuzz.token_set_ratio(q, rules_lower) if rules_lower else 0.0
        best = max(best, 0.7 * title_score + 0.3 * rules_score)
    return best


def fuzzy_best_many(queries: list[str], titles: Sequence[str], rules: Sequence[str]) -> list[float]:
    """``fuzzy_best`` over a whole catalogue at once. One score per title, same order.

    Batched here beside the scalar form so the ``0.7 * title + 0.3 * rules`` weighting and the
    scorer choice stay in ONE module: inlining them at the call site would create a second live
    copy that silently diverges from the settlement-join channel's re-rank.

    ``process.cdist`` rather than a Python loop because the loop does not actually free the event
    loop. The whole point of ``build_pool``'s ``to_thread`` hop is to get ~9,762 events x ~17
    queries off the loop, but every ``token_set_ratio`` call returns to Python bytecode holding
    the GIL, so the offload converted one long freeze into sustained sub-quantum starvation:
    measured 0.46-0.59s of scoring with loop lag p50 rising 1.3ms -> 15-20ms, and at the
    ``DEFAULT_MAX_CONCURRENT_RESEARCH = 6`` the code actually runs at, 3.25s wall with p50 56ms
    and the loop receiving 7.8% of its ticks — the window in which sibling questions' soft
    deadlines advance while their tasks cannot run. ``cdist`` releases the GIL and threads
    internally: 0.055-0.106s for the same work, loop lag back to idle, and verified BIT-identical
    to the scalar form at ``dtype=np.float64`` (the float32 default drifts ~5e-06, which would
    reorder ties). The win survives a 2-core runner — it is the GIL release, not the core count.

    The empty guard is LOAD-BEARING, not defensive noise: ``deterministic_queries("")`` returns
    ``[]``, and ``.max(axis=0)`` over a zero-row array raises ``ValueError``.
    """
    if not queries or not titles:
        return [0.0] * len(titles)
    lowered = [query.lower() for query in queries]
    title_scores = process.cdist(
        lowered, [title.lower() for title in titles], scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64
    )
    rules_scores = process.cdist(
        lowered, [rule.lower() for rule in rules], scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float64
    )
    return (0.7 * title_scores + 0.3 * rules_scores).max(axis=0).tolist()
