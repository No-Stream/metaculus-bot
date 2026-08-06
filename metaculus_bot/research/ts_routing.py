"""Deterministic routing for the time-series-anchor provider: question → resolving series.

No LLM anywhere. Two paths, in precedence order:
  (a) URL extraction from resolution_criteria + fine_print (FRED series / Yahoo
      ticker URLs — the resolving source, highest precedence);
  (b) a conservative curated title-keyword registry.
Ambiguous → None + log. Two Yahoo tickers with relative-return wording → a
relative-return spread route; other two-ticker framings (a ratio, a price difference, or a
single ticker's level) skip — a mean-zero relative-log-return band in percentage points is
wrong-unit for those.

``route_question`` is the entry point and ``_Route`` is what it returns. The estimator math
lives in ``ts_estimators``, rendering in ``ts_render``, and the provider factory plus the
public entry points in ``timeseries_anchor``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import unquote

from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.research.ts_fetch import FRED_NON_REVISING_SERIES, SeriesSpec

logger = logging.getLogger(__name__)

YfColumn = Literal["Close", "High", "Low", "Open"]
# How the fetched raw series is turned into the quantity the question resolves on.
# Ported from the Phase-A replay's ``derive_config`` (scratch/ts_anchor_replay_2026-07-16):
#   level       — raw period-end value (optionally unit-scaled, e.g. BOPGTB ÷1000).
#   mom_diff    — month-over-month first difference (PAYEMS jobs added; scale ×1000).
#   mom_pct     — month-over-month % change (CPI MoM inflation).
#   monthly_avg — monthly mean of a higher-frequency series (weekly gasoline → month).
Derivation = Literal["level", "mom_diff", "mom_pct", "monthly_avg"]

# Mirror financial_data's URL extraction regexes (kept local to avoid importing
# financial_data's yfinance/fredapi module chain just for two patterns).
_FRED_SERIES_URL_RE = re.compile(r"fred\.stlouisfed\.org/series/([A-Za-z0-9_]+)")
_YAHOO_TICKER_URL_RE = re.compile(r"finance\.yahoo\.com/quote/([A-Za-z0-9.^=\-]+)")
# Metaculus injects markdown backslash escapes into rendered URLs (à la
# resolution_source.strip_markdown_escapes); undo them before matching.
_MARKDOWN_ESCAPE_RE = re.compile(r"\\([_&.\-#()])")
# "Highest / peak / maximum" framings resolve on the forward-window max, not the
# period-end level.
_MAX_KEYWORD_RE = re.compile(r"\b(highest|peak|maximum|max)\b", re.IGNORECASE)
# The two-ticker spread route builds a mean-zero relative-log-return band in percentage
# points. That is the right quantity ONLY for the relative-return family ("X's returns
# exceed Y's"). A two-ticker question asking for a ratio, a price difference, or a single
# ticker's level resolves in a different unit entirely (a ~85x ratio, dollars), so the
# spread band would be actively wrong-unit — gate the route on this wording and skip
# otherwise. Kept conservative: return / outperform / relative only (not bare "vs", which
# appears in ratio and level comparisons too).
#
# Word-boundary matching (not a bare substring scan): the "return(s) TO <level>"
# construction is a LEVEL question ("will the price return to $400"), not a relative
# return, so it must NOT route to the spread band. The `(?!\s+to\b)` lookahead excludes
# exactly that phrasing while keeping the observed relative-return wordings — "X's returns
# exceed Y's" (plural, followed by "exceed") and "return(URL) minus return(URL)" (singular,
# followed by "(" not " to"). `outperform` stays a prefix match (outperform/outperforms/
# outperformed); `relative` stays a whole-word match (kept broad — genuine "performance of
# X relative to Y" phrasings must still route, and the bounds backstop already catches
# ratio/level shapes that slip through).
_RELATIVE_RETURN_RE = re.compile(r"\breturns?\b(?!\s+to\b)|\boutperform|\brelative\b", re.IGNORECASE)

# Wording that says the question resolves on a DIFFERENCE between two series, or on a
# CHANGE in one, rather than on a level. Every registry entry serves a level (or a declared
# derivation of one), so a single-series level band on one of these shapes is the wrong
# quantity — usually the wrong unit too (a two-leg Treasury spread resolves in basis points
# while the leg's level is in percent).
#
# The URL branch already refuses the two-URL version of this via its ambiguity check, but a
# question that names both legs in prose and cites no URL reaches the keyword registry with
# nothing to catch it, and the magnitude backstop does not: a 4.4-4.95 percent level band
# against a −50..50 basis-point displayed range scores INSIDE that range once the open-bound
# tolerance widens it (pinned in the tests).
#
# Applied as a ROUTE-level guard rather than per-entry `exclude_keywords`, because an exclude
# removes the entry from ambiguity detection too: excluding DGS10 on "the 10-year treasury
# yield versus the high yield spread" left the HY-OAS entry as the sole match, so a two-leg
# question that used to skip as ambiguous would have routed to one leg. The guard runs AFTER
# the ambiguity check, so it can only ever turn a route into a skip.
_TWO_LEG_OR_CHANGE_RE = re.compile(
    r"\bbasis point|\bbps\b|\bspread\b|\bminus\b|\bversus\b|\bvs\.?\b|\bdifference between\b", re.IGNORECASE
)

# Registry entries whose own series IS the published spread/difference, so the wording above
# describes exactly what they serve rather than a quantity they cannot express.
_SPREAD_NATIVE_SERIES: frozenset[str] = frozenset({"BAMLH0A0HYM2"})

# The non-revising FRED allowlist lives in ts_fetch (fetch-layer knowledge shared with
# financial_data); imported above as FRED_NON_REVISING_SERIES.


@dataclass(frozen=True)
class _TemplateEntry:
    keywords: tuple[str, ...]
    series_id: str
    source: Literal["fred", "yfinance"]
    label: str
    # How the question resolves relative to the fetched series. Non-"level" entries
    # (PAYEMS MoM change, CPI MoM inflation, gasoline monthly average) fit the band on
    # the DERIVED quantity — the level-with-caveat hack is gone.
    derivation: Derivation = "level"
    # Unit conversion applied inside the derivation (BOPGTB is millions of USD, the
    # trade-balance question resolves in billions → scale=0.001; PAYEMS diff ×1000 is
    # carried by the mom_diff branch, applied on top of scale).
    scale: float = 1.0
    # Extra keyword logic to disambiguate near-collisions under substring matching. An
    # entry matches iff (any `keywords`) AND (all `require_keywords`) AND (any
    # `require_any_keywords`) AND (no `exclude_keywords`). `require_keywords` (all-of) keeps
    # US vs. Australian unemployment questions from both matching (they share "unemployment
    # rate"). `require_any_keywords` (any-of) is the DERIVATION gate: a non-"level" entry
    # fits only when the question's own wording asks for the derived quantity (MoM change /
    # MoM % / monthly average). Without it a question that merely NAMES the series but
    # resolves on a different quantity — an index-LEVEL, year-over-year, or foreign-country
    # CPI question, a payroll LEVEL question — gets an empirical band in the wrong units.
    require_keywords: tuple[str, ...] = ()
    require_any_keywords: tuple[str, ...] = ()
    exclude_keywords: tuple[str, ...] = ()
    model_target: bool = True
    note: str = ""
    # NOTE: whether a fred series revises is NOT stored here — it is derived from the
    # single source of truth `FRED_NON_REVISING_SERIES` (ts_fetch) in `_single_spec`
    # (default: revises=True unless allowlisted), so URL-routed and registry-routed series
    # share one revises decision. A per-entry flag here would silently do nothing (and mask
    # a leakage bug if it disagreed with the allowlist).


# Wording that scopes a quantity to a whole MONTH rather than a point in time. GASREGW is a
# WEEKLY series, so the two gasoline entries below split on exactly these tokens: the
# monthly-average entry REQUIRES one of them, the weekly-level entry EXCLUDES all of them.
# Sharing one tuple is what makes them exact complements rather than two keyword lists that
# happen to agree — the gap between a required-token list and a separately-maintained excluded
# one is where the point-in-time gasoline family previously fell through both gates and landed
# nowhere. Any phrasing reaches exactly one entry, so neither derivation can leak into the
# other's family and the pair can never co-match into an ambiguity skip.
_MONTH_SCOPED_KEYWORDS = ("for the month", "monthly average", "during the month", "monthly")
_GASOLINE_KEYWORDS = ("regular gasoline", "regular gas price", "gasoline price")

# Conservative curated registry: title keyword(s) → resolving series. Kept small
# and unambiguous on purpose; anything not here (or matching >1 entry) yields "".
_TEMPLATE_REGISTRY: tuple[_TemplateEntry, ...] = (
    _TemplateEntry(
        (
            "10-year treasury",
            "10 year treasury",
            "10-yr treasury",
            "10y treasury",
            "ten-year treasury",
            "10-year yield",
            # The recurring "ending value of the UST 10Y Yield for these biweekly periods"
            # family names the series in a form none of the DGS10 keywords above cover. Those
            # questions DO cite the DGS10 FRED URL in their resolution criteria and already
            # route through the URL branch, so these two tokens recover no question observed
            # today (all 9 of that family route pre-fix on real text). They are wording
            # robustness for a family whose criteria could drop the link, not new coverage.
            "ust 10y",
            "ust 10-year",
        ),
        "DGS10",
        "fred",
        "10-Year Treasury constant-maturity yield (%)",
    ),
    _TemplateEntry(
        ("high yield oas", "high-yield oas", "hy oas", "ice bofa", "high-yield spread", "high yield spread"),
        "BAMLH0A0HYM2",
        "fred",
        "ICE BofA US High Yield OAS (%)",
    ),
    _TemplateEntry(("vix",), "^VIX", "yfinance", "CBOE Volatility Index (VIX)"),
    _TemplateEntry(
        _GASOLINE_KEYWORDS,
        "GASREGW",
        "fred",
        "US regular gasoline price — monthly average ($/gal)",
        derivation="monthly_avg",
        # Gate: fit the monthly-average band only on questions that ask for a value over a
        # MONTH (the recurring "national average price ... for the month of X" family). Bare
        # "average" would misfire — the GEOGRAPHIC "national average" descriptor is in every
        # gasoline question, including spot/point-in-time ones, which must not inherit
        # monthly_avg (and the bounds backstop can't catch it: both are ~$3/gal). The tokens
        # here key on the monthly PERIOD framing instead.
        require_any_keywords=_MONTH_SCOPED_KEYWORDS,
    ),
    _TemplateEntry(
        _GASOLINE_KEYWORDS,
        "GASREGW",
        "fred",
        "US regular gasoline price — weekly level ($/gal)",
        # The complement of the entry above. GASREGW is weekly, so a point-in-time gasoline
        # question ("the average price ... on August 17") resolves on the weekly LEVEL, which
        # this series answers directly. Without this sibling the monthly gate correctly refused
        # those questions and they had nowhere else to land, so the whole point-in-time family
        # went dark while reading identically to a family that was never in scope.
        exclude_keywords=_MONTH_SCOPED_KEYWORDS,
    ),
    _TemplateEntry(("brent",), "DCOILBRENTEU", "fred", "Brent crude spot ($/bbl)"),
    _TemplateEntry(("wti", "west texas intermediate"), "CL=F", "yfinance", "WTI crude front-month ($/bbl)"),
    _TemplateEntry(
        ("cpi", "consumer price index"),
        "CPIAUCSL",
        "fred",
        "CPI-U all items — month-over-month % change (SA)",
        derivation="mom_pct",
        # Gate: CPIAUCSL is US headline CPI, and this entry fits ONLY the recurring US
        # month-over-month inflation family ("seasonally adjusted month over month headline
        # CPI inflation ... in the United States", qids 39567/41681). Require MoM language
        # so an index-LEVEL question, a year-over-year question, or a foreign-country CPI
        # question (UK 12-month YoY, Egypt urban CPI — real tournament questions that route
        # here today) does NOT get a US-MoM-% band in the wrong units / wrong country. The
        # excludes are belt-and-suspenders for the YoY / level / foreign markers that lack
        # MoM language anyway; " uk " uses both spaces so it never fires on "ukraine".
        require_any_keywords=("month-over-month", "month over month", "mom "),
        exclude_keywords=(
            "year-over-year",
            "year over year",
            "yoy",
            "12-month",
            "annual",
            "index level",
            "united kingdom",
            " uk ",
            "egypt",
        ),
    ),
    _TemplateEntry(
        ("unemployment rate",),
        "UNRATE",
        "fred",
        "US unemployment rate (%)",
        exclude_keywords=("australia", "australian"),
    ),
    _TemplateEntry(
        ("nonfarm payroll", "non-farm payroll", "payroll employment", "payrolls"),
        "PAYEMS",
        "fred",
        "Nonfarm payrolls — month-over-month change (jobs added, persons)",
        derivation="mom_diff",
        scale=1000.0,  # PAYEMS is in thousands of persons; the question resolves in persons.
        # Gate: the recurring family resolves on the CHANGE in payroll employment ("change
        # in seasonally adjusted nonfarm payroll employment", qids 40100/40099/38829). A
        # payroll LEVEL question (~160M persons) must not inherit the mom_diff band, whose
        # ±hundreds-of-thousands scale is a different quantity entirely.
        require_any_keywords=("change", "jobs added", "added", "gain", "gained"),
    ),
    _TemplateEntry(
        ("consumer sentiment", "michigan sentiment", "umich sentiment", "consumer confidence"),
        "UMCSENT",
        "fred",
        "U. Michigan consumer sentiment index",
    ),
    # --- Replay-validated additions (scratch/ts_anchor_replay_2026-07-16/series_map.json) ---
    _TemplateEntry(
        ("s&p 500", "s&p500", "sp 500", "sp500", "standard & poor's 500", "standard and poor's 500"),
        "^GSPC",
        "yfinance",
        "S&P 500 index level",
    ),
    _TemplateEntry(
        ("bitcoin", "btc-usd", "price of btc"),
        "BTC-USD",
        "yfinance",
        "Bitcoin price ($; daily High on max/highest questions)",
    ),
    _TemplateEntry(
        ("gold price", "price of gold", "spot gold", "gold futures"),
        "GC=F",
        "yfinance",
        "Gold front-month futures ($/oz; daily High on max/highest questions)",
    ),
    _TemplateEntry(
        ("silver price", "price of silver", "spot silver", "silver futures"),
        "SI=F",
        "yfinance",
        "Silver front-month futures ($/oz; daily High on max/highest questions)",
    ),
    _TemplateEntry(
        ("case-shiller", "case shiller", "national home price", "home price index", "house price index"),
        "CSUSHPISA",
        "fred",
        "S&P/Case-Shiller US National Home Price Index (SA)",
    ),
    _TemplateEntry(
        ("australia", "australian"),
        "LRHUTTTTAUM156S",
        "fred",
        "Australia unemployment rate (%, SA; OECD/FRED mirror)",
        require_keywords=("unemployment",),
    ),
    _TemplateEntry(
        ("federal funds target", "fed funds target", "federal funds rate", "fed funds rate"),
        "DFEDTARU",
        "fred",
        "Federal funds target range, upper limit (%)",
    ),
    _TemplateEntry(
        ("average weekly hours", "weekly hours"),
        "AWHAETP",
        "fred",
        "US average weekly hours, all employees, total private (SA)",
    ),
    _TemplateEntry(
        (
            "goods trade balance",
            "advance goods trade",
            "trade balance",
            "goods trade deficit",
            "trade deficit in goods",
        ),
        "BOPGTB",
        "fred",
        "US advance goods trade balance ($B)",
        scale=0.001,  # FRED reports BOPGTB in millions of USD; the question resolves in billions.
    ),
)


@dataclass(frozen=True)
class _Route:
    """A resolved routing decision: what series to fetch and how to render it."""

    kind: Literal["single", "spread"]
    spec: SeriesSpec
    label: str
    spec_b: SeriesSpec | None = None  # second leg for spreads
    label_b: str = ""
    model_target: bool = True
    is_max: bool = False  # forward-window-max question (VIX/Brent "highest") vs period-end level
    derivation: Derivation = "level"  # how the raw series maps to the resolved quantity
    scale: float = 1.0  # unit conversion inside the derivation (BOPGTB ÷1000, PAYEMS diff ×1000)
    note: str = ""


def _extract_url_identifiers(criteria: str, fine_print: str) -> tuple[list[str], list[str]]:
    """Return (fred_series, yahoo_tickers) cited in the question's resolution text.

    URL-decodes (``%5E`` → ``^``) and unescapes Metaculus markdown backslashes
    before matching, matching financial_data / resolution_source handling.
    Order-preserving dedup.
    """
    text = _MARKDOWN_ESCAPE_RE.sub(r"\1", unquote(f"{criteria}\n{fine_print}"))
    fred = list(dict.fromkeys(_FRED_SERIES_URL_RE.findall(text)))
    # The Yahoo char class swallows a sentence-final '.'; strip trailing dots only
    # (internal dots like DX-Y.NYB are preserved).
    tickers = list(dict.fromkeys(t.rstrip(".") for t in _YAHOO_TICKER_URL_RE.findall(text)))
    return fred, tickers


def _wants_max(text: str) -> bool:
    return bool(_MAX_KEYWORD_RE.search(text))


def _wants_relative_return(text: str) -> bool:
    """True when the question wording asks for a relative return between the two tickers
    (the only quantity the spread band represents). Word-boundary match on
    ``_RELATIVE_RETURN_RE``, which excludes the "return(s) to <level>" level-question
    construction while keeping return / outperform / relative wordings."""
    return bool(_RELATIVE_RETURN_RE.search(text))


def _entry_quantity_ok(entry: _TemplateEntry, lowered: str) -> bool:
    """The derivation gate, independent of title-keyword matching: at least one
    ``require_any_keywords`` present (when any are declared) AND no ``exclude_keywords``.

    Split out so the URL route can reuse it: a URL pins WHICH series resolves the question,
    but a non-"level" derivation still only fits when the question actually asks for the
    derived quantity — so a year-over-year / index-level / foreign-country CPI question
    citing the US-MoM-CPI FRED URL is skipped rather than handed a wrong-units band."""
    if entry.require_any_keywords and not any(kw in lowered for kw in entry.require_any_keywords):
        return False
    return not any(kw in lowered for kw in entry.exclude_keywords)


def _entry_matches(entry: _TemplateEntry, lowered: str) -> bool:
    """An entry matches iff any of its keywords appear AND every require_keyword appears
    AND the derivation gate (``_entry_quantity_ok``) passes (all case-folded).
    require/require_any/exclude disambiguate substring collisions (US vs. Australian
    'unemployment rate') and keep a derived-quantity entry off a wrong-units question
    (MoM vs. YoY/level CPI, payroll change vs. level)."""
    if not any(kw in lowered for kw in entry.keywords):
        return False
    if not all(kw in lowered for kw in entry.require_keywords):
        return False
    return _entry_quantity_ok(entry, lowered)


def _registry_entry_for_series(
    series_id: str, source: Literal["fred", "yfinance"], lowered: str
) -> _TemplateEntry | None:
    """Registry entry whose series_id matches (case-insensitive) a URL-cited series.

    A URL pins WHICH series resolves the question, but the registry still holds HOW the
    raw series maps to the resolved quantity (derivation / scale / model_target / note /
    label). Without this, a question citing a FRED URL for e.g. PAYEMS or CPIAUCSL would
    render a raw unscaled level band for a MoM-change / MoM-% quantity — actively
    misleading. None when no registry entry declares this series.

    One series may carry several entries that differ only in derivation (GASREGW has a
    monthly-average and a weekly-level sibling), so pick the one whose quantity gate the
    question's own wording satisfies. Those siblings are built as exact complements, so at most
    one can pass. When none passes, return the first candidate so the caller still reports a
    concrete derivation in its skip log."""
    sid = series_id.upper()
    candidates = [e for e in _TEMPLATE_REGISTRY if e.source == source and e.series_id.upper() == sid]
    if not candidates:
        return None
    return next((e for e in candidates if _entry_quantity_ok(e, lowered)), candidates[0])


def _single_spec(series_id: str, source: Literal["fred", "yfinance"], text: str) -> SeriesSpec:
    """Build the SeriesSpec for one leg, honoring ALFRED-revising and daily-High."""
    if source == "fred":
        # Default to ALFRED vintages (revises=True) for every fred series except the
        # curated non-revising allowlist. Unknown / URL-cited series therefore fetch
        # point-in-time, which is leakage-safe (see FRED_NON_REVISING_SERIES).
        revises = series_id.upper() not in FRED_NON_REVISING_SERIES
        return SeriesSpec(source="fred", series_id=series_id, revises=revises)
    column: YfColumn = "High" if _wants_max(text) else "Close"
    return SeriesSpec(source="yfinance", series_id=series_id, column=column)


def _route_from_entry(entry: _TemplateEntry, spec: SeriesSpec, *, is_max: bool) -> _Route:
    """Single-series route carrying a registry entry's render metadata.

    Shared by the URL branch and the keyword branch so the eight fields only have to agree
    in one place (the two used to construct ``_Route`` identically, side by side)."""
    return _Route(
        kind="single",
        spec=spec,
        label=entry.label,
        model_target=entry.model_target,
        is_max=is_max,
        derivation=entry.derivation,
        scale=entry.scale,
        note=entry.note,
    )


def _single_url_route(series_id: str, source: Literal["fred", "yfinance"], text: str, *, is_max: bool) -> _Route | None:
    """Route a single URL-cited series. The URL pins WHICH series; a matching registry
    entry still supplies HOW to derive/scale/label it (else a MoM-change or unit-scaled
    question renders a misleading raw-level band). Non-matching series keep dataclass
    defaults (level, scale=1.0).

    Returns None when the cited series has a NON-"level" registry derivation but the
    question text fails that entry's quantity gate (``_entry_quantity_ok``): a MoM-%,
    MoM-change, or monthly-average band on a question asking for a level / YoY / foreign
    quantity is worse than no anchor, and we deliberately do NOT fall back to a raw-level
    band (see ``_registry_entry_for_series`` on why that would be misleading). A series with
    a level sibling (GASREGW) reaches that sibling instead of skipping, because the sibling is
    a genuine route for the question's quantity rather than a fallback."""
    lowered = text.lower()
    spec = _single_spec(series_id, source, text)
    entry = _registry_entry_for_series(series_id, source, lowered)
    if entry is None:
        return _Route(kind="single", spec=spec, label=series_id, is_max=is_max)
    if entry.derivation != "level" and not _entry_quantity_ok(entry, lowered):
        logger.info(
            "ts_anchor: URL-cited %s carries registry derivation %r but the question text lacks the "
            "matching quantity language (or hits an exclude) — skipping to avoid a wrong-units band",
            series_id,
            entry.derivation,
        )
        return None
    return _route_from_entry(entry, spec, is_max=is_max)


def route_question(question: MetaculusQuestion) -> _Route | None:
    """Deterministically route a question to a series, or None if it doesn't map.

    URL extraction wins over the keyword registry (the cited source is the ground
    truth). Two Yahoo tickers → a relative-return spread. A single cited series →
    a level/max anchor. Anything ambiguous (>1 series that isn't the 2-ticker
    spread, or >1 registry keyword match) returns None with a log line.
    """
    text = f"{question.question_text or ''}\n{question.resolution_criteria or ''}\n{question.fine_print or ''}"
    is_max = _wants_max(text)
    fred, tickers = _extract_url_identifiers(question.resolution_criteria or "", question.fine_print or "")

    total_ids = len(fred) + len(tickers)
    if total_ids >= 1:
        if len(tickers) == 2 and not fred:
            # The spread band is a mean-zero relative-log-return in pp — right only for the
            # relative-return family. Skip a ratio / price-difference / single-level
            # two-ticker question rather than inject a wrong-unit band.
            if not _wants_relative_return(text):
                logger.info(
                    "ts_anchor: two Yahoo tickers %s but no relative-return wording "
                    "(return/outperform/relative) for qid=%s — skipping (spread band would be wrong-unit)",
                    tickers,
                    getattr(question, "id_of_question", None),
                )
                return None
            return _Route(
                kind="spread",
                spec=SeriesSpec(source="yfinance", series_id=tickers[0], column="Close"),
                label=tickers[0],
                spec_b=SeriesSpec(source="yfinance", series_id=tickers[1], column="Close"),
                label_b=tickers[1],
            )
        if total_ids == 1:
            if fred:
                return _single_url_route(fred[0], "fred", text, is_max=is_max)
            return _single_url_route(tickers[0], "yfinance", text, is_max=is_max)
        logger.info(
            "ts_anchor: ambiguous URL routing (fred=%s tickers=%s) for qid=%s — skipping",
            fred,
            tickers,
            getattr(question, "id_of_question", None),
        )
        return None

    lowered = text.lower()
    matches = [e for e in _TEMPLATE_REGISTRY if _entry_matches(e, lowered)]
    if not matches:
        return None
    if len(matches) > 1:
        logger.info(
            "ts_anchor: ambiguous keyword routing (%s) for qid=%s — skipping",
            [m.series_id for m in matches],
            getattr(question, "id_of_question", None),
        )
        return None

    entry = matches[0]
    if entry.series_id not in _SPREAD_NATIVE_SERIES and _TWO_LEG_OR_CHANGE_RE.search(text):
        logger.info(
            "ts_anchor: keyword-matched %s but the question wording asks for a spread/difference/change "
            "(qid=%s) — skipping rather than anchoring a single-series level band on a different quantity",
            entry.series_id,
            getattr(question, "id_of_question", None),
        )
        return None
    return _route_from_entry(entry, _single_spec(entry.series_id, entry.source, text), is_max=is_max)
