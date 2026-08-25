"""The rendered snapshot: a markdown table in the ranker's order, plus the rules bullets.

Five things here are contracts rather than formatting choices:

- **Zero rendered rows returns ``""`` before any preamble is emitted.** That early return is
  what produces ``status="empty"`` downstream and the attempted-vs-succeeded distinction
  residual analysis reads off the archive. Under ranked selection a zero-row render is a
  legitimate outcome (the model is allowed to say nothing bears on the question), so this is
  a hot path, not a theoretical one. The seam (``prediction_market.format_snapshot_for_research``)
  substitutes ``render_no_relevant_market_line`` for exactly that deliberate-zero case, so the
  forecaster can tell a considered empty answer from an outage; every failure path still
  returns ``""``.
- **The rows are rendered in the ranker's order, verbatim.** No venue interleave, no
  fairness pass, no per-venue cap, no re-scoring. Round-robin venue fairness is exactly what
  lost 43 of 58 wanted rows in the measurement this port is built on.
- **A multi-outcome market renders one ``↳`` sub-row per outcome, in the venue adapter's order,
  also verbatim.** The same rule one level down: the adapter knows which of its outcomes are worth
  keeping (price on the three price-bearing venues, ballot order on PredictIt) and the renderer
  truncates from the END, so re-sorting here would silently change what survives the budget.
- **Every number in the ``prob`` column is a probability, unless it is labelled as something
  else.** The one exception is a scalar market's own value, and it renders as ``value N (scale LOW
  to HIGH)`` for exactly that reason — the forecaster prompts tell a model to anchor on this cell,
  so an unlabelled non-probability here is not a cosmetic defect. Manifold shipped one: a
  ``PSEUDO_NUMERIC`` market's ``probability`` field is a position on its value scale, and the cell
  read ``0.48`` on a market whose estimate was 120.97 years. ``_price_cell`` is the only writer.
- **The rules-bullet section keeps its byte shape** (``- **{platform}** <{url}>: {body}`` at
  h3). The MC per-model option parser scans for ``- <name>: NN%`` and the bullets sit close
  enough to a per-model region for that to matter, so nothing percentage-shaped may ever be
  APPENDED to a bullet. Nothing is: a body is the market's rules text verbatim. A multi-outcome
  market's answer probabilities used to ride inside it, parenthesised for exactly that reason,
  and moved out to the sub-rows — see ``_bullet_body``.

The preamble/legend strings mirror the ones the provider ships today, minus the fuzzy-match
vocabulary the ranker replaces. Their asserted substrings are load-bearing: the ranked design
changes how a row is chosen, not what a forecaster is told to verify.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from metaculus_bot.research.market_retrieval.ranking import (
    DEGRADED_RANKING_MARKER,
    STRONG_TIERS,
    TIER_UNSPECIFIED,
    WHY_CHARS,
)
from metaculus_bot.research.market_retrieval.types import (
    MarketChild,
    MarketMatch,
    MarketSnapshot,
    ScalarEstimate,
    _child_liquidity_label,
    _liquidity_label,
    format_scalar_number,
)

# Significant digits for a scalar market's VALUE, which is deliberately shorter than the bounds
# beside it (`types.SCALAR_BOUND_SIG_DIGITS`, which lives there because the ranker's candidate line
# renders the same bounds). The two differ because the numbers do: a value is a noisy crowd estimate,
# so six digits already say more than it means (120.96691732988944 -> `120.967`) and a short cell
# reads better, while a bound is a parameter the market's author chose and rounding it would
# misstate the market's own scale.
SCALAR_VALUE_SIG_DIGITS = 6

# Separate from, and smaller than, the ranker prompt's per-venue caps: this text ships to the
# forecaster inside a research section that has its own character budget, and the two must not be
# collapsed into one constant. A bullet is now rules text alone — a multi-outcome market's outcome
# names and prices moved out of the body and into `↳` sub-rows on 2026-08-05, where each gets its
# own price, volume, close date and status instead of a parenthesised percentage.
RAW_BULLET_BODY_MAX_CHARS = 200

# The `↳` sub-row bounds. Three separate ceilings, because they fail differently.
#
# `MAX_CHILD_ROWS_PER_MARKET` bounds ONE market: a Manifold threshold ladder ships 17 answers and a
# Kalshi strike family can carry more, and past the first handful (each adapter orders by what makes
# an outcome worth keeping) the rungs are the tail of a distribution whose shape the leaders already
# gave. Generous rather than tight — the operator's directive is maximum information per market.
#
# `MAX_CHILD_ROWS_PER_SNAPSHOT` bounds the SECTION, and it is the one that has to exist: 86.5% of
# the Kalshi catalogue is multi-strike, so a realistic 8-row slate is mostly parents with children,
# and the per-market cap alone would license 80 sub-rows — roughly doubling a section that goes to
# the expensive forecaster models on every question. Its value is chosen against `RENDER_BUDGET`
# rather than picked: at 24 against 8 rendered rows it guarantees three full rounds, and the
# guarantee that matters is the FIRST one — while this is >= `RENDER_BUDGET`, no multi-outcome market
# can ever render priceless, which is the whole failure this expansion exists to end. Every market
# whose outcomes were cut says so in a marker row, so a thinned table never reads as a complete one.
#
# `CHILD_TITLE_MAX_CHARS` is well under the parent's `TITLE_MAX_CHARS`: an outcome label is a rung,
# not a question ("Republican Party", "Before Nov 1, 2026", "0 (0 bps)", "$3.80 - $4.19" — 9-18
# chars measured across all four venues' committed fixtures), so this only bounds a pathological one.
MAX_CHILD_ROWS_PER_MARKET = 10
MAX_CHILD_ROWS_PER_SNAPSHOT = 24
CHILD_TITLE_MAX_CHARS = 48

# The continuation glyph, in the `platform` cell: a sub-row's venue is its parent's, one line up, and
# repeating it would spend ~10 chars a row restating that. It occupies the platform column rather
# than prefixing the title so the title cell holds nothing but the outcome's own label.
CHILD_ROW_MARKER = "↳"

TABLE_COLUMNS: tuple[str, ...] = (
    "platform",
    "title",
    "prob",
    "total_vol",
    "OI",
    "signal",
    "close",
    "status",
    "relation",
    "why",
)

TITLE_MAX_CHARS = 80

# Shared trailing legend, shipped INSIDE the research section. The liquidity half is unchanged;
# the retired `relevance` column's content-overlap explanation is replaced by ONE sentence per
# new axis — the ranker's relation + why, and what RESOLVED means about a price. This text is
# fixed overhead on every rendered snapshot and the snapshot goes to the expensive forecaster
# models, so it earns its length: every label a cell can hold is named (a legend that omits one
# teaches forecasters to guess at it) and nothing else is.
MARKET_SIGNAL_LEGEND = (
    "The `signal` column labels each market's liquidity/participation "
    "(thin/decent/deep for real-money venues, thin/decent/high for Manifold's play-money bettor count); "
    "`total_vol` and `OI` are that market's traded volume and open interest in approximate USD on the "
    "real-money venues, and play-money mana on Manifold. "
    "`no-liquidity-data` means the venue publishes no volume figures at all (PredictIt) — it says nothing "
    "about how liquid the market is, so treat it as unknown rather than as thin. Treat "
    "deep/high-liquidity markets as a strong anchor and discount thin ones (low volume, few participants) as noisy. "
    "Rows are ordered by EVIDENTIAL VALUE, best first, and `relation` grades how each bears on THIS question — "
    "`same_quantity_same_date`, then `same_quantity_other_cut`, then `driver_or_consequence`, then `weak` "
    "(`unspecified` if ungraded); only the first two measure the quantity asked about, and `why` is the "
    "one-phrase reason. `status` is `open` or `RESOLVED`; a RESOLVED price is a realized outcome, not a forecast. "
    "A `↳` row is one OUTCOME of the market above it (strike, bracket, ballot line) with its own price; the parent "
    "row has none, so anchor on the outcome that matches this question. "
    "A `prob` cell prefixed `value` is a SCALAR market's estimate of the quantity itself, in the market's own units "
    "on the scale shown beside it — not a probability."
)

# Strong-evidence framing — used when at least one rendered row measures the same quantity
# (relation tier 1 or 2). "MAY be relevant" and "verify each market's resolution criteria" are
# asserted verbatim elsewhere; the old "the match below is fuzzy" clause is gone, because the
# rows are now selected by a model reading each market's rules rather than by word overlap. The
# old "a poorly-matched market may be worth little or nothing" clause is gone too — the
# per-row `relation` grade now says that per row, and saying it twice is the verbosity the
# forecaster-facing budget exists to stop.
MARKET_PREAMBLE_STRONG = (
    "The following prediction markets MAY be relevant — each was selected and ranked for THIS question, so "
    "verify each market's resolution criteria, resolution date, and topic against THIS question before "
    "weighting. A market whose criteria and date match this question is extremely strong evidence — anchor on "
    "its price; on a related but different event, date, or threshold, name the specific mismatch and discount "
    "accordingly. "
)

# Neutral framing — used when NO rendered row measures the same quantity, so the table is
# context at best. "may all be off-topic" is asserted verbatim elsewhere.
MARKET_PREAMBLE_NEUTRAL = (
    "The following prediction markets were retrieved for this question and may all be off-topic — none was "
    "judged to measure the quantity THIS question asks about, so treat them as leads to verify, not as "
    "evidence. Weight a market only after you confirm its resolution criteria, date, and topic match this "
    "question; otherwise disregard it. "
)


def render_no_relevant_market_line(pool_size: int) -> str:
    """The section body for a DELIBERATE zero-row ranking over a non-empty candidate pool.

    Without it the section vanishes wholesale on an adaptive-width-zero answer, which reads
    exactly like a provider outage — the run log records the difference (``outcome=ranked
    rows=0``) but the forecaster prompt still ships the full relation/liquidity weighting
    clauses for a table that isn't there. One sentence closes that: the empty table becomes a
    considered judgment the forecaster can lean on rather than an absence to guess about. Only
    the ranker's own empty answer earns it; every failure path still renders nothing (see
    ``prediction_market.format_snapshot_for_research``).
    """
    return (
        f"No sufficiently relevant market among {pool_size} candidates — prediction markets were "
        "retrieved and reviewed for this question, and none was judged to bear on it. This is a "
        "deliberate empty result, not a provider outage."
    )


def _cell(text: str, *, limit: int | None = None) -> str:
    """One table cell: pipes neutralised, newlines flattened, optionally truncated.

    A raw `|` from a market title or a model-authored phrase would split the row into extra
    columns and silently shift every later cell — which is why the rendering tests select
    cells BY HEADER NAME rather than by index.
    """
    cleaned = (text or "").replace("|", "/").replace("\n", " ").strip()
    return cleaned if limit is None else cleaned[:limit]


def _price_cell(implied_prob_yes: float | None, scalar_estimate: ScalarEstimate | None) -> str:
    """The ``prob`` cell: a two-decimal probability, a labelled scalar value, or ``-``.

    A scalar market's number is prefixed with ``value`` and followed by its scale precisely so it
    CANNOT be read as a probability. A bare figure would not be enough: the failure this fixes
    rendered ``0.48`` for a market whose value was 120.97, and the reason that got through is that
    ``0.48`` is a perfectly plausible probability. Magnitude alone does not save the general case
    either — a scalar market on a 0-to-1 scale trades values that look exactly like probabilities —
    so the label does the work rather than the reader's arithmetic.

    A row can never hold both numbers — the venue parser fills one or the other — so the ordering of
    these branches is not a precedence rule. A value whose venue omitted the bounds still renders,
    labelled and without them: the number is the market's answer and the scale is context for it.
    """
    if implied_prob_yes is not None:
        return f"{implied_prob_yes:.2f}"
    if scalar_estimate is None:
        return "-"
    value = format_scalar_number(scalar_estimate.value, sig_digits=SCALAR_VALUE_SIG_DIGITS)
    bounds = scalar_estimate.bounds_text()
    return f"value {value} ({scalar_estimate.scale_label} {bounds})" if bounds else f"value {value}"


def _priced_cells(
    *,
    implied_prob_yes: float | None,
    total_volume: float | None,
    open_interest: float | None,
    close_time: datetime | None,
    is_resolved: bool,
    signal: str,
    scalar_estimate: ScalarEstimate | None = None,
) -> dict[str, str]:
    """The six columns a parent row and a ``↳`` sub-row format identically.

    Shared so the two can never disagree about how a price or a date reads: a sub-row that rounded
    its probability to three places, or wrote its dates the other way round, would look like a
    different kind of number rather than the same measurement one level down.

    ``scalar_estimate`` defaults to absent because only a parent row can carry one — no venue
    publishes a scalar OUTCOME inside a multi-outcome market — but it is formatted here rather than
    at the parent's call site so there stays exactly one place that decides what the ``prob`` column
    may contain.
    """
    return {
        "prob": _price_cell(implied_prob_yes, scalar_estimate),
        "total_vol": f"{total_volume:.0f}" if total_volume is not None else "-",
        "OI": f"{open_interest:.0f}" if open_interest is not None else "-",
        "signal": signal,
        "close": close_time.strftime("%Y-%m-%d") if close_time else "-",
        "status": "RESOLVED" if is_resolved else "open",
    }


def _row_cells(match: MarketMatch) -> dict[str, str]:
    return {
        "platform": match.platform,
        "title": _cell(match.market_title, limit=TITLE_MAX_CHARS),
        **_priced_cells(
            implied_prob_yes=match.implied_prob_yes,
            total_volume=match.total_volume,
            open_interest=match.open_interest,
            close_time=match.close_time,
            is_resolved=match.is_resolved,
            signal=_liquidity_label(match),
            scalar_estimate=match.scalar_estimate,
        ),
        "relation": _cell(match.relation_tier) or TIER_UNSPECIFIED,
        "why": _cell(match.relevance_label, limit=WHY_CHARS) or "-",
    }


def _child_cells(platform: str, child: MarketChild) -> dict[str, str]:
    """One outcome's sub-row: its own price and liquidity, its parent's venue and relation.

    ``relation`` and ``why`` are the RANKER's grades and the ranker never saw this outcome — it
    graded the market. Repeating the parent's grade on every sub-row would spend ~30 chars a row
    implying a judgement that was not made, so both read ``-`` and the indentation says whose
    grade applies.
    """
    return {
        "platform": CHILD_ROW_MARKER,
        "title": _cell(child.title, limit=CHILD_TITLE_MAX_CHARS),
        **_priced_cells(
            implied_prob_yes=child.implied_prob_yes,
            total_volume=child.total_volume,
            open_interest=child.open_interest,
            close_time=child.close_time,
            is_resolved=child.is_resolved,
            signal=_child_liquidity_label(platform, child),
        ),
        "relation": "-",
        "why": "-",
    }


def _child_allowances(matches: Sequence[MarketMatch]) -> list[int]:
    """How many sub-rows each row may render, filling ONE ROUND AT A TIME across the whole slate.

    So every multi-outcome market shows its leading outcome — the most-traded strike, the most
    probable bracket — before any market shows a second, and a market with fewer outcomes than the
    round count hands its unused slots back to the others. Draining the budget in rank order instead
    would spend it all on the top two rows and leave rows 6-8 rendering a bare ``-``, which is the
    exact defect this whole expansion exists to remove.

    NOT the venue round-robin the module docstring forbids, and the distinction is worth being
    precise about because the vocabulary collides. That one reordered the RANKED ROWS for venue
    fairness and cost 43 of 58 wanted rows. This changes no row's position and drops no row: the
    slate is still the ranker's order verbatim, and the only thing shared out is how deep each
    market's own outcome list is allowed to go.
    """
    allowances = [0] * len(matches)
    budget = MAX_CHILD_ROWS_PER_SNAPSHOT
    for _ in range(MAX_CHILD_ROWS_PER_MARKET):
        for index, match in enumerate(matches):
            if budget <= 0:
                return allowances
            if allowances[index] < len(match.children):
                allowances[index] += 1
                budget -= 1
    return allowances


def _table_row(cells: dict[str, str]) -> str:
    """One markdown row, columns in ``TABLE_COLUMNS`` order. A ``KeyError`` here is the point:
    every row shape — parent, sub-row, marker — must fill every column or the table skews."""
    return "| " + " | ".join(cells[column] for column in TABLE_COLUMNS) + " |"


def _omitted_children_cells(omitted_children: Sequence[MarketChild]) -> dict[str, str]:
    """A marker sub-row naming how many outcomes the budget cut, and the price mass they carried.

    Shaped as a table row rather than a trailing note because a bare line between rows would end
    the markdown table and orphan everything after it. The wording mirrors the resolution-source
    fetcher's ``[N additional source(s) omitted — section budget]``: same bracketed shape, same
    "which budget" clause, so a forecaster meeting either recognises the other.

    The price-mass clause is what turns a one-sided cut from silent into visible: a single
    surviving bracket read as an equality constraint on a tail is exactly the q45189 failure, and
    the price-descending child sort alone cannot fix it — a model must also know how much of the
    distribution it is NOT seeing before it treats the visible rungs as exhaustive. The figure sums
    only the outcomes that carry a price, so on an unquoted family it honestly reads 0.00.
    """
    omitted = len(omitted_children)
    priced_mass = sum(child.implied_prob_yes for child in omitted_children if child.implied_prob_yes is not None)
    plural = "" if omitted == 1 else "s"
    return {
        "platform": CHILD_ROW_MARKER,
        "title": f"[{omitted} more outcome{plural} omitted — render budget, {priced_mass:.2f} of priced mass]",
        "prob": "-",
        "total_vol": "-",
        "OI": "-",
        "signal": "-",
        "close": "-",
        "status": "-",
        "relation": "-",
        "why": "-",
    }


def _bullet_body(match: MarketMatch) -> str:
    """A bullet's text: the market's rules, verbatim and flattened to one line.

    A multi-outcome market's answers used to LEAD this body as ``answers: Over $4.60 (50%), ...``,
    because they were the row's only price and the table had nowhere to put them. The ``↳`` sub-rows
    are that place now, and each answer gets a real ``prob`` cell plus its own volume, close date and
    status there, so carrying them here too would pay twice for the same numbers and crowd out the
    rules text the sub-rows cannot show. That also retires the parenthesisation guard: the reason
    ``(50%)`` was parenthesised is that the per-model MC option parser reads ``- <name>: NN%`` and a
    bare percentage after a colon could arm that shape. Nothing the formatter adds to a bullet is
    percentage-shaped any more, and a table cell cannot be mistaken for a bullet.

    One duplication is knowingly left: a PredictIt bullet reads ``contracts: A, B, C`` and its
    sub-rows name the same contracts. The names are in ``raw_rules`` because the RANKER needs them
    (a PredictIt title is often just "Which party will win ..."), so splitting them out would change
    a candidate line to save ~150 chars on the venue that contributes the fewest rendered rows.
    """
    return (match.raw_rules or "").strip().replace("\n", " ")


def _preamble(matches: Sequence[MarketMatch]) -> str:
    """Strong when any rendered row measures the same quantity, neutral otherwise.

    The selector is the ranker's own top tier, replacing the content-overlap + confidence bar
    it retired. A fail-open slate carries no tier on any row, so it selects neutral — which
    is right: an unranked slate must never get an authoritative header.
    """
    return MARKET_PREAMBLE_STRONG if any(m.relation_tier in STRONG_TIERS for m in matches) else MARKET_PREAMBLE_NEUTRAL


def render_snapshot(snapshot: MarketSnapshot, *, ranking_degraded: bool = False) -> str:
    """The markdown block for the research bundle, or ``""`` when there is nothing to show.

    ``ranking_degraded`` prefixes the degraded marker, because the preamble and legend both
    tell the forecaster the rows are in evidential order — which is false for a fail-open
    slate, and a silently-wrong ordering claim is worse than a visibly degraded one.

    A multi-outcome market contributes one row plus a ``↳`` sub-row per outcome it was allowed, and
    a marker row when the budget cut some. The rules-bullet section stays ONE bullet per market
    regardless: every outcome inside a market shares its settlement rule, so a bullet per sub-row
    would repeat the same text up to ten times.
    """
    matches = snapshot.matches
    if not matches:
        return ""

    lines: list[str] = []
    if ranking_degraded:
        lines.append(DEGRADED_RANKING_MARKER)
        lines.append("")
    lines.append(_preamble(matches) + MARKET_SIGNAL_LEGEND)
    lines.append("")
    lines.append("| " + " | ".join(TABLE_COLUMNS) + " |")
    lines.append("|" + "---|" * len(TABLE_COLUMNS))
    allowances = _child_allowances(matches)
    for match, allowance in zip(matches, allowances):
        lines.append(_table_row(_row_cells(match)))
        lines.extend(_table_row(_child_cells(match.platform, child)) for child in match.children[:allowance])
        omitted_children = match.children[allowance:]
        if omitted_children:
            lines.append(_table_row(_omitted_children_cells(omitted_children)))

    lines.append("")
    lines.append("### Resolution criteria / rules")
    for match in matches:
        body = _bullet_body(match)
        if len(body) > RAW_BULLET_BODY_MAX_CHARS:
            body = body[:RAW_BULLET_BODY_MAX_CHARS] + "..."
        link = f" <{match.market_url}>" if match.market_url else ""
        lines.append(f"- **{match.platform}**{link}: {body}")

    return "\n".join(lines)
