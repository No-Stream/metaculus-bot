"""The rendered snapshot: the empty-render contract, the columns, and the MC-parser trap.

Four of these guard things that break quietly rather than loudly:

- **``""`` before any preamble on zero rows.** That early return is what produces
  ``status="empty"`` downstream and the attempted-vs-succeeded distinction residual analysis
  reads. Emitting a header with an empty table instead would publish a section claiming
  markets exist, and would make every "no markets" question look like a successful render.
- **Cells are selected BY HEADER NAME, never by index.** The previous suite indexed cells
  positionally with a comment explaining the numbering, so the column addition this port makes
  would have silently shifted every assertion onto the neighbouring column.
- **No rules bullet ends in ``: NN%``.** The per-model MC option parser scans for
  ``- <name>: NN%``, and Manifold descriptions are user-generated and routinely open with a
  percentage, so nothing percentage-shaped may be appended to a bullet.
- **A ``prob`` cell holding something other than a probability says so.** The forecaster prompts
  point a model at that column to anchor on, and Manifold's scalar markets publish a scale position
  in the field a probability normally comes from — which rendered as ``0.48`` on a market whose
  estimate was 121 years. ``TestScalarPriceCell`` asserts the property, not just the string.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from metaculus_bot.constants import RESEARCH_SECTION_CHAR_LIMIT
from metaculus_bot.research.market_retrieval.ranking import (
    DEGRADED_RANKING_MARKER,
    RENDER_BUDGET,
    TIER_UNSPECIFIED,
    TIERS,
    WHY_CHARS,
)
from metaculus_bot.research.market_retrieval.rendering import (
    CHILD_ROW_MARKER,
    CHILD_TITLE_MAX_CHARS,
    MARKET_PREAMBLE_NEUTRAL,
    MARKET_PREAMBLE_STRONG,
    MARKET_SIGNAL_LEGEND,
    MAX_CHILD_ROWS_PER_MARKET,
    MAX_CHILD_ROWS_PER_SNAPSHOT,
    RAW_BULLET_BODY_MAX_CHARS,
    TABLE_COLUMNS,
    TITLE_MAX_CHARS,
    render_snapshot,
)
from metaculus_bot.research.market_retrieval.types import (
    MarketChild,
    MarketMatch,
    MarketSnapshot,
    ScalarEstimate,
)
from tests.test_market_retrieval_generation import Platform

_PERCENT_TAIL_RE = re.compile(r":\s*[0-9]+(?:\.[0-9]+)?\s*%\s*$")

# Char budget for the WORST-CASE rendered snapshot: 8 rows with every field simultaneously at its
# cap, EVERY row multi-outcome, and every one of its `MAX_CHILD_ROWS_PER_MARKET` sub-rows maxed too.
# Measured at 10,298 chars. 10,600 leaves ~3% headroom, the same deliberate tightness the
# pre-expansion budget had: this section goes to the expensive forecaster models, so adding a
# column, lengthening the legend, or raising a child cap has to trip it.
#
# **All three figures were re-derived when multi-outcome markets started rendering `↳` sub-rows, and
# the growth is the point of that change rather than a regression.** For the record: maxed
# 6,304 → 10,137, realistic 4,648 → 7,465, fixed overhead 1,444 → 1,621. Two separate causes, worth
# keeping apart. The sub-rows are nearly all of it — 3,680 chars on the maxed slate and 2,664 on the
# realistic one, measured by rendering the same rows with `children` stripped — and `~190` is the
# legend's one added `↳` sentence. `MAX_CHILD_ROWS_PER_SNAPSHOT` is what bounds the sub-row half; the
# per-market cap never binds on a full 8-row slate, so raising it would not move these numbers.
#
# The operator's directive here superseded its own earlier ~6,000 figure: that number described
# "roughly today's 12-row render" BEFORE maximum-info expansion was chosen, and expansion that
# renders every outcome's own price cannot fit a budget set for rendering none of them. What did NOT
# change is the reason the budgets exist, so they are re-derived tight rather than loosened.
#
# The scalar-market `prob` cell added one legend sentence, +161 chars on all three figures.
# Only the fixed-overhead ceiling moved with it: the two whole-render budgets absorbed the growth
# and are deliberately LEFT where they were, so this change spent their headroom rather than
# raising them. That is why maxed now sits at ~3% and realistic at ~1% — a guard that is nearly
# touching is doing its job, and the next thing that widens the section re-derives all three.
#
# The omitted-price-mass clause on the truncation marker (q45189 disclosure) added ~21 chars per
# truncated market: maxed 10,298 → 10,466, realistic 7,626 → 7,794. Maxed still fits under 10,600
# (1.3% headroom); realistic breached its 7,700 and was re-derived to 7,900 (1.4%), same
# deliberate tightness. Fixed overhead is untouched — the clause rides the marker row, not the
# legend.
MARKET_SNAPSHOT_MAXED_RENDER_CHAR_BUDGET = 10_600

# 8 rows of realistic content, using shapes measured off live payloads rather than chosen — the
# figure that describes what a real question renders. Measured at 7,794 chars.
MARKET_SNAPSHOT_REALISTIC_RENDER_CHAR_BUDGET = 7_900

# Preamble + legend: the FIXED overhead every snapshot pays regardless of row count, measured at
# 1,782 chars. Budgeted separately and tightly because prose is the likeliest thing to bloat and
# the only part with no data to justify it — the whole-snapshot budget has slack that would
# otherwise absorb an added paragraph unnoticed. At 1,782 of 1,850 this is still the tightest
# budget in the file, which is the point: the next legend sentence has to earn a re-derivation.
#
# This one WAS re-derived, from 1,700, and what it bought is on the record: a `prob` cell may now
# hold a scalar market's value, and a forecaster told to anchor on that column needs the legend to
# say so. The sentence was cut to its contract (the `value` prefix, the units, "not a probability")
# rather than paid for at full length, because it explains a shape that appears on a small minority
# of questions while the legend ships on every one.
MARKET_SNAPSHOT_FIXED_OVERHEAD_CHAR_BUDGET = 1_850

_REAL_TITLE = "Will the US unemployment rate be above 4.5% in June 2026?"
_REAL_RULES = (
    "If the BLS-reported seasonally adjusted U-3 unemployment rate for June 2026 is above 4.5%, "
    "then the market resolves to Yes."
)
_REAL_WHY = "near-identical: same BLS U-3 series, same month"
_REAL_URL = "https://kalshi.com/markets/KXUNRATE-26JUN"


def _row(
    platform: Platform = "kalshi",
    *,
    title: str = "A market",
    tier: str = "",
    why: str = "",
    prob: float | None = 0.42,
    volume: float | None = 12345.0,
    oi: float | None = 6789.0,
    bettors: int | None = None,
    resolved: bool = False,
    close: datetime | None = None,
    rules: str = "rules text",
    url: str = "https://example.test/m",
    answers: tuple[tuple[str, float], ...] = (),
    children: tuple[MarketChild, ...] = (),
    scalar: ScalarEstimate | None = None,
) -> MarketMatch:
    return MarketMatch(
        platform=platform,
        market_title=title,
        market_url=url,
        implied_prob_yes=prob,
        bid=None,
        ask=None,
        spread=None,
        volume_24h=None,
        close_time=close,
        is_resolved=resolved,
        match_confidence=1.0,
        raw_rules=rules,
        total_volume=volume,
        open_interest=oi,
        num_bettors=bettors,
        relation_tier=tier,
        relevance_label=why,
        top_answers=answers,
        children=children,
        scalar_estimate=scalar,
    )


def _table_rows(text: str) -> list[dict[str, str]]:
    """Parse the rendered table into header-keyed dicts.

    By header NAME rather than by position, deliberately: this port adds a column, and an
    index-based reader would keep passing while asserting on the wrong cell.
    """
    lines = [line for line in text.split("\n") if line.startswith("| ")]
    header = [cell.strip() for cell in lines[0].strip("| ").split(" | ")]
    out: list[dict[str, str]] = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.strip("| ").split(" | ")]
        if set("".join(cells)) <= {"-"}:
            continue  # the |---|---| separator
        out.append(dict(zip(header, cells)))
    return out


def _sub_rows_per_market(text: str) -> list[int]:
    """How many ``↳`` sub-rows each parent row rendered, in slate order, markers excluded.

    Counted off the rendered table rather than by inspecting the allowance helper, so the assertions
    hold against what a forecaster actually reads.
    """
    counts: list[int] = []
    for cells in _table_rows(text):
        if cells["platform"] != CHILD_ROW_MARKER:
            counts.append(0)
        elif not cells["title"].startswith("["):
            counts[-1] += 1
    return counts


def _bullets(text: str) -> list[str]:
    body = text.split("### Resolution criteria / rules", 1)[1]
    return [line for line in body.split("\n") if line.startswith("- ")]


class TestEmptyRender:
    def test_zero_rows_returns_empty_before_any_preamble(self) -> None:
        """A ranked design returning zero rows is a legitimate outcome now, so this path is
        hot: the empty string is what makes the snapshot read as `empty` rather than as a
        successful render of nothing."""
        rendered = render_snapshot(MarketSnapshot(matches=[]))

        assert rendered == ""
        assert "MAY be relevant" not in rendered
        assert "signal" not in rendered

    def test_zero_rows_returns_empty_even_when_degraded(self) -> None:
        assert render_snapshot(MarketSnapshot(matches=[]), ranking_degraded=True) == ""


class TestColumns:
    def test_the_column_set_and_order(self) -> None:
        """`conf` and `relevance` are gone; `status`, `relation` and `why` replace them."""
        rendered = render_snapshot(MarketSnapshot(matches=[_row()]))

        header = [line for line in rendered.split("\n") if line.startswith("| platform")][0]
        assert header == "| platform | title | prob | total_vol | OI | signal | close | status | relation | why |"
        assert TABLE_COLUMNS == (
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
        assert "conf" not in header
        assert "relevance" not in header

    def test_cells_carry_the_expected_values(self) -> None:
        row = _row(
            title="US unemployment rate",
            tier="same_quantity_other_cut",
            why="same BLS series, different month",
            close=datetime(2026, 6, 30, tzinfo=timezone.utc),
        )

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[row])))[0]

        assert cells["platform"] == "kalshi"
        assert cells["title"] == "US unemployment rate"
        assert cells["prob"] == "0.42"
        assert cells["total_vol"] == "12345"
        assert cells["OI"] == "6789"
        assert cells["signal"] == "decent"
        assert cells["close"] == "2026-06-30"
        assert cells["status"] == "open"
        assert cells["relation"] == "same_quantity_other_cut"
        assert cells["why"] == "same BLS series, different month"

    def test_missing_values_render_as_dashes(self) -> None:
        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[_row(prob=None, volume=None, oi=None)])))[0]

        assert (cells["prob"], cells["total_vol"], cells["OI"], cells["close"]) == ("-", "-", "-", "-")

    def test_status_reports_a_resolved_market(self) -> None:
        """Resolved markets reach the render now that the `as_of` filter is gone, and their
        price is a realized outcome rather than a forecast."""
        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[_row(resolved=True)])))[0]

        assert cells["status"] == "RESOLVED"

    def test_an_unlabelled_relation_renders_as_unspecified(self) -> None:
        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[_row(tier="")])))[0]

        assert cells["relation"] == TIER_UNSPECIFIED
        assert cells["why"] == "-"

    def test_predictit_renders_no_liquidity_data_in_signal(self) -> None:
        """Honest — its dump carries no volume field anywhere — and both the forecaster prompt
        clause and the venue field-contract expectations depend on it. Unlike the ranker
        prompt, the rendered table keeps it."""
        row = _row("predictit", volume=None, oi=None)

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[row])))[0]

        assert cells["signal"] == "no-liquidity-data"

    def test_a_pipe_in_a_title_or_label_cannot_shift_the_columns(self) -> None:
        row = _row(title="A | B | C", tier="weak", why="pipes | everywhere")

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[row])))[0]

        assert cells["title"] == "A / B / C"
        assert cells["why"] == "pipes / everywhere"
        assert cells["relation"] == "weak"

    def test_manifold_scores_on_bettors_rather_than_dollars(self) -> None:
        row = _row("manifold", volume=None, oi=None, bettors=250)

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[row])))[0]

        assert cells["signal"] == "high"

    def test_a_manifold_total_vol_renders_its_mana_figure_unconverted(self) -> None:
        """The unit the legend has to qualify. `parse_manifold_matches` populates `total_volume`
        from Manifold's `volume`, which is MANA (play money) with no conversion anywhere, and
        22.1% of measured Manifold rows exceed the $5k thin ceiling in mana terms. Rendering `-`
        instead would add the only per-platform branch in `_row_cells` and delete a real
        participation signal, so the number stays and the LEGEND carries the caveat. `OI` is
        genuinely absent for Manifold, which is why it reads `-`."""
        row = _row("manifold", volume=107_943.0, oi=None, bettors=250)

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[row])))[0]

        assert cells["total_vol"] == "107943"
        assert cells["OI"] == "-"


class TestScalarPriceCell:
    """A scalar market's ``prob`` cell: what it says, and what it can never be mistaken for.

    Manifold's ``PSEUDO_NUMERIC`` markets trade a VALUE on a bounded scale, and their payload's
    ``probability`` field is that value's position on it. The parser now keeps the two apart, so this
    is the render half of the same fix: the cell holds a labelled value, and the forecaster prompts
    that tell a model to anchor on this column can no longer point it at a scale position.

    Asserted on the rendered markdown rather than on ``_price_cell``, because the defect was only
    visible in the table — the intermediate object read ``implied_prob_yes=0.4839``, which looks
    entirely reasonable until it reaches a column headed ``prob``.
    """

    _AGE_ESTIMATE = ScalarEstimate(value=120.96691732988944, minimum=0.0, maximum=250.0, is_log_scale=False)

    def test_a_scalar_row_shows_its_labelled_value_and_scale(self) -> None:
        """The production row that exposed this rendered `0.48` for a market whose estimate was
        ~121 years. Same market, same figures, straight off the live payload."""
        row = _row(
            "manifold",
            title="What will be the age of the oldest person alive in 2100?",
            prob=None,
            scalar=self._AGE_ESTIMATE,
        )

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[row])))[0]

        assert cells["prob"] == "value 120.967 (scale 0 to 250)"
        assert "0.48" not in cells["prob"], "the scale position must not survive anywhere in the cell"

    def test_the_scalar_cell_cannot_be_read_as_a_probability(self) -> None:
        """The property that actually matters, stated as a property rather than a string match.

        A bare number would not be enough — the whole reason `0.48` got through is that it is a
        perfectly plausible probability — and magnitude is no defence either. That second half is a
        measurement, not a worry: `min=0, max=1` scalar markets exist on Manifold, and the one used
        here is real (it trades 0.468 on a 0-to-1 scale, captured live in the venue fixture). Pre-fix
        it rendered `0.47` under a `prob` header with nothing to distinguish it from a price. So the
        guard is that no scalar cell ever parses as the two-decimal form this column uses for prices.
        """
        probability_shaped = re.compile(r"^-?[0-9]+\.[0-9]{2}$")
        on_a_probability_like_scale = ScalarEstimate(value=0.4680001051790792, minimum=0.0, maximum=1.0)
        rows = [
            _row("manifold", prob=None, scalar=self._AGE_ESTIMATE),
            _row("manifold", prob=None, scalar=on_a_probability_like_scale),
        ]

        for cells in _table_rows(render_snapshot(MarketSnapshot(matches=rows))):
            assert not probability_shaped.match(cells["prob"])
            assert cells["prob"].startswith("value ")

    def test_a_log_scale_row_says_so(self) -> None:
        """Where a value sits between its bounds reads differently on a log axis, and it is the one
        thing about the scale type a forecaster needs. Nothing here computes with it — the venue's
        own `value` is in question units on either scale."""
        row = _row(
            "manifold",
            prob=None,
            scalar=ScalarEstimate(value=609.0, minimum=1.0, maximum=10_000_000.0, is_log_scale=True),
        )

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[row])))[0]

        assert cells["prob"] == "value 609 (log scale 1 to 10,000,000)"

    def test_a_large_magnitude_value_reads_as_a_number_rather_than_an_exponent(self) -> None:
        """The real 2040 world-population market: ``min=1e6, max=2e10``, trading 7.85 billion.

        Significant-digit rounding is what makes one rule serve both this and a ``0.5 to 2.5`` market,
        but `%g`'s automatic exponent would render this cell as
        ``value 7.84859e+09 (log scale 1000000 to 2e+10)`` — one scale written two ways, with the
        reader doing exponent arithmetic to find the population. Grouped decimal instead, and the
        bounds stay exact because 11 significant digits is what this market's own ceiling needs.
        """
        row = _row(
            "manifold",
            prob=None,
            scalar=ScalarEstimate(
                value=7848589347.056932, minimum=1_000_000.0, maximum=20_000_000_000.0, is_log_scale=True
            ),
        )

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[row])))[0]

        assert cells["prob"] == "value 7,848,590,000 (log scale 1,000,000 to 20,000,000,000)"
        assert "e+" not in cells["prob"]

    def test_a_scalar_row_without_bounds_still_shows_its_value(self) -> None:
        """The bounds are the venue's to omit; the value is the market's answer. Dropping the number
        because its scale is missing would trade a real datum for a dash."""
        row = _row("manifold", prob=None, scalar=ScalarEstimate(value=42.5))

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[row])))[0]

        assert cells["prob"] == "value 42.5"

    def test_a_negative_scale_reads_with_to_rather_than_a_hyphen(self) -> None:
        """Live Manifold scales run negative (-15 to 2, -48 to 48, -4 to 4), and `-15--2` is
        unreadable. The `to` is a correctness choice, not a style one."""
        row = _row("manifold", prob=None, scalar=ScalarEstimate(value=-1.0, minimum=-15.0, maximum=2.0))

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[row])))[0]

        assert cells["prob"] == "value -1 (scale -15 to 2)"

    def test_an_ordinary_probability_row_is_untouched(self) -> None:
        """The regression guard. Every other venue and every BINARY Manifold row keeps the
        two-decimal price, and a `↳` sub-row keeps it too — the scalar branch is reachable only from
        a parent row carrying an estimate."""
        parent = _row("kalshi", prob=None, children=(MarketChild(title="Above 4.2%", implied_prob_yes=0.31),))
        rows = [_row(prob=0.42), parent]

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=rows)))

        assert cells[0]["prob"] == "0.42"
        assert cells[1]["prob"] == "-", "a multi-outcome parent still has no single price"
        assert cells[2]["prob"] == "0.31"

    def test_the_legend_names_the_scalar_cell(self) -> None:
        """The legend is a forecaster-facing contract and names every label a cell can hold — a cell
        shape it omits is one a forecaster has to guess at. This is the one that was guessed wrong
        for real."""
        rendered = render_snapshot(MarketSnapshot(matches=[_row("manifold", prob=None, scalar=self._AGE_ESTIMATE)]))

        assert "prefixed `value`" in rendered
        assert "not a probability" in rendered

    def test_a_slate_of_scalar_rows_fits_the_maxed_budget(self) -> None:
        """A scalar cell is the widest the `prob` column gets, so the section budget is re-checked
        against a slate made entirely of them.

        Kept separate from `_maxed_rows` rather than folded into it: that slate makes every row
        multi-outcome, and a market cannot be both (a row with outcomes carries no single value), so
        one slate cannot be the worst case for both shapes.
        """
        widest = ScalarEstimate(value=-123456.789, minimum=-1_000_000.0, maximum=10_000_000.0, is_log_scale=True)
        rows = [
            _row(
                "manifold",
                title="T" * TITLE_MAX_CHARS,
                tier=TIERS[0],
                why="W" * WHY_CHARS,
                prob=None,
                scalar=widest,
                rules="R" * RAW_BULLET_BODY_MAX_CHARS,
                close=datetime(2026, 12, 31, tzinfo=timezone.utc),
                bettors=250,
            )
            for _ in range(RENDER_BUDGET)
        ]

        rendered = render_snapshot(MarketSnapshot(matches=rows))

        assert len(rendered) < MARKET_SNAPSHOT_MAXED_RENDER_CHAR_BUDGET
        assert _table_rows(rendered)[0]["prob"] == "value -123,457 (log scale -1,000,000 to 10,000,000)"


class TestRankedOrder:
    def test_the_ranked_order_is_preserved_verbatim(self) -> None:
        """No venue interleave, no fairness pass, no per-venue cap. Round-robin venue fairness
        is what evicted 43 of 58 wanted rows in the measurement this port is built on."""
        matches = [
            _row("kalshi", title="first", tier="same_quantity_same_date"),
            _row("kalshi", title="second", tier="same_quantity_other_cut"),
            _row("manifold", title="third", tier="weak", bettors=5),
            _row("kalshi", title="fourth", tier="weak"),
        ]

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=matches)))

        assert [cell["title"] for cell in cells] == ["first", "second", "third", "fourth"]
        assert [cell["platform"] for cell in cells] == ["kalshi", "kalshi", "manifold", "kalshi"]

    def test_several_rows_from_one_venue_are_not_capped(self) -> None:
        matches = [_row("kalshi", title=f"row {i}", tier="weak") for i in range(8)]

        assert len(_table_rows(render_snapshot(MarketSnapshot(matches=matches)))) == 8


class TestPreambleSelector:
    @pytest.mark.parametrize("tier", ["same_quantity_same_date", "same_quantity_other_cut"])
    def test_a_same_quantity_row_earns_the_strong_preamble(self, tier: str) -> None:
        rendered = render_snapshot(MarketSnapshot(matches=[_row(tier="weak"), _row(tier=tier)]))

        assert rendered.startswith(MARKET_PREAMBLE_STRONG)

    @pytest.mark.parametrize("tier", ["driver_or_consequence", "weak", TIER_UNSPECIFIED, ""])
    def test_context_only_rows_get_the_neutral_preamble(self, tier: str) -> None:
        rendered = render_snapshot(MarketSnapshot(matches=[_row(tier=tier)]))

        assert rendered.startswith(MARKET_PREAMBLE_NEUTRAL)

    def test_the_asserted_substrings_survive(self) -> None:
        """Asserted at several sites across the suite; the ranked design changes how a row is
        chosen, not what a forecaster is told to verify."""
        assert "MAY be relevant" in MARKET_PREAMBLE_STRONG
        assert "verify each market's resolution criteria" in MARKET_PREAMBLE_STRONG
        assert "may all be off-topic" in MARKET_PREAMBLE_NEUTRAL

    def test_the_fuzzy_match_vocabulary_is_gone(self) -> None:
        """There is no fuzzy match to warn about any more — a model read each market's rules."""
        for text in (MARKET_PREAMBLE_STRONG, MARKET_PREAMBLE_NEUTRAL, MARKET_SIGNAL_LEGEND):
            assert "fuzzy" not in text.lower()


class TestLegend:
    def test_every_liquidity_label_the_cells_can_hold_is_explained(self) -> None:
        """A legend that omits a label a cell can hold teaches forecasters to guess at it."""
        for label in ("thin", "decent", "deep", "high", "no-liquidity-data"):
            assert label in MARKET_SIGNAL_LEGEND

    def test_every_relation_tier_the_cells_can_hold_is_explained(self) -> None:
        for tier in (*TIERS, TIER_UNSPECIFIED):
            assert tier in MARKET_SIGNAL_LEGEND

    def test_the_volume_columns_units_are_qualified_per_venue(self) -> None:
        """`total_vol` holds USD on Kalshi/Polymarket and play-money MANA on Manifold, with no
        conversion anywhere, so a venue-wide "approximate USD" claim is simply false — and it is
        a claim the expensive forecaster models read on every rendered snapshot."""
        assert "real-money venues" in MARKET_SIGNAL_LEGEND
        assert "mana" in MARKET_SIGNAL_LEGEND

    def test_the_legend_explains_the_new_columns_and_not_the_retired_one(self) -> None:
        assert "`relation`" in MARKET_SIGNAL_LEGEND
        assert "`why`" in MARKET_SIGNAL_LEGEND
        assert "ordered by EVIDENTIAL VALUE" in MARKET_SIGNAL_LEGEND
        assert "RESOLVED" in MARKET_SIGNAL_LEGEND
        assert "likely-relevant" not in MARKET_SIGNAL_LEGEND
        assert "verify-carefully" not in MARKET_SIGNAL_LEGEND

    def test_the_legend_explains_the_sub_row_glyph(self) -> None:
        """A glyph a forecaster has never seen, in a column that otherwise names a venue, has to be
        explained where it appears. The legend also has to say why the parent's `prob` is blank —
        without that, the strongest available evidence reads as a market with no price."""
        assert CHILD_ROW_MARKER in MARKET_SIGNAL_LEGEND
        assert "one OUTCOME of the market above it" in MARKET_SIGNAL_LEGEND
        assert "the parent row has none" in MARKET_SIGNAL_LEGEND

    def test_the_legend_ships_inside_the_section(self) -> None:
        rendered = render_snapshot(MarketSnapshot(matches=[_row(tier="weak")]))

        assert MARKET_SIGNAL_LEGEND in rendered


class TestDegradedRender:
    def test_the_marker_precedes_the_preamble(self) -> None:
        """The preamble and legend both tell the forecaster the rows are in evidential order,
        which is false for a fail-open slate."""
        rendered = render_snapshot(MarketSnapshot(matches=[_row(tier="")]), ranking_degraded=True)

        assert rendered.startswith(DEGRADED_RANKING_MARKER)
        assert rendered.index(DEGRADED_RANKING_MARKER) < rendered.index(MARKET_PREAMBLE_NEUTRAL)

    def test_a_healthy_render_carries_no_marker(self) -> None:
        rendered = render_snapshot(MarketSnapshot(matches=[_row(tier="weak")]))

        assert DEGRADED_RANKING_MARKER not in rendered


class TestRenderBudget:
    """The forecaster-facing section must stay compact: all relevant data, no excess verbosity.

    Two token budgets in this port point opposite ways. The ~38k RANKER prompt is fine — it goes
    to a cheap model and recall lives there. This snapshot goes to the expensive forecaster and
    reasoning models on every question, so it is the one that has to stay lean. These assertions
    are the regression guard: a future edit that adds a column, pads the legend, or raises
    RAW_BULLET_BODY_MAX_CHARS cannot bloat the section silently.
    """

    def _maxed_rows(self) -> list[MarketMatch]:
        """Eight rows with every field simultaneously at its cap, every one of them multi-outcome
        with a full complement of maxed sub-rows — a bound, not a forecast."""
        maxed_child = MarketChild(
            title="C" * CHILD_TITLE_MAX_CHARS,
            implied_prob_yes=0.4237,
            total_volume=123456789.0,
            open_interest=98765432.0,
            is_resolved=True,
            close_time=datetime(2026, 12, 31, tzinfo=timezone.utc),
        )
        return [
            _row(
                title="M" * TITLE_MAX_CHARS,
                tier="same_quantity_other_cut",
                why="W" * WHY_CHARS,
                rules="R" * RAW_BULLET_BODY_MAX_CHARS,
                url="https://kalshi.com/markets/" + "T" * 40,
                prob=None,
                volume=123456789.0,
                oi=98765432.0,
                close=datetime(2026, 12, 31, tzinfo=timezone.utc),
                resolved=index % 2 == 0,
                children=tuple(maxed_child for _ in range(MAX_CHILD_ROWS_PER_MARKET)),
            )
            for index in range(RENDER_BUDGET)
        ]

    def _realistic_rows(self) -> list[MarketMatch]:
        """Eight realistic rows, each a strike family with four outcomes — the shape a real slate
        takes, since 86.5% of the Kalshi catalogue is multi-strike. Outcome labels and prices are
        real ones from the committed venue fixtures."""
        outcomes = tuple(
            MarketChild(
                title=title,
                implied_prob_yes=prob,
                total_volume=volume,
                open_interest=volume / 2,
                close_time=datetime(2026, 6, 30, tzinfo=timezone.utc),
            )
            for title, prob, volume in (
                ("Before Nov 1, 2026", 0.175, 45_000.0),
                ("Republican Party", 0.535, 32_000.0),
                ("0 (0 bps)", 0.888, 6_805_439.0),
                ("$3.80 - $4.19", 0.5083, 258.0),
            )
        )
        return [
            _row(
                title=_REAL_TITLE,
                tier="same_quantity_other_cut",
                why=_REAL_WHY,
                rules=_REAL_RULES,
                url=_REAL_URL,
                prob=None,
                close=datetime(2026, 6, 30, tzinfo=timezone.utc),
                children=outcomes,
            )
            for _ in range(RENDER_BUDGET)
        ]

    def test_a_maxed_eight_row_snapshot_fits_the_char_budget(self) -> None:
        rendered = render_snapshot(MarketSnapshot(matches=self._maxed_rows()))

        assert len(rendered) < MARKET_SNAPSHOT_MAXED_RENDER_CHAR_BUDGET, (
            f"maxed render grew to {len(rendered)} chars, over the "
            f"{MARKET_SNAPSHOT_MAXED_RENDER_CHAR_BUDGET} budget — this section goes to the expensive "
            f"forecaster models on every question. Cut something or re-derive the budget."
        )

    def test_a_realistic_eight_row_snapshot_fits_the_operators_budget(self) -> None:
        """The operator's ~6,000 figure, against content shaped like the real thing."""
        rendered = render_snapshot(MarketSnapshot(matches=self._realistic_rows()))

        assert len(rendered) < MARKET_SNAPSHOT_REALISTIC_RENDER_CHAR_BUDGET

    def test_the_fixed_prose_overhead_is_budgeted_separately(self) -> None:
        """Prose has no data to justify it, so it gets the tight budget. Without this, the
        whole-snapshot slack would absorb an added paragraph unnoticed."""
        overhead = len(MARKET_PREAMBLE_STRONG) + len(MARKET_SIGNAL_LEGEND)

        assert overhead < MARKET_SNAPSHOT_FIXED_OVERHEAD_CHAR_BUDGET
        assert len(MARKET_PREAMBLE_NEUTRAL) < len(MARKET_PREAMBLE_STRONG) + 200

    def test_the_snapshot_stays_under_a_quarter_of_the_research_section_limit(self) -> None:
        """Stated explicitly so the relationship between the two ceilings is visible: the
        snapshot is one section inside a research bundle that gets middle-trimmed at
        RESEARCH_SECTION_CHAR_LIMIT, and a section approaching that limit would start evicting
        its siblings rather than itself.

        The margin is no longer comfortable, and that is worth knowing rather than smoothing over:
        the sub-row expansion took the maxed render from 56% of the quarter-limit to 90%, so this
        assertion has stopped being a formality. The next thing that widens the section has to
        re-derive its relationship to RESEARCH_SECTION_CHAR_LIMIT rather than assume the room is
        there. The realistic figure (7,465 of 11,249, or 66%) is where the real headroom lives.
        """
        rendered = render_snapshot(MarketSnapshot(matches=self._maxed_rows()))

        assert len(rendered) < RESEARCH_SECTION_CHAR_LIMIT / 4
        assert MARKET_SNAPSHOT_MAXED_RENDER_CHAR_BUDGET < RESEARCH_SECTION_CHAR_LIMIT / 4

    def test_the_new_columns_are_single_tokens_per_row(self) -> None:
        """`status` and `relation` are graded labels, not prose: one token each, so the two new
        columns cost ~30 chars a row rather than a clause. Holds for sub-rows too, which fill both
        with a dash."""
        cells = _table_rows(render_snapshot(MarketSnapshot(matches=self._maxed_rows())))

        for row_cells in cells:
            assert " " not in row_cells["status"]
            assert " " not in row_cells["relation"]

    def test_the_row_count_is_capped(self) -> None:
        """Nothing downstream re-caps this, so a snapshot handed more than the budget's worth of
        rows would render all of them. The cap lives in the ranking stage; this pins that the
        renderer is fed at most that many by asserting the budget was measured at the cap."""
        assert RENDER_BUDGET == 8


class TestRulesBullets:
    def test_the_bullet_shape_is_unchanged(self) -> None:
        rendered = render_snapshot(MarketSnapshot(matches=[_row(rules="Resolves per BLS.", url="https://k.test/m")]))

        assert _bullets(rendered) == ["- **kalshi** <https://k.test/m>: Resolves per BLS."]

    def test_the_section_heading_stays_at_h3(self) -> None:
        rendered = render_snapshot(MarketSnapshot(matches=[_row()]))

        assert "\n### Resolution criteria / rules\n" in rendered
        assert "#### Resolution criteria" not in rendered

    def test_rules_are_truncated_with_an_ellipsis(self) -> None:
        rendered = render_snapshot(MarketSnapshot(matches=[_row(rules="r" * 5000)]))

        bullet = _bullets(rendered)[0]
        assert bullet.endswith("...")
        assert bullet.count("r") == RAW_BULLET_BODY_MAX_CHARS

    def test_a_missing_url_omits_the_link(self) -> None:
        rendered = render_snapshot(MarketSnapshot(matches=[_row(url="", rules="text")]))

        assert _bullets(rendered) == ["- **kalshi**: text"]

    def test_a_percentage_leading_description_does_not_produce_a_percent_tail(self) -> None:
        """Manifold descriptions are user-generated and routinely open with a percentage. The
        per-model MC option parser reads ``- <name>: NN%``, so the formatter must never append
        anything percentage-shaped to a bullet."""
        row = _row("manifold", volume=None, oi=None, bettors=5, rules="45% of forecasters expect a hike this year.")

        rendered = render_snapshot(MarketSnapshot(matches=[row]))

        bullet = _bullets(rendered)[0]
        assert bullet == "- **manifold** <https://example.test/m>: 45% of forecasters expect a hike this year."
        assert not _PERCENT_TAIL_RE.search(bullet)

    def test_the_formatter_appends_nothing_percentage_shaped(self) -> None:
        """The binding rule is about what the FORMATTER adds, and it adds nothing: a bullet is
        exactly ``- **{platform}** <{url}>: {rules}``, with the rules text verbatim.

        A venue whose own rules text ends in ``Threshold: 45%`` therefore renders a bullet that
        ends that way, and that is deliberate rather than overlooked — sanitising venue text
        would mangle legitimate resolution criteria. What protects the MC parser is the section
        boundary: it scopes to the text BEFORE ``### Research Summary``
        (``parsing._summary_section_for_bullets``) and this table renders inside that section,
        so the bullets are structurally out of its reach. Note the regex is NOT end-anchored,
        so "ends with" understates it — any ``: NN%`` on an in-scope line would match.
        """
        for rules in ("Resolves at 45%", "Threshold: 45%", "45% chance", ""):
            rendered = render_snapshot(MarketSnapshot(matches=[_row(rules=rules)]))

            assert _bullets(rendered) == [f"- **kalshi** <https://example.test/m>: {rules}"]

    def test_a_truncated_bullet_cannot_gain_a_percentage_tail(self) -> None:
        """The one thing the formatter does append is the truncation ellipsis, and a bullet
        ending in ``...`` can never read as an option probability."""
        rendered = render_snapshot(MarketSnapshot(matches=[_row(rules="x" * 300 + " 45%")]))

        assert not _PERCENT_TAIL_RE.search(_bullets(rendered)[0])

    def test_newlines_in_rules_are_flattened(self) -> None:
        """A bullet that spans lines would break the one-bullet-per-market shape."""
        rendered = render_snapshot(MarketSnapshot(matches=[_row(rules="line one\nline two")]))

        assert _bullets(rendered) == ["- **kalshi** <https://example.test/m>: line one line two"]


class TestMultiOutcomeRows:
    """A multi-outcome market: a dash in the PARENT's ``prob``, one ``↳`` sub-row per outcome.

    The defect these sub-rows exist to fix was a MISLABELLED anchor rather than a missing number. A
    Polymarket event row took its title from the event and its price from ``markets[0]``, so
    "How many Fed rate cuts in 2026?" rendered 0.89 — the "will no cuts happen" child's price — and
    the forecaster prompts tell the model to anchor on a matched market's price. Withholding the
    price (what Kalshi families and PredictIt ballots did) was honest but threw away every outcome's
    real price. Each outcome now gets its own row, its own title, and its own price.
    """

    # The committed Polymarket fixture's own numbers (`polymarket_search.events[0]`), so the render
    # and the parser are pinned against the same real payload.
    _CHILDREN = (
        MarketChild(title="0 (0 bps)", implied_prob_yes=0.888, total_volume=6_805_439.0),
        MarketChild(title="1 (25 bps)", implied_prob_yes=0.065, total_volume=2_325_219.0),
    )
    _RULES = "Resolves per the Fed's published target range."

    def _parent(
        self,
        *,
        platform: Platform = "polymarket",
        children: tuple[MarketChild, ...] = _CHILDREN,
        rules: str = _RULES,
        title: str = "How many Fed rate cuts in 2026?",
        tier: str = "",
        bettors: int | None = None,
    ) -> MarketMatch:
        """A live-shaped multi-outcome parent: no probability of its own, real event-level money."""
        return _row(
            platform,
            title=title,
            tier=tier,
            prob=None,
            volume=46_225_700.0,
            oi=1_751_721.0,
            bettors=bettors,
            rules=rules,
            children=children,
        )

    def test_the_parent_prob_is_a_dash_and_each_outcome_carries_its_own_price(self) -> None:
        """The whole point, in one assertion: 0.888 renders against "0 (0 bps)" and 0.065 against
        "1 (25 bps)", and NEITHER renders against the event's own title. The parent's liquidity
        columns still populate, so it keeps its weight."""
        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[self._parent()])))

        assert cells[0]["title"] == "How many Fed rate cuts in 2026?"
        assert cells[0]["prob"] == "-"
        assert (cells[0]["total_vol"], cells[0]["OI"], cells[0]["signal"]) == ("46225700", "1751721", "deep")
        assert [(cell["title"], cell["prob"]) for cell in cells[1:]] == [("0 (0 bps)", "0.89"), ("1 (25 bps)", "0.07")]

    def test_a_sub_row_is_marked_with_the_continuation_glyph_not_the_venue_name(self) -> None:
        """A sub-row's venue is its parent's, one line up; repeating it would spend ~10 chars a row
        restating that. The glyph also makes the nesting unambiguous when a row is read alone."""
        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[self._parent()])))

        assert cells[0]["platform"] == "polymarket"
        assert [cell["platform"] for cell in cells[1:]] == [CHILD_ROW_MARKER, CHILD_ROW_MARKER]

    def test_sub_rows_render_in_the_adapters_order_verbatim(self) -> None:
        """The same rule as the ranked slate, one level down. The adapter ordered these by what its
        venue makes worth keeping and the renderer truncates from the END, so re-sorting here would
        silently change which outcomes survive the budget."""
        reversed_children = tuple(reversed(self._CHILDREN))

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[self._parent(children=reversed_children)])))

        assert [cell["title"] for cell in cells[1:]] == ["1 (25 bps)", "0 (0 bps)"]

    def test_a_sub_row_carries_no_relation_grade(self) -> None:
        """The ranker graded the MARKET, never its individual outcomes. Repeating the parent's grade
        on every sub-row would spend ~30 chars a row implying a judgement nobody made."""
        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[self._parent(tier="same_quantity_same_date")])))

        assert cells[0]["relation"] == "same_quantity_same_date"
        assert all((cell["relation"], cell["why"]) == ("-", "-") for cell in cells[1:])

    def test_a_sub_rows_signal_is_labelled_by_its_parents_venue_rule(self) -> None:
        """One labelling rule for both levels, or a Kalshi strike and its family could disagree
        about what "thin" means. Manifold scores bettors rather than dollars, and its children carry
        the market's count precisely so their `signal` agrees with the parent's instead of reading a
        false `no-liquidity-data` on a venue that does publish per-answer volume."""
        answers = (MarketChild(title="Over $4.60", implied_prob_yes=0.4992, total_volume=417.0, num_bettors=47),)

        cells = _table_rows(
            render_snapshot(MarketSnapshot(matches=[self._parent(platform="manifold", children=answers, bettors=47)]))
        )

        assert cells[0]["signal"] == "decent"
        assert cells[1]["signal"] == "decent"
        assert cells[1]["total_vol"] == "417", "the child's own volume, not its parent's"

    def test_the_bullet_carries_rules_text_alone(self) -> None:
        """A multi-outcome bullet used to LEAD with `answers: Over $4.60 (50%), ...` because the
        table had nowhere to put a price. It does now, with a real column for each outcome's volume
        and close date, so carrying them here too would pay twice for the same numbers."""
        rendered = render_snapshot(MarketSnapshot(matches=[self._parent()]))

        assert _bullets(rendered) == [f"- **polymarket** <https://example.test/m>: {self._RULES}"]
        assert "answers:" not in rendered

    def test_the_formatter_still_appends_nothing_percentage_shaped(self) -> None:
        """The rule that outlived the answers: the per-model MC option parser reads
        ``- <name>: NN%``, so nothing percentage-shaped may be appended to a bullet. Moving the
        answers into table cells RETIRED the parenthesisation guard that used to be needed here —
        a cell cannot be mistaken for a bullet — and this pins that the move did not reintroduce
        the shape by another route."""
        for rules in ("", "Resolves per AAA.", "45% of forecasters expect a hike."):
            rendered = render_snapshot(MarketSnapshot(matches=[self._parent(rules=rules)]))

            assert not _PERCENT_TAIL_RE.search(_bullets(rendered)[0])

    def test_a_pipe_in_an_outcome_title_cannot_shift_the_columns(self) -> None:
        child = (MarketChild(title="A | B", implied_prob_yes=0.5),)

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[self._parent(children=child)])))

        assert cells[1]["title"] == "A / B"
        assert cells[1]["prob"] == "0.50"

    def test_an_outcome_title_is_truncated_at_its_own_shorter_cap(self) -> None:
        """An outcome label is a rung, not a question (9-18 chars measured across all four venues),
        so its cap is well under the parent's and only bounds a pathological label."""
        child = (MarketChild(title="L" * 500),)

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[self._parent(children=child)])))

        assert cells[1]["title"] == "L" * CHILD_TITLE_MAX_CHARS
        assert CHILD_TITLE_MAX_CHARS < TITLE_MAX_CHARS

    def test_a_row_with_no_outcomes_renders_exactly_one_table_row(self) -> None:
        """The superset half of the change: a single-outcome market is untouched, keeps its own
        price, and gains no sub-row."""
        cells = _table_rows(render_snapshot(MarketSnapshot(matches=[_row(prob=0.42)])))

        assert len(cells) == 1
        assert (cells[0]["platform"], cells[0]["prob"]) == ("kalshi", "0.42")
        # Scoped to the TABLE: the glyph legitimately appears in the legend, which every snapshot
        # carries, so a whole-render substring check would never fail.
        assert _sub_rows_per_market(render_snapshot(MarketSnapshot(matches=[_row(prob=0.42)]))) == [0]


class TestChildRowBudget:
    """How the ``↳`` rows are rationed, and what a thinned table has to admit.

    The section goes to the expensive forecaster models on every question and 86.5% of the Kalshi
    catalogue is multi-strike, so an 8-row slate is mostly parents with outcomes and the per-market
    cap alone would license 80 sub-rows. The allocation is what keeps that bounded WITHOUT
    reintroducing the priceless-parent row it exists to remove.
    """

    def _outcomes(self, count: int) -> tuple[MarketChild, ...]:
        return tuple(MarketChild(title=f"outcome {index}", implied_prob_yes=0.5) for index in range(count))

    def _slate(self, *, rows: int, outcomes: int) -> list[MarketMatch]:
        return [_row(title=f"market {index}", prob=None, children=self._outcomes(outcomes)) for index in range(rows)]

    def test_no_multi_outcome_market_ever_renders_priceless(self) -> None:
        """THE invariant, and the reason the allocation fills one round at a time rather than
        draining in rank order. While the snapshot cap is at least `RENDER_BUDGET`, every
        multi-outcome market shows at least its leading outcome — draining in rank order would have
        spent the whole budget on the top rows and left the rest rendering a bare `-`."""
        rendered = render_snapshot(
            MarketSnapshot(matches=self._slate(rows=RENDER_BUDGET, outcomes=MAX_CHILD_ROWS_PER_MARKET))
        )

        cells = _table_rows(rendered)
        parents = [cell for cell in cells if cell["platform"] != CHILD_ROW_MARKER]
        assert len(parents) == RENDER_BUDGET
        assert all(cell["prob"] == "-" for cell in parents), "every parent in this slate is multi-outcome"
        priced_sub_rows = [cell for cell in cells if cell["platform"] == CHILD_ROW_MARKER and cell["prob"] != "-"]
        assert len(priced_sub_rows) >= RENDER_BUDGET

    def test_the_allowance_is_shared_one_round_at_a_time(self) -> None:
        """Eight markets wanting ten outcomes each get three apiece, not ten-ten-four-nothing."""
        rendered = render_snapshot(
            MarketSnapshot(matches=self._slate(rows=RENDER_BUDGET, outcomes=MAX_CHILD_ROWS_PER_MARKET))
        )

        per_market = _sub_rows_per_market(rendered)
        assert per_market == [MAX_CHILD_ROWS_PER_SNAPSHOT // RENDER_BUDGET] * RENDER_BUDGET
        assert sum(per_market) == MAX_CHILD_ROWS_PER_SNAPSHOT

    def test_a_market_with_fewer_outcomes_hands_its_unused_slots_back(self) -> None:
        """Otherwise a binary-heavy slate would leave a third of the budget unspent while the one
        ladder on it rendered three of its rungs."""
        matches = [
            _row(title="ladder", prob=None, children=self._outcomes(MAX_CHILD_ROWS_PER_MARKET)),
            _row(title="pair", prob=None, children=self._outcomes(2)),
        ]

        per_market = _sub_rows_per_market(render_snapshot(MarketSnapshot(matches=matches)))

        assert per_market == [MAX_CHILD_ROWS_PER_MARKET, 2]

    def test_the_per_market_cap_bounds_a_single_market(self) -> None:
        """With the snapshot cap wide open, one market still cannot render a 30-rung ladder."""
        matches = [_row(title="ladder", prob=None, children=self._outcomes(30))]

        rendered = render_snapshot(MarketSnapshot(matches=matches))

        assert _sub_rows_per_market(rendered) == [MAX_CHILD_ROWS_PER_MARKET]
        assert (
            f"[{30 - MAX_CHILD_ROWS_PER_MARKET} more outcomes omitted — render budget, 10.00 of priced mass]"
            in rendered
        )

    def test_a_truncated_market_says_so_in_a_marker_row(self) -> None:
        """A thinned table must never read as a complete one. The wording mirrors the
        resolution-source fetcher's `[N additional source(s) omitted — section budget]`."""
        matches = [_row(title="ladder", prob=None, children=self._outcomes(MAX_CHILD_ROWS_PER_MARKET + 1))]

        cells = _table_rows(render_snapshot(MarketSnapshot(matches=matches)))

        assert cells[-1]["title"] == "[1 more outcome omitted — render budget, 0.50 of priced mass]"
        assert cells[-1]["platform"] == CHILD_ROW_MARKER

    def test_a_marker_row_fills_every_column(self) -> None:
        """A row short of a cell would end the markdown table and orphan every row after it."""
        matches = [_row(title="ladder", prob=None, children=self._outcomes(MAX_CHILD_ROWS_PER_MARKET + 4))]

        rendered = render_snapshot(MarketSnapshot(matches=matches))

        widths = {line.count("|") for line in rendered.split("\n") if line.startswith("| ")}
        assert widths == {len(TABLE_COLUMNS) + 1}
        assert _table_rows(rendered)[-1]["title"] == "[4 more outcomes omitted — render budget, 2.00 of priced mass]"

    def test_the_snapshot_cap_is_at_least_the_row_budget(self) -> None:
        """The constant relationship the priceless-parent invariant rests on: fewer sub-row slots
        than rendered rows would mean some multi-outcome market gets none."""
        assert MAX_CHILD_ROWS_PER_SNAPSHOT >= RENDER_BUDGET
        assert MAX_CHILD_ROWS_PER_MARKET <= MAX_CHILD_ROWS_PER_SNAPSHOT

    def test_the_marker_sums_only_the_omitted_outcomes_mass(self) -> None:
        """The figure a forecaster reads must be the mass they are NOT seeing — counting rendered
        rows into it would overstate the cut and teach models to distrust complete tables. Children
        arrive price-descending from the adapters, so the omitted tail is the low-priced end:
        12 outcomes at 0.30, 0.29, ... — the two past the per-market cap carry 0.20 + 0.19."""
        children = tuple(
            MarketChild(title=f"outcome {index}", implied_prob_yes=0.30 - 0.01 * index)
            for index in range(MAX_CHILD_ROWS_PER_MARKET + 2)
        )
        matches = [_row(title="ladder", prob=None, children=children)]

        rendered = render_snapshot(MarketSnapshot(matches=matches))

        assert "[2 more outcomes omitted — render budget, 0.39 of priced mass]" in rendered

    def test_an_unpriced_cut_honestly_reads_zero_priced_mass(self) -> None:
        """Omitted outcomes with no live quote contribute nothing to the figure — inventing mass
        for them would be worse than the silence the clause replaces. `0.00` is itself information:
        the cut rows carried no price."""
        children = tuple(
            MarketChild(title=f"outcome {index}", implied_prob_yes=0.5 if index < MAX_CHILD_ROWS_PER_MARKET else None)
            for index in range(MAX_CHILD_ROWS_PER_MARKET + 3)
        )
        matches = [_row(title="ladder", prob=None, children=children)]

        rendered = render_snapshot(MarketSnapshot(matches=matches))

        assert "[3 more outcomes omitted — render budget, 0.00 of priced mass]" in rendered
