"""The rendered snapshot: the empty-render contract, the columns, and the MC-parser trap.

Three of these guard things that break quietly rather than loudly:

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
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from metaculus_bot.research.market_retrieval.ranking import DEGRADED_RANKING_MARKER, TIER_UNSPECIFIED, TIERS
from metaculus_bot.research.market_retrieval.rendering import (
    MARKET_PREAMBLE_NEUTRAL,
    MARKET_PREAMBLE_STRONG,
    MARKET_SIGNAL_LEGEND,
    RAW_RULES_MAX_CHARS,
    TABLE_COLUMNS,
    render_snapshot,
)
from metaculus_bot.research.market_retrieval.types import MarketMatch, MarketSnapshot
from tests.test_market_retrieval_generation import Platform

_PERCENT_TAIL_RE = re.compile(r":\s*[0-9]+(?:\.[0-9]+)?\s*%\s*$")


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

    def test_the_legend_explains_the_new_columns_and_not_the_retired_one(self) -> None:
        assert "`relation`" in MARKET_SIGNAL_LEGEND
        assert "`why`" in MARKET_SIGNAL_LEGEND
        assert "ordered by EVIDENTIAL VALUE" in MARKET_SIGNAL_LEGEND
        assert "RESOLVED" in MARKET_SIGNAL_LEGEND
        assert "likely-relevant" not in MARKET_SIGNAL_LEGEND
        assert "verify-carefully" not in MARKET_SIGNAL_LEGEND

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
        assert bullet.count("r") == RAW_RULES_MAX_CHARS

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
