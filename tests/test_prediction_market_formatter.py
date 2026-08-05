"""The seam's formatter, a thin delegate over the market-retrieval renderer.

What belongs here is only what the SEAM decides: that an empty snapshot renders nothing, that the
degraded-ranking marker is derived from the snapshot's own ranking token rather than passed in,
which preamble a rendered row's top tier selects, and the liquidity-label thresholds. The render
itself — columns, cell formatting, budgets — is tested in `tests/test_market_retrieval_rendering.py`.

The `market_row` helper these use lives in `tests/market_retrieval_fakes.py`.
"""

from __future__ import annotations

import pytest

from metaculus_bot.research.market_retrieval.ranking import DEGRADED_RANKING_MARKER
from metaculus_bot.research.market_retrieval.rendering import TABLE_COLUMNS
from metaculus_bot.research.prediction_market import MarketSnapshot, _liquidity_label, format_snapshot_for_research
from tests.market_retrieval_fakes import market_row as _row


class TestFormatterDelegate:
    def test_an_empty_snapshot_renders_nothing(self):
        """The early return is what produces `status="empty"` downstream and the
        attempted-vs-succeeded distinction residual analysis reads off the archive. Under ranked
        selection a zero-row render is a legitimate outcome, so this is a hot path."""
        assert format_snapshot_for_research(MarketSnapshot(matches=[])) == ""

    def test_the_degraded_marker_is_derived_from_the_snapshots_own_ranking_token(self):
        """Derived, not passed in, so the render is reproducible from an archived snapshot
        alone — which matters because `record_raw_research` archives the snapshot and a replay
        tool renders from it."""
        rows = [_row("A market", tier="")]

        healthy = format_snapshot_for_research(MarketSnapshot(matches=rows, sources={"ranking": "ok(1)"}))
        degraded = format_snapshot_for_research(MarketSnapshot(matches=rows, sources={"ranking": "error(x)"}))

        assert DEGRADED_RANKING_MARKER not in healthy
        assert degraded.startswith(DEGRADED_RANKING_MARKER)

    def test_a_missing_ranking_token_reads_as_degraded(self):
        """The conservative direction: an unlabelled snapshot may be in retrieval order, and
        claiming evidential order falsely is worse than a marker on a healthy table."""
        assert DEGRADED_RANKING_MARKER in format_snapshot_for_research(MarketSnapshot(matches=[_row("A market")]))

    def test_cells_are_addressable_by_header_name(self):
        """Read by HEADER NAME, never by index. The previous version of this test carried a
        comment enumerating cell positions, and every column addition silently shifted them."""
        snapshot = MarketSnapshot(matches=[_row("Will X win?", platform="predictit")], sources={"ranking": "ok(1)"})

        formatted = format_snapshot_for_research(snapshot)
        header = next(line for line in formatted.splitlines() if line.startswith("| platform |"))
        columns = [cell.strip() for cell in header.strip("|").split("|")]
        row = next(line for line in formatted.splitlines() if line.startswith("| predictit"))
        cells = dict(zip(columns, [cell.strip() for cell in row.strip("|").split("|")], strict=True))

        assert columns == list(TABLE_COLUMNS)
        assert cells["signal"] == "no-liquidity-data"
        assert cells["status"] == "open"
        assert cells["relation"] == "same_quantity_other_cut"
        assert cells["why"] == "same series, adjacent month"

    def test_the_preamble_follows_the_ranker_top_tier(self):
        """Strong when a rendered row measures the same quantity, neutral otherwise. The selector
        is the ranker's own grade, replacing the content-overlap + confidence bar it retired."""
        strong = format_snapshot_for_research(
            MarketSnapshot(matches=[_row("A", tier="same_quantity_same_date")], sources={"ranking": "ok(1)"})
        )
        neutral = format_snapshot_for_research(
            MarketSnapshot(matches=[_row("A", tier="weak")], sources={"ranking": "ok(1)"})
        )

        assert "extremely strong evidence" in strong.lower()
        assert "may all be off-topic" in neutral.lower()
        assert "extremely strong evidence" not in neutral.lower()
        # The retired vocabulary must not come back: rows are chosen by a model reading each
        # market's rules, not by word overlap.
        for rendered in (strong, neutral):
            assert "fuzzy" not in rendered.lower()
            assert "verify-carefully" not in rendered
            assert "likely-relevant" not in rendered

    @pytest.mark.parametrize(
        ("total_volume", "expected"), [(1_000.0, "thin"), (20_000.0, "decent"), (100_000.0, "deep")]
    )
    def test_liquidity_label_real_money_thresholds(self, total_volume, expected):
        row = _row("x")
        row.total_volume = total_volume
        row.open_interest = None
        assert _liquidity_label(row) == expected

    @pytest.mark.parametrize(("num_bettors", "expected"), [(5, "thin"), (50, "decent"), (200, "high")])
    def test_liquidity_label_manifold_bettor_thresholds(self, num_bettors, expected):
        row = _row("x", platform="manifold")
        row.total_volume = None
        row.open_interest = None
        row.num_bettors = num_bettors
        assert _liquidity_label(row) == expected
