"""The spot-peer-primary convention for Metaculus's own platform scores.

The tournament leaderboard ranks on ``spot_peer_score``; ``peer_score`` is the same
quantity scaled by coverage, which for a submit-once bot is largely a function of how
early it submitted. These tests pin the convention at every layer it can be lost:
the accessors, the ranking preference, the sort-tier separation that stops the two
quantities interleaving, the rendered label, and the report table.
"""

from __future__ import annotations

import logging

import pytest

from metaculus_bot.performance_analysis import analysis
from metaculus_bot.performance_analysis.platform_scores import (
    FALLBACK_TIER,
    NO_SCORE_TIER,
    PEER_FIELD,
    RANKING_FIELDS,
    SPOT_PEER_FIELD,
    baseline_score,
    coverage,
    log_ranking_score_sources,
    peer_score,
    platform_score_fragments,
    platform_scores,
    ranking_score,
    score_field,
    spot_baseline_score,
    spot_peer_score,
)

# The exact shape collector._our_forecast writes onto a record, so these tests break if
# the collector's key names drift away from the accessors.
_COLLECTOR_SHAPED_SCORES: dict[str, float] = {
    "peer_score": -15.0,
    "spot_peer_score": -38.8,
    "baseline_score": -9.0,
    "spot_baseline_score": -23.0,
    "coverage": 0.387,
    "weighted_coverage": 0.4,
    "relative_legacy_score": 0.1,
}


def _record(**scores: float | None) -> dict:
    """A binary record carrying only platform scores.

    The bot-side score keys are present-but-None so the record can also flow through
    ``generate_report``, whose other sections index them directly.
    """
    return {
        "post_id": 1,
        "type": "binary",
        "brier_score": None,
        "log_score": None,
        "numeric_log_score": None,
        "mc_log_score": None,
        "resolution_parsed": None,
        "metaculus_scores": dict(scores),
    }


class TestAccessors:
    def test_reads_every_field_off_a_collector_shaped_record(self):
        record = {"post_id": 1, "metaculus_scores": dict(_COLLECTOR_SHAPED_SCORES)}
        assert spot_peer_score(record) == pytest.approx(-38.8)
        assert peer_score(record) == pytest.approx(-15.0)
        assert spot_baseline_score(record) == pytest.approx(-23.0)
        assert baseline_score(record) == pytest.approx(-9.0)
        assert coverage(record) == pytest.approx(0.387)

    def test_missing_block_and_null_values_read_as_none_not_zero(self):
        """A null score is "not measured"; reading it as 0.0 would publish a fabricated
        exactly-average result into a ranking."""
        assert spot_peer_score({"post_id": 1}) is None
        assert spot_peer_score({"post_id": 1, "metaculus_scores": None}) is None
        assert spot_peer_score(_record(spot_peer_score=None)) is None
        assert peer_score(_record(spot_peer_score=-1.0)) is None

    def test_platform_scores_block_is_a_dict_even_when_absent(self):
        assert platform_scores({"post_id": 1}) == {}
        assert platform_scores({"post_id": 1, "metaculus_scores": None}) == {}

    def test_score_field_coerces_ints_to_float(self):
        assert score_field(_record(spot_peer_score=7), SPOT_PEER_FIELD) == pytest.approx(7.0)


class TestRankingScore:
    def test_spot_peer_wins_when_both_are_present(self):
        ranked = ranking_score(_record(spot_peer_score=-38.8, peer_score=-15.0))
        assert ranked is not None
        assert ranked.field == SPOT_PEER_FIELD
        assert ranked.value == pytest.approx(-38.8)
        assert ranked.is_spot
        assert ranked.tier == 0

    def test_falls_back_to_coverage_scaled_peer_only_when_spot_is_absent(self):
        ranked = ranking_score(_record(peer_score=-15.0))
        assert ranked is not None
        assert ranked.field == PEER_FIELD
        assert not ranked.is_spot
        assert ranked.tier == 1

    def test_none_when_the_record_carries_neither(self):
        assert ranking_score(_record(coverage=0.5)) is None
        assert ranking_score({"post_id": 1}) is None

    def test_tiers_leave_room_for_a_callers_own_fallback_and_for_no_score(self):
        """Spot and peer occupy tiers 0/1; a caller's Brier or log-score fallback sorts
        behind both, and "nothing to rank on" sorts last."""
        assert RANKING_FIELDS == (SPOT_PEER_FIELD, PEER_FIELD)
        assert FALLBACK_TIER == 2
        assert NO_SCORE_TIER == 3


class TestLogRankingScoreSources:
    def test_counts_each_source(self):
        records = [
            _record(spot_peer_score=1.0, peer_score=0.5),
            _record(spot_peer_score=-2.0),
            _record(peer_score=-1.0),
            {"post_id": 4},
        ]
        counts = log_ranking_score_sources(records, cut="test")
        assert counts == {SPOT_PEER_FIELD: 2, PEER_FIELD: 1, "none": 1}

    def test_peer_only_records_raise_a_warning_not_a_silent_info(self, caplog):
        with caplog.at_level(logging.WARNING):
            log_ranking_score_sources([_record(peer_score=-1.0)], cut="test")
        assert any("coverage-scaled" in r.message for r in caplog.records if r.levelno == logging.WARNING)

    def test_an_all_spot_cohort_warns_about_nothing(self, caplog):
        with caplog.at_level(logging.WARNING):
            log_ranking_score_sources([_record(spot_peer_score=-1.0)], cut="test")
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


class TestRenderedFragments:
    def test_spot_leads_bolded_and_peer_is_labelled_coverage_scaled(self):
        fragments = platform_score_fragments(_record(spot_peer_score=-38.8, peer_score=-15.0))
        assert fragments == ["spot peer **-38.8**", "peer -15.0 (coverage-scaled, secondary)"]

    def test_peer_alone_still_carries_its_label(self):
        """An unlabelled peer figure reads as the leaderboard number, which on this pair
        would understate the miss by 24 points."""
        assert platform_score_fragments(_record(peer_score=-15.0)) == ["peer -15.0 (coverage-scaled, secondary)"]

    def test_no_fragments_when_the_record_carries_no_platform_score(self):
        assert platform_score_fragments({"post_id": 1}) == []


class TestPlatformScoreSummary:
    @staticmethod
    def _data() -> list[dict]:
        return [
            _record(spot_peer_score=-38.8, peer_score=-15.0, spot_baseline_score=-23.0, baseline_score=-9.0),
            _record(spot_peer_score=10.0, peer_score=5.0, coverage=0.5),
            _record(),
        ]

    def test_counts_each_field_over_the_records_that_carry_it(self):
        summary = analysis.platform_score_summary(self._data())
        assert summary["count"] == 3
        assert summary["spot_peer"] == {
            "count": 2,
            "mean": pytest.approx(-14.4),
            "median": pytest.approx(-14.4),
        }
        assert summary["peer"]["count"] == 2
        assert summary["spot_baseline"]["count"] == 1
        assert summary["baseline"]["count"] == 1
        assert summary["mean_coverage"] == pytest.approx(0.5)

    def test_empty_input_reports_no_scores_rather_than_zeros(self):
        summary = analysis.platform_score_summary([])
        assert summary["count"] == 0
        for key in ("spot_peer", "peer", "spot_baseline", "baseline"):
            assert summary[key] == {"count": 0, "mean": None, "median": None}
        assert summary["mean_coverage"] is None


class TestReportSection:
    def test_section_labels_spot_as_primary_and_peer_as_coverage_scaled(self):
        section = "\n".join(analysis._platform_score_section_lines(TestPlatformScoreSummary._data()))
        assert "## Metaculus Platform Scores" in section
        spot_row = next(line for line in section.splitlines() if line.startswith("| spot peer"))
        peer_row = next(line for line in section.splitlines() if line.startswith("| peer"))
        assert "PRIMARY" in spot_row
        assert "coverage-scaled, secondary" in peer_row
        # Spot must be rendered ABOVE peer — a reader takes the first row as the headline.
        assert section.index(spot_row) < section.index(peer_row)

    def test_section_is_absent_when_no_record_carries_a_platform_score(self):
        assert analysis._platform_score_section_lines([_record()]) == []

    def test_generate_report_leads_with_the_platform_scores(self):
        report = analysis.generate_report(TestPlatformScoreSummary._data())
        assert "## Metaculus Platform Scores" in report
