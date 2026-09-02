"""Tests for the in-place re-resolution detector (performance_analysis/rescore_diff.py).

The situation modelled is q44798 (post 44645, "Halo: Campaign Evolved Metascore"), which
Metaculus resolved at 80 and then edited to 82 without ``resolution_set_time`` moving --
the stored stamp (2026-08-31T21:38:45Z) actually PRECEDES the pull that still read 80. The
record's spot peer went +5.41 -> -5.42 between two consecutive rounds and nothing noticed,
so the round's published tables for that question were stale. A value-level diff of the two
pulls is the only thing that catches it.

Fully offline: these are pure dict-to-dict comparisons, no API and no disk.
"""

import logging

from metaculus_bot.performance_analysis.rescore_diff import (
    RESCORE_ATOL,
    diff_platform_rescores,
    render_rescore_summary,
)


def _record(
    *,
    question_id: int = 44798,
    post_id: int = 44645,
    resolution_raw: str = "80",
    resolution_parsed: float | str | None = 80.0,
    spot_peer: float | None = 5.4070,
    peer: float | None = 3.4008,
    extra_scores: dict | None = None,
) -> dict:
    """One performance record, trimmed to the fields the diff reads."""
    scores = {
        "spot_peer_score": spot_peer,
        "peer_score": peer,
        "spot_baseline_score": 73.5477,
        "coverage": 0.6602,
    }
    if extra_scores:
        scores.update(extra_scores)
    return {
        "question_id": question_id,
        "post_id": post_id,
        "type": "discrete",
        "resolution_raw": resolution_raw,
        "resolution_parsed": resolution_parsed,
        "metaculus_scores": scores,
    }


class TestUnchangedPull:
    """The common case: a re-pull that reproduces the prior round exactly."""

    def test_nothing_moved_reports_zero_rescored(self):
        prior = [_record()]
        new = [_record()]

        diff = diff_platform_rescores(prior, new)

        assert (diff.compared, diff.rescored, diff.unmatched) == (1, 0, 0)
        assert diff.changes == ()

    def test_an_unchanged_record_is_tagged_False_not_None(self):
        """False means 'compared, nothing moved'; None means 'never compared'. A cut that
        conflates them counts unmatched records as confirmed-stable."""
        new = [_record()]

        diff_platform_rescores([_record()], new)

        assert new[0]["platform_rescored"] is False
        assert new[0]["platform_rescored_fields"] == []
        assert new[0]["prior_resolution"] is None
        assert new[0]["prior_metaculus_scores"] is None

    def test_float_noise_below_the_tolerance_is_not_a_change(self):
        new = [_record(spot_peer=5.4070 + RESCORE_ATOL / 10)]

        diff = diff_platform_rescores([_record()], new)

        assert diff.rescored == 0
        assert new[0]["platform_rescored"] is False


class TestResolutionValueChange:
    """The q44798 shape: the resolution itself was edited, and the scores followed."""

    def _diff(self):
        prior = [_record(resolution_raw="80", resolution_parsed=80.0, spot_peer=5.4070, peer=3.4008)]
        new = [_record(resolution_raw="82", resolution_parsed=82.0, spot_peer=-5.4190, peer=-3.8224)]
        return prior, new, diff_platform_rescores(prior, new)

    def test_the_record_is_tagged_rescored(self):
        _prior, new, diff = self._diff()

        assert diff.rescored == 1
        assert new[0]["platform_rescored"] is True

    def test_every_moved_field_is_named(self):
        _prior, new, _diff = self._diff()

        assert new[0]["platform_rescored_fields"] == [
            "resolution_raw",
            "resolution_parsed",
            "metaculus_scores.spot_peer_score",
            "metaculus_scores.peer_score",
        ]

    def test_the_prior_resolution_and_scores_are_carried(self):
        _prior, new, _diff = self._diff()

        assert new[0]["prior_resolution"] == "80"
        assert new[0]["prior_metaculus_scores"]["spot_peer_score"] == 5.4070

    def test_the_changes_carry_old_and_new_values(self):
        _prior, _new, diff = self._diff()
        by_field = {change.field: change for change in diff.changes}

        assert (by_field["resolution_raw"].old, by_field["resolution_raw"].new) == ("80", "82")
        assert by_field["metaculus_scores.spot_peer_score"].old == 5.4070
        assert by_field["metaculus_scores.spot_peer_score"].new == -5.4190
        assert diff.rescored_question_ids == (44798,)

    def test_one_warn_per_changed_field_names_the_question(self, caplog):
        with caplog.at_level(logging.WARNING):
            self._diff()

        warns = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warns) == 4
        assert all(w.startswith("PLATFORM_RESCORED: question=44798 post=44645 field=") for w in warns)
        assert any("old=80 new=82" in w for w in warns)


class TestScoreOnlyChange:
    """Metaculus re-scored without touching the resolution — still a stale prior table."""

    def test_a_score_move_alone_is_a_rescore(self):
        new = [_record(spot_peer=1.25)]

        diff = diff_platform_rescores([_record()], new)

        assert diff.rescored == 1
        assert new[0]["platform_rescored_fields"] == ["metaculus_scores.spot_peer_score"]

    def test_the_prior_resolution_is_carried_even_when_the_resolution_held(self):
        """Equal to the current value on purpose: that is how a reader tells a score-only
        re-score from a re-resolution without re-reading the prior file."""
        new = [_record(spot_peer=1.25)]

        diff_platform_rescores([_record()], new)

        assert new[0]["prior_resolution"] == new[0]["resolution_raw"] == "80"

    def test_a_score_appearing_where_the_prior_had_none_is_a_change(self):
        new = [_record(spot_peer=5.4070)]

        diff = diff_platform_rescores([_record(spot_peer=None)], new)

        assert [c.field for c in diff.changes] == ["metaculus_scores.spot_peer_score"]
        assert (diff.changes[0].old, diff.changes[0].new) == (None, 5.4070)

    def test_a_score_field_only_the_new_pull_carries_is_diffed(self):
        """The field list is the union of both sides, so a field Metaculus adds later needs
        no edit here to be noticed."""
        new = [_record(extra_scores={"some_future_score": 12.0})]

        diff = diff_platform_rescores([_record()], new)

        assert [c.field for c in diff.changes] == ["metaculus_scores.some_future_score"]


class TestMatching:
    def test_a_record_with_no_prior_is_tagged_None_and_counted_unmatched(self):
        new = [_record(question_id=45400, post_id=45300)]

        diff = diff_platform_rescores([_record()], new)

        assert (diff.compared, diff.unmatched) == (0, 1)
        assert new[0]["platform_rescored"] is None
        assert new[0]["platform_rescored_fields"] is None

    def test_matching_needs_both_ids_not_either(self):
        """Question ids and post ids share one integer space on Metaculus, so a match on one
        id alone can pair two different questions."""
        prior = [_record(question_id=44873, post_id=44724, resolution_raw="yes")]
        # A different question whose POST id happens to equal the prior record's QUESTION id.
        new = [_record(question_id=99999, post_id=44873, resolution_raw="no")]

        diff = diff_platform_rescores(prior, new)

        assert (diff.compared, diff.rescored, diff.unmatched) == (0, 0, 1)

    def test_duplicate_prior_keys_are_reported_and_the_last_wins(self, caplog):
        prior = [_record(spot_peer=1.0), _record(spot_peer=2.0)]
        new = [_record(spot_peer=2.0)]

        with caplog.at_level(logging.WARNING):
            diff = diff_platform_rescores(prior, new)

        assert diff.duplicate_prior_keys == ((44798, 44645),)
        assert diff.rescored == 0, "the last prior record wins, and it matches"
        assert any("duplicated" in r.message for r in caplog.records)

    def test_an_empty_prior_leaves_every_record_uncompared(self):
        new = [_record(), _record(question_id=1, post_id=2)]

        diff = diff_platform_rescores([], new)

        assert (diff.compared, diff.unmatched) == (0, 2)
        assert all(record["platform_rescored"] is None for record in new)


class TestRenderRescoreSummary:
    def test_a_clean_diff_says_prior_tables_are_current(self):
        new = [_record()]
        diff_platform_rescores([_record()], new)

        text = "\n".join(render_rescore_summary(new))

        assert "1 of 1 record(s) matched a prior pull, 0 re-scored" in text
        assert "Prior-round tables remain current" in text

    def test_a_rescore_is_rendered_with_both_values_and_the_staleness_warning(self):
        new = [_record(resolution_raw="82", resolution_parsed=82.0, spot_peer=-5.4190)]
        diff_platform_rescores([_record()], new)

        lines = render_rescore_summary(new)
        text = "\n".join(lines)

        assert "1 re-scored or re-resolved" in lines[0]
        assert "q44798 (post 44645) resolution_raw: '80' -> '82'" in text
        assert "metaculus_scores.spot_peer_score: 5.407 -> -5.419" in text
        assert "is stale" in text

    def test_untagged_records_render_as_uncompared(self):
        """Reading a dataset that never went through the diff must not claim stability."""
        text = "\n".join(render_rescore_summary([_record()]))

        assert "0 of 1 record(s) matched a prior pull" in text
