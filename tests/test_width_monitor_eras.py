"""Which config era a record is bucketed into, and what the era boundaries are pinned to.

Two concerns that have to stay separate: the bucketing convention (half-open intervals, timestamp
shapes, unparseable timestamps) and the boundary VALUES, which are merge-to-main committer
timestamps and are anchored here to facts the constants cannot define themselves.
"""

from datetime import UTC, datetime, timedelta

import pytest

from metaculus_bot.performance_analysis.eras import B4E9DF0_MERGED_AT, GRID_SCALED_MAX_STEP_MERGED_AT
from metaculus_bot.performance_analysis.width_monitor import (
    TS_ANCHOR_ENABLE,
    WIDENING_FLIP,
    assign_era,
    default_eras,
)


class TestEraAssignment:
    def test_boundaries(self):
        """Bucketing at literal wall-clock instants either side of each boundary.

        These used ``WIDENING_FLIP.isoformat()`` / ``TS_ANCHOR_ENABLE.isoformat()``
        as inputs, which fed each constant back to itself and made the assertion
        pass for any value the constants happened to hold (2099-01-01 included).
        That is how the wrong TS-anchor boundary survived a green suite. Literal
        timestamps here; the constants' own values are pinned to their merge
        commits in ``TestEraBoundariesAreMergeDates``.
        """
        eras = default_eras()
        assert assign_era({"bot_comment_created_at": "2026-05-11T23:59:59Z"}, eras) == "widening_on (k_tail=1.25)"
        assert assign_era({"bot_comment_created_at": "2026-05-18T17:21:19Z"}, eras) == "widening_off (k_tail=1.0)"
        assert assign_era({"bot_comment_created_at": "2026-07-01T00:00:00Z"}, eras) == "widening_off (k_tail=1.0)"
        assert assign_era({"bot_comment_created_at": "2026-07-21T17:07:37Z"}, eras) == "ts_anchor (sharpen)"
        assert assign_era({"bot_comment_created_at": "2026-08-01T00:00:00Z"}, eras) == "ts_anchor (sharpen)"

    def test_boundary_instant_is_half_open(self):
        """``[start, end)``: the boundary instant itself belongs to the LATER era,
        and one microsecond earlier to the earlier one.

        Value-independent by construction — it asserts the interval convention,
        not the dates — so it is deliberately kept separate from the assertions
        that pin the dates themselves.
        """
        eras = default_eras()
        for boundary, earlier_label, later_label in (
            (WIDENING_FLIP, "widening_on (k_tail=1.25)", "widening_off (k_tail=1.0)"),
            (TS_ANCHOR_ENABLE, "widening_off (k_tail=1.0)", "ts_anchor (sharpen)"),
        ):
            assert assign_era({"bot_comment_created_at": boundary.isoformat()}, eras) == later_label
            just_before = boundary - timedelta(microseconds=1)
            assert assign_era({"bot_comment_created_at": just_before.isoformat()}, eras) == earlier_label

    def test_missing_timestamp(self):
        assert assign_era({"bot_comment_created_at": None}, default_eras()) == "no_timestamp"
        assert assign_era({}, default_eras()) == "no_timestamp"

    def test_unparseable_timestamp_is_not_attributed_to_an_era(self):
        """An unparseable timestamp lands in no_timestamp, never silently in the first or last era.

        Mis-attributing one record's config era is exactly what the shared parser exists to
        prevent.
        """
        for raw in ("not-a-date", "2026-13-45T99:00:00Z", ""):
            assert assign_era({"bot_comment_created_at": raw}, default_eras()) == "no_timestamp"

    def test_offset_and_naive_timestamps_land_in_the_same_era(self):
        """The three ISO shapes the archive carries (Z, explicit offset, naive) must agree.

        A naive read of a -07:00 instant is 7 hours off and can cross a boundary.
        """
        eras = default_eras()
        instant_utc = "2026-07-21T18:07:37Z"
        same_instant_offset = "2026-07-21T11:07:37-07:00"
        naive_utc = "2026-07-21T18:07:37"
        labels = {
            assign_era({"bot_comment_created_at": raw}, eras) for raw in (instant_utc, same_instant_offset, naive_utc)
        }
        assert labels == {"ts_anchor (sharpen)"}


class TestEraBoundariesAreMergeDates:
    """Era boundaries must be MERGE-TO-MAIN timestamps, never authoring dates.

    Prod runs from ``main``, so a config change is live only once its merge
    commit lands there. Every assertion here anchors to a fact the constant
    cannot define — either the merge commit's committer timestamp or the roster
    that the same merge retired — because the pre-existing boundary test fed the
    constant back to itself and therefore passed for any value (including
    2099-01-01).
    """

    def test_boundaries_equal_merge_commit_timestamps(self):
        """Both constants equal the committer date of the merge that landed them.

        Re-derive with ``TZ=UTC git log -1 --date=iso-local --format='%h %cd' <sha>``:

          * ``0e85e1b`` 2026-05-18 17:21:19 +0000 — flipped ``TAIL_WIDEN_K_TAIL``
            1.25 -> 1.0 (confirmed by value across ``0e85e1b^1``/``0e85e1b``).
            Authored ``b8d730f`` 2026-05-12, six days earlier.
          * ``b4e9df0`` 2026-07-21 17:07:37 +0000 — the july15 bundle: TS anchor
            provider + prompt clause + the ``TS_ANCHOR_ENABLED: 'true'`` yaml
            flip, all authored 2026-07-17, four days earlier.
        """
        assert datetime(2026, 5, 18, 17, 21, 19, tzinfo=UTC) == WIDENING_FLIP
        assert datetime(2026, 7, 21, 17, 7, 37, tzinfo=UTC) == TS_ANCHOR_ENABLE

    def test_pre_merge_roster_record_is_not_in_post_merge_era(self):
        """A record that provably ran the retired 6-model roster cannot be in the
        post-``b4e9df0`` era.

        This is qid 44795 verbatim: published 2026-07-17T21:16:47Z, four days
        after the anchor was authored and four days before it reached ``main``.
        Its own comment names ``gpt-5.5``, ``claude-opus-4.6`` and ``grok-4.5``
        — and ``b4e9df0`` dropped the roster from six models to the
        latest-per-vendor triple in the same merge that landed the anchor, so
        that combination is impossible post-merge. The assertion therefore holds
        independently of what value the constant happens to carry.
        """
        record = {
            "bot_comment_created_at": "2026-07-17T21:16:47.573093+00:00",
            "bot_comment": (
                "*Forecaster 1 (gpt-5.6-sol)*: 12.0\n"
                "*Forecaster 2 (gpt-5.5)*: 13.0\n"
                "*Forecaster 3 (claude-opus-4.8)*: 11.5\n"
                "*Forecaster 4 (claude-opus-4.6)*: 12.5\n"
                "*Forecaster 5 (gemini-3.1-pro-preview)*: 12.2\n"
                "*Forecaster 6 (grok-4.5)*: 14.0\n"
            ),
        }
        assert assign_era(record, default_eras()) == "widening_off (k_tail=1.0)"

    @pytest.mark.parametrize(
        "created_at",
        [
            "2026-07-17T03:24:24Z",  # the anchor provider's own authoring instant
            "2026-07-19T12:00:00Z",
            "2026-07-21T00:00:00Z",
            "2026-07-21T17:07:36Z",  # one second before the merge landed
        ],
    )
    def test_july15_gap_window_buckets_pre_anchor(self, created_at):
        """Nothing on ``main`` changed between the 2026-07-12 merge (``f084bf7``)
        and ``b4e9df0``, so every run in the author-to-merge gap used the
        identical pre-anchor config and belongs in ``widening_off``."""
        assert assign_era({"bot_comment_created_at": created_at}, default_eras()) == "widening_off (k_tail=1.0)"

    def test_every_b4e9df0_gate_reads_the_same_instant(self):
        """The monitor's era boundary and the clamp screen's era gate are the SAME
        merge, so they are aliases of one constant.

        Both mark ``b4e9df0``: the era split the width rows are bucketed by, and the
        instant after which a coarse discrete grid's max-step cap stopped being a flat
        0.2. Two independently-edited copies could drift, which would file one record
        into the anchor era while screening it under the pre-fix cap.
        """
        assert TS_ANCHOR_ENABLE is B4E9DF0_MERGED_AT
        assert GRID_SCALED_MAX_STEP_MERGED_AT is B4E9DF0_MERGED_AT

    @pytest.mark.parametrize(
        "created_at",
        [
            "2026-05-12T10:32:02Z",  # b8d730f authoring instant
            "2026-05-15T12:00:00Z",
            "2026-05-18T17:21:18Z",  # one second before 0e85e1b landed
        ],
    )
    def test_widening_gap_window_buckets_pre_flip(self, created_at):
        """Same defect class as the TS-anchor boundary, six days wide. Zero
        resolved records fall in this window today, so it is latent — a future
        backfill recovering May 12-18 records would activate it silently."""
        assert assign_era({"bot_comment_created_at": created_at}, default_eras()) == "widening_on (k_tail=1.25)"
