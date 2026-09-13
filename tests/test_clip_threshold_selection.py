"""Tests for the selection-aware readings of "pick the best floor, then apply it".

The argmax is usually a plateau, a censored candidate may never win it, the out-of-sample carry and
the out-of-bag mean price the selection itself, and the bootstrap under all of them is seeded.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from metaculus_bot import bootstrap
from metaculus_bot.performance_analysis.clip_threshold_report import render_report
from metaculus_bot.performance_analysis.clip_threshold_selection import censored_rows_at_argmax_score, oob_argmax
from metaculus_bot.performance_analysis.clip_threshold_sweep import (
    BINARY_FLOOR_GRID,
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    ClipRecord,
    argmax_row,
    argmax_rows,
    bootstrap_mean_ci,
    build_clip_records,
    sweep_row,
)
from metaculus_bot.performance_analysis.clip_threshold_tables import MIN_OOS_COMPLEMENT_N, compute_report, oos_row
from metaculus_bot.performance_analysis.clip_threshold_windows import WINDOW_ALL, build_windows
from tests.clip_threshold_fakes import AS_OF, binary_record, one_binary


class TestBootstrap:
    """The CI is deterministic under the fixed seed and brackets the mean."""

    def test_deterministic_and_brackets_the_mean(self):
        """Re-seeding per call is what makes a row reproducible; the index cache is only a speed measure,
        so it is cleared between the two calls or the second would trivially return the first's array."""
        deltas = [-3.0, 0.0, 1.5, 8.0, -12.0, 4.0, 0.5, 2.0, -1.0, 6.0]
        first = bootstrap_mean_ci(deltas)
        bootstrap._INDEX_CACHE.clear()
        second = bootstrap_mean_ci(deltas)
        assert first == second
        lo, hi = first
        assert lo is not None
        assert hi is not None
        assert lo < sum(deltas) / len(deltas) < hi

    def test_index_cache_is_opt_in_and_the_sweep_opts_in(self):
        """The ablation harness derives a distinct seed per scoring group, so a cache that filled on
        every call would grow for the process lifetime with zero reuse; only the sweep, which repeats
        ``(n, B, seed)`` across cells, asks for it. Cached and uncached draws are the same matrix."""
        bootstrap._INDEX_CACHE.clear()
        uncached = bootstrap.bootstrap_indices(7, n_bootstrap=5, seed=3)
        assert bootstrap._INDEX_CACHE == {}
        cached = bootstrap.bootstrap_indices(7, n_bootstrap=5, seed=3, cache=True)
        assert list(bootstrap._INDEX_CACHE) == [(7, 5, 3)]
        assert (uncached == cached).all()
        bootstrap._INDEX_CACHE.clear()
        bootstrap_mean_ci([-3.0, 0.0, 1.5, 8.0])
        assert list(bootstrap._INDEX_CACHE) == [(4, BOOTSTRAP_B, BOOTSTRAP_SEED)]

    def test_degenerate_sample_gives_a_point_interval(self):
        lo, hi = bootstrap_mean_ci([0.0, 0.0, 0.0])
        assert (lo, hi) == (0.0, 0.0)

    def test_empty_sample_has_no_interval(self):
        assert bootstrap_mean_ci([]) == (None, None)


def _floor_rows(window, *, question_type: str = "binary", grid=BINARY_FLOOR_GRID):
    """The window's own floor-only sweep, one row per candidate, as ``_type_report`` hands it to ``oos_row``."""
    return [
        sweep_row(window.records, question_type=question_type, side="floor_only", window=window.label, c=c)
        for c in grid
    ]


class TestOutOfSample:
    """A thin complement cannot fit a clip level, and says so."""

    @staticmethod
    def _records(n: int, *, start: datetime) -> tuple[ClipRecord, ...]:
        return build_clip_records(
            [
                binary_record(
                    question_id=5000 + i,
                    p_yes=0.02,
                    resolution=(i % 4 == 0),
                    created_at=(start + timedelta(days=i)).isoformat().replace("+00:00", "Z"),
                )
                for i in range(n)
            ],
            "binary",
        ).records

    def test_thin_complement_reports_no_fit(self):
        records = self._records(12, start=datetime(2026, 6, 1, tzinfo=UTC))
        [window] = [
            w for w in build_windows(records, question_type="binary", as_of=AS_OF, last_n=(5,)) if w.label == "last_5"
        ]
        assert len(window.complement) < MIN_OOS_COMPLEMENT_N
        row = oos_row(window, question_type="binary", in_window=_floor_rows(window))
        assert row.underpowered is True
        assert row.c_star is None
        assert row.carried_sum_delta is None
        # The in-window argmax is still reported: it is a description, not a fit.
        assert row.in_window_c_star is not None

    def test_fit_and_carry_are_both_reported_when_the_complement_is_thick(self):
        records = self._records(60, start=datetime(2026, 6, 1, tzinfo=UTC))
        [window] = [
            w for w in build_windows(records, question_type="binary", as_of=AS_OF, last_n=(20,)) if w.label == "last_20"
        ]
        row = oos_row(window, question_type="binary", in_window=_floor_rows(window))
        assert row.underpowered is False
        assert row.c_star in BINARY_FLOOR_GRID
        assert row.carried_sum_delta is not None
        assert row.in_window_c_star in BINARY_FLOOR_GRID


class TestArgmaxPlateau:
    """The winning candidate is usually a TIE, and the report has to say so.

    Every candidate at or below a window's in-force floor scores exactly 0 when no publish in
    that window was clamped, so reporting one representative as "the" argmax would read as a
    preference for loosening when the evidence is indifference.
    """

    def test_every_candidate_at_or_below_the_floor_ties_at_zero(self):
        """Published 0.30 post-flip: nothing at or below 0.02 can move it and nothing at 0.025 or
        above can either, so the tie runs from the smallest candidate up to 0.0200.
        """
        records = [one_binary(question_id=7100, p_yes=0.30, resolution=True)]
        rows = [
            sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=c)
            for c in BINARY_FLOOR_GRID
        ]
        tied = argmax_rows(rows)
        assert [row.c for row in tied] == [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10]
        assert argmax_row(rows) is tied[0]

    def test_censored_rows_never_win(self):
        """At 0.01 this record is censored, published AT the post-flip 0.02 floor, so that row is
        excluded even though its ``sum_delta`` of 0 beats every tightening candidate.
        """
        records = [one_binary(question_id=7101, p_yes=0.02, resolution=False)]
        rows = [
            sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=c)
            for c in BINARY_FLOOR_GRID
        ]
        tied = argmax_rows(rows)
        assert all(row.censored_n == 0 for row in tied)
        assert 0.005 not in {row.c for row in tied}
        best = argmax_row(rows)
        assert best is not None
        assert best.c == 0.02

    def test_no_exact_row_means_no_argmax(self):
        record = one_binary(question_id=7102, p_yes=0.02, resolution=False)
        censored_only = [sweep_row([record], question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.005)]
        assert argmax_rows(censored_only) == []
        assert argmax_row(censored_only) is None

    def test_member_censored_only_rows_never_win_either(self):
        """Members 0.02 / 0.03 publish 0.025 post-flip: the published-value rule sees nothing at
        the floor, but the 0.02 member is one of the two middle values, so every loosening
        candidate's ``sum_delta`` of 0 is unobservable. Such a row must lose the argmax on
        censoring and be named in the censored ties, not compete as if it were neutral."""
        records = [one_binary(question_id=7103, p_yes=0.025, resolution=False, per_model={"a": "2.0%", "b": "3.0%"})]
        rows = [
            sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=c)
            for c in BINARY_FLOOR_GRID
        ]
        loosening = {row.c for row in rows if row.n_loosening}
        assert loosening == {0.005, 0.01, 0.015}
        for row in rows:
            if row.c in loosening:
                assert (row.censored_n, row.member_censored_n, row.exact) == (0, 1, False)
        tied = argmax_rows(rows)
        assert {row.c for row in tied} == {0.02, 0.025}
        assert not ({row.c for row in tied} & loosening)
        assert {row.c for row in censored_rows_at_argmax_score(rows)} == loosening

    def test_carry_gap_is_zero_when_the_labels_disagree_but_the_scores_do_not(self):
        records = TestOutOfSample._records(60, start=datetime(2026, 6, 1, tzinfo=UTC))
        [window] = [
            w for w in build_windows(records, question_type="binary", as_of=AS_OF, last_n=(20,)) if w.label == "last_20"
        ]
        row = oos_row(window, question_type="binary", in_window=_floor_rows(window))
        assert row.carry_gap is not None
        assert row.carry_gap == pytest.approx(
            (row.in_window_sum_delta or 0.0) - (row.carried_sum_delta or 0.0), abs=1e-12
        )
        assert row.n_tied_in_window >= 1

    def test_carry_gap_is_none_when_no_fit_was_made(self):
        records = TestOutOfSample._records(12, start=datetime(2026, 6, 1, tzinfo=UTC))
        [window] = [
            w for w in build_windows(records, question_type="binary", as_of=AS_OF, last_n=(5,)) if w.label == "last_5"
        ]
        row = oos_row(window, question_type="binary", in_window=_floor_rows(window))
        assert row.carry_gap is None


class TestOutOfBagArgmax:
    """The selection-corrected value of "pick the best floor, then apply it"."""

    @staticmethod
    def _rows(records):
        return [
            sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=c)
            for c in BINARY_FLOOR_GRID
        ]

    def test_identity_window_has_zero_oob_value(self):
        records = [one_binary(question_id=9050 + i, p_yes=0.30 + 0.01 * i, resolution=True) for i in range(6)]
        result = oob_argmax(records, self._rows(records), side="floor_only", window=WINDOW_ALL)
        assert result.oob_mean_delta == pytest.approx(0.0, abs=1e-12)
        assert result.shrinkage == pytest.approx(0.0, abs=1e-12)
        assert result.n_iterations > 0

    def test_deterministic_and_bracketed(self):
        records = [one_binary(question_id=9060 + i, p_yes=0.02, resolution=(i % 3 == 0)) for i in range(12)]
        rows = self._rows(records)
        first = oob_argmax(records, rows, side="floor_only", window=WINDOW_ALL)
        second = oob_argmax(records, rows, side="floor_only", window=WINDOW_ALL)
        assert first == second
        assert first.in_window_c is not None
        assert first.oob_ci_lo is not None
        assert first.oob_ci_hi is not None
        assert first.oob_mean_delta is not None
        assert first.oob_ci_lo <= first.oob_mean_delta <= first.oob_ci_hi
        # Only exact rows compete: the censored 0.005 candidate is never a fit.
        assert first.n_candidates == sum(1 for row in rows if row.exact)

    def test_selection_shrinks_a_lucky_small_window(self):
        """Six YES at 0.02 and five NO at 0.02: in-window the 0.10 floor looks like a gain, but a
        resample that draws mostly NOs fits a lower floor and scores it on the YES-heavy remainder
        (and vice versa), so the out-of-bag mean sits below the in-window mean."""
        records = [one_binary(question_id=9070 + i, p_yes=0.02, resolution=(i < 6)) for i in range(11)]
        result = oob_argmax(records, self._rows(records), side="floor_only", window=WINDOW_ALL)
        assert result.in_window_mean_delta is not None
        assert result.in_window_mean_delta > 0.0
        assert result.shrinkage is not None
        assert result.shrinkage > 0.0


class TestCensoredTies:
    def test_censored_candidate_at_the_argmax_score_is_named(self):
        records = [one_binary(question_id=9130, p_yes=0.02, resolution=False)]
        rows = [
            sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=c)
            for c in BINARY_FLOOR_GRID
        ]
        censored = censored_rows_at_argmax_score(rows)
        assert [row.c for row in censored] == [0.005, 0.01, 0.015]
        report = compute_report(
            [binary_record(question_id=9131, p_yes=0.02, resolution=False)],
            dataset_path="synthetic",
            as_of=AS_OF,
            exclude_qids=frozenset(),
        )
        assert "censored at the same score" in render_report(report)
