"""Tests for the sweep's window vocabulary: nested suffix windows against disjoint era slices."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from metaculus_bot.performance_analysis.clip_threshold_sweep import BINARY_FLOOR_GRID, ClipRecord, build_clip_records
from metaculus_bot.performance_analysis.clip_threshold_tables import nesting_rows, regime_span
from metaculus_bot.performance_analysis.clip_threshold_windows import (
    LOOKBACK_DAYS,
    WINDOW_ALL,
    WINDOW_CURRENT_CLAMP,
    WINDOW_ERA_POST_FLIP,
    WINDOW_ERA_PRE_FLIP,
    WINDOW_LAST_90D,
    WINDOW_TRIPLE_ERA,
    build_windows,
    nested_windows,
    window_labels,
)
from metaculus_bot.performance_analysis.eras import B4E9DF0_MERGED_AT, WIDENING_FLIP_MERGED_AT
from tests.clip_threshold_fakes import AFTER_FT_0292, AFTER_WIDENING, AS_OF, BEFORE_WIDENING, binary_record


class TestWindows:
    """Suffix windows, the dated windows and the oversize disclosure."""

    @staticmethod
    def _dated_records(n: int) -> tuple[ClipRecord, ...]:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        return build_clip_records(
            [
                binary_record(
                    question_id=1000 + i,
                    p_yes=0.5,
                    resolution=True,
                    created_at=(base + timedelta(days=i)).isoformat().replace("+00:00", "Z"),
                )
                for i in range(n)
            ],
            "binary",
        ).records

    def _by_label(self, records, **kwargs) -> dict:
        return {w.label: w for w in build_windows(records, question_type="binary", as_of=AS_OF, **kwargs)}

    def test_last_n_takes_the_newest_n(self):
        records = self._dated_records(20)
        windows = {w.label: w for w in build_windows(records, question_type="binary", as_of=AS_OF, last_n=(10, 5))}
        assert [r.question_id for r in windows["last_5"].records] == ["1015", "1016", "1017", "1018", "1019"]
        assert len(windows["last_10"].records) == 10
        # The complement is exactly the records the fit may see.
        assert len(windows["last_5"].complement) == 15

    def test_oversize_window_reports_its_actual_n(self):
        records = self._dated_records(8)
        windows = {w.label: w for w in build_windows(records, question_type="binary", as_of=AS_OF, last_n=(300,))}
        window = windows["last_300"]
        assert window.oversize is True
        assert len(window.records) == 8
        assert window.requested_n == 300
        assert window.complement == ()

    def test_last_90d_honours_as_of(self):
        base = datetime(2026, 9, 1, tzinfo=UTC)
        records = build_clip_records(
            [
                binary_record(
                    question_id=2000 + i,
                    p_yes=0.5,
                    resolution=True,
                    created_at=(base - timedelta(days=30 * i)).isoformat().replace("+00:00", "Z"),
                )
                for i in range(6)
            ],
            "binary",
        ).records
        windows = self._by_label(records)
        # 0, 30, 60 days back are inside a 90-day lookback from 2026-09-02; 90+ are not.
        assert len(windows[WINDOW_LAST_90D].records) == 3
        assert len(windows[WINDOW_LAST_90D].complement) == 3

    def test_clamp_regime_and_triple_era_use_the_shared_constants(self):
        records = build_clip_records(
            [
                binary_record(question_id=3001, p_yes=0.5, resolution=True, created_at=BEFORE_WIDENING),
                binary_record(question_id=3002, p_yes=0.5, resolution=True, created_at=AFTER_WIDENING),
                binary_record(question_id=3003, p_yes=0.5, resolution=True, created_at=AFTER_FT_0292),
            ],
            "binary",
        ).records
        windows = self._by_label(records)
        assert windows[WINDOW_CURRENT_CLAMP].start == WIDENING_FLIP_MERGED_AT
        assert len(windows[WINDOW_CURRENT_CLAMP].records) == 2
        assert windows[WINDOW_TRIPLE_ERA].start == B4E9DF0_MERGED_AT
        assert [r.question_id for r in windows[WINDOW_TRIPLE_ERA].records] == ["3003"]

    def test_undated_records_are_in_all_and_out_of_every_dated_window(self):
        records = build_clip_records(
            [
                binary_record(question_id=4001, p_yes=0.5, resolution=True, created_at=None),
                binary_record(question_id=4002, p_yes=0.5, resolution=True, created_at=AFTER_FT_0292),
            ],
            "binary",
        ).records
        windows = self._by_label(records)
        assert len(windows[WINDOW_ALL].records) == 2
        for label in (WINDOW_LAST_90D, WINDOW_CURRENT_CLAMP, WINDOW_TRIPLE_ERA):
            assert [r.question_id for r in windows[label].records] == ["4002"]

    def test_mc_window_set_adds_last_50(self):
        assert "last_50" in window_labels("multiple_choice")
        assert "last_50" not in window_labels("binary")

    def test_the_lookback_label_is_derived_from_its_own_constant(self):
        """A change to LOOKBACK_DAYS must not leave a window labelled with the old span."""
        assert f"last_{LOOKBACK_DAYS}d" == WINDOW_LAST_90D


class TestEraSlices:
    """The disjoint config-era windows, beside the nested ones."""

    def _records(self):
        return build_clip_records(
            [
                binary_record(question_id=9040, p_yes=0.5, resolution=True, created_at=BEFORE_WIDENING),
                binary_record(question_id=9041, p_yes=0.5, resolution=True, created_at=AFTER_WIDENING),
                binary_record(question_id=9042, p_yes=0.5, resolution=True, created_at=AFTER_FT_0292),
                binary_record(question_id=9043, p_yes=0.5, resolution=True, created_at=None),
            ],
            "binary",
        ).records

    def test_era_slices_partition_the_dated_records_with_triple_era(self):
        windows = {w.label: w for w in build_windows(self._records(), question_type="binary", as_of=AS_OF)}
        pre = [r.question_id for r in windows[WINDOW_ERA_PRE_FLIP].records]
        post = [r.question_id for r in windows[WINDOW_ERA_POST_FLIP].records]
        triple = [r.question_id for r in windows[WINDOW_TRIPLE_ERA].records]
        assert (pre, post, triple) == (["9040"], ["9041"], ["9042"])
        assert windows[WINDOW_ERA_PRE_FLIP].end == WIDENING_FLIP_MERGED_AT
        assert windows[WINDOW_ERA_POST_FLIP].start == WIDENING_FLIP_MERGED_AT
        assert windows[WINDOW_ERA_POST_FLIP].end == B4E9DF0_MERGED_AT
        assert windows[WINDOW_ERA_PRE_FLIP].is_era_slice
        assert windows[WINDOW_ERA_POST_FLIP].is_era_slice
        assert not windows[WINDOW_TRIPLE_ERA].is_era_slice

    def test_era_complements_are_the_older_dated_records(self):
        windows = {w.label: w for w in build_windows(self._records(), question_type="binary", as_of=AS_OF)}
        assert windows[WINDOW_ERA_PRE_FLIP].complement == ()
        assert [r.question_id for r in windows[WINDOW_ERA_POST_FLIP].complement] == ["9040"]

    def test_nested_windows_exclude_the_era_slices_and_labels_include_them(self):
        windows = build_windows(self._records(), question_type="binary", as_of=AS_OF)
        nested = {w.label for w in nested_windows(windows)}
        assert WINDOW_ERA_PRE_FLIP not in nested
        assert WINDOW_ERA_POST_FLIP not in nested
        assert WINDOW_ALL in nested
        assert WINDOW_TRIPLE_ERA in nested
        assert {WINDOW_ERA_PRE_FLIP, WINDOW_ERA_POST_FLIP} <= set(window_labels("binary"))


class TestNestingAndRegimeSpan:
    def test_distinct_count_equals_the_all_count(self):
        records = build_clip_records(
            [
                binary_record(question_id=9100, p_yes=0.02, resolution=False, created_at=BEFORE_WIDENING),
                binary_record(question_id=9101, p_yes=0.03, resolution=False, created_at=AFTER_WIDENING),
                binary_record(question_id=9102, p_yes=0.04, resolution=False, created_at=AFTER_FT_0292),
            ],
            "binary",
        ).records
        windows = build_windows(records, question_type="binary", as_of=AS_OF)
        rows = {row.c: row for row in nesting_rows(windows, grid=BINARY_FLOOR_GRID)}
        assert rows[0.05].n_distinct == 3 == rows[0.05].n_affected_by_window[WINDOW_ALL]
        assert rows[0.05].n_affected_by_window[WINDOW_TRIPLE_ERA] == 1
        assert WINDOW_ERA_PRE_FLIP not in rows[0.05].n_affected_by_window

    def test_regime_span_reports_whether_the_live_clamp_bound_anything(self):
        records = build_clip_records(
            [
                binary_record(question_id=9110, p_yes=0.30, resolution=True, created_at=AFTER_WIDENING),
                binary_record(question_id=9111, p_yes=0.90, resolution=True, created_at=AFTER_WIDENING),
            ],
            "binary",
        ).records
        [current] = [
            w for w in build_windows(records, question_type="binary", as_of=AS_OF) if w.label == WINDOW_CURRENT_CLAMP
        ]
        span = regime_span(current, question_type="binary")
        assert (span.floor, span.ceiling) == (0.02, 0.98)
        assert (span.min_value, span.max_value) == (0.30, 0.90)
        assert (span.n_at_or_below_floor, span.n_at_or_above_ceiling) == (0, 0)
        bound = build_clip_records([binary_record(question_id=9112, p_yes=0.02, resolution=False)], "binary").records
        [current_bound] = [
            w for w in build_windows(bound, question_type="binary", as_of=AS_OF) if w.label == WINDOW_CURRENT_CLAMP
        ]
        assert regime_span(current_bound, question_type="binary").n_at_or_below_floor == 1
