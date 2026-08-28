"""Tests for the weekly close-margin watch view (scripts/close_margin_watch.py).

Exercises the pure aggregation functions over synthetic CLOSE_MARGIN records shaped
exactly like the telemetry-archive parser emits (float margin_frac, ISO submitted_at,
int qid/margin_s). ISO-week labels are ground truth from datetime.isocalendar.
"""

import pytest

from scripts.close_margin_watch import percentile, summarize_weeks, week_key


def _rec(submitted_at, margin_frac, *, qid=1, margin_s=1000):
    return {
        "marker": "close_margin",
        "submitted_at": submitted_at,
        "margin_frac": margin_frac,
        "qid": qid,
        "margin_s": margin_s,
    }


class TestWeekKey:
    def test_iso_week_labels(self):
        assert week_key("2026-07-06T00:00:00+00:00") == "2026-W28"
        assert week_key("2026-07-13T00:00:00+00:00") == "2026-W29"
        assert week_key("2026-07-19T12:00:00+00:00") == "2026-W29"
        assert week_key("2026-07-20T00:00:00+00:00") == "2026-W30"

    def test_bad_inputs_return_none(self):
        assert week_key("n/a") is None
        assert week_key(None) is None
        assert week_key(12345) is None


class TestPercentile:
    def test_linear_interpolation(self):
        values = [float(i) for i in range(10)]  # 0.0 .. 9.0
        assert percentile(values, 0.50) == pytest.approx(4.5)
        assert percentile(values, 0.10) == pytest.approx(0.9)

    def test_unsorted_input_is_sorted(self):
        assert percentile([0.40, 0.05, 0.10], 0.50) == pytest.approx(0.10)
        assert percentile([0.40, 0.05, 0.10], 0.10) == pytest.approx(0.06)

    def test_single_value(self):
        assert percentile([0.5], 0.10) == 0.5

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="percentile of an empty sequence"):
            percentile([], 0.5)


class TestSummarizeWeeks:
    def _records(self):
        return [
            _rec("2026-07-06T00:00:00+00:00", 0.50, qid=1),  # W28
            _rec("2026-07-13T00:00:00+00:00", 0.40, qid=2),  # W29
            _rec("2026-07-19T00:00:00+00:00", 0.10, qid=3, margin_s=3600),  # W29, below red
            _rec("2026-07-19T12:00:00+00:00", 0.05, qid=4, margin_s=1800),  # W29, below red
            _rec("2026-07-19T00:00:00+00:00", None, qid=5),  # skipped: no window fraction
            _rec("n/a", 0.20, qid=6),  # skipped: unparseable submit time
        ]

    def test_weekly_buckets_and_stats(self):
        summaries, _below_red, _skipped = summarize_weeks(self._records(), red_line=0.30)
        assert [s.week for s in summaries] == ["2026-W28", "2026-W29"]

        w28, w29 = summaries
        assert w28.n == 1
        assert w28.p50 == pytest.approx(0.50)
        assert w28.n_below_red == 0

        assert w29.n == 3
        assert w29.p50 == pytest.approx(0.10)
        assert w29.p10 == pytest.approx(0.06)
        assert w29.minimum == pytest.approx(0.05)
        assert w29.n_below_red == 2

    def test_below_red_and_skipped(self):
        _, below_red, skipped = summarize_weeks(self._records(), red_line=0.30)
        assert sorted(rec["qid"] for rec in below_red) == [3, 4]
        assert skipped == 2

    def test_red_line_is_configurable(self):
        # A looser red line (0.45) pulls the 0.40 W29 question in too.
        _, below_red, _ = summarize_weeks(self._records(), red_line=0.45)
        assert sorted(rec["qid"] for rec in below_red) == [2, 3, 4]
