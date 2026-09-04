"""Reading a Wayback Machine snapshot URL: the route in, the timestamp out, the disclosure.

Pure parsing and rendering, no network. The rung that uses these lives in
``tests/resolution_source/test_resolution_source_escalation.py``; what is pinned here is the
route shape the live probe confirmed on 2026-09-03 and the two-sided freshness rule.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from metaculus_bot.research.wayback import (
    WaybackSnapshot,
    parse_snapshot_url,
    snapshot_age_days,
    wayback_lead,
    wayback_snapshot_url,
)

_NOW = datetime(2026, 9, 4, 1, 8, 28, tzinfo=UTC)


class TestSnapshotRequestUrl:
    def test_the_original_url_is_carried_verbatim(self):
        """The archive's path format carries the original URL as-is; percent-encoding it makes
        the archive treat it as a different resource."""
        assert wayback_snapshot_url("https://www.bls.gov/wsp/") == (
            "https://web.archive.org/web/2026id_/https://www.bls.gov/wsp/"
        )

    def test_a_query_string_survives(self):
        assert wayback_snapshot_url("https://x.gov/data?series=CPI&y=2026").endswith(
            "https://x.gov/data?series=CPI&y=2026"
        )


class TestParseSnapshotUrl:
    def test_the_shape_the_live_probe_returned(self):
        """Live-verified 2026-09-03: `web/2026id_/https://www.bls.gov/wsp/` answered 302 with
        this Location and `x-archive-redirect-reason: found capture at 20260828221347`."""
        parsed = parse_snapshot_url("https://web.archive.org/web/20260828221347id_/https://www.bls.gov/wsp/")

        assert parsed is not None
        assert parsed.captured_at == datetime(2026, 8, 28, 22, 13, 47, tzinfo=UTC)
        assert parsed.inner_url == "https://www.bls.gov/wsp/"

    @pytest.mark.parametrize("modifier", ["", "if_", "im_"])
    def test_other_capture_modifiers_parse(self, modifier):
        parsed = parse_snapshot_url(f"https://web.archive.org/web/20260828221347{modifier}/https://x.gov/p")
        assert parsed is not None
        assert parsed.inner_url == "https://x.gov/p"

    def test_an_unredirected_four_digit_request_is_undatable(self):
        """The archive answering our own request URL means it never landed on a capture, and a
        copy with no usable date cannot carry the age disclosure that makes it admissible."""
        assert parse_snapshot_url("https://web.archive.org/web/2026id_/https://x.gov/p") is None

    def test_a_non_archive_url_is_not_a_snapshot(self):
        assert parse_snapshot_url("https://example.test/web/20260828221347id_/https://x.gov/p") is None

    def test_a_snapshot_with_no_inner_url_is_rejected(self):
        assert parse_snapshot_url("https://web.archive.org/web/20260828221347id_/") is None

    def test_an_impossible_timestamp_is_rejected(self):
        assert parse_snapshot_url("https://web.archive.org/web/20261399999999id_/https://x.gov/p") is None


class TestSnapshotAge:
    def test_a_recent_capture_ages_in_days(self):
        snapshot = WaybackSnapshot(captured_at=_NOW - timedelta(days=7, hours=3), inner_url="https://x.gov/p")
        age = snapshot_age_days(snapshot, _NOW)
        assert age is not None
        assert 7.1 < age < 7.2

    def test_a_capture_inside_clock_skew_reads_as_brand_new(self):
        snapshot = WaybackSnapshot(captured_at=_NOW + timedelta(hours=2), inner_url="https://x.gov/p")
        assert snapshot_age_days(snapshot, _NOW) == 0.0

    def test_a_capture_far_in_the_future_is_unusable(self):
        """A broken clock or a misparse on one side, not the freshest possible copy — the same
        two-sided rule the Datawrapper freshness guard applies, because the lead asserts a date."""
        snapshot = WaybackSnapshot(captured_at=_NOW + timedelta(days=400), inner_url="https://x.gov/p")
        assert snapshot_age_days(snapshot, _NOW) is None


class TestWaybackLead:
    def test_every_clause_the_disclosure_owes(self):
        snapshot = WaybackSnapshot(
            captured_at=datetime(2026, 8, 28, 22, 13, 47, tzinfo=UTC), inner_url="https://www.bls.gov/wsp/"
        )
        lead = wayback_lead(snapshot, 6.1, "blocked")

        assert lead == (
            "[Archived copy from the Wayback Machine, captured 2026-08-28, 6 days before this "
            "forecast; the live page could not be fetched (blocked).]"
        )
