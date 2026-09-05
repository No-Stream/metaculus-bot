"""Reading a Wayback Machine snapshot URL: the route in, the timestamp out, the disclosure.

Pure parsing and rendering, no network. The rung that uses these lives in
``tests/resolution_source/test_resolution_source_wayback_rung.py``; what is pinned here is the
route shape the live probe confirmed on 2026-09-03 and the two-sided freshness rule.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from metaculus_bot.research.wayback import (
    WaybackSnapshot,
    innermost_url,
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
        assert wayback_snapshot_url("https://www.bls.gov/wsp/", now=_NOW) == (
            "https://web.archive.org/web/2026id_/https://www.bls.gov/wsp/"
        )

    def test_a_query_string_survives(self):
        assert wayback_snapshot_url("https://x.gov/data?series=CPI&y=2026", now=_NOW).endswith(
            "https://x.gov/data?series=CPI&y=2026"
        )

    def test_the_request_stamp_follows_the_clock(self):
        """The year is read off the caller's clock and cannot be a literal.

        The archive pads a bare year UP to that year's end, so a year written into the source
        asks every run after it for a capture from a year that has already finished — and the
        freshness bound then withholds every one of those captures, retiring the rung silently
        while it still spends two archive round trips per question.
        """
        this_year = wayback_snapshot_url("https://x.gov/p", now=_NOW)
        next_year = wayback_snapshot_url("https://x.gov/p", now=_NOW.replace(year=_NOW.year + 1))

        assert this_year != next_year
        assert this_year == "https://web.archive.org/web/2026id_/https://x.gov/p"
        assert next_year == "https://web.archive.org/web/2027id_/https://x.gov/p"

    def test_the_request_url_is_not_itself_a_datable_capture(self):
        """Year granularity rather than a full 14-digit stamp, and that is load-bearing.

        The rung tells "the archive never landed on a capture" apart from "it served one we
        cannot use" by whether the FINAL url parses as a capture. A 14-digit request url would
        parse as one dated at the request instant, so an archive 404 — or an unredirected 200 —
        would read as a capture taken seconds ago and be served as brand new.
        """
        assert parse_snapshot_url(wayback_snapshot_url("https://x.gov/p", now=_NOW)) is None


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


class TestInnermostUrl:
    """Unwrapping is repeated until nothing matches, because a capture OF a capture presents
    `web.archive.org` as its inner host and would clear a hostname check at one level."""

    def test_a_plain_url_is_its_own_innermost(self):
        assert innermost_url("https://www.bls.gov/wsp/") == "https://www.bls.gov/wsp/"

    def test_one_level_unwraps_to_the_captured_page(self):
        assert innermost_url("https://web.archive.org/web/20260828221347id_/https://www.bls.gov/wsp/") == (
            "https://www.bls.gov/wsp/"
        )

    def test_a_nested_capture_unwraps_all_the_way(self):
        nested = (
            "https://web.archive.org/web/20260901000000id_/"
            "https://web.archive.org/web/20240101000000/https://www.metaculus.com/questions/45001/"
        )
        assert innermost_url(nested) == "https://www.metaculus.com/questions/45001/"

    def test_an_undated_archive_request_is_not_unwrapped(self):
        """Only a dated capture URL is a snapshot; our own four-digit request shape is not."""
        request = "https://web.archive.org/web/2026id_/https://x.gov/p"
        assert innermost_url(request) == request


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
