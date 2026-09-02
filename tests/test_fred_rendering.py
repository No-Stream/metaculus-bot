"""Tests for the FRED fetch and render module: values, the block, and the vintage table.

Note the patch targets: every ``Fred`` and ``fetch_series`` patch aims at
``metaculus_bot.research.fred_rendering``, the module that constructs the client and reads
``Fred.earliest_realtime_start`` / ``latest_realtime_end`` off the class. fredapi's real class
carries those same literals, so a patch aimed at a module that does not build the client would
leave these tests green while they silently stopped proving anything.
"""

import re
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.error import URLError
from xml.etree.ElementTree import ParseError

import numpy as np
import pandas as pd
import pytest

from metaculus_bot.research.financial_data import financial_data_provider
from metaculus_bot.research.fred_rendering import (
    UnknownFredSeries,
    _fetch_fred_data,
    _fetch_fred_data_ceiling,
    _format_fred_change,
    _format_fred_value,
    _render_fred_series,
    is_unknown_fred_series_error,
)
from scripts.telemetry.markers import MARKER_SPECS
from tests.financial_fakes import _BENCH_OPEN_TIME, _make_q, _monthly_fred


class TestFetchFredData:
    """Tests for _fetch_fred_data."""

    def test_valid_series_returns_markdown_with_key_fields(self) -> None:
        dates = pd.date_range(end="2026-03-01", periods=60, freq="MS")
        values = np.linspace(3.5, 4.2, 60)
        mock_series = pd.Series(values, index=dates, name="UNRATE")

        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.return_value = mock_series
        mock_fred_instance.get_series_info.return_value = pd.DataFrame(
            {"title": ["Unemployment Rate"]}, index=["UNRATE"]
        )

        with patch("metaculus_bot.research.fred_rendering.Fred") as mock_fred_class:
            mock_fred_class.return_value = mock_fred_instance

            result = _fetch_fred_data("UNRATE", "fake_api_key")

        assert result != ""
        assert "UNRATE" in result
        # Should contain the latest value
        assert "4.2" in result or "4.20" in result

    def test_fred_exception_returns_empty_string(self) -> None:
        with patch("metaculus_bot.research.fred_rendering.Fred") as mock_fred_class:
            mock_fred_class.return_value.get_series.side_effect = Exception("API error")

            result = _fetch_fred_data("INVALID", "fake_api_key")

        assert result == ""


class TestUnknownFredSeries:
    """A series id FRED says does not exist is its own outcome, not a soft-failed empty.

    q45363 (the Boliviano-USD rate) lost its entire financial block to the hallucinated id
    ``DEXBOUS``, and the only trace was the diagnostics token ``DEXBOUS:empty`` — byte-identical
    to a live series that happens to hold no observations in the window. fredapi surfaces FRED's
    own 400 body as ``ValueError(message)``, so that phrase is the evidence, and everything else
    keeps soft-failing rather than being attributed to a bad id on a guess.
    """

    # FRED's own 400 body, read off a live request for the hallucinated id q45363 published on.
    FRED_400_MESSAGE: ClassVar[str] = "The series does not exist."

    def test_the_fred_400_message_classifies_as_an_unknown_series(self) -> None:
        assert is_unknown_fred_series_error(ValueError(self.FRED_400_MESSAGE))

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("Bad Request. The value for variable api_key is not registered."),
            URLError("connection reset"),
            ParseError("not xml"),
            Exception("The series does not exist."),  # right words, wrong type — fredapi raises ValueError
        ],
    )
    def test_every_other_failure_stays_a_generic_soft_fail(self, exc: Exception) -> None:
        assert not is_unknown_fred_series_error(exc)

    @pytest.mark.parametrize(
        ("is_resolving_source", "expected_proposer"),
        [(False, "classifier"), (True, "resolution_url")],
    )
    def test_the_fetch_raises_and_warns_naming_the_id_and_its_proposer(
        self,
        caplog: pytest.LogCaptureFixture,
        is_resolving_source: bool,
        expected_proposer: str,
    ) -> None:
        with (
            patch("metaculus_bot.research.fred_rendering.Fred") as mock_fred_class,
            caplog.at_level("WARNING"),
        ):
            mock_fred_class.return_value.get_series.side_effect = ValueError(self.FRED_400_MESSAGE)

            with pytest.raises(UnknownFredSeries) as excinfo:
                _fetch_fred_data("DEXBOUS", "fake_api_key", is_resolving_source=is_resolving_source)

        assert excinfo.value.series_id == "DEXBOUS"
        assert f"FRED_UNKNOWN_SERIES: series_id=DEXBOUS proposed_by={expected_proposer}" in caplog.text

    def test_the_warning_parses_under_its_marker_spec(self, caplog: pytest.LogCaptureFixture) -> None:
        """The archive harvests this line, so its shape is pinned to the registry regex."""
        with (
            patch("metaculus_bot.research.fred_rendering.Fred") as mock_fred_class,
            caplog.at_level("WARNING"),
        ):
            mock_fred_class.return_value.get_series.side_effect = ValueError(self.FRED_400_MESSAGE)
            with pytest.raises(UnknownFredSeries):
                _fetch_fred_data("DEXBOUS", "fake_api_key")

        spec = next(s for s in MARKER_SPECS if s.name == "fred_unknown_series")
        match = spec.regex.search(caplog.text)
        assert match is not None
        assert match.group("series_id") == "DEXBOUS"
        assert match.group("proposed_by") == "classifier"


class TestRenderFredSeriesYoY:
    """The year-over-year line must use a DATE-based ~365-day lookup, not a fixed
    13-observation offset (F8). On a daily series 13 observations is ~2.5 weeks, so
    the old offset mislabeled a two-and-a-half-week move as year-over-year."""

    @staticmethod
    def _yoy_change_from(markdown: str) -> float:
        """Pull the signed YoY change value out of the rendered markdown line."""
        for line in markdown.splitlines():
            if line.startswith("- Year-over-year change:"):
                # "- Year-over-year change: +12.3 (+4.56%)" -> "+12.3"
                return float(line.split(":", 1)[1].strip().split(" ")[0])
        raise AssertionError(f"no year-over-year line in:\n{markdown}")

    def test_daily_series_uses_365d_ago_value_not_obs_minus_13(self) -> None:
        # 800 business days ending 2026-03-02. The value is a linear ramp from 0.0
        # to 799.0 (one unit per observation), so the value at any date equals its
        # integer offset from the start — making the two lookups trivially distinct.
        dates = pd.bdate_range(end="2026-03-02", periods=800)
        data = pd.Series(np.arange(800.0), index=dates, name="DGS10")

        latest_value = float(data.iloc[-1])  # 799.0
        # obs[-13] (the OLD, wrong behavior) is ~2.5 weeks back, not a year.
        wrong_offset_value = float(data.iloc[-13])  # 787.0
        # The date-based lookup: last observation at or before ~365 days ago.
        year_ago = data.index[-1] - pd.Timedelta(days=365)
        correct_value = float(data.loc[:year_ago].iloc[-1])

        markdown = _render_fred_series("DGS10", data, "10Y Treasury rate")
        rendered_yoy = self._yoy_change_from(markdown)

        assert rendered_yoy == pytest.approx(latest_value - correct_value, abs=1e-6)
        # And is materially different from the old fixed-offset result.
        assert rendered_yoy != pytest.approx(latest_value - wrong_offset_value, abs=1e-6)

    def test_monthly_series_still_correct(self) -> None:
        # 60 monthly observations; the value ~12 months back is one year ago.
        dates = pd.date_range(end="2026-03-01", periods=60, freq="MS")
        data = pd.Series(np.arange(60.0), index=dates, name="UNRATE")

        latest_value = float(data.iloc[-1])
        year_ago = data.index[-1] - pd.Timedelta(days=365)
        expected_prior = float(data.loc[:year_ago].iloc[-1])

        markdown = _render_fred_series("UNRATE", data, "unemployment rate")
        rendered_yoy = self._yoy_change_from(markdown)

        assert rendered_yoy == pytest.approx(latest_value - expected_prior, abs=1e-6)

    def test_short_series_omits_yoy_line(self) -> None:
        # Only ~3 months of monthly data: nothing is ~365 days back, so the YoY
        # line is omitted rather than reaching for a nonexistent observation.
        dates = pd.date_range(end="2026-03-01", periods=3, freq="MS")
        data = pd.Series(np.arange(3.0), index=dates, name="UNRATE")

        markdown = _render_fred_series("UNRATE", data, "unemployment rate")

        assert "Year-over-year change" not in markdown


class TestRenderFredSeriesZeroBasePercent:
    """A base of exactly 0 has no percent change; it must not render as 0.00%.

    FRED spread series cross zero routinely (T10Y2Y inverted through 2023-24), and the
    old ``else 0`` put a fabricated "unchanged" percentage next to a genuine absolute
    move in a forecaster prompt.
    """

    def test_zero_previous_observation_omits_the_percent_clause(self) -> None:
        dates = pd.date_range(end="2026-03-01", periods=3, freq="MS")
        data = pd.Series([0.5, 0.0, 0.31], index=dates, name="T10Y2Y")

        markdown = _render_fred_series("T10Y2Y", data, "10Y-2Y spread")
        change_line = next(line for line in markdown.splitlines() if line.startswith("- Change from previous:"))

        assert change_line == "- Change from previous: +0.31"
        assert "0.00%" not in markdown

    def test_zero_year_ago_observation_omits_only_the_yoy_percent(self) -> None:
        # 25 monthly observations so the ~365-day lookup lands on a real row, which is
        # set to exactly 0. The month-over-month clause is unaffected.
        dates = pd.date_range(end="2026-03-01", periods=25, freq="MS")
        values = [1.0] * 25
        values[12] = 0.0  # the observation ~365 days before the last one
        data = pd.Series(values, index=dates, name="T10Y3M")
        data.iloc[-1] = 0.4
        data.iloc[-2] = 0.2

        markdown = _render_fred_series("T10Y3M", data, "10Y-3M spread")
        yoy_line = next(line for line in markdown.splitlines() if line.startswith("- Year-over-year change:"))
        mom_line = next(line for line in markdown.splitlines() if line.startswith("- Change from previous:"))

        assert yoy_line == "- Year-over-year change: +0.4"
        assert "(+100.00%)" in mom_line

    def test_a_nonzero_base_still_renders_its_percent(self) -> None:
        dates = pd.date_range(end="2026-03-01", periods=2, freq="MS")
        data = pd.Series([2.0, 3.0], index=dates, name="UNRATE")

        markdown = _render_fred_series("UNRATE", data, "unemployment rate")

        assert "- Change from previous: +1 (+50.00%)" in markdown


class TestFredValuePrecision:
    """FRED levels must render at the precision the agency published them at.

    q44944 resolved on a Case-Shiller print of 331.893 inside a displayed range four index
    points wide with 0.02-point buckets, and the provider — the one component designed to
    read the resolving series directly — rendered it through `:.4g` as "331.9". The exact
    value reached the forecasters only because two gap-fill passes independently quoted the
    FRED page.
    """

    def test_a_case_shiller_scale_level_keeps_all_its_digits(self) -> None:
        data = _monthly_fred([330.873, 331.359, 331.020, 331.893])

        markdown = _render_fred_series("CSUSHPISA", data, "Case-Shiller home price index")

        assert "- Latest value: 331.893 (2026-06-01)" in markdown
        assert "- Previous value: 331.02" in markdown
        assert "  - 2026-05-01: 331.02" in markdown
        assert "331.9\n" not in markdown, "the `:.4g` rounding must be gone from every line"

    def test_the_change_line_keeps_its_precision_and_loses_float_noise(self) -> None:
        data = _monthly_fred([331.020, 331.893])

        markdown = _render_fred_series("CSUSHPISA", data, "Case-Shiller home price index")

        # 331.893 - 331.02 is 0.8729999999999905 in binary floating point.
        assert "- Change from previous: +0.873 (+0.26%)" in markdown

    def test_a_large_level_never_renders_in_scientific_notation(self) -> None:
        """WALCL (the Fed balance sheet, in millions) went out as "6.7e+06" under `:.4g`."""
        data = _monthly_fred([6_698_123.0, 6_699_580.0], name="WALCL")

        markdown = _render_fred_series("WALCL", data, "Fed balance sheet")

        assert "- Latest value: 6699580 (2026-06-01)" in markdown
        assert "e+0" not in markdown

    def test_a_rate_still_renders_without_trailing_zeros(self) -> None:
        data = _monthly_fred([4.15, 4.2], name="DGS10")

        markdown = _render_fred_series("DGS10", data, "10Y Treasury rate")

        assert "- Latest value: 4.2 (2026-06-01)" in markdown
        assert "- Change from previous: +0.05" in markdown

    def test_the_formatters_own_contract(self) -> None:
        """Unit-level, because the interesting inputs (float noise, a magnitude that rounds
        away, a true zero) do not arise from real FRED decimals. A negative sign on a
        magnitude that rounds to zero would not be information."""
        assert _format_fred_value(331.893) == "331.893"
        assert _format_fred_value(6_699_580.0) == "6699580"
        assert _format_fred_value(4.2) == "4.2"
        assert _format_fred_value(0.0) == "0"
        assert _format_fred_value(331.893 - 331.020) == "0.873"
        assert _format_fred_change(-0.749) == "-0.749"
        assert _format_fred_change(0.0) == "+0"
        assert _format_fred_change(-0.0) == "+0"
        assert _format_fred_change(-1e-9) == "+0"


class TestFredFirstReleaseTable:
    """The first-release-vs-current-vintage table for a revising resolving series.

    q44944's resolving quantity was the FIRST published Case-Shiller print while every level
    the provider rendered was today's revised vintage; anchoring on a revision-adjusted May
    was worth +66.6 spot peer. The table turns the revision channel from a symmetric-noise
    assumption into a signed input — and carries the double-count guard, because stacking it
    on a same-source leading indicator overshot by 0.7 index points and lost 15 spot peer.
    """

    @staticmethod
    def _fred_mock(
        current: pd.Series, first_releases: pd.Series | None, vintage_error: Exception | None = None
    ) -> MagicMock:
        """A fredapi mock whose get_series answers the plain and initial-release calls.

        The initial-release call is the one carrying ``output_type``; recorded on
        ``mock.first_release_calls`` so a test can assert the request shape."""
        first_release_calls: list[dict] = []

        def get_series(series_id: str, **kwargs) -> pd.Series:
            del series_id
            if "output_type" in kwargs:
                first_release_calls.append(kwargs)
                if first_releases is None:
                    raise vintage_error or ValueError("Bad Request. Invalid output_type.")
                return first_releases
            return current

        instance = MagicMock()
        instance.get_series.side_effect = get_series
        instance.get_series_info.return_value = pd.DataFrame({"title": ["S&P Case-Shiller"]}, index=["CSUSHPISA"])
        instance.first_release_calls = first_release_calls
        return instance

    # Five months of Case-Shiller, current vintage against first release. The last four
    # pairs are the table's rows: +0.873, +0.43, 0.0 (unrevised) and -0.749 — the same
    # both-directions ±0.3-0.8 revision channel the dossier measured across three instances.
    _CURRENT: ClassVar[list[float]] = [330.44, 330.873, 331.359, 331.020, 331.893]
    _FIRST: ClassVar[list[float]] = [330.16, 331.622, 331.359, 330.590, 331.020]

    def _fetch(
        self,
        *,
        is_resolving_source: bool,
        first_releases: pd.Series | None = None,
        vintage_error: Exception | None = None,
    ) -> tuple[str, MagicMock]:
        current = _monthly_fred(self._CURRENT)
        if vintage_error is not None:
            releases = None
        else:
            releases = _monthly_fred(self._FIRST) if first_releases is None else first_releases
        instance = self._fred_mock(current, releases, vintage_error)
        with patch("metaculus_bot.research.fred_rendering.Fred", return_value=instance) as fred_class:
            fred_class.earliest_realtime_start = "1776-07-04"
            fred_class.latest_realtime_end = "9999-12-31"
            markdown = _fetch_fred_data("CSUSHPISA", "fake_key", is_resolving_source=is_resolving_source)
        return markdown, instance

    def test_a_resolving_series_renders_the_table_with_the_double_count_guard(self) -> None:
        markdown, instance = self._fetch(is_resolving_source=True)

        assert "- First release vs current vintage" in markdown
        assert "  - 2026-06-01: first release 331.02 → current vintage 331.893 (revised +0.873)" in markdown
        assert "  - 2026-05-01: first release 330.59 → current vintage 331.02 (revised +0.43)" in markdown
        # An unrevised print says so rather than rendering "revised +0".
        assert "  - 2026-04-01: first release 331.359 → current vintage 331.359 (unrevised)" in markdown
        # 4 prints: +0.873, +0.43, 0.0 (331.359 unrevised), -0.749.
        assert "Of these 4 prints, 2 were revised up, 1 down and 1 not at all" in markdown
        assert "mean revision +0.1385" in markdown
        assert "⚠ Do not double-count" in markdown
        assert "Apply one of them, not both" in markdown
        assert len(instance.first_release_calls) == 1

    def test_the_initial_release_request_opens_the_full_real_time_window(self) -> None:
        """Both real-time bounds default to TODAY, which would restrict the answer to prints
        that were never revised — exactly the ones with nothing to report."""
        _markdown, instance = self._fetch(is_resolving_source=True)

        (kwargs,) = instance.first_release_calls
        assert kwargs["output_type"] == 4
        assert kwargs["realtime_start"] == "1776-07-04"
        assert kwargs["realtime_end"] == "9999-12-31"
        # Bounded to the prints the table renders, read off the dates already in hand.
        assert kwargs["observation_start"] == pd.Timestamp("2026-03-01")

    def test_an_out_of_order_fred_response_still_renders_ascending(self) -> None:
        """`_render_fred_series` documents "sorted ascending by date" as a precondition and
        nothing used to establish it: the live path only dropna'd. Two things break on an
        out-of-order response, and neither is cosmetic. The year-over-year lookup slices
        `data.loc[:year_ago]`, which RAISES on a non-monotonic DatetimeIndex with a
        non-existing key, so the outer soft-fail swallows the whole series block; and
        `pd.concat(join="inner")` takes the LEFT operand's order, so the table's `tail(4)`
        would pick four arbitrary prints under a "recent prints" label. Sorting the first
        releases (which the fetch already did) fixes neither, since they are the right operand.
        """
        shuffled = _monthly_fred(self._CURRENT).iloc[np.array([0, 4, 1, 3, 2])]
        assert not shuffled.index.is_monotonic_increasing, "the fixture must actually be out of order"
        instance = self._fred_mock(shuffled, _monthly_fred(self._FIRST))
        with patch("metaculus_bot.research.fred_rendering.Fred", return_value=instance) as fred_class:
            fred_class.earliest_realtime_start = "1776-07-04"
            fred_class.latest_realtime_end = "9999-12-31"
            markdown = _fetch_fred_data("CSUSHPISA", "fake_key", is_resolving_source=True)

        assert markdown != "", "an out-of-order response must not soft-fail the whole block"
        # The same four rows, in the same order, as the already-sorted fixture renders.
        assert "- Latest value: 331.893 (2026-06-01)" in markdown
        row_dates = re.findall(r"^ {2}- (\d{4}-\d{2}-\d{2}): first release", markdown, flags=re.MULTILINE)
        assert row_dates == sorted(row_dates), f"table rows are not date-ascending: {row_dates}"
        assert row_dates == ["2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01"]

    def test_a_classifier_only_series_makes_no_vintage_request(self) -> None:
        """The revision channel matters for the series a question GRADES against; every
        other identifier would just be another HTTP round trip inside the fetch thread."""
        markdown, instance = self._fetch(is_resolving_source=False)

        assert "### CSUSHPISA" in markdown
        assert "First release vs current vintage" not in markdown
        assert instance.first_release_calls == []

    def test_a_non_revising_series_makes_no_vintage_request(self) -> None:
        """DGS10 is a market rate on the non-revising allowlist: a first-release table there
        would be a column of zeros dressed as a finding."""
        current = _monthly_fred([4.15, 4.2], name="DGS10")
        instance = self._fred_mock(current, current)
        with patch("metaculus_bot.research.fred_rendering.Fred", return_value=instance):
            markdown = _fetch_fred_data("DGS10", "fake_key", is_resolving_source=True)

        assert "### DGS10" in markdown
        assert instance.first_release_calls == []

    @pytest.mark.parametrize(
        "vintage_error",
        [
            # fredapi re-raises the API's own error message as ValueError...
            ValueError("Bad Request. Invalid output_type."),
            # ...a transport failure arrives as URLError, an OSError...
            URLError("connection reset"),
            # ...and a non-XML body (a proxy or status page answering instead) reaches
            # ET.fromstring, whose ParseError is a SyntaxError and matches neither above.
            ParseError("syntax error: line 1, column 0"),
        ],
        ids=["api_error", "transport", "unparseable_body"],
    )
    def test_a_failed_vintage_fetch_leaves_the_primary_block_standing(self, vintage_error: Exception) -> None:
        """The table is enrichment; the series itself is the source, and no failure mode of
        the extra call may take it down."""
        markdown, instance = self._fetch(is_resolving_source=True, vintage_error=vintage_error)

        assert "- Latest value: 331.893 (2026-06-01)" in markdown
        assert "First release vs current vintage" not in markdown
        assert len(instance.first_release_calls) == 1

    def test_a_response_missing_the_latest_print_drops_the_table(self) -> None:
        """The guard on the one inference in this path — that opening the real-time window
        really does return revised prints' first releases. If FRED ever answers with only
        never-revised observations, the newest print is absent and a table of older ones
        under a "recent prints" label would be a different claim than the one being made."""
        stale_releases = _monthly_fred(self._FIRST[:-1], end="2026-05-01")
        markdown, _instance = self._fetch(is_resolving_source=True, first_releases=stale_releases)

        assert "- Latest value: 331.893 (2026-06-01)" in markdown
        assert "First release vs current vintage" not in markdown

    def test_the_benchmarking_path_renders_the_same_shape_without_a_table(self) -> None:
        """One renderer for both paths. The keyless ALFRED CSV serves a series AS OF a
        vintage, not each print's first release, so a backtest cannot measure this feature —
        the same limitation prediction_market and resolution_source carry."""
        current = _monthly_fred(self._CURRENT)
        with patch("metaculus_bot.research.fred_rendering.fetch_series", return_value=current):
            markdown = _fetch_fred_data_ceiling("CSUSHPISA", _BENCH_OPEN_TIME)

        assert "- Latest value: 331.893 (2026-06-01)" in markdown
        assert "First release vs current vintage" not in markdown

    @pytest.mark.asyncio
    async def test_the_provider_marks_url_extracted_series_as_resolving(self) -> None:
        """End-to-end wiring: only the URL-extracted series gets is_resolving_source=True,
        the classifier's extra context series does not."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: NONE\nFRED_SERIES: HOUST"
        question = _make_q(
            "Where will the Case-Shiller index print for June?",
            resolution_criteria="Resolves per https://fred.stlouisfed.org/series/CSUSHPISA.",
        )
        seen: dict[str, bool] = {}

        def _fred(series_id: str, api_key: str, *, is_resolving_source: bool = False) -> str:
            del api_key
            seen[series_id] = is_resolving_source
            return f"### {series_id} (stub)"

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data._fetch_fred_data", side_effect=_fred),
        ):
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("FRED_API_KEY", "fake_key")
            try:
                await financial_data_provider()(question)
            finally:
                monkeypatch.undo()

        assert seen == {"HOUST": False, "CSUSHPISA": True}
