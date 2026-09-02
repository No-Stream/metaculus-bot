"""FRED series fetch and render: value formatting, the block, and the vintage table.

Both fetch paths and every formatting decision for the ``### <series> (<title>)`` block a
forecaster reads. Split out of ``financial_data``, which keeps the classifier, the identifier
extraction and capping, the job fan-out, and the yfinance side.

The two fetchers ride along with the renderers deliberately. ``_fetch_fred_first_releases``
reads ``Fred.earliest_realtime_start`` / ``latest_realtime_end`` off the CLASS, and its test
proves that by patching ``Fred`` in the module that constructs it — fredapi's real class
carries the identical literals, so a patch aimed at a module that no longer builds the client
would leave the test green while it silently stopped proving anything. Client construction and
the class-attribute reads therefore stay in one patchable namespace.

Values are rendered fixed-point, up to six decimals with trailing zeros stripped, never
scientific notation: `:.4g` turned a Case-Shiller print of 331.893 into "331.9" on a question
whose displayed range was four index points wide, and the Fed balance sheet into "6.7e+06".
"""

import logging
from datetime import datetime
from typing import cast
from xml.etree.ElementTree import ParseError

import pandas as pd
from fredapi import Fred

from metaculus_bot.constants import FINANCIAL_FRED_VINTAGE_PRINTS
from metaculus_bot.research.ts_fetch import FRED_NON_REVISING_SERIES, FetchError, SeriesSpec, fetch_series

logger: logging.Logger = logging.getLogger(__name__)


def _pct_clause(change: float, base: float) -> str:
    """`` (+1.23%)`` for a percent change off ``base``, or "" when there is no percent.

    A base of exactly 0 has NO percent change — it is undefined, not 0.00%. FRED spread
    and rate-difference series (T10Y2Y, T10Y3M) cross zero routinely, and the old
    ``else 0`` rendered ``+0.31 (+0.00%)`` there: a fabricated "unchanged" reading sitting
    beside a genuine absolute move, in a forecaster prompt. Omitting the clause leaves the
    absolute change, which is the part that was actually measured.
    """
    base_value = float(base)
    if base_value == 0:
        return ""
    return f" ({(change / abs(base_value)) * 100:+.2f}%)"


# Decimals kept on a rendered FRED level or change. Six covers everything FRED publishes
# (index levels at three, most rates at two, a few series at four) and trailing zeros are
# stripped, so a rate still renders "4.2". Replaces `:.4g`, which rounded a Case-Shiller
# level of 331.893 to "331.9" on q44944 — a question whose displayed range was four index
# points wide with 0.02-point buckets, so the digits `:.4g` threw away were the whole
# forecast. `:.4g` also flipped to scientific notation on large series (WALCL's ~6.7e6
# rendered as "6.7e+06"), which this format never does.
_FRED_VALUE_DECIMALS = 6


def _format_fred_value(value: float) -> str:
    """A FRED level at its published precision: fixed-point, no scientific notation.

    Also cleans up float subtraction artifacts for free — a change computed as
    331.893 - 331.02 = 0.8729999999999905 renders "0.873".
    """
    text = f"{float(value):.{_FRED_VALUE_DECIMALS}f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-", "-0"} else "0"


def _format_fred_change(change: float) -> str:
    """A signed change at the same precision (``:+`` cannot drive a custom formatter).

    A change that rounds to zero at this precision renders "+0", never "-0": the sign of a
    quantity too small to display is not information.
    """
    magnitude = _format_fred_value(abs(float(change)))
    sign = "-" if float(change) < 0 and magnitude != "0" else "+"
    return f"{sign}{magnitude}"


def _first_release_lines(data: pd.Series, first_releases: pd.Series) -> list[str]:
    """The first-release-versus-current-vintage table for the most recent prints.

    A question on a revising FRED series resolves on the value the agency PUBLISHES, i.e.
    the first release, while every level rendered above it is today's revised vintage —
    q44944 resolved on a first-release Case-Shiller print, and a revision-adjusted anchor
    was worth +66.6 spot peer there. The gap between the two is a signed, measurable
    quantity, so it is rendered rather than left as a symmetric-noise assumption.

    Empty list when the two series share no dated observation, or when the LATEST print we
    render a level for has no first release in hand: a table of older prints under a
    "recent prints" label would be a different claim than the one being made.
    """
    paired = pd.concat([data.rename("current"), first_releases.rename("first")], axis=1, join="inner").dropna()
    if paired.empty or data.index[-1] not in paired.index:
        return []
    recent = paired.tail(FINANCIAL_FRED_VINTAGE_PRINTS)
    dates = pd.DatetimeIndex(recent.index).strftime("%Y-%m-%d")
    current_values = recent["current"].to_numpy(dtype="float64")
    first_values = recent["first"].to_numpy(dtype="float64")
    revisions = current_values - first_values
    rows = [
        f"  - {obs_date}: first release {_format_fred_value(first)} → current vintage "
        f"{_format_fred_value(current)} "
        + ("(unrevised)" if _format_fred_change(revision) == "+0" else f"(revised {_format_fred_change(revision)})")
        for obs_date, first, current, revision in zip(dates, first_values, current_values, revisions, strict=True)
    ]
    revised_up = int((revisions > 0).sum())
    revised_down = int((revisions < 0).sum())
    unchanged = len(revisions) - revised_up - revised_down
    direction = (
        f"  - Of these {len(revisions)} prints, {revised_up} were revised up, {revised_down} down and "
        f"{unchanged} not at all; mean revision {_format_fred_change(float(revisions.mean()))}."
    )
    return [
        "- First release vs current vintage (this series revises: a question resolving on the published print "
        "resolves on the FIRST release, while every level above is today's revised vintage):",
        "\n".join([*rows, direction]),
        # Mandatory guard from the q44944 dossier's "two levers still on the table" table:
        # each lever alone roughly doubled the score, stacking them overshot by 0.7 index
        # points and lost 15 spot-peer.
        "- ⚠ Do not double-count: adjusting for the revision direction and leaning on a same-source leading "
        "indicator (e.g. ICE HPI for Case-Shiller) partly measure the SAME underlying data. Apply one of them, "
        "not both.",
    ]


def _render_fred_series(series_id: str, data: pd.Series, title: str, *, first_releases: pd.Series | None = None) -> str:
    """Render the derived-stat markdown block for a FRED series.

    Shared by the live (fredapi) and benchmarking (keyless ts_fetch) paths so the two
    render identically — latest/previous value, MoM + YoY change, last 6 observations, and
    the first-release table when the caller could fetch one. ``data`` must already be
    dropna'd and sorted ascending by date.
    """
    parts = [f"### {series_id} ({title})"]

    latest_value = data.iloc[-1]
    latest_date = data.index[-1]
    parts.append(f"- Latest value: {_format_fred_value(latest_value)} ({latest_date.strftime('%Y-%m-%d')})")

    if len(data) >= 2:
        previous_value = data.iloc[-2]
        parts.append(f"- Previous value: {_format_fred_value(previous_value)}")

    # Change from the previous OBSERVATION — a row step, whatever the series'
    # cadence (monthly CPI, quarterly GDP, weekly claims) — which is exactly what
    # the rendered "Change from previous" label claims and no more.
    if len(data) >= 2:
        mom_change = latest_value - data.iloc[-2]
        parts.append(
            f"- Change from previous: {_format_fred_change(mom_change)}{_pct_clause(mom_change, data.iloc[-2])}"
        )

    # Year-over-year change via a DATE-based lookup, not a fixed observation
    # offset: `data.iloc[-13]` is one year back only on a monthly series; on a
    # daily FRED series (DGS10, DGS2, T10Y2Y, ...) 13 observations is ~2.5 weeks,
    # which would be mislabeled "year-over-year" in a live forecaster prompt. Take
    # the most recent observation at or before ~365 days ago (label slice is
    # inclusive; data is sorted ascending). Omit the line entirely when no such
    # observation exists (series shorter than a year).
    year_ago = latest_date - pd.Timedelta(days=365)
    prior = data.loc[:year_ago]
    if not prior.empty:
        yoy_value = prior.iloc[-1]
        yoy_change = latest_value - yoy_value
        parts.append(f"- Year-over-year change: {_format_fred_change(yoy_change)}{_pct_clause(yoy_change, yoy_value)}")

    # Last 6 observations
    last_6 = data.tail(6)
    obs_lines = [
        f"  - {cast(pd.Timestamp, date).strftime('%Y-%m-%d')}: {_format_fred_value(val)}"
        for date, val in last_6.items()
    ]
    parts.append("- Recent observations:\n" + "\n".join(obs_lines))

    if first_releases is not None:
        parts.extend(_first_release_lines(data, first_releases))

    return "\n".join(parts)


def _fetch_fred_first_releases(fred: Fred, series_id: str, observation_start: pd.Timestamp) -> pd.Series | None:
    """First-published value per observation since ``observation_start``, or None.

    ``output_type=4`` is ALFRED's "Observations, Initial Release Only" format: one row per
    observation carrying the value first released for it, revisions omitted (ALFRED's
    download-data help, read 2026-09-01: "This output format only contains the first
    released values for each observation ... the realtime_start_date column contains the
    dates when initial values were released to the public"). One row per observation is what
    makes this the safe request shape — ``fredapi.get_series_first_release`` instead pulls
    EVERY revision of every observation and takes the first per date, which on a series that
    restates its whole history each month (Case-Shiller's seasonal factors) is six figures
    of rows against FRED's 100k response cap, silently truncated at the oldest end.

    The real-time window is opened to FRED's full range because both bounds default to
    TODAY, and a real-time period of today would restrict the answer to values whose
    real-time period still contains today — i.e. only the prints that were never revised,
    which are exactly the ones with nothing to report. ``_first_release_lines`` re-checks
    that the latest rendered observation is present, so if that reading of the parameter
    interaction is ever wrong the table is dropped rather than rendered stale.

    ``None`` on any FRED error, so the series' primary block still renders — this table is
    enrichment and must never be able to take the source itself down. The three ways this
    call can fail are ``ValueError`` (``fredapi`` re-raises the API's own error message that
    way), ``OSError`` (``URLError``/``HTTPError`` transport failures), and ``ParseError``
    (``fredapi`` runs ``ET.fromstring`` over the response body, including an error body,
    which is not XML if a proxy or status page answers instead).
    """
    try:
        first_releases = fred.get_series(
            series_id,
            observation_start=observation_start,
            output_type=4,
            realtime_start=Fred.earliest_realtime_start,
            realtime_end=Fred.latest_realtime_end,
        )
    except (ValueError, OSError, ParseError):
        logger.warning(f"FRED first-release (ALFRED vintage) fetch failed for {series_id=}", exc_info=True)
        return None
    cleaned = first_releases.dropna()
    if cleaned.empty:
        logger.info(f"FRED first-release fetch returned no observations for {series_id=}")
        return None
    return cleaned.sort_index()


def _fetch_fred_data(series_id: str, api_key: str, *, is_resolving_source: bool = False) -> str:
    """Fetch economic data for a single FRED series (live path, fredapi).

    Sync function -- caller wraps in asyncio.to_thread().
    Returns formatted markdown or "" on any failure.

    ``is_resolving_source`` marks a series the QUESTION resolves on (URL-extracted from the
    resolution criteria, not merely named by the classifier). Those get the extra
    first-release/vintage fetch, since the revision channel only matters for the series the
    question grades against, and every extra identifier would otherwise cost another HTTP
    round trip inside this thread.
    """
    try:
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id)

        if data.empty:
            logger.warning(f"FRED returned empty data for {series_id=}")
            return ""

        # Sorted here, not just dropna'd, because everything downstream reads position as
        # date order: `iloc[-1]`/`iloc[-2]` as latest/previous, the YoY label slice (which
        # RAISES on a non-monotonic DatetimeIndex and would soft-fail the whole block), and
        # the first-release table's `tail()` — `pd.concat(join="inner")` takes the LEFT
        # operand's order, so sorting the vintage series alone establishes nothing.
        data = data.dropna().sort_index()
        if data.empty:
            return ""

        # Title is best-effort enrichment; fall back to the raw series_id if FRED metadata lookup fails.
        title = series_id
        try:
            info_df = fred.get_series_info(series_id)
            if isinstance(info_df, pd.DataFrame) and "title" in info_df.columns:
                title = cast(pd.Series, info_df["title"]).iloc[0]
            elif isinstance(info_df, pd.Series) and "title" in info_df.index:
                title = info_df["title"]
        except Exception:  # HARNESS-SCAN-EXEMPT-broad-except  # cosmetic title lookup, logged
            logger.debug(f"FRED series title lookup failed for {series_id=}", exc_info=True)

        first_releases = None
        # Series on the non-revising allowlist cannot revise, so a first-release table there
        # would be a column of zeros dressed as a finding.
        if is_resolving_source and series_id.upper() not in FRED_NON_REVISING_SERIES and len(data) >= 2:
            # Bound the request to the prints the table renders, taken off the dates we
            # already hold — cadence-agnostic, so it is the last four months on CPI and the
            # last four quarters on GDP without either being hardcoded.
            observation_start = cast(pd.Timestamp, data.index[-min(FINANCIAL_FRED_VINTAGE_PRINTS, len(data))])
            first_releases = _fetch_fred_first_releases(fred, series_id, observation_start)

        return _render_fred_series(series_id, data, title, first_releases=first_releases)

    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except  # provider soft-fail boundary, logged
        logger.warning(f"FRED fetch failed for {series_id=}", exc_info=True)
        return ""


def _fetch_fred_data_ceiling(series_id: str, as_of: datetime) -> str:
    """Point-in-time FRED fetch for backtests, via the keyless ts_fetch path.

    Sync function -- caller wraps in asyncio.to_thread(). Reuses ``ts_fetch.fetch_series``
    (fredgraph / ALFRED-vintage), so revised macro series (CPI, payrolls, GDP) return the
    vintage KNOWN at ``as_of`` instead of today's revisions — a plain observation_end on
    fredapi would still leak those. Keyless, so it works in CI without FRED_API_KEY.

    Non-title header: get_series_info needs an API key, so the backtest block reuses the
    series_id as the title — identical to the live path's metadata-failure fallback.
    Returns formatted markdown or "" on any fetch/data error.

    No first-release/vintage table here, so a backtest cannot measure that feature (the same
    limitation prediction_market and resolution_source carry). The keyless alfredgraph CSV
    serves exactly one thing — the series AS OF a given vintage date — so pinning each
    print's FIRST release would need one fetch per print at release dates this path does not
    know; the real-time query that answers it in one request needs FRED_API_KEY, which this
    path exists to do without.
    """
    try:
        # Default to ALFRED vintages for every revising macro series; only the curated
        # non-revising allowlist is safe on plain fredgraph (same decision the anchor makes).
        revises = series_id.upper() not in FRED_NON_REVISING_SERIES
        spec = SeriesSpec(source="fred", series_id=series_id, revises=revises)
        data = fetch_series(spec, as_of.date())
        return _render_fred_series(series_id, data, series_id)
    except (FetchError, ValueError):
        logger.warning(f"FRED ceilinged fetch failed for {series_id=}", exc_info=True)
        return ""
