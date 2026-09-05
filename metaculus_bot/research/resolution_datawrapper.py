"""Response policy and result ordering for resolution-source Datawrapper datasets."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from metaculus_bot.constants import (
    RESOLUTION_SOURCE_CLOCK_SKEW_TOLERANCE,
    RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS,
    RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS,
    RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
)
from metaculus_bot.research.http_fetch import (
    DatawrapperChartRef,
    decode_text_body,
    parse_http_last_modified,
    read_body_capped,
)
from metaculus_bot.research.resolution_body_text import _truncate_csv_middle, strip_html_tags
from metaculus_bot.research.resolution_fetch_result import (
    _NON_OK_FETCH_STATUS,
    FetchResult,
    FetchStatus,
    http_failure_class,
    server_header_token,
    vacuous_body_status,
)

logger = logging.getLogger(__name__)


def _datawrapper_hop_status(status: int) -> FetchStatus:
    """Map the CDN's HTTP status onto a FetchStatus (200 -> ``success``)."""
    return "success" if status == 200 else _NON_OK_FETCH_STATUS.get(status, "error")


def _datawrapper_last_modified(resp: Any) -> datetime | None:
    """The dataset's parsed ``Last-Modified``, or None when absent or unparseable."""
    raw = resp.headers.get("Last-Modified")
    return parse_http_last_modified(raw) if raw else None


def _datawrapper_freshness_failure(last_modified: datetime | None) -> str | None:
    """Why ``last_modified`` fails the freshness guard, or None when it passes.

    Two-sided, deliberately. The lead this stamp authorizes asserts a
    publication date, and a FUTURE one means a broken clock or a misparse on
    one side — so it is unusable as a freshness claim, not maximally fresh.
    The old one-sided check let any future date through as the freshest
    possible dataset.
    """
    if last_modified is None:
        return "no parseable Last-Modified"
    now = datetime.now(UTC)
    if last_modified - now > RESOLUTION_SOURCE_CLOCK_SKEW_TOLERANCE:
        return f"published {last_modified.isoformat()}, which is in the FUTURE"
    if now - last_modified > timedelta(days=RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS):
        return (
            f"published {last_modified.isoformat()}, age {(now - last_modified).days}d "
            f"> {RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS}d bound"
        )
    return None


def _datawrapper_success_text(
    chart: DatawrapperChartRef, parent_url: str, url: str, *, dataset_text: str, published: datetime
) -> str:
    """The liveness lead plus the budgeted CSV rows."""
    # Every claim in this lead is now checked: the timestamp by the
    # freshness guard above, and "dataset" itself by the row-shape
    # check — an authoritative `published <ts>` stamp over an empty or
    # soft-404 body was the same defect class as a manufactured price.
    title_part = f" ({chart.title!r})" if chart.title else ""
    lead = (
        f'Live "Get the data" dataset for Datawrapper chart {chart.chart_id}{title_part} '
        f"embedded in {parent_url}. Dataset published {published.isoformat()}."
    )
    # The DATASET cap, not the page cap: datasets budget against their own
    # section allowance so a chart's rows can never evict cited page text.
    # Tags are stripped BEFORE truncation so the budget buys rows, not markup.
    csv_budget = RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS - len(lead) - 2
    return f"{lead}\n\n{_truncate_csv_middle(dataset_text, csv_budget, url)}"


async def _datawrapper_dataset_outcome(resp: Any, chart: DatawrapperChartRef, parent_url: str, url: str) -> FetchResult:
    """Turn the CDN response into a FetchResult, serving the dataset live or not at all."""
    status = resp.status
    content_type = (resp.headers.get("Content-Type") or "").lower()
    hop_status = _datawrapper_hop_status(status)
    if hop_status != "success":
        return FetchResult(
            url=url,
            status=hop_status,
            text="",
            http_status=status,
            content_type=content_type or None,
            chart_id=chart.chart_id,
            chart_title=chart.title,
            parent_url=parent_url,
            failure_class=http_failure_class(status),
            server=server_header_token(resp.headers.get("Server")),
        )

    body = await read_body_capped(
        resp,
        max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
        label=f"resolution_source datawrapper {chart.chart_id}",
    )
    if body is None:
        return FetchResult(
            url=url,
            status="error",
            text="",
            http_status=status,
            content_type=content_type or None,
            chart_id=chart.chart_id,
            chart_title=chart.title,
            parent_url=parent_url,
        )

    # Content BEFORE freshness, deliberately: an empty or non-CSV CDN
    # body is a failed hop whatever its Last-Modified says, and
    # `stale_data` is reported to diagnostics as the benign `none`
    # (the freshness guard working as designed), which would hide it.
    # Row-shape is decided on the PRE-strip text: looks_like_csv_rows
    # rejects markup by its leading `<`, and stripping first would remove
    # exactly the allow-listed fragment tags (`<p>`, `<div>`) a CDN
    # soft-404 opens with, letting an error page carry the authoritative
    # "Dataset published" lead if its prose holds a comma.
    dataset_text, undecodable_ratio = decode_text_body(body, content_type)
    vacuous = vacuous_body_status(dataset_text, undecodable_ratio, require_csv_rows=True)
    dataset_text = strip_html_tags(dataset_text).strip()
    if vacuous is not None:
        logger.warning(
            f"resolution_source datawrapper hop {chart.chart_id}: dataset body is not a usable "
            f"dataset ({vacuous}: {len(body)} bytes, undecodable={undecodable_ratio:.2f}) — "
            f"withheld rather than stamped live"
        )
        return FetchResult(
            url=url,
            status=vacuous,
            text="",
            http_status=status,
            content_type=content_type or None,
            chart_id=chart.chart_id,
            chart_title=chart.title,
            parent_url=parent_url,
        )

    last_modified = _datawrapper_last_modified(resp)
    freshness_failure = _datawrapper_freshness_failure(last_modified)
    if freshness_failure is not None:
        logger.warning(
            f"resolution_source datawrapper hop {chart.chart_id}: dataset failed the "
            f"freshness guard ({freshness_failure}) — withheld, not served as live"
        )
        return FetchResult(
            url=url,
            status="stale_data",
            text="",
            http_status=status,
            content_type=content_type or None,
            chart_id=chart.chart_id,
            chart_title=chart.title,
            parent_url=parent_url,
            data_last_modified=last_modified.isoformat() if last_modified else None,
        )

    assert last_modified is not None  # a passing freshness guard implies a parsed timestamp
    return FetchResult(
        url=url,
        status="success",
        text=_datawrapper_success_text(chart, parent_url, url, dataset_text=dataset_text, published=last_modified),
        http_status=status,
        content_type=content_type or None,
        chart_id=chart.chart_id,
        chart_title=chart.title,
        parent_url=parent_url,
        data_last_modified=last_modified.isoformat(),
    )


def _select_datawrapper_charts(page_results: list[FetchResult]) -> list[tuple[int, DatawrapperChartRef]]:
    """Pick the charts to hop to, as ``(parent_index, chart)`` pairs.

    Page order first, then document order within a page (tracker pages put the
    hero/resolving chart first), deduped by chart id across pages, capped
    globally at ``RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS``.
    """
    picks: list[tuple[int, DatawrapperChartRef]] = []
    seen: set[str] = set()
    for idx, r in enumerate(page_results):
        for chart in r.datawrapper_charts:
            if chart.chart_id in seen:
                continue
            seen.add(chart.chart_id)
            picks.append((idx, chart))
            if len(picks) >= RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS:
                return picks
    return picks


def _interleave_dataset_results(
    page_results: list[FetchResult],
    picks: list[tuple[int, DatawrapperChartRef]],
    dataset_results: list[FetchResult],
) -> list[FetchResult]:
    """Place each dataset result directly after its parent page's result, so
    the rendered section (and the total-budget trimming order) keeps a chart's
    data adjacent to the page that embeds it."""
    by_parent: dict[int, list[FetchResult]] = {}
    for (idx, _chart), ds in zip(picks, dataset_results, strict=False):
        by_parent.setdefault(idx, []).append(ds)
    merged: list[FetchResult] = []
    for idx, r in enumerate(page_results):
        merged.append(r)
        merged.extend(by_parent.get(idx, []))
    return merged
