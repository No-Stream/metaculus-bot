"""Inline chart-config reading for the resolution-source section.

One responsibility: recover the numbers a cited page publishes ONLY inside a
client-side chart config, and render them as a compact labelled block. Zero LLM
calls, no network, no HTML parser — an HTML-entity unescape plus ``json.loads``
over the config the page already ships to the browser.

Why this exists (qid 43949, IOM Missing Migrants). The resolving page fetches
fine — the repo's own Tier-1 path returns HTTP 200 with the full
``BROWSER_HEADERS`` set — and trafilatura extracts ~80k chars of incident rows
and prose that contain none of ``1342`` / ``Total Dead and Missing`` / ``2026``.
The resolving annual series lives in
``<div class="charts-highchart" data-chart="{...}">``, whose JSON carries
``xAxis[0].categories`` and ``series[].data``. A Wayback snapshot 25 days BEFORE
that forecast carries the same markup with 2026 = 1,240, so the series was
machine-readable from the page before the question opened; the published
forecast instead sat ~340 above the true level. This is an EXTRACTOR gap, not an
egress one, and it is the same shape the Tier-2 Datawrapper hop already fixes
for one vendor — except that here no second request is needed, because the
values are in the page we already hold.

Two config shapes are read:

- ``data-chart="…"`` / ``data-chart='…'`` — an HTML-escaped JSON config on the
  container element. This is the receipt-backed form (Drupal's Charts module,
  which the IOM page uses) and it is always strict JSON once unescaped.
- ``Highcharts.chart(…)`` / ``new Highcharts.Chart(…)`` / ``Highcharts.stockChart(…)``
  — the inline-script form. Its argument is a JavaScript object literal, which is
  strict JSON only when the page happens to emit it that way (quoted keys, no
  trailing commas, no function values). We parse it when it is and skip it when
  it is not: a JS-literal parser is a much larger surface than this rung earns,
  and nothing in the archive needs one yet.

Non-negotiables:

- Nothing raises. A malformed config, a truncated attribute, a config that is
  not an object, a series carrying callbacks instead of numbers — every one of
  those is skipped with a DEBUG line. A cited page must never fail to render
  because its decoration did not parse.
- Only numbers are rendered, labelled by their own category where the config
  supplies one. Nothing is summed, interpolated, unit-converted or re-derived:
  the block states the values the page's own chart holds, so a forecaster reading
  it is reading the page, not our arithmetic.
- Every bound is a constant (charts, series per chart, points per series,
  candidate configs scanned, bytes per config, and the block's own char cap), so
  a page with fifty charts cannot spend a question's whole research budget on
  decoration.
"""

from __future__ import annotations

import html as html_entities
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from metaculus_bot.constants import (
    RESOLUTION_SOURCE_CHART_BLOCK_MAX_CHARS,
    RESOLUTION_SOURCE_CHART_MAX_CANDIDATES,
    RESOLUTION_SOURCE_CHART_MAX_CHARTS,
    RESOLUTION_SOURCE_CHART_MAX_CONFIG_CHARS,
    RESOLUTION_SOURCE_CHART_MAX_POINTS,
    RESOLUTION_SOURCE_CHART_MAX_SERIES,
)
from metaculus_bot.research.number_format import format_decimal_value

logger = logging.getLogger(__name__)

# The lead line the block carries. Says what we DID (read the page's own chart
# config) and why the figures are missing from the text beside it, without
# claiming the numbers appear nowhere else on the page — a claim we cannot check.
CHART_DATA_LEAD = (
    "[Chart data read from this page's inline chart configuration — the charts render "
    "client-side, so the page text below does not include them.]"
)

# Attribute form. Both quote styles, because the escaping that makes the JSON
# safe inside a double-quoted attribute (`&quot;`) is exactly what a
# single-quoted attribute does not need.
_DATA_CHART_RE = re.compile(r"""data-chart\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE)

# Inline-script form. `stockChart` is included because Highstock pages use it for
# exactly the long daily series a resolution source tends to be graded on.
_HIGHCHARTS_CALL_RE = re.compile(r"(?:new\s+)?Highcharts\.(?:chart|stockChart)\s*\(", re.IGNORECASE)

# Cheap "is it worth scanning at all?" gate. A regex rather than two `in` checks because
# both patterns above are case-insensitive and `html_text.lower()` would copy a body up
# to the 5 MiB response cap to answer a question one linear scan already answers.
_ANY_CHART_HINT_RE = re.compile(r"data-chart|highcharts", re.IGNORECASE)


# A whole string literal (either quote style, backslash escapes consumed) OR a single
# brace. Matching strings as one token is what keeps a chart title reading
# `"Deaths {2014-2026}"` from closing the object early, and the alternations are
# unambiguous, so the scan is linear rather than backtracking (the same discipline
# `resolution_body_text`'s tag regex documents at length).
_JSON_BRACE_OR_STRING_RE = re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[{}]")


def _balanced_object(text: str, start: int) -> str | None:
    """The ``{...}`` run beginning at ``start``, or None if it does not close in bounds.

    Bounded by ``RESOLUTION_SOURCE_CHART_MAX_CONFIG_CHARS`` so a page with one
    unclosed brace costs a fixed scan rather than the whole body.
    """
    if start >= len(text) or text[start] != "{":
        return None
    end = min(len(text), start + RESOLUTION_SOURCE_CHART_MAX_CONFIG_CHARS)
    depth = 0
    for match in _JSON_BRACE_OR_STRING_RE.finditer(text, start, end):
        matched = match.group()
        if matched == "{":
            depth += 1
        elif matched == "}":
            depth -= 1
            if depth == 0:
                return text[start : match.end()]
    return None


def _candidate_configs(html_text: str) -> list[str]:
    """Raw config strings found in ``html_text``, attribute form first.

    Attribute form leads because it is the receipt-backed one and it is always
    valid JSON; a page carrying both would otherwise have its readable configs
    crowded out of the candidate budget by unparseable script literals.

    ``RESOLUTION_SOURCE_CHART_MAX_CANDIDATES`` bounds the sites EXAMINED rather
    than the configs kept, and one counter spans both loops. Counting kept configs
    bounded nothing on the script form: a ``Highcharts.chart(`` whose braces never
    close appends no candidate, so every such site on the page paid its own
    ``_balanced_object`` scan and the page's cost was sites x the config-char
    bound. An examined site now costs its bounded scan once and then retires
    budget like any other.
    """
    candidates: list[str] = []
    examined = 0
    for match in _DATA_CHART_RE.finditer(html_text):
        examined += 1
        raw = match.group(1) if match.group(1) is not None else match.group(2)
        if raw:
            candidates.append(html_entities.unescape(raw))
        if examined >= RESOLUTION_SOURCE_CHART_MAX_CANDIDATES:
            return candidates
    for match in _HIGHCHARTS_CALL_RE.finditer(html_text):
        if examined >= RESOLUTION_SOURCE_CHART_MAX_CANDIDATES:
            break
        examined += 1
        brace = html_text.find("{", match.end())
        if brace == -1:
            continue
        obj = _balanced_object(html_text, brace)
        if obj:
            candidates.append(obj)
    return candidates


def _format_number(value: Any) -> str | None:
    """``value`` as a plain decimal string, or None when it is not a number.

    Floats go through ``number_format.format_decimal_value``, the same rule the FRED
    block renders by and for the same reason: a resolution source graded on an index
    level reads ``331.893``, not ``331.9`` or ``3.31893e+02``. That shared home is also
    where the "-0" guard lives, so a delta of ``-1e-7`` renders "0" rather than putting a
    minus sign on a quantity too small to display. ``bool`` is excluded explicitly
    because it is an ``int`` subclass and a ``True`` in a data array is a flag, not an
    observation.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):  # NaN / infinity
            return None
        return format_decimal_value(value)
    if isinstance(value, str):
        # Some configs quote their numbers. Accept only a clean numeric literal;
        # anything else is a label that landed in the value slot.
        try:
            return _format_number(float(value))
        except ValueError:
            return None
    return None


def _format_epoch_ms(value: Any) -> str | None:
    """An x value on a declared ``datetime`` axis as a UTC ``YYYY-MM-DD``, else None.

    Highcharts defines a datetime axis in MILLISECONDS since the Unix epoch, UTC, so
    this is the axis's own contract rather than a guess — and it only runs when the
    config declares ``type: "datetime"``. Without it a daily resolving series renders
    ``1756771200000=42``, which is the shape most likely to matter (a tracker's own
    Highstock chart) rendered as noise. Out-of-range or non-numeric x values fall back
    to the numeric label rather than inventing a date.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=UTC).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return None


def _point_label_and_value(point: Any, *, x_is_datetime: bool) -> tuple[str | None, str | None]:
    """One data point as ``(own label, value)``; either half may be None.

    Highcharts admits four point shapes and three of them can carry a label of
    their own: a bare number (no label — the caller supplies the category), an
    object with ``name``/``y``, and an ``[x, y]`` pair whose x is a category name
    or a number.
    """
    if isinstance(point, dict):
        label = point.get("name")
        if not isinstance(label, str) and x_is_datetime:
            label = _format_epoch_ms(point.get("x"))
        return (label if isinstance(label, str) else None), _format_number(point.get("y"))
    if isinstance(point, list | tuple) and len(point) == 2:
        x, y = point
        return _pair_x_label(x, x_is_datetime=x_is_datetime), _format_number(y)
    return None, _format_number(point)


def _pair_x_label(x: Any, *, x_is_datetime: bool) -> str | None:
    """The label an ``[x, y]`` pair's x half contributes: its own string, a UTC date on a
    declared datetime axis, else the number itself."""
    if isinstance(x, str):
        return x
    if x_is_datetime:
        return _format_epoch_ms(x) or _format_number(x)
    return _format_number(x)


def _x_axis(config: dict[str, Any]) -> dict[str, Any]:
    """``xAxis[0]``, tolerating the single-object ``xAxis`` form; ``{}`` when absent."""
    x_axis = config.get("xAxis")
    if isinstance(x_axis, list) and x_axis and isinstance(x_axis[0], dict):
        x_axis = x_axis[0]
    return x_axis if isinstance(x_axis, dict) else {}


def _categories(config: dict[str, Any]) -> list[Any]:
    categories = _x_axis(config).get("categories")
    return categories if isinstance(categories, list) else []


def _x_is_datetime(config: dict[str, Any]) -> bool:
    return _x_axis(config).get("type") == "datetime"


def _chart_title(config: dict[str, Any]) -> str:
    """The config's own title text, or ``""``.

    Deliberately only the config's title: the human-readable heading above an
    IOM-shaped chart lives in a sibling ``<h2>`` outside the JSON, and pairing a
    chart with a heading found by HTML proximity is a guess we would be printing
    as a label. The series names carry the meaning instead.
    """
    title = config.get("title")
    if isinstance(title, dict) and isinstance(title.get("text"), str):
        return title["text"].strip()
    return ""


def _render_series(series: Any, categories: list[Any], *, x_is_datetime: bool) -> str | None:
    """One ``series`` entry as an indented ``name: label=value, …`` line, or None."""
    if not isinstance(series, dict):
        return None
    data = series.get("data")
    if not isinstance(data, list) or not data:
        return None
    kept = data[-RESOLUTION_SOURCE_CHART_MAX_POINTS:]
    offset = len(data) - len(kept)
    cells: list[str] = []
    for i, point in enumerate(kept):
        label, value = _point_label_and_value(point, x_is_datetime=x_is_datetime)
        if value is None:
            # A gap (Highcharts `null`), a callback, or a label in the value slot.
            continue
        if label is None:
            index = offset + i
            if index < len(categories) and isinstance(categories[index], str):
                label = categories[index]
        cells.append(f"{label}={value}" if label else value)
    if not cells:
        return None
    name = series.get("name")
    name = name.strip() if isinstance(name, str) and name.strip() else "series"
    # State the window explicitly when the series was cut, so a forecaster reading
    # "2015=…" knows the chart starts earlier and is not reading its whole history.
    window = f" (last {len(kept)} of {len(data)} points)" if offset else ""
    return f"  {name}{window}: " + ", ".join(cells)


def _render_config(raw: str, ordinal: int) -> str | None:
    """One candidate config as a labelled block, or None when it carries no series."""
    if len(raw) > RESOLUTION_SOURCE_CHART_MAX_CONFIG_CHARS:
        logger.debug(f"resolution_source chart config skipped: {len(raw)} chars over the config bound")
        return None
    try:
        config = json.loads(raw)
    except (ValueError, TypeError) as e:
        # The script-literal form is the expected failure here (a JS object
        # literal is usually not strict JSON). DEBUG, not WARN: a page whose
        # decoration does not parse is not a defect in the fetch.
        logger.debug(f"resolution_source chart config not JSON-parseable: {e}")
        return None
    if not isinstance(config, dict):
        return None
    series_list = config.get("series")
    if not isinstance(series_list, list) or not series_list:
        return None
    categories = _categories(config)
    x_is_datetime = _x_is_datetime(config)
    lines = [
        line
        for line in (
            _render_series(s, categories, x_is_datetime=x_is_datetime)
            for s in series_list[:RESOLUTION_SOURCE_CHART_MAX_SERIES]
        )
        if line is not None
    ]
    if not lines:
        return None
    title = _chart_title(config)
    heading = f"Chart {ordinal}" + (f" — {title}" if title else "")
    return "\n".join([heading, *lines])


def _omitted_charts_note(n: int) -> str:
    return f"[{n} further chart(s) on this page omitted — chart-data budget]"


def render_inline_chart_data(html_text: str) -> str:
    """The page's inline chart series as a bounded labelled block, or ``""``.

    ``""`` means "nothing to add": no config, none parseable, or none carrying
    numeric series. Callers treat a non-empty return as CONTENT — it is what
    keeps a page whose text is pure chrome from being withheld, and what puts the
    resolving figures in front of a forecaster on a page whose prose has none.

    The return always leads with :data:`CHART_DATA_LEAD`.

    Invariant: ``len(return) <= RESOLUTION_SOURCE_CHART_BLOCK_MAX_CHARS``. Whole
    charts are dropped rather than cut — a half-row reads like a complete series,
    which is the stale-as-live failure in another coat — and the omitted count is
    stated. The omission note is reserved at its worst-case width up front (the
    same trick ``_truncate_csv_middle`` uses), which is what makes the bound
    provable rather than nearly true.
    """
    if not html_text or _ANY_CHART_HINT_RE.search(html_text) is None:
        return ""
    budget = (
        RESOLUTION_SOURCE_CHART_BLOCK_MAX_CHARS
        - len(CHART_DATA_LEAD)
        - len(_omitted_charts_note(RESOLUTION_SOURCE_CHART_MAX_CANDIDATES))
    )
    blocks: list[str] = []
    omitted_charts = 0
    used = 0
    for raw in _candidate_configs(html_text):
        block = _render_config(raw, len(blocks) + 1)
        if block is None:
            continue
        # The chart cap counts as an omission, exactly like the char budget: both
        # leave a readable chart off the page, and a silent `break` here made the
        # docstring's "the omitted count is stated" false on the common shape (a
        # page with more readable charts than the cap) and made the count
        # UNDER-state on a page whose over-budget chart came first. Only readable
        # configs are counted — a config we could not parse was never a chart we
        # left out.
        if len(blocks) >= RESOLUTION_SOURCE_CHART_MAX_CHARTS or used + len(block) + 1 > budget:
            omitted_charts += 1
            continue
        blocks.append(block)
        used += len(block) + 1
    if not blocks:
        return ""
    parts = [CHART_DATA_LEAD, *blocks]
    if omitted_charts:
        parts.append(_omitted_charts_note(omitted_charts))
    return "\n".join(parts)
