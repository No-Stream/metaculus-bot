"""Raw-body text shaping for the resolution-source section.

One responsibility: turn a non-HTML body — plain text, CSV, or a Tier-2
Datawrapper dataset — into the bounded text a forecaster reads. Strip the
allow-listed markup that would otherwise spend the char budget on tags, then
truncate to a cap with a marker saying the snapshot is partial. Trafilatura owns
real HTML pages; nothing here touches them.

Split out of ``research.resolution_source`` because these five names carry
non-obvious invariants that read better away from the fetch machinery: both
truncators guarantee ``len(return) <= cap``, the CSV variant keeps BOTH ends so
the newest rows survive whichever direction the series runs, and the tag regex
carries a measured catastrophic-backtracking guard (3.4 s at 200 KiB in the
naive form, synchronously on the event loop).
"""

from __future__ import annotations

import re

# Tag names an allow-list, not `[A-Za-z]+`, and the character after the name
# must be `>`, `/`, or whitespace-then-an-attribute-assignment. Both halves are
# load-bearing: the naive `</?[A-Za-z][^>]*>` form eats a CSV cell reading
# `x <a and y > b` (measured on the 2026-08-26 diagnosis), and requiring an `=`
# in the attribute region is what keeps that same cell out of the allow-listed
# form — `<a and y >` names no attribute. A real tag either closes immediately
# (`<br/>`, `</a>`, `<td>`) or assigns something (`href=`, `style=`).
_HTML_TAG_NAMES: tuple[str, ...] = (
    "a",
    "b",
    "i",
    "u",
    "p",
    "em",
    "strong",
    "span",
    "div",
    "br",
    "td",
    "tr",
    "th",
    "table",
    "font",
    "small",
    "sub",
    "sup",
)
# The pre-`=` attribute region excludes `=` so there is exactly ONE way to reach the
# first delimiter. With `[^<>]*` on both sides of the `=`, a non-matching body (an
# allow-listed lookalike like `<b ` followed by an angle-bracket-free run of URL cells
# carrying query-string `=` signs) backtracks quadratically: 3.4s at 200 KiB, ~35 min
# at the 5 MiB response cap — synchronously on the event loop, wedging the sibling
# fetches past every wall timeout. Same matched language: any match via a later `=`
# is also a match via the first one, since the post-`=` region admits more `=`s.
_HTML_TAG_RE = re.compile(
    r"</?(?:" + "|".join(_HTML_TAG_NAMES) + r")(?:\s*/?>|\s+[^<>=]*=[^<>]*>)",
    re.IGNORECASE,
)

# An anchor whose inner text is empty carries its content in the href (a bare
# link cell), so the href is kept where the inner text would have been.
_EMPTY_ANCHOR_RE = re.compile(
    r"<a\s+[^<>]*href\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s<>]+))[^<>]*>\s*</a\s*>",
    re.IGNORECASE,
)


def strip_html_tags(text: str) -> str:
    """Remove allow-listed HTML tags from ``text``, keeping their inner text.

    For the RAW-body branches only (CSV / plain text) — trafilatura owns real
    HTML pages. Datawrapper poll tables embed a styled ``<a href=…>`` per
    pollster row, and 69% of one live tracker's 33k-char CSV was tag markup: at
    that run's actual 2,853-char budget, 9 rows survived with the tags against 30
    with them stripped (measured 2026-08-26). The tags carry nothing a forecaster
    reads — the pollster name inside them is the content — so the budget should
    buy rows.

    A no-op (byte-identical) on any text with no allow-listed tag in it, which
    covers every numeric tracker CSV checked (zero ``<`` characters).
    """
    if "<" not in text:
        return text
    without_bare_links = _EMPTY_ANCHOR_RE.sub(
        lambda m: m.group(1) or m.group(2) or m.group(3) or "",
        text,
    )
    return _HTML_TAG_RE.sub("", without_bare_links)


def _truncate_with_marker(text: str, cap: int, url: str) -> str:
    """Return ``text`` bounded at ``cap`` chars; on truncation, append a marker
    line naming the cap and URL so forecasters can tell the snapshot is partial.

    Invariant: ``len(return) <= cap``. When truncation fires, the emitted text
    is trimmed to ``cap - len(marker)`` before the marker is appended so the
    total stays within budget. If the marker itself is longer than the cap
    (pathologically small cap in tests), returns the raw truncation without
    the marker rather than emitting only-marker text.
    """
    if cap <= 0:
        # No budget at all — the caller's arithmetic (section allowance minus a lead line
        # minus a very long parent URL) can go negative. ``text[:cap]`` on a negative cap
        # returns nearly the WHOLE text while the invariant claims a bound, so a
        # zero-budget slot would have rendered a full page.
        return ""
    if len(text) <= cap:
        return text
    marker = f"\n[truncated at {cap} chars — full source at {url}]"
    if len(marker) >= cap:
        # Cap is too small to even fit the marker; degrade to plain truncation.
        return text[:cap]
    body_budget = cap - len(marker)
    return text[:body_budget].rstrip() + marker


def _truncate_csv_middle(text: str, cap: int, url: str) -> str:
    """Bound a CSV at ``cap`` chars, keeping the header + BOTH ends of the rows.

    The resolution-relevant rows are the most recent ones, but Datawrapper
    datasets run in either direction — the tracker model-average series are
    chronological (newest LAST) while the poll-input tables on the same pages
    are newest FIRST (observed live on both natesilver.net trackers,
    2026-08-25). Keeping both ends is ordering-agnostic: the newest rows
    survive whichever end they sit at, and only the middle is omitted. Plain
    head truncation would cut the current level off an ascending series — the
    stale-as-live failure in a different coat.

    Invariant: ``len(return) <= cap``. Degrades to plain head truncation when
    the text has too few lines to be row-shaped or the cap is too small to fit
    the header + marker + at least one row.
    """
    if len(text) <= cap:
        return text
    lines = text.rstrip("\n").split("\n")
    if len(lines) < 4:
        return _truncate_with_marker(text, cap, url)
    header = lines[0]
    rows = lines[1:]
    marker_template = "[... {} middle rows omitted — full data at {}]"
    # Reserve marker space at its worst-case width (all rows omitted).
    worst_marker = marker_template.format(len(rows), url)
    row_budget = cap - len(header) - len(worst_marker) - 4  # joining newlines
    if row_budget <= 0:
        return _truncate_with_marker(text, cap, url)
    head_budget = row_budget // 2
    head: list[str] = []
    used_head = 0
    for line in rows:
        cost = len(line) + 1
        if used_head + cost > head_budget:
            break
        head.append(line)
        used_head += cost
    tail: list[str] = []
    used_tail = 0
    for line in reversed(rows[len(head) :]):
        cost = len(line) + 1
        if used_head + used_tail + cost > row_budget:
            break
        tail.append(line)
        used_tail += cost
    if not head and not tail:
        return _truncate_with_marker(text, cap, url)
    tail.reverse()
    marker = marker_template.format(len(rows) - len(head) - len(tail), url)
    return "\n".join([header, *head, marker, *tail])
