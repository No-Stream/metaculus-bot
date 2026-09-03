"""The one markdown-table emitter for the residual-analysis CLIs.

``width_monitor``, ``outer_tail`` and the clip-threshold report all print into the same
round's output, so table syntax lives here rather than in each of them. Cells arrive already
formatted: the callers own every rendering convention (``n/a`` for an unmeasured quantity,
``identity`` for a do-nothing row), this module only lays them out.
"""

from __future__ import annotations

from collections.abc import Sequence


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Header line, separator, then one line per row, with the column count taken from ``header``."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines
