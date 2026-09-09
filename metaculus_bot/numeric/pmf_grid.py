"""How a forecaster names a bin when a question is elicited per bin.

On an enumerable grid (``config.elicit_per_bin``) each forecaster declares one probability per bin,
keyed by a label, plus the reserved ``below_range`` / ``above_range`` keys where a bound is open.
The bins are the platform's right-closed intervals ``(edge_k, edge_{k+1}]``, the first also owning
its left edge; ``cdf[0]`` is the mass below the range and ``1 - cdf[N]`` the mass above it. The
edges are ``build_cdf_value_grid`` on the question's own axis, which reproduces the platform's
``continuous_range`` exactly, so a label here names the same bin the platform scores.

One label style per grid, chosen by :func:`pmf_grid`:

* ``day`` / ``week``: the ISO date of the first UTC calendar day the bin covers, the way the
  platform itself displays a day-granularity date question (post 651 renders ``2026-09-08`` through
  ``2026-09-19``, and the last label is the API's ``nominal_max``).
* ``timestamp``: ``start to end`` in ``YYYY-MM-DDTHH:MM:SSZ`` for a date grid whose edges fall at
  arbitrary times of day (the legacy 200-bin shape). No such grid is coarse enough to be elicited
  per bin in the corpus; the style exists so no input is unsupported.
* ``center``: the bin centre, on a linear quantity grid whose displayed lower bound is the first
  bin's centre (the discrete convention on both platforms: ``range_min = nominal_min - step / 2``);
  a count question labels its bins ``0``, ``1``, .... Only the LOWER offset is tested: 158 of 159
  linear Mantic discrete grids also put ``nominal_max`` half a step inside the range, but post 650
  declares it one step too high (100000 against a ``range_max`` of 99950), and a coarse count grid
  built that way would otherwise be labelled ``-0.5 to 0.5`` where ``0`` is the bin's only honest name.
* ``interval``: ``a to b`` in displayed values, for every other quantity grid (a ``zero_point``
  axis, or nominal bounds on the range edges as on the two coarse Series 1 numeric questions).

:func:`fold_bin_label` is the matching side: the extraction ladder folds both a model's key and the
grid's keys through it, so ``"7.0"``, ``" 7 "`` and ``"55,000"`` land on ``"7"`` and ``"55000"``, while
a non-ISO date spelling falls through to the later rungs like an unmatched multiple-choice key.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.constants import PMF_ABOVE_RANGE_KEY, PMF_BELOW_RANGE_KEY
from metaculus_bot.mc_processing import fold_option_label
from metaculus_bot.numeric.config import grid_bin_width
from metaculus_bot.numeric.date_axis import EpochDateQuestion, format_epoch
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.numeric.utils import nominal_bounds
from metaculus_bot.numeric.validation import resolve_zero_point

__all__ = ["BinLabelStyle", "PmfGrid", "fold_bin_label", "format_bin_value", "pmf_grid"]

BinLabelStyle = Literal["center", "interval", "day", "week", "timestamp"]

# The platform derives ``range_min = nominal_min - step / 2`` in float arithmetic (post 619: 77.15 from 77.2 and 0.1).
_CENTER_ALIGNMENT_RTOL = 1e-9

# The server's own PMF rounding precision: drops edge residue (``77.35000000000001``), keeps ``123456.7`` intact.
_LABEL_DECIMALS = 9

# The calendar-day styles are the platform's own granularity tokens; an empty granularity means edges at any time of day.
_CALENDAR_DAY_STYLES: dict[str, BinLabelStyle] = {"day": "day", "week": "week"}


@dataclass(frozen=True)
class PmfGrid:
    """The bins of one question as the forecaster sees them: a label per bin, in grid order."""

    labels: tuple[str, ...]
    edges: tuple[float, ...]
    open_lower_bound: bool
    open_upper_bound: bool
    style: BinLabelStyle

    @property
    def keys(self) -> tuple[str, ...]:
        """The block's keys in prompt order: ``below_range`` (open lower) + labels + ``above_range`` (open upper)."""
        below = (PMF_BELOW_RANGE_KEY,) if self.open_lower_bound else ()
        above = (PMF_ABOVE_RANGE_KEY,) if self.open_upper_bound else ()
        return below + self.labels + above


def format_bin_value(value: float) -> str:
    """Render a quantity as the platform displays it: no trailing zeros, no float residue, never ``-0``.

    The same idiom as ``research/number_format.format_decimal_value``, at the server's 9-decimal PMF
    rounding precision instead of FRED's published 6; not called directly because the import-linter
    contract keeps ``numeric`` independent of ``research`` (the ``ts_render._fmt`` precedent).
    """
    rendered = f"{round(value, _LABEL_DECIMALS):.{_LABEL_DECIMALS}f}".rstrip("0").rstrip(".")
    return "0" if rendered == "-0" else rendered


def fold_bin_label(label: str) -> str:
    """The comparison form of a bin key: the option-label fold, commas dropped, a number re-rendered canonically."""
    folded = fold_option_label(label).replace(",", "")
    try:
        numeric = float(folded)
    except ValueError:
        return folded
    return format_bin_value(numeric)


def pmf_grid(view: NumericQuestion) -> PmfGrid:
    """The labelled grid of ``view``, the numeric-pipeline view of a numeric, discrete or date question."""
    zero_point = resolve_zero_point(view)
    edges = tuple(
        float(edge) for edge in build_cdf_value_grid(view.lower_bound, view.upper_bound, zero_point, view.cdf_size)
    )
    style, labels = _label_bins(view, edges, zero_point)
    grid = PmfGrid(
        labels=labels,
        edges=edges,
        open_lower_bound=view.open_lower_bound,
        open_upper_bound=view.open_upper_bound,
        style=style,
    )
    _require_distinct_keys(grid, view)
    return grid


def _require_distinct_keys(grid: PmfGrid, view: NumericQuestion) -> None:
    """A guard, not a repair: bins narrower than the 9-decimal label precision fold onto one key, and no Mantic grid is that fine."""
    folded = Counter(fold_bin_label(key) for key in grid.keys)
    colliding = sorted(key for key, count in folded.items() if count > 1)
    if colliding:
        raise ValueError(
            f"pmf_grid: question {view.id_of_question} has {len(grid.labels)} bins whose keys fold onto the same key "
            f"{colliding}; bins this narrow cannot be named per bin"
        )


def _label_bins(
    view: NumericQuestion, edges: tuple[float, ...], zero_point: float | None
) -> tuple[BinLabelStyle, tuple[str, ...]]:
    if isinstance(view, EpochDateQuestion):
        granularity = view.date_granularity
        calendar_style = _CALENDAR_DAY_STYLES.get(granularity)
        if calendar_style is not None:
            return calendar_style, tuple(format_epoch(edge, granularity) for edge in edges[:-1])
        return "timestamp", _interval_labels(edges, lambda edge: format_epoch(edge, granularity))
    step = grid_bin_width(view.lower_bound, view.upper_bound, view.cdf_size)
    if zero_point is None and _is_center_aligned(view, step):
        return "center", tuple(format_bin_value(view.lower_bound + (k + 0.5) * step) for k in range(len(edges) - 1))
    return "interval", _interval_labels(edges, format_bin_value)


def _is_center_aligned(view: NumericQuestion, step: float) -> bool:
    """True when the displayed lower bound sits half a step inside the range, the discrete convention."""
    _, nominal_lower = nominal_bounds(view)
    tolerance = _CENTER_ALIGNMENT_RTOL * max(1.0, view.upper_bound - view.lower_bound)
    return abs((nominal_lower - view.lower_bound) - step / 2) <= tolerance


def _interval_labels(edges: tuple[float, ...], render: Callable[[float], str]) -> tuple[str, ...]:
    return tuple(f"{render(low)} to {render(high)}" for low, high in pairwise(edges))
