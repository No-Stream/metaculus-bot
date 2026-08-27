"""Chart-image side-channel for the time-series-anchor provider (Phase B).

Renders a single 800x400 PNG showing what a forecaster's vision model sees when
the anchor image is enabled: the resolution series' last ~2 years, a vertical
rule at the forecast time (``as_of``), and a shaded P10-P90 band projected as a
horizontal ribbon to the horizon end with the P50 marked.

Ported from ``scratch/ts_anchor_replay_2026-07-16/make_charts.py`` but simplified
for prod: NO resolution marker (prod has no realized value at forecast time), and
the series/band come in as plain arguments rather than being re-derived from a
replay results JSON.

Rendering uses the matplotlib **Agg backend via the OO API** (``Figure`` +
``FigureCanvasAgg``, never pyplot) with bundled DejaVu fonts only — no global
pyplot state, so it is safe to call from an ``asyncio.to_thread`` worker without
the pyplot global-figure-registry race. Deterministic across machines.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib as mpl
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Bundled DejaVu fonts only — deterministic across machines, no font-cache
# surprises, and the same choice make_charts.py made for the replay corpus.
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.unicode_minus"] = False

FIG_W_IN, FIG_H_IN, DPI = 8.0, 4.0, 100  # 800x400 px
LOOKBACK_DAYS = 730  # last ~2y of the series shown

SERIES_COLOR = "#2c3e50"
BAND_COLOR = "#3498db"
AS_OF_COLOR = "#7f8c8d"
P50_COLOR = "#1f6dad"

_TITLE_LIMIT = 72


def _truncate_title(title: str, limit: int = _TITLE_LIMIT) -> str:
    return title if len(title) <= limit else title[: limit - 1] + "…"


def render_anchor_chart(
    series: pd.Series,
    *,
    as_of: pd.Timestamp,
    horizon_end: pd.Timestamp,
    band: tuple[float, float, float],
    title: str,
) -> str:
    """Render the anchor chart and return it as a base64-encoded PNG string.

    ``series`` is the full resolution series (only the last ~2 years before
    ``as_of`` are plotted). ``band`` is (P10, P50, P90) in the series' units,
    drawn as a horizontal ribbon from ``as_of`` to ``horizon_end``. The returned
    string is bare base64 (no ``data:`` URI prefix) — the caller wraps it in
    ``VisionMessageData(b64_image=...)`` which prepends the data URI itself.
    """
    as_of_ts = pd.Timestamp(as_of)
    horizon_ts = pd.Timestamp(horizon_end)
    index = pd.DatetimeIndex(series.index)
    recent = series[index >= as_of_ts - pd.Timedelta(days=LOOKBACK_DAYS)]
    if recent.size < 2:
        recent = series.iloc[-min(series.size, 60) :]

    p10, p50, p90 = band

    fig = Figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    canvas = FigureCanvasAgg(fig)
    # matplotlib's Axes stubs type axvline/plot x-args as float, but they accept
    # datetime/Timestamp at runtime via unit conversion (the replay corpus's
    # make_charts.py relied on the same). Typing ``ax`` as Any sidesteps the
    # over-strict stub without contorting the call sites.
    ax: Any = fig.add_subplot(111)

    ax.plot(recent.index, recent.to_numpy(dtype="float64"), color=SERIES_COLOR, lw=1.1, label="series")
    ax.axvline(as_of_ts, color=AS_OF_COLOR, ls="--", lw=1.0, label="forecast time")
    ax.fill_between(
        [as_of_ts, horizon_ts],
        [p10, p10],
        [p90, p90],
        color=BAND_COLOR,
        alpha=0.22,
        label="anchor P10–P90",  # noqa: RUF001  # en dash is deliberate range typography in the chart legend
    )
    ax.plot([as_of_ts, horizon_ts], [p50, p50], color=P50_COLOR, lw=1.4, ls=":", label="anchor P50")

    _finish_axes(ax, fig, recent, band=band, title=title)

    buf = io.BytesIO()
    canvas.print_png(buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _finish_axes(ax, fig: Figure, recent: pd.Series, *, band: tuple[float, float, float], title: str) -> None:
    ax.set_title(_truncate_title(title), fontsize=9, loc="left")
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(visible=True, alpha=0.25, lw=0.5)
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")

    # Keep the projected band in view even when it sits outside the recent range.
    ys = list(band)
    lo_data = float(recent.min())
    hi_data = float(recent.max())
    lo, hi = min(lo_data, *ys), max(hi_data, *ys)
    pad = 0.06 * (hi - lo if hi > lo else abs(hi) + 1.0)
    ax.set_ylim(lo - pad, hi + pad)

    ax.legend(fontsize=7, loc="best", framealpha=0.85)
    fig.subplots_adjust(bottom=0.2, left=0.1, right=0.97, top=0.9)
