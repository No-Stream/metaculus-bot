"""Era-bucketed numeric-width / calibration monitor (READ-ONLY, free).

Tracks how wide the bot's published numeric distributions are, and how well
that width is calibrated, split by config era. The bot has historically
oscillated between too-wide and too-narrow numeric forecasts:

  * Until 2026-05-12 the pipeline INTENTIONALLY widened tails (``k_tail=1.25``
    in the tail-widening pass).
  * On 2026-05-12 widening was turned off (``k_tail=1.0``, identity) after a
    calibration study showed the widened tails were too fat.
  * On 2026-07-17 the Time-Series-Anchor prompt clause landed. It pushes
    "sharpen, don't widen" (published low-tail coverage was ~0.03 vs a 0.10
    target — badly too wide), so the forward risk flips toward over-sharpening.
    This monitor is the loop-closer for that transition.

Per era it reports, on the bot's PUBLISHED 201-point CDF:

  * central-80% coverage  = fraction of PIT in [0.10, 0.90]  (calibrated 0.80)
  * central-50% coverage  = fraction of PIT in [0.25, 0.75]  (calibrated 0.50)
    both with Beta-Binomial / Jeffreys-prior 95% CIs.
  * cov@10 = P(PIT <= 0.10)  (calibrated 0.10; low-tail coverage)
  * cov@50 = P(PIT <= 0.50)  (calibrated 0.50; directional bias / below-median)
  * cov@90 = P(PIT <= 0.90)  (calibrated 0.90; high-tail coverage)
  * PIT std (calibrated Uniform(0,1) std = 1/sqrt(12) ~= 0.289; smaller => PIT
    piled in the center => distributions too WIDE; larger => piled at the
    extremes => too NARROW).
  * median relative band width = median over questions of (P90 - P10) / |P50|,
    read off the published CDF. This is the RAW sharpness metric and does not
    depend on resolutions — it answers "how wide are we, in absolute terms",
    complementing the coverage metrics which answer "is that width calibrated".

PIT is F_bot(resolution) evaluated on the canonical Metaculus value grid
(``build_cdf_value_grid``); out-of-bounds resolutions map to PIT 0.0 (below
lower) / 1.0 (above upper). Method mirrors
``scratch/calibration_audit_2026-07-16/mc_numeric_calibration.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.performance_analysis.collector import build_performance_dataset, load_dataset
from metaculus_bot.performance_analysis.scaling import grid_zero_point as _grid_zero_point

logger: logging.Logger = logging.getLogger(__name__)

NUMERIC_TYPES: tuple[str, ...] = ("numeric", "discrete")

# Calibrated reference values, surfaced in the legend so a reader knows which
# direction "off" points.
UNIFORM_PIT_STD: float = 1.0 / np.sqrt(12.0)  # ~0.2887


@dataclass(frozen=True)
class Era:
    """A config era: records whose bot_comment_created_at falls in
    ``[start, end)`` belong to this era. ``None`` bounds are open (-inf / +inf).
    """

    label: str
    start: datetime | None
    end: datetime | None

    def contains(self, dt: datetime) -> bool:
        if self.start is not None and dt < self.start:
            return False
        if self.end is not None and dt >= self.end:
            return False
        return True


# Config-flip boundaries that plausibly shift the numeric width distribution.
# These are the ONLY width-relevant flips (per CLAUDE.md era-bucketing guidance:
# bucket by pipeline-behavior changes, not every git hash).
WIDENING_FLIP = datetime(2026, 5, 12, tzinfo=timezone.utc)  # k_tail 1.25 -> 1.0
TS_ANCHOR_ENABLE = datetime(2026, 7, 17, tzinfo=timezone.utc)  # "sharpen, don't widen" clause landed


def default_eras() -> list[Era]:
    """The three width-relevant config eras, oldest first.

    ``ts_anchor`` is the active era from 2026-07-17 onward: the timeseries-anchor
    "sharpen, don't widen" clause is enabled in prod as of that date, so records
    published on/after it land in this bucket instead of contaminating the
    ``widening_off`` baseline. (It stays empty only for as long as no post-enable
    question has resolved and been pulled.)
    """
    return [
        Era("widening_on (k_tail=1.25)", None, WIDENING_FLIP),
        Era("widening_off (k_tail=1.0)", WIDENING_FLIP, TS_ANCHOR_ENABLE),
        Era("ts_anchor (sharpen)", TS_ANCHOR_ENABLE, None),
    ]


NO_TIMESTAMP_LABEL = "no_timestamp"


def _parse_created_at(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def assign_era(record: dict, eras: list[Era]) -> str:
    """Return the era label for a record, or ``NO_TIMESTAMP_LABEL`` when the
    bot-comment timestamp is missing/unparseable (can't be era-attributed)."""
    dt = _parse_created_at(record.get("bot_comment_created_at"))
    if dt is None:
        return NO_TIMESTAMP_LABEL
    for era in eras:
        if era.contains(dt):
            return era.label
    return NO_TIMESTAMP_LABEL


def jeffreys_ci(k: int, n: int, cl: float = 0.95) -> tuple[float, float, float]:
    """Beta-Binomial posterior mean + equal-tailed CI under a Jeffreys(0.5, 0.5)
    prior. Mirrors ``bb`` in mc_numeric_calibration.py."""
    a = 0.5 + k
    b = 0.5 + (n - k)
    mean = a / (a + b)
    lo = float(stats.beta.ppf((1 - cl) / 2, a, b))
    hi = float(stats.beta.ppf(1 - (1 - cl) / 2, a, b))
    return mean, lo, hi


def _cdf_and_grid(record: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (cdf, value_grid) for a numeric/discrete record, or None if the
    record lacks the bounds / CDF needed to build the grid.

    Prefers the API's grid-exact ``scaling.continuous_range`` when present (it
    already encodes log-vs-linear spacing, so no scale has to be re-derived from
    ``zero_point``); falls back to reconstructing via ``build_cdf_value_grid``
    with a zero_point interpretation that handles the ``zero_point == 0`` log
    case (see ``_grid_zero_point``).
    """
    cdf = record.get("our_forecast_values")
    scaling = record.get("scaling") or {}
    lo, hi = scaling.get("range_min"), scaling.get("range_max")
    if cdf is None or lo is None or hi is None or len(cdf) < 3:
        return None
    cdf_arr = np.asarray(cdf, dtype=float)
    api_grid = scaling.get("continuous_range")
    if api_grid is not None and len(api_grid) == len(cdf):
        return cdf_arr, np.asarray(api_grid, dtype=float)
    zp = _grid_zero_point(scaling.get("zero_point"), float(lo))
    grid = build_cdf_value_grid(float(lo), float(hi), zp, len(cdf))
    return cdf_arr, grid


def compute_pit(record: dict) -> float | None:
    """PIT = F_bot(resolution) on the canonical value grid. Out-of-bounds
    resolutions map to 0.0 / 1.0. Returns None when the record can't be scored."""
    built = _cdf_and_grid(record)
    if built is None:
        return None
    cdf, grid = built
    res = record.get("resolution_parsed")
    if res == "below_lower_bound":
        return 0.0
    if res == "above_upper_bound":
        return 1.0
    if isinstance(res, (int, float)) and not isinstance(res, bool):
        return float(np.interp(float(res), grid, cdf))
    return None


def relative_band_width(record: dict, *, median_floor: float = 1e-9) -> float | None:
    """(P90 - P10) / |P50| read off the published CDF (resolution-independent).

    Returns None when the record lacks a usable CDF, or when |P50| is below
    ``median_floor`` (the ratio blows up for questions centred on ~0, e.g. a
    signed-change quantity; those are excluded and counted rather than
    poisoning the median)."""
    built = _cdf_and_grid(record)
    if built is None:
        return None
    cdf, grid = built
    # Invert the (monotone) CDF: value at quantile q = interp of grid over cdf.
    p10, p50, p90 = (float(np.interp(q, cdf, grid)) for q in (0.10, 0.50, 0.90))
    if abs(p50) < median_floor:
        return None
    return (p90 - p10) / abs(p50)


@dataclass
class EraWidthMetrics:
    label: str
    n_pit: int
    n_eff: int
    n_width: int
    n_oob_low: int
    n_oob_high: int
    cov80: tuple[float, float, float]
    cov50: tuple[float, float, float]
    cov_at_10: float
    cov_at_50: float
    cov_at_90: float
    pit_std: float
    mean_pit: float
    median_rel_width: float | None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n_pit": self.n_pit,
            "n_eff": self.n_eff,
            "n_width": self.n_width,
            "n_oob_low": self.n_oob_low,
            "n_oob_high": self.n_oob_high,
            "cov80": {"mean": self.cov80[0], "lo": self.cov80[1], "hi": self.cov80[2]},
            "cov50": {"mean": self.cov50[0], "lo": self.cov50[1], "hi": self.cov50[2]},
            "cov_at_10": self.cov_at_10,
            "cov_at_50": self.cov_at_50,
            "cov_at_90": self.cov_at_90,
            "pit_std": self.pit_std,
            "mean_pit": self.mean_pit,
            "median_rel_width": self.median_rel_width,
        }


def _n_effective_clusters(post_ids: list[object]) -> int:
    """Count distinct question families for the CI's effective sample size.

    Records sharing a ``post_id`` are one correlated family (same series/window,
    multiple sub-questions per post). A record with no ``post_id`` (``None``) is
    treated as its own family — assigned a unique sentinel by position so it is
    never merged with another None-post record — since we can't prove it shares
    a family with anything else.
    """
    clusters: set[object] = set()
    for i, pid in enumerate(post_ids):
        clusters.add(pid if pid is not None else f"__no_post_{i}")
    return len(clusters)


def compute_era_metrics(label: str, records: list[dict]) -> EraWidthMetrics | None:
    """Compute width/calibration metrics for one era's records. Returns None if
    no numeric/discrete records in the era yield a PIT."""
    pits: list[float] = []
    pit_post_ids: list[object] = []
    widths: list[float] = []
    for r in records:
        if r.get("type") not in NUMERIC_TYPES:
            continue
        pit = compute_pit(r)
        if pit is not None:
            pits.append(pit)
            pit_post_ids.append(r.get("post_id"))
        w = relative_band_width(r)
        if w is not None:
            widths.append(w)

    if not pits:
        return None

    arr = np.asarray(pits, dtype=float)
    n = len(arr)
    n_oob_low = int((arr == 0.0).sum())
    n_oob_high = int((arr == 1.0).sum())
    cov80_k = int(((arr >= 0.10) & (arr <= 0.90)).sum())
    cov50_k = int(((arr >= 0.25) & (arr <= 0.75)).sum())

    # Coverage CIs use n_eff (distinct post_ids), not the raw question count:
    # ~62% of records share a post (same series/window, multiple sub-questions
    # per post) and are correlated, so a naive n=question-count Jeffreys CI runs
    # ~26% too narrow. Cluster on post_id only — the one grouping key already on
    # every record; a record missing a post_id counts as its own cluster (via a
    # unique sentinel) so it is never merged with another. The point estimate is
    # unchanged (cov_k / n); only the CI width widens to reflect n_eff clusters,
    # via jeffreys_ci(round(cov_k * n_eff / n), n_eff).
    n_eff = _n_effective_clusters(pit_post_ids)
    cov80 = jeffreys_ci(round(cov80_k * n_eff / n), n_eff)
    cov50 = jeffreys_ci(round(cov50_k * n_eff / n), n_eff)

    return EraWidthMetrics(
        label=label,
        n_pit=n,
        n_eff=n_eff,
        n_width=len(widths),
        n_oob_low=n_oob_low,
        n_oob_high=n_oob_high,
        cov80=cov80,
        cov50=cov50,
        cov_at_10=float((arr <= 0.10).mean()),
        cov_at_50=float((arr <= 0.50).mean()),
        cov_at_90=float((arr <= 0.90).mean()),
        pit_std=float(arr.std()),
        mean_pit=float(arr.mean()),
        median_rel_width=(float(np.median(widths)) if widths else None),
    )


def compute_all_eras(data: list[dict], eras: list[Era] | None = None) -> list[EraWidthMetrics]:
    """Bucket records by era and compute per-era metrics. Eras with no scorable
    numeric records are omitted. Emits an ``all`` row spanning every era."""
    if eras is None:
        eras = default_eras()
    order = [e.label for e in eras] + [NO_TIMESTAMP_LABEL]
    buckets: dict[str, list[dict]] = {lbl: [] for lbl in order}
    numeric_records: list[dict] = []
    for r in data:
        if r.get("type") not in NUMERIC_TYPES:
            continue
        numeric_records.append(r)
        buckets[assign_era(r, eras)].append(r)

    results: list[EraWidthMetrics] = []
    for lbl in order:
        m = compute_era_metrics(lbl, buckets[lbl])
        if m is not None:
            results.append(m)
    overall = compute_era_metrics("all", numeric_records)
    if overall is not None:
        results.append(overall)
    return results


def _fmt_ci(ci: tuple[float, float, float]) -> str:
    m, lo, hi = ci
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def render_markdown(metrics: list[EraWidthMetrics]) -> str:
    """Compact markdown table of the per-era width/calibration metrics."""
    lines: list[str] = []
    lines.append("## Numeric width / calibration monitor (per config era)")
    lines.append("")
    lines.append(
        "Calibrated targets: cov80=0.80, cov50=0.50, cov@10=0.10, cov@50=0.50, "
        f"cov@90=0.90, PIT std={UNIFORM_PIT_STD:.3f}. "
        "PIT std below target => too WIDE; above => too NARROW. "
        "cov@10 below 0.10 => low tail too wide; median rel width = (P90-P10)/|P50| (raw sharpness). "
        "cov80/cov50 CIs are computed at n_eff (distinct post_ids), not n: questions cluster into "
        "correlated families (multiple sub-questions per post), so a naive n-based CI is too narrow."
    )
    lines.append("")
    header = (
        "| era | n | n_eff | cov80 [95% CI] | cov50 [95% CI] | cov@10 | cov@50 | cov@90 "
        "| PIT std | mean PIT | med rel width (n) | OOB lo/hi |"
    )
    sep = "|" + "|".join(["---"] * 12) + "|"
    lines.append(header)
    lines.append(sep)
    for m in metrics:
        rel = f"{m.median_rel_width:.3f} ({m.n_width})" if m.median_rel_width is not None else f"n/a ({m.n_width})"
        lines.append(
            f"| {m.label} | {m.n_pit} | {m.n_eff} | {_fmt_ci(m.cov80)} | {_fmt_ci(m.cov50)} "
            f"| {m.cov_at_10:.3f} | {m.cov_at_50:.3f} | {m.cov_at_90:.3f} "
            f"| {m.pit_std:.3f} | {m.mean_pit:.3f} | {rel} | {m.n_oob_low}/{m.n_oob_high} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Era-bucketed numeric width / calibration monitor (read-only)")
    parser.add_argument(
        "--cached",
        default="scratch/coherence_2026-07-15/perf_all_tagged.json",
        help="Path to a cached performance dataset JSON (list of records). Default: %(default)s",
    )
    parser.add_argument(
        "--tournament",
        default=None,
        help="Instead of --cached, pull a tournament live (read-only, free). Overrides --cached when set.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to also write the metrics as JSON.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)

    if args.tournament:
        data = build_performance_dataset(tournament=args.tournament)
    else:
        data = load_dataset(args.cached)

    metrics = compute_all_eras(data)
    print(render_markdown(metrics))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump([m.to_dict() for m in metrics], f, indent=2)
        logger.info(f"Wrote {len(metrics)} era rows to {args.output_json}")


if __name__ == "__main__":
    main()
