"""Lookback windows for the clip-threshold sweep, and the complement each one may fit on.

Two families of window, and the difference is the point. The suffix and dated windows
(``all``, ``last_N``, ``last_90d``, ``current_clamp_regime``, ``triple_era``) are NESTED:
each is a subset of ``all``, so their rows are one measurement re-counted at several sample
sizes and cannot disagree. The era slices (``era_pre_flip``, ``era_post_flip``, with
``triple_era`` as the third) are DISJOINT: they partition the dated records, so agreement
across them carries information the nested rows cannot. Config eras are keyed on the
roster/pipeline boundaries the rest of ``performance_analysis`` uses, for both types.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from metaculus_bot.performance_analysis.analysis import (
    B4E9DF0_MERGED_AT,
    FT_0292_MERGED_AT,
    WIDENING_FLIP_MERGED_AT,
)
from metaculus_bot.performance_analysis.clip_threshold_sweep import BINARY, MULTIPLE_CHOICE, ClipRecord

LOOKBACK_DAYS: int = 90

WINDOW_ALL = "all"
WINDOW_CURRENT_CLAMP = "current_clamp_regime"
WINDOW_TRIPLE_ERA = "triple_era"
# Derived so a change to LOOKBACK_DAYS cannot leave a window labelled with the old span.
WINDOW_LAST_90D = f"last_{LOOKBACK_DAYS}d"

ERA_WINDOW_PREFIX = "era_"
WINDOW_ERA_PRE_FLIP = f"{ERA_WINDOW_PREFIX}pre_flip"
WINDOW_ERA_POST_FLIP = f"{ERA_WINDOW_PREFIX}post_flip"

# Suffix sizes per type. MC carries a last_50 because its whole archive is under 100
# records, so last_100 and above are the full set and must say so (``Window.oversize``).
LAST_N_BY_TYPE: dict[str, tuple[int, ...]] = {
    BINARY: (300, 200, 100),
    MULTIPLE_CHOICE: (300, 200, 100, 50),
}

# The start of the clamp regime now in force, per type; the sweep's ``_CLAMP_HISTORY`` asserts
# its newest row against the live constants, so a constant change has to land in both places.
_CLAMP_REGIME_START: dict[str, datetime] = {
    BINARY: WIDENING_FLIP_MERGED_AT,
    MULTIPLE_CHOICE: FT_0292_MERGED_AT,
}

# Any instant works for label derivation; the windows' membership is what depends on the clock.
_LABEL_PROBE_INSTANT = datetime(2000, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class Window:
    """A lookback window plus the records a fit is allowed to see (its complement)."""

    label: str
    records: tuple[ClipRecord, ...]
    complement: tuple[ClipRecord, ...]
    requested_n: int | None
    oversize: bool
    """The window asked for more records than the type has, so it IS ``all`` and says so."""
    start: datetime | None
    end: datetime | None = None
    """Exclusive upper bound; set only on the disjoint era slices."""

    @property
    def is_era_slice(self) -> bool:
        return self.label.startswith(ERA_WINDOW_PREFIX)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n": len(self.records),
            "n_complement": len(self.complement),
            "requested_n": self.requested_n,
            "oversize": self.oversize,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "is_era_slice": self.is_era_slice,
        }


def _dated_between(
    records: Sequence[ClipRecord], start: datetime | None, end: datetime | None
) -> tuple[ClipRecord, ...]:
    return tuple(
        r
        for r in records
        if r.created_at is not None and (start is None or r.created_at >= start) and (end is None or r.created_at < end)
    )


def build_windows(
    records: Sequence[ClipRecord],
    *,
    question_type: str,
    as_of: datetime,
    last_n: tuple[int, ...] | None = None,
) -> list[Window]:
    """The type's windows, plus each window's complement (the records a fit may see).

    Undated records live in ``all`` only: no dated window can honestly claim them, and the
    header reports how many there are so their absence is visible rather than inferred. The
    complement is likewise dated-only, which is why it can be two records smaller than a
    naive set difference against ``all``.
    """
    sizes = LAST_N_BY_TYPE[question_type] if last_n is None else last_n
    dated = [r for r in records if r.created_at is not None]
    windows: list[Window] = [
        Window(label=WINDOW_ALL, records=tuple(records), complement=(), requested_n=None, oversize=False, start=None)
    ]
    for size in sizes:
        inside = dated[-size:] if size < len(dated) else dated
        windows.append(
            Window(
                label=f"last_{size}",
                records=tuple(inside),
                complement=tuple(dated[: len(dated) - len(inside)]),
                requested_n=size,
                oversize=size > len(dated),
                start=(inside[0].created_at if inside else None),
            )
        )
    for label, start in (
        (WINDOW_LAST_90D, as_of - timedelta(days=LOOKBACK_DAYS)),
        (WINDOW_CURRENT_CLAMP, _CLAMP_REGIME_START[question_type]),
        (WINDOW_TRIPLE_ERA, B4E9DF0_MERGED_AT),
    ):
        windows.append(
            Window(
                label=label,
                records=_dated_between(dated, start, None),
                complement=_dated_between(dated, None, start),
                requested_n=None,
                oversize=False,
                start=start,
            )
        )
    for label, era_start, era_end in (
        (WINDOW_ERA_PRE_FLIP, None, WIDENING_FLIP_MERGED_AT),
        (WINDOW_ERA_POST_FLIP, WIDENING_FLIP_MERGED_AT, B4E9DF0_MERGED_AT),
    ):
        windows.append(
            Window(
                label=label,
                records=_dated_between(dated, era_start, era_end),
                complement=() if era_start is None else _dated_between(dated, None, era_start),
                requested_n=None,
                oversize=False,
                start=era_start,
                end=era_end,
            )
        )
    return windows


def window_labels(question_type: str) -> tuple[str, ...]:
    """Every window label this type reports, in output order.

    Derived from :func:`build_windows` over an empty cohort rather than listed a second time:
    the report diffs these labels against the windows that carry records to print its
    "Empty windows" audit, so a label declared in one place and not the other would either
    hide a silently-absent window or print a phantom one. Labels depend on neither the
    records nor ``as_of``.
    """
    return tuple(w.label for w in build_windows((), question_type=question_type, as_of=_LABEL_PROBE_INSTANT))


def nested_windows(windows: Sequence[Window]) -> list[Window]:
    """The windows that are subsets of ``all`` (everything but the disjoint era slices)."""
    return [w for w in windows if not w.is_era_slice]
