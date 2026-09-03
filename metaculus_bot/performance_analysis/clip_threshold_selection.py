"""Selection-aware readings of the clip sweep: argmax ties, out-of-bag value, nesting, power.

Every row of the sweep is exact for its own candidate, but the ARGMAX over the grid is a
choice, and a choice made on a small window carries a winner's curse the row's own CI does
not see. This module holds the corrections for that: which censored candidates tied with the
winner and lost only for being censored, what the fit-then-apply policy is worth out of bag,
which questions the nested windows are re-counting, and the binomial power behind a
"0 hits in n" reading.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from metaculus_bot.performance_analysis.clip_threshold_sweep import (
    ARGMAX_TIE_ATOL,
    BOOTSTRAP_CL,
    BOOTSTRAP_SEED,
    ClipRecord,
    ClipSide,
    SweepRow,
    argmax_row,
    clip_delta,
)

# The out-of-bag check refits the argmax on every resample, so it is dearer per iteration
# than the plain CI and runs at half the draws.
OOB_BOOTSTRAP_B: int = 2000


def censored_rows_at_argmax_score(rows: Sequence[SweepRow]) -> list[SweepRow]:
    """Censored rows whose ``sum_delta`` equals the argmax's, so the reader sees WHY they lost.

    A candidate excluded for censoring rather than for scoring worse is a different fact from
    a strict win, and a bare ``tied: -`` hides it.
    """
    best = argmax_row(rows)
    if best is None:
        return []
    return sorted(
        (row for row in rows if not row.exact and abs(row.sum_delta - best.sum_delta) <= ARGMAX_TIE_ATOL),
        key=lambda row: row.c,
    )


@dataclass(frozen=True, slots=True)
class OobArgmax:
    """Selection-corrected value of "fit the argmax floor, then apply it", for one window.

    The CI on the in-window argmax row ignores that the row was CHOSEN over the grid, so on a
    small window it can read as a real gain when the choice was noise. This refits the argmax
    on each bootstrap resample and scores the fitted candidate on the records the resample
    left out, which is the honest price of the selection.
    """

    window: str
    n: int
    in_window_c: float | None
    in_window_mean_delta: float | None
    oob_mean_delta: float | None
    """Mean over iterations of the fitted candidate's mean delta on the out-of-bag records."""
    oob_ci_lo: float | None
    oob_ci_hi: float | None
    n_iterations: int
    n_candidates: int
    """How many exact candidates the fit chose among (censored candidates never compete)."""

    @property
    def shrinkage(self) -> float | None:
        """In-window mean minus out-of-bag mean: the part of the apparent gain that is selection."""
        if self.in_window_mean_delta is None or self.oob_mean_delta is None:
            return None
        return self.in_window_mean_delta - self.oob_mean_delta

    def to_dict(self) -> dict:
        return asdict(self) | {"shrinkage": self.shrinkage}


def oob_argmax(records: Sequence[ClipRecord], rows: Sequence[SweepRow], *, side: ClipSide, window: str) -> OobArgmax:
    """Bootstrap the argmax-then-apply policy over ``records`` (see :class:`OobArgmax`).

    ``rows`` are the window's sweep rows for this side, one per grid candidate; only EXACT
    rows compete, matching :func:`argmax_rows`. Ties inside a resample go to the smallest
    ``c``, again matching the in-window rule.
    """
    exact = sorted((row for row in rows if row.exact), key=lambda row: row.c)
    best = argmax_row(rows)
    n = len(records)
    empty = OobArgmax(
        window=window,
        n=n,
        in_window_c=(best.c if best else None),
        in_window_mean_delta=(best.mean_delta if best else None),
        oob_mean_delta=None,
        oob_ci_lo=None,
        oob_ci_hi=None,
        n_iterations=0,
        n_candidates=len(exact),
    )
    if not exact or n < 2:
        return empty
    deltas = np.array([[clip_delta(r, row.c, side=side).delta for r in records] for row in exact], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    oob_means: list[float] = []
    for _ in range(OOB_BOOTSTRAP_B):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        in_bag = deltas @ counts
        fitted = int(np.flatnonzero(in_bag >= in_bag.max() - ARGMAX_TIE_ATOL)[0])
        out_of_bag = counts == 0
        if not out_of_bag.any():
            continue
        oob_means.append(float(deltas[fitted, out_of_bag].mean()))
    if not oob_means:
        return empty
    half = (1.0 - BOOTSTRAP_CL) / 2.0
    lo, hi = np.quantile(oob_means, [half, 1.0 - half])
    return OobArgmax(
        window=window,
        n=n,
        in_window_c=(best.c if best else None),
        in_window_mean_delta=(best.mean_delta if best else None),
        oob_mean_delta=float(np.mean(oob_means)),
        oob_ci_lo=float(lo),
        oob_ci_hi=float(hi),
        n_iterations=len(oob_means),
        n_candidates=len(exact),
    )


def affected_question_ids(records: Sequence[ClipRecord], c: float, *, side: ClipSide) -> frozenset[str]:
    """Which questions candidate ``c`` moves; the unit of the nesting check."""
    return frozenset(clip.question_id for clip in (clip_delta(r, c, side=side) for r in records) if clip.affected)


def binomial_cdf(k: int, n: int, rate: float) -> float:
    """``P(X <= k)`` for ``X ~ Binomial(n, rate)``, with ``k`` clipped at ``n``; 1.0 on an empty sample."""
    if n == 0:
        return 1.0
    return float(stats.binom.cdf(min(k, n), n, rate))
