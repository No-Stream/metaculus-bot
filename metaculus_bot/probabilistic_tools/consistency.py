from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

from metaculus_bot.probabilistic_tools.base_rate import implied_likelihood_ratio
from metaculus_bot.probabilistic_tools.distributions import (
    FitType,
    LognormalFit,
    eval_cdf,
    fit_lognormal_from_percentiles,
    fit_normal_from_percentiles,
    fit_student_t_from_percentiles,
)

logger = logging.getLogger(__name__)


FamilyLiteral = Literal["normal", "lognormal", "student_t", "skew_normal", "beta", "other"]
EvidenceStrength = Literal["strong", "moderate", "weak", "none"]

# Log-likelihood-ratio bands for declared-evidence-strength consistency checks.
# These follow a Jeffreys-style scheme (ln(BF) bands of roughly 1, 3, 10) with
# a custom tightening for "none"-evidence at log(1.25): declaring NO evidence
# is inconsistent with even a mild prior→posterior shift. Tune via backtest.
_DEFAULT_STUDENT_T_DF: float = 5.0
_NONE_EVIDENCE_LR_BOUND: float = math.log(1.25)  # custom: no-evidence must be tight
_WEAK_EVIDENCE_LR_BOUND: float = math.log(3.0)
_MODERATE_EVIDENCE_LR_BOUND: float = math.log(10.0)

# Ratio of claimed-family SSE to best-family SSE above which we flag. 2.0 is
# a mild threshold: declared family must fit materially worse than the best
# available fit before we raise a signal.
_FAMILY_SSE_FLAG_RATIO: float = 2.0


@dataclass(frozen=True)
class ConsistencyResult:
    flag: bool
    flag_reason: str | None
    details: dict


def _sse_for_fit(fit: FitType, declared: dict[float, float]) -> float:
    # Preserve prior heuristic: declare lognormal unfittable whenever *any*
    # declared value is non-positive, so the best-fit comparison rejects it
    # cleanly rather than treating x<=0 as "cdf=0 contributes (0-p)^2".
    if isinstance(fit, LognormalFit) and any(x <= 0 for x in declared.values()):
        return math.inf
    sse = 0.0
    for p, x in declared.items():
        cdf_val = eval_cdf(fit, x)
        sse += (cdf_val - p) ** 2
    return sse


def percentile_family_consistency(
    declared_percentiles: dict[float, float],
    claimed_family: FamilyLiteral | None,
    student_t_df: float | None = None,
) -> ConsistencyResult:
    """Compare declared percentiles against normal/lognormal/Student-t fits.

    ``student_t_df`` defaults to ``_DEFAULT_STUDENT_T_DF`` (5.0). Callers that
    thread through a forecaster's declared df (``NumericStructured.student_t_df``)
    get an apples-to-apples comparison with their chosen heavy-tailed shape.
    """
    if not declared_percentiles or len(declared_percentiles) < 2:
        raise ValueError("declared_percentiles must have at least 2 entries")
    for p in declared_percentiles.keys():
        if not (0.0 < float(p) < 1.0):
            raise ValueError(f"percentile keys must be in (0, 1) (got {p})")

    df = _DEFAULT_STUDENT_T_DF if student_t_df is None else float(student_t_df)
    if df <= 0:
        raise ValueError(f"student_t_df must be > 0 (got {df})")

    sse_by_family: dict[str, float] = {}
    fits_by_family: dict[str, FitType] = {}

    try:
        normal_fit = fit_normal_from_percentiles(declared_percentiles)
        fits_by_family["normal"] = normal_fit
        sse_by_family["normal"] = _sse_for_fit(normal_fit, declared_percentiles)
    except ValueError:
        sse_by_family["normal"] = math.inf

    all_positive = all(v > 0 for v in declared_percentiles.values())
    if all_positive:
        try:
            lognorm_fit = fit_lognormal_from_percentiles(declared_percentiles)
            fits_by_family["lognormal"] = lognorm_fit
            sse_by_family["lognormal"] = _sse_for_fit(lognorm_fit, declared_percentiles)
        except ValueError:
            sse_by_family["lognormal"] = math.inf
    else:
        sse_by_family["lognormal"] = math.inf

    try:
        t_fit = fit_student_t_from_percentiles(declared_percentiles, df=df)
        fits_by_family["student_t"] = t_fit
        sse_by_family["student_t"] = _sse_for_fit(t_fit, declared_percentiles)
    except ValueError:
        sse_by_family["student_t"] = math.inf

    best_family = min(sse_by_family.items(), key=lambda kv: kv[1])[0]

    details = {
        "best_fit_family": best_family,
        "sse_by_family": sse_by_family,
        "fits_by_family": fits_by_family,
        "claimed_family": claimed_family,
    }

    if claimed_family is None:
        return ConsistencyResult(flag=False, flag_reason=None, details=details)

    if claimed_family not in sse_by_family:
        # Families we cannot fit here (skew_normal, beta, mixture, other): be
        # conservative and don't flag rather than comparing apples to oranges.
        return ConsistencyResult(
            flag=False,
            flag_reason=None,
            details=details,
        )

    if claimed_family == best_family:
        return ConsistencyResult(flag=False, flag_reason=None, details=details)

    sse_claimed = sse_by_family[claimed_family]
    sse_best = sse_by_family[best_family]
    # best has SSE <= claimed; flag when claimed is materially worse.
    inverse_ratio = math.inf if sse_best == 0.0 else sse_claimed / sse_best
    details["sse_ratio_claimed_over_best"] = inverse_ratio

    if inverse_ratio > _FAMILY_SSE_FLAG_RATIO:
        reason = (
            f"claimed family {claimed_family!r} fits worse than {best_family!r} "
            f"(SSE {sse_claimed:.6g} vs {sse_best:.6g}, ratio {inverse_ratio:.2f})"
        )
        return ConsistencyResult(flag=True, flag_reason=reason, details=details)
    return ConsistencyResult(flag=False, flag_reason=None, details=details)


def stated_base_rate_consistency(
    stated_base_rate_prob: float,
    stated_posterior_prob: float,
    evidence_strength_max: EvidenceStrength,
) -> ConsistencyResult:
    """Flag when the log implied LR exceeds the Bayes-factor band for the
    declared evidence strength. Uses Jeffreys-style log-BF bands (custom
    tightening at 'none'). Uniform log-LR rule across all four strengths.

    When either probability is at the boundary (0 or 1) the LR is undefined;
    we return an unflagged result with ``details["implied_lr"] = None`` and
    a boundary reason string rather than raising.
    """
    # Runtime check guards against callers bypassing the EvidenceStrength
    # Literal (e.g., Python doesn't enforce Literal types at runtime).
    if evidence_strength_max not in ("strong", "moderate", "weak", "none"):
        raise ValueError(f"unknown evidence_strength_max: {evidence_strength_max}")

    try:
        lr = implied_likelihood_ratio(stated_base_rate_prob, stated_posterior_prob)
    except ValueError:
        # Degenerate prior/posterior (0 or 1): LR undefined. Don't raise —
        # return a non-flagged consistency result with an explicit reason so
        # the tool runner can still surface the probabilities.
        return ConsistencyResult(
            flag=False,
            flag_reason="degenerate prior/posterior at boundary (LR undefined)",
            details={
                "implied_lr": None,
                "evidence_strength_max": evidence_strength_max,
            },
        )
    log_lr = math.log(lr)

    details = {
        "implied_lr": lr,
        "evidence_strength_max": evidence_strength_max,
    }

    flag = False
    reason: str | None = None

    if evidence_strength_max == "none":
        if abs(log_lr) > _NONE_EVIDENCE_LR_BOUND:
            flag = True
            reason = f"evidence=none but implied LR={lr:.3f} exceeds |log(1.25)| bound"
    elif evidence_strength_max == "weak":
        if abs(log_lr) > _WEAK_EVIDENCE_LR_BOUND:
            flag = True
            reason = f"evidence=weak but implied LR={lr:.3f} exceeds |log(3)| bound"
    elif evidence_strength_max == "moderate":
        if abs(log_lr) > _MODERATE_EVIDENCE_LR_BOUND:
            flag = True
            reason = f"evidence=moderate but implied LR={lr:.3f} exceeds |log(10)| bound"
    # evidence=strong: never flag.

    return ConsistencyResult(flag=flag, flag_reason=reason, details=details)


def percentile_monotonicity_check(declared_percentiles: dict[float, float]) -> ConsistencyResult:
    if not declared_percentiles:
        raise ValueError("declared_percentiles must be non-empty")
    items = sorted(((float(k), float(v)) for k, v in declared_percentiles.items()), key=lambda x: x[0])

    first_violation: tuple[float, float, float] | None = None
    for i in range(1, len(items)):
        (_, prev_v) = items[i - 1]
        this_p, this_v = items[i]
        if not (this_v > prev_v):
            first_violation = (this_p, prev_v, this_v)
            break

    if first_violation is None:
        return ConsistencyResult(
            flag=False,
            flag_reason=None,
            details={"first_violation": None},
        )
    reason = (
        f"percentile {first_violation[0]} has value {first_violation[2]} "
        f"not strictly greater than previous value {first_violation[1]}"
    )
    return ConsistencyResult(
        flag=True,
        flag_reason=reason,
        details={"first_violation": first_violation},
    )
