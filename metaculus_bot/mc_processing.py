from __future__ import annotations

from collections.abc import Sequence

from forecasting_tools import PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

from metaculus_bot.constants import MC_PROB_MAX, MC_PROB_MIN
from metaculus_bot.simple_types import OptionProbability

# The zero-mass branches below are unreachable while ``MC_PROB_MIN > 0`` (the clamp puts
# a positive floor under every option before any total is summed). They raise rather than
# return ``[1/n] * n``: a uniform ballot is a FORECAST, and manufacturing one from a
# vector carrying no probability mass at all is the defect class this module was audited
# for. The day ``MC_PROB_MIN`` is set to 0 they fire loudly instead of publishing 1/n.
_ZERO_MASS_MESSAGE = (
    "MC option probabilities carry no mass across {n} options after clamping; refusing to impute a uniform ballot"
)

# Slack on the ``n * lo > 1.0`` degenerate test so an exactly feasible ballot never trips it on rounding.
FLOOR_FEASIBILITY_ATOL: float = 1e-12


def clamp_and_renormalize_probs(
    probabilities: Sequence[float],
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> list[float]:
    """Clamp probabilities into ``[lo, hi]`` and renormalize to sum 1.

    ``lo``/``hi`` default to the live ``MC_PROB_MIN``/``MC_PROB_MAX``, so every pipeline
    caller behaves exactly as it did before the kwargs existed. They are read through a
    ``None`` sentinel rather than as literal parameter defaults on purpose: a default value
    binds at function-definition time, which would silently break the monkeypatch surface
    the suite uses to pin the floor-is-zero behaviour, and would freeze the clamp against a
    later change to the constant. The kwargs exist for the offline clip-threshold sweep
    (``performance_analysis.clip_threshold``), which reprices archived ballots at candidate
    floors; nothing in the live pipeline passes them.

    **In-bounds output is guaranteed only when ``n * lo <= 1.0``** (n <= 100 at the 0.01
    floor; n <= 10 at a 0.10 sweep floor). Above that no in-bounds sum-1 solution exists —
    101 options each at least 0.01 already exceed 1 — so the degenerate branch below returns
    a plain clamp-then-renormalize whose values sit BELOW ``lo`` (verified: 200 uniform
    options come back at 0.005 apiece). AT the boundary (``n * lo == 1.0``) the unique
    in-bounds solution is the uniform vector and the normal path converges to it; the guard
    is a strict comparison for that reason. It used to be ``>=``, which sent an exactly
    feasible ballot down the degenerate branch and returned sub-floor values: dead at the
    live 0.01 floor (a 100-option ballot never happens) but live for the sweep, where
    ``10 * 0.10 == 1.0`` exactly and the archive holds 10-option ballots. The one
    behaviour delta from that change is at n = 100 with a concentrated ballot, which now
    returns the uniform 0.01 rather than a sub-floor vector ft's validator would have
    rejected. The ceiling side has the mirror-image precondition, ``n * hi >= 1.0`` and
    ``lo < hi``; neither is validated (no caller can pass a violating pair — the live
    pipeline passes no kwargs and the sweep passes ``lo = c <= 0.10`` with ``hi = 1 - c``),
    and a violating pair returns a clamped vector that does not sum to 1 rather than
    raising. ft's validator then moves sub-floor values, which is the correct behavior —
    there is nothing better to return — but it is not the unconditional guarantee this
    docstring used to claim.

    Callers construct ``PredictedOptionList`` from the result; ft 0.2.92's validator
    re-clamps to ``[0.01, 0.99]`` + renormalizes + raises when any option moves > 0.05,
    so feeding it already-in-bounds, sum-1 values makes that validator a no-op. That
    no-op property likewise holds only under the feasibility condition above: on a
    high-cardinality ballot with a DOMINANT option (200 options with one at 0.9; 120 with
    0.5/0.3) ft's re-clamp moves the dominant option by more than 0.05 and raises
    ``ValidationError`` at construction. Uniform high-cardinality is fine even above the
    boundary — ft clamps 0.005 up to 0.01 and renormalizes straight back to 0.005, a zero
    net move — so the publish-time loss needs high cardinality AND concentration. It is
    not reachable by relaxing our floor: ft clamps to its OWN hard-coded 0.01 regardless of
    what we send, so a lower floor here changes nothing about its move (measured — for
    n=200 ft accepts a top option up to ~0.06 and no floor choice lifts that). Metaculus MC
    ballots are far below this cardinality in practice; see
    tests/test_ft_pin_mc_high_cardinality.py for the pinned shapes.

    The naive clamp-then-divide can push a clamped option back out of bounds: when a
    dominant option keeps the post-clamp total above 1, dividing drags the floored
    siblings below ``MC_PROB_MIN`` again (e.g. ``0.984 + 8x0.002`` -> the eight floor to
    0.01 but the >1 total divides them back under it). We keep the cheap clamp-then-divide
    for the common in-budget case, then repair any residual violation by pinning the
    offending options at their bound and rescaling only the still-free mass, iterating
    until the invariant holds.
    """
    floor = MC_PROB_MIN if lo is None else lo
    ceiling = MC_PROB_MAX if hi is None else hi
    n = len(probabilities)
    if n == 0:
        return []
    # Degenerate (> 100 options at a 0.01 floor): no in-bounds sum-1 solution, so keep sum ~= 1.
    if n * floor > 1.0 + FLOOR_FEASIBILITY_ATOL:
        # floor > 0 here, so the total is positive by construction (it used to return [1/n] * n).
        clamped_degenerate = [max(floor, min(ceiling, p)) for p in probabilities]
        total_degenerate = sum(clamped_degenerate)
        return [p / total_degenerate for p in clamped_degenerate]

    probs = [max(floor, min(ceiling, p)) for p in probabilities]
    total = sum(probs)
    if total <= 0:
        raise ValueError(_ZERO_MASS_MESSAGE.format(n=n))
    probs = [p / total for p in probs]

    return _repair_bound_violations(probs, floor, ceiling)


def _repair_bound_violations(probs: list[float], floor: float, ceiling: float) -> list[float]:
    """Pin out-of-bounds options at their bound and rescale only the still-free mass.

    Iterates because pinning changes the free budget, which can push a previously in-bounds
    option out. Mutates and returns ``probs``; the caller has already clamped + renormalized
    once, so this only fires when the naive divide dragged a floored option back under
    ``floor``. The bounds arrive as arguments rather than being read off the module globals
    so a caller sweeping candidate floors gets the same repair the live clamp gets.
    """
    n = len(probs)
    pinned = [False] * n
    for _ in range(n):
        violators = [i for i in range(n) if not pinned[i] and not (floor <= probs[i] <= ceiling)]
        if not violators:
            break
        for i in violators:
            probs[i] = floor if probs[i] < floor else ceiling
            pinned[i] = True
        free = [i for i in range(n) if not pinned[i]]
        if not free:
            break
        budget = 1.0 - sum(probs[i] for i in range(n) if pinned[i])
        free_sum = sum(probs[i] for i in free)
        if budget <= 0:
            for i in free:
                probs[i] = floor
        elif free_sum > 0:
            scale = budget / free_sum
            for i in free:
                probs[i] *= scale
        else:
            # Free mass to distribute but every free option is exactly 0 — impossible
            # while the floor is > 0. Splitting the budget evenly here would invent a
            # sub-ballot; see _ZERO_MASS_MESSAGE.
            raise ValueError(_ZERO_MASS_MESSAGE.format(n=n))
    return probs


def _normalize_name(name: str) -> str:
    # Trim common prefixes like "Option X:" while preserving canonical names when matching
    stripped = name.strip()
    # Remove leading "Option" labels if present
    lowered = stripped.lower()
    if lowered.startswith(("option ", "option:")):
        # drop leading token up to colon/space
        parts = stripped.split(":", 1)
        if len(parts) == 2:
            return parts[1].strip().lower()
        # fallback: remove first word
        return " ".join(stripped.split(" ")[1:]).strip().lower()
    return stripped.lower()


def accumulate_declared_option_probs(
    raw_options: Sequence[OptionProbability],
    allowed_options: Sequence[str],
) -> list[tuple[str, float]]:
    """``(canonical_option, summed_probability)`` pairs in ``allowed_options`` order, UNCLAMPED.

    The one place loosely parsed MC options are matched onto a question's option set:
    names are matched through ``_normalize_name`` on BOTH sides (it strips a leading
    "Option " token, so a canonical option literally named "Option A" must be keyed the
    same way its parsed spelling arrives, or it never matches), duplicates are summed, and
    unmatched options are skipped. ``build_mc_prediction`` clamps these pairs into a
    constructible ``PredictedOptionList``; the extraction ladder reads the same pairs for
    the MEMBER_FORECAST marker's pre-clamp ``raw`` vector, so the two line up index for
    index by construction rather than by parallel code.
    """
    allowed_norm_to_canonical = {_normalize_name(opt): opt for opt in allowed_options}
    accum: dict[str, float] = {}
    for item in raw_options:
        canonical = allowed_norm_to_canonical.get(_normalize_name(item.option_name))
        if canonical is not None:
            accum[canonical] = accum.get(canonical, 0.0) + float(item.probability)
    return [(name, accum[name]) for name in allowed_options if name in accum]


def build_mc_prediction(
    raw_options: Sequence[OptionProbability],
    allowed_options: Sequence[str],
) -> PredictedOptionList:
    """Convert loosely parsed MC options to a strict PredictedOptionList.

    - Filters to allowed option names (case-insensitive match against provided canonicals).
    - Aggregates duplicates by summing probabilities.
    - Clamps to [MC_PROB_MIN, MC_PROB_MAX] and renormalizes to sum 1.0 BEFORE
      constructing the PredictedOptionList, so ft 0.2.92's clamp-and-renormalize
      validator is a no-op on construction (no publish-time ValueError).
    - Preserves the order of `allowed_options` in the final list.
    - **Raises when NOTHING matched.** It used to return an even distribution over
      the allowed options instead, which is a forecast the model never made: the
      extraction ladder's rung-3 parser validates an EMPTY option list, so a
      salvage over a rationale with no ballot in it produced exact 1/n on every
      option, and that uniform passes every downstream check (full option set,
      in bounds, sums to 1) — it clamped as a no-op, went into the ensemble
      median, and published with only ``rung=llm`` (which legitimate salvages
      also emit) to distinguish it. Raising instead sends the ladder to its
      typed failure, so the forecaster is dropped and attributed like any other
      extraction failure.
    """
    pairs = accumulate_declared_option_probs(raw_options, allowed_options)

    if not pairs:
        raise ValueError(
            f"no parsed option matched the question's options {list(allowed_options)} "
            f"(parsed names: {[item.option_name for item in raw_options]}); refusing to impute a uniform ballot"
        )

    # Clamp + renormalize the floats BEFORE construction so ft 0.2.92's validator
    # (which clamps to [0.01, 0.99] + renormalizes + raises on any >0.05 move) sees
    # already-in-bounds, sum-1 values and is a no-op.
    clamped = clamp_and_renormalize_probs([p for _, p in pairs])
    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name=n, probability=p) for (n, _), p in zip(pairs, clamped, strict=True)
        ]
    )


__all__ = ["accumulate_declared_option_probs", "build_mc_prediction", "clamp_and_renormalize_probs"]
