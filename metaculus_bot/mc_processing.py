from __future__ import annotations

from collections.abc import Sequence

from forecasting_tools import PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

from metaculus_bot.constants import MC_PROB_MAX, MC_PROB_MIN
from metaculus_bot.simple_types import OptionProbability


def clamp_and_renormalize_probs(probabilities: Sequence[float]) -> list[float]:
    """Clamp probabilities into ``[MC_PROB_MIN, MC_PROB_MAX]`` and renormalize to sum 1.

    **In-bounds output is guaranteed only when ``n * MC_PROB_MIN < 1.0``** (n <= 100 at the
    0.01 floor). Above that no in-bounds sum-1 solution exists — 101 options each at least
    0.01 already exceed 1 — so the degenerate branch below returns a plain
    clamp-then-renormalize whose values sit BELOW ``MC_PROB_MIN`` (verified: 200 uniform
    options come back at 0.005 apiece; 100 uniform come back at exactly 0.01, so the
    boundary for the uniform case is n > 100, not n >= 100). ft's validator then moves
    them, which is the correct behavior — there is nothing better to return — but it is
    not the unconditional guarantee this docstring used to claim.

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
    n = len(probabilities)
    if n == 0:
        return []
    # Degenerate: the floor alone cannot fit under a unit sum (>= 100 options at a 0.01
    # floor). No in-bounds sum-1 solution exists; fall back to a single clamp+renorm,
    # which at least keeps sum ~= 1 for the upstream sum gate.
    if n * MC_PROB_MIN >= 1.0:
        clamped_degenerate = [max(MC_PROB_MIN, min(MC_PROB_MAX, p)) for p in probabilities]
        total_degenerate = sum(clamped_degenerate)
        return [p / total_degenerate for p in clamped_degenerate] if total_degenerate > 0 else [1.0 / n] * n

    probs = [max(MC_PROB_MIN, min(MC_PROB_MAX, p)) for p in probabilities]
    total = sum(probs)
    if total <= 0:
        return [1.0 / n] * n
    probs = [p / total for p in probs]

    pinned = [False] * n
    for _ in range(n):
        violators = [i for i in range(n) if not pinned[i] and not (MC_PROB_MIN <= probs[i] <= MC_PROB_MAX)]
        if not violators:
            break
        for i in violators:
            probs[i] = MC_PROB_MIN if probs[i] < MC_PROB_MIN else MC_PROB_MAX
            pinned[i] = True
        free = [i for i in range(n) if not pinned[i]]
        if not free:
            break
        budget = 1.0 - sum(probs[i] for i in range(n) if pinned[i])
        free_sum = sum(probs[i] for i in free)
        if budget <= 0:
            for i in free:
                probs[i] = MC_PROB_MIN
        elif free_sum > 0:
            scale = budget / free_sum
            for i in free:
                probs[i] *= scale
        else:
            share = budget / len(free)
            for i in free:
                probs[i] = share
    return probs


def _normalize_name(name: str) -> str:
    # Trim common prefixes like "Option X:" while preserving canonical names when matching
    stripped = name.strip()
    # Remove leading "Option" labels if present
    lowered = stripped.lower()
    if lowered.startswith("option ") or lowered.startswith("option:"):
        # drop leading token up to colon/space
        parts = stripped.split(":", 1)
        if len(parts) == 2:
            return parts[1].strip().lower()
        # fallback: remove first word
        return " ".join(stripped.split(" ")[1:]).strip().lower()
    return stripped.lower()


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
    """
    # Map normalized allowed names to canonical. Both sides must normalize the
    # SAME way: incoming items run through _normalize_name (which strips a leading
    # "Option " token), so the allowed lookup must too — otherwise a canonical
    # option literally named "Option A" is keyed "option a" here but arrives as
    # "a", never matches, and gets silently dropped into the even-distribution
    # fallback (a fabricated uniform forecast). Keying by _normalize_name makes
    # the match symmetric.
    allowed_norm_to_canonical = {_normalize_name(opt): opt for opt in allowed_options}

    # Aggregate by canonical option name
    accum: dict[str, float] = {}
    for item in raw_options:
        norm = _normalize_name(item.option_name)
        if norm in allowed_norm_to_canonical:
            canonical = allowed_norm_to_canonical[norm]
            accum[canonical] = accum.get(canonical, 0.0) + float(item.probability)

    # Create list in allowed order, skipping truly missing options
    pairs: list[tuple[str, float]] = [(name, accum[name]) for name in allowed_options if name in accum]

    # If everything was filtered out, fall back to an even distribution over allowed options
    if not pairs and allowed_options:
        even = 1.0 / len(allowed_options)
        pairs = [(name, even) for name in allowed_options]

    # Clamp + renormalize the floats BEFORE construction so ft 0.2.92's validator
    # (which clamps to [0.01, 0.99] + renormalizes + raises on any >0.05 move) sees
    # already-in-bounds, sum-1 values and is a no-op.
    clamped = clamp_and_renormalize_probs([p for _, p in pairs])
    return PredictedOptionList(
        predicted_options=[PredictedOption(option_name=n, probability=p) for (n, _), p in zip(pairs, clamped)]
    )


__all__ = ["build_mc_prediction", "clamp_and_renormalize_probs"]
